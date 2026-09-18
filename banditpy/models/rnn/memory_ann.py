import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import core
from ._mle_fit_base import RNNFit2ArmBase


class MemoryANNModel(nn.Module):
    """
    Literal port of the 'Memory-ANN' architecture from Eckstein et al. (2026,
    Nature Human Behaviour), Fig. 3c. Compare with 'VanillaRNNModel'
    (rnn_fit.py), the unconstrained baseline it sits below.

    Two separate recurrent modules, each blind to the other's input, matching
    the paper's stated constraints ('reward processing does not have access
    to past or present actions and vice versa for action processing'):

      reward module : hidden state s^(r), input r_t only
      action module : hidden state s^(a), input onehot(a_t) only

    At each trial, each module's small feedforward readout ('reward ANN' /
    'action ANN' in the paper) maps its own (input, hidden state) onto a new
    value for the *chosen* action only:

        s_t^(r) = reward_cell(r_t, s_{t-1}^(r))
        Q_{t+1}(a_t) = reward_readout(r_t, s_t^(r))

        s_t^(a) = action_cell(onehot(a_t), s_{t-1}^(a))
        c_{t+1}(a_t) = action_readout(onehot(a_t), s_t^(a))

    'the same update applies regardless of which action is being updated'
    (i.e. one shared readout, not per-action parameters), and 'the values of
    all unchosen actions decay strictly exponentially' toward learnable
    reference points 'q_init' / 'c_init' — this is a fixed decay, not
    something the readout networks produce:

        Q_{t+1}(a') = q_init + decay_q * (Q_t(a') - q_init),  a' != a_t
        c_{t+1}(a') = c_init + decay_c * (c_t(a') - c_init),  a' != a_t

    Choice logits are 'the outputs of reward and action processing combined
    by simple addition': logits_t = Q_t + c_t, fed straight into
    'F.cross_entropy' (no separate temperature — same convention as
    'VanillaRNNModel').

    Notably absent: any hand-imposed sigmoid on the reward readout. The
    paper's sigmoidal Q(r) mapping (their Fig. 3f) was an *empirical finding*
    from probing the trained network, not an architectural constraint, so
    'reward_readout' here is a free small MLP and should discover that shape
    itself if the data supports it — probing it post-hoc (sweep r_t at fixed
    s^(r), color by PC1 of s^(r)) is the natural next step once this is fit,
    to check whether the same mechanism re-emerges in your task.

    'init_state' / 'step' expose the per-trial update as a single-step API —
    'forward' just loops 'step' over a segment, and 'MemoryANNFit2Arm.
    simulate_posterior_predictive' reuses the same 'step' for its
    trial-by-trial rollout, so the update rule is written in exactly one
    place.
    """

    def __init__(
        self,
        num_actions: int = 2,
        hidden_size_r: int = 8,
        hidden_size_a: int = 8,
        mlp_hidden: int = 16,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_size_r = hidden_size_r
        self.hidden_size_a = hidden_size_a

        self.reward_cell = nn.RNNCell(1, hidden_size_r)
        self.reward_readout = nn.Sequential(
            nn.Linear(1 + hidden_size_r, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, 1),
        )

        self.action_cell = nn.RNNCell(num_actions, hidden_size_a)
        self.action_readout = nn.Sequential(
            nn.Linear(num_actions + hidden_size_a, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, 1),
        )

        self.q_init = nn.Parameter(torch.zeros(()))
        self.c_init = nn.Parameter(torch.zeros(()))
        self.decay_q_raw = nn.Parameter(torch.zeros(()))
        self.decay_c_raw = nn.Parameter(torch.zeros(()))

        self._init_weights()

    def _init_weights(self):
        for cell in (self.reward_cell, self.action_cell):
            for name, param in cell.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)
        for readout in (self.reward_readout, self.action_readout):
            for layer in readout:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight.data)
                    nn.init.constant_(layer.bias.data, 0)

    def init_state(self, device):
        """(s_r, s_a, Q, c) at the start of a segment (before any trial)."""
        A = self.num_actions
        s_r = torch.zeros(1, self.hidden_size_r, device=device)
        s_a = torch.zeros(1, self.hidden_size_a, device=device)
        Q = self.q_init * torch.ones(A, device=device)
        c = self.c_init * torch.ones(A, device=device)
        return s_r, s_a, Q, c

    def step(self, state, chosen_idx, reward):
        """Advance state by one trial's own outcome (chosen_idx, reward).

        Parameters
        ----------
        state : (s_r, s_a, Q, c) as returned by 'init_state' or a prior 'step'.
        chosen_idx : 0-indexed action chosen this trial (int or 0-d LongTensor).
        reward : reward received this trial (float or 0-d/1x1 FloatTensor).

        Returns
        -------
        (s_r, s_a, Q, c) — the state to use for predicting the *next* trial.
        """
        s_r, s_a, Q, c = state
        A = self.num_actions
        device = Q.device
        action_range = torch.arange(A, device=device)

        decay_q = torch.sigmoid(self.decay_q_raw)
        decay_c = torch.sigmoid(self.decay_c_raw)

        if torch.is_tensor(reward):
            r_t = reward.view(1, 1).float()
        else:
            r_t = torch.tensor([[float(reward)]], device=device)
        s_r = self.reward_cell(r_t, s_r)
        val_r = self.reward_readout(torch.cat([r_t, s_r], dim=-1)).squeeze()

        chosen_idx_t = (
            chosen_idx
            if torch.is_tensor(chosen_idx)
            else torch.tensor(chosen_idx, device=device)
        )
        a_onehot = F.one_hot(chosen_idx_t, num_classes=A).float().view(1, -1)
        s_a = self.action_cell(a_onehot, s_a)
        val_a = self.action_readout(torch.cat([a_onehot, s_a], dim=-1)).squeeze()

        chosen_mask = (action_range == chosen_idx_t).float()
        decay_target_q = self.q_init + decay_q * (Q - self.q_init)
        decay_target_c = self.c_init + decay_c * (c - self.c_init)
        Q = chosen_mask * val_r + (1 - chosen_mask) * decay_target_q
        c = chosen_mask * val_a + (1 - chosen_mask) * decay_target_c

        return (s_r, s_a, Q, c)

    def forward(self, choices_idx: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        choices_idx : LongTensor, shape (T,)
            0-indexed chosen action on every trial of the segment.
        rewards : FloatTensor, shape (T,)
            Reward received on every trial of the segment.

        Returns
        -------
        logits : FloatTensor, shape (T, num_actions)
            logits[t] predicts choices_idx[t] using only trials < t (i.e. the
            state going into trial t) — same causal convention as
            'VanillaRNNModel' ('l_t = [a_{t-1}, r_{t-1}]').
        """
        T = choices_idx.shape[0]
        device = choices_idx.device
        state = self.init_state(device)

        logits_list = []
        for t in range(T):
            _, _, Q, c = state
            logits_list.append((Q + c).unsqueeze(0))
            state = self.step(state, choices_idx[t], rewards[t])

        return torch.cat(logits_list, dim=0)


class MemoryANNFit2Arm(RNNFit2ArmBase):
    """Fit 'MemoryANNModel' to observed choices via maximum likelihood.

    Shared fitting/evaluation/cross-validation machinery lives in
    'RNNFit2ArmBase' (see '_mle_fit_base.py') — this class only supplies the
    model and the segment/forward-pass glue, same as 'VanillaRNNFit2Arm'
    (rnn_fit.py). Compare 'nll_per_trial' between the two on the same task:
    the gap (if any) is the cost of Memory-ANN's extra architectural
    constraints, exactly as Fig. 3d in the paper compares Memory-ANN against
    Vanilla RNN.

    Examples
    --------
    >>> fit = MemoryANNFit2Arm(task, hidden_size_r=8, hidden_size_a=8)
    >>> fit.fit(n_epochs=500)
    >>> proba = fit.predict_proba()      # shape (n_trials, n_actions)
    >>> nll   = fit.nll_per_trial        # scalar
    """

    def __init__(
        self,
        task: core.Bandit2Arm,
        hidden_size_r: int = 8,
        hidden_size_a: int = 8,
        mlp_hidden: int = 16,
        segment_starts="session",
        device=None,
    ):
        model = MemoryANNModel(
            num_actions=task.n_ports,
            hidden_size_r=hidden_size_r,
            hidden_size_a=hidden_size_a,
            mlp_hidden=mlp_hidden,
        )
        super().__init__(
            task,
            model,
            init_kwargs={
                "hidden_size_r": hidden_size_r,
                "hidden_size_a": hidden_size_a,
                "mlp_hidden": mlp_hidden,
            },
            segment_starts=segment_starts,
            device=device,
        )

    # ------------------------------------------------------------------

    def _segment_tensors(self, seg_choices, seg_rewards):
        choices_idx = torch.tensor(
            seg_choices - 1, dtype=torch.long, device=self.device
        )
        rewards_t = torch.tensor(seg_rewards.astype(np.float32), device=self.device)
        return choices_idx, rewards_t

    def _forward_segment(self, segment):
        choices_idx, rewards = segment
        logits = self.model(choices_idx, rewards)  # (T, n_actions)
        return logits, choices_idx

    # ------------------------------------------------------------------

    def simulate_posterior_predictive(self, seed: int = None):
        """Run the fitted model autoregressively through the actual task structure.

        Mirrors 'VanillaRNNFit2Arm.simulate_posterior_predictive': choices are
        sampled from the model's policy at each trial; rewards are drawn from
        'self.task.probs'. Hidden states reset at every segment boundary.
        Reuses 'MemoryANNModel.step' for the trial-by-trial state update.

        Returns
        -------
        Bandit2Arm
        """
        rng = np.random.default_rng(seed)
        task = self.task
        n_trials = task.n_trials
        A = self.n_ports

        choices_sim = np.zeros(n_trials, dtype=int)
        rewards_sim = np.zeros(n_trials, dtype=int)

        self.model.eval()
        m = self.model

        with torch.no_grad():
            state = m.init_state(self.device)
            for t in range(n_trials):
                if self._seg_mask[t]:
                    state = m.init_state(self.device)

                _, _, Q, c = state
                probs = F.softmax(Q + c, dim=-1).cpu().numpy()
                choice0 = rng.choice(A, p=probs)  # 0-indexed
                reward = int(rng.random() < task.probs[t, choice0])

                choices_sim[t] = choice0 + 1
                rewards_sim[t] = reward

                chosen_idx = torch.tensor(choice0, device=self.device)
                state = m.step(state, chosen_idx, float(reward))

        return core.Bandit2Arm(
            probs=task.probs.copy(),
            choices=choices_sim,
            rewards=rewards_sim,
            session_ids=task.session_ids.copy(),
            block_ids=None if task.block_ids is None else task.block_ids.copy(),
            window_ids=None if task.window_ids is None else task.window_ids.copy(),
            starts=None if task.starts is None else task.starts.copy(),
            stops=None if task.stops is None else task.stops.copy(),
            datetime=None if task.datetime is None else task.datetime.copy(),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, extra: dict = None):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "num_actions": self.model.num_actions,
            "hidden_size_r": self.model.hidden_size_r,
            "hidden_size_a": self.model.hidden_size_a,
            **self._common_checkpoint_fields(extra),
        }
        torch.save(checkpoint, path)
        print(f"Saved to {path}")

    @staticmethod
    def load(path: str, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        choices = checkpoint["choices"]
        probs = checkpoint.get(
            "probs",
            np.zeros((len(choices), checkpoint["num_actions"]), dtype=np.float32),
        )
        task = core.Bandit2Arm(
            probs=probs,
            choices=choices,
            rewards=checkpoint["rewards"],
            session_ids=checkpoint["session_ids"],
            block_ids=checkpoint.get("block_ids"),
            window_ids=checkpoint.get("window_ids"),
        )

        fitter = MemoryANNFit2Arm(
            task=task,
            hidden_size_r=checkpoint["hidden_size_r"],
            hidden_size_a=checkpoint["hidden_size_a"],
            segment_starts=checkpoint["seg_mask"],
            device=device,
        )
        fitter.model.load_state_dict(checkpoint["model_state_dict"])
        fitter.model.eval()
        fitter.nll_history = checkpoint.get("nll_history", [])
        fitter._loaded_nll_per_trial = checkpoint.get("nll_per_trial", None)
        fitter._loaded_predict_proba = checkpoint.get("predict_proba", None)
        fitter.extra = checkpoint.get("extra", {})
        return fitter

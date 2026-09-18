import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import core
from ..model import _logit_dynamics_dataframe, _plot_logit_dynamics_ax
from ._mle_fit_base import RNNFit2ArmBase


class VanillaRNNModel(nn.Module):
    """Vanilla RNN for the two-armed bandit task.

    Architecture (Findling et al.):

        l_t  = [ a_{t-1} (one-hot),  r_{t-1} (scalar) ]

        s_t  = tanh( W_1 · l_t  +  W_hh · s_{t-1}  +  b_1 )   # recurrent hidden state

        h_t  = W_2 · s_t  +  b_2                                 # choice logits

        p_t  = softmax( h_t )                                     # action probabilities

    Input size = n_actions + 1  (one-hot action + reward scalar).
    For a 2-arm task: input_size = 3.
    """

    def __init__(self, input_size, hidden_size, num_actions):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)

        scale = 1.0 / math.sqrt(self.hidden_size)
        for head in (self.policy_head, self.value_head):
            nn.init.normal_(head.weight.data, mean=0.0, std=scale)
            nn.init.constant_(head.bias.data, 0)

    def forward(self, x, hidden_state=None):
        """
        Args:
            x            : (batch, seq_len, input_size)
            hidden_state : (1, batch, hidden_size) or None

        Returns:
            policy_logits : (batch, seq_len, num_actions)
            value         : (batch, seq_len)
            hidden_state  : (1, batch, hidden_size)
        """
        rnn_out, hidden_state = self.rnn(x, hidden_state)
        policy_logits = self.policy_head(rnn_out)
        value = self.value_head(rnn_out).squeeze(-1)
        return policy_logits, value, hidden_state


class VanillaRNNFit2Arm(RNNFit2ArmBase):
    """Fit a Vanilla RNN to observed animal choices via maximum likelihood.

    The model predicts the animal's next choice at each trial from the previous
    choice and reward:

        l_t  = [ a_{t-1} (one-hot),  r_{t-1} (scalar) ]        # l_0 = zeros
        s_t  = tanh( W_1 · l_t  +  W_hh · s_{t-1}  +  b_1 )
        h_t  = W_2 · s_t  +  b_2
        p_t  = softmax( h_t )

    Parameters are optimized to minimize the negative log-likelihood (NLL) of
    the observed choice sequence.  The hidden state resets to zero at the start
    of each segment (session / window / block, or a custom boolean array).

    Shared fitting/evaluation/cross-validation machinery lives in
    'RNNFit2ArmBase' (see '_mle_fit_base.py') — this class only supplies the
    model and the segment/forward-pass glue.

    Parameters
    ----------
    task : core.Bandit2Arm
    hidden_size : int
    segment_starts : str or array-like of bool
        When to reset the hidden state.  Same convention as
        ``CompressibilityRatio2Arm.compute()``:
        ``"session"`` (default), ``"window"``, ``"block"``, or a boolean array
        of length ``n_trials`` with ``True`` at segment-start positions.

    Examples
    --------
    >>> fit = VanillaRNNFit2Arm(task, hidden_size=48)
    >>> fit.fit(n_epochs=500)
    >>> proba = fit.predict_proba()      # shape (n_trials, n_actions)
    >>> nll   = fit.nll_per_trial        # scalar
    """

    def __init__(
        self,
        task: core.Bandit2Arm,
        hidden_size: int = 48,
        segment_starts="session",
        device=None,
    ):
        model = VanillaRNNModel(
            input_size=task.n_ports + 1,  # one-hot action + scalar reward
            hidden_size=hidden_size,
            num_actions=task.n_ports,
        )
        super().__init__(
            task,
            model,
            init_kwargs={"hidden_size": hidden_size},
            segment_starts=segment_starts,
            device=device,
        )

    # ------------------------------------------------------------------

    def _segment_tensors(self, seg_choices, seg_rewards):
        """(x_seq, y_seq): x_seq = [one_hot(a_{t-1}), r_{t-1}] (l_0 = zeros)."""
        n_ports = self.n_ports
        seg_len = len(seg_choices)

        x = np.zeros((seg_len, n_ports + 1), dtype=np.float32)
        for t in range(1, seg_len):
            x[t, seg_choices[t - 1] - 1] = 1.0  # one-hot (0-indexed)
            x[t, n_ports] = float(seg_rewards[t - 1])

        y = (seg_choices - 1).astype(np.int64)  # 0-indexed targets

        x_t = torch.tensor(x, device=self.device).unsqueeze(0)  # (1, T, D)
        y_t = torch.tensor(y, device=self.device)  # (T,)
        return x_t, y_t

    def _forward_segment(self, segment):
        x_seq, y_seq = segment
        logits, _, _ = self.model(x_seq)  # (1, T, n_actions)
        return logits.squeeze(0), y_seq

    # ------------------------------------------------------------------
    # Vanilla-RNN-specific analyses (no Memory-ANN equivalent yet)
    # ------------------------------------------------------------------

    def compute_logit_dynamics(
        self,
        n_bins: int = 15,
        logit_range=(-5.0, 5.0),
        min_trials_per_bin: int = 5,
    ) -> pd.DataFrame:
        """Conditional "logit-change" dynamics, as in Li et al., *Discovering
        cognitive strategies with tiny recurrent neural networks*.

        For every trial ``t`` define the model's preference for action 1 as
        ``logit_t = log(p1_t / p2_t)`` (teacher-forced on the real observed
        history, i.e. ``predict_proba()``).  The one-step update

            ``logit_change_t = logit_{t+1} - logit_t``

        is grouped by the action actually taken and the reward actually
        received at trial ``t`` (4 conditions: A1/R0, A1/R1, A2/R0, A2/R1)
        and averaged within bins of ``logit_t``.  Transitions that cross a
        segment boundary (hidden state reset) are excluded since ``logit_{t+1}``
        would not be a genuine continuation of trial ``t``.

        Parameters
        ----------
        n_bins : int
            Number of bins spanning ``logit_range``.
        logit_range : (float, float)
            Lower/upper bound of the logit axis.
        min_trials_per_bin : int
            Bins with fewer than this many trials are dropped (avoids noisy
            single-trial estimates, especially at extreme logit values that
            are rarely visited in real behavior).

        Returns
        -------
        pd.DataFrame
            Columns: ``bin_center``, ``action`` (1 or 2), ``reward`` (0 or 1),
            ``mean_change``, ``sem_change``, ``n``.

        Notes
        -----
        Real behavior often under-samples extreme logit values.  For denser,
        smoother curves, fit a fresh ``VanillaRNNFit2Arm`` on a simulated task
        (e.g. ``fit.simulate(reward_schedule=...)``), load this model's state
        dict into it, and call ``compute_logit_dynamics()`` on that instead.
        """
        return _logit_dynamics_dataframe(
            self.predict_proba(),
            self.task.choices,
            self.task.rewards,
            self._seg_mask,
            n_bins=n_bins,
            logit_range=logit_range,
            min_trials_per_bin=min_trials_per_bin,
        )

    def plot_logit_dynamics(self, ax=None, n_bins: int = 15, logit_range=(-5.0, 5.0)):
        """Plot the conditional logit-change dynamics (see ``compute_logit_dynamics``).

        Reproduces the style of Li et al.'s dynamical-portrait figure: one
        line per (action, reward) condition, colored by action and shaded by
        reward outcome.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        n_bins, logit_range
            Passed to ``compute_logit_dynamics``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        df = self.compute_logit_dynamics(n_bins=n_bins, logit_range=logit_range)
        return _plot_logit_dynamics_ax(df, ax=ax)

    def simulate_posterior_predictive(
        self, seed: int = None, return_hidden: bool = False
    ):
        """Run the fitted RNN autoregressively through the actual task structure.

        Mirrors ``DecisionModel.simulate_posterior_predictive``: choices are
        sampled from the model's policy at each trial; rewards are drawn from
        ``self.task.probs``.  The hidden state resets at every segment boundary
        (same ``_seg_mask`` used during fitting).

        Parameters
        ----------
        seed : int, optional
            RNG seed for reproducibility.
        return_hidden : bool
            If True, also return the hidden-state trajectory.

        Returns
        -------
        Bandit2Arm
            Simulated task with the same structure (probs, session/block/window
            IDs, timestamps) as the original but with model-generated choices
            and rewards.
        np.ndarray, shape (n_trials, hidden_size)
            Hidden state at each trial (only when ``return_hidden=True``).
        """
        rng = np.random.default_rng(seed)
        task = self.task
        n_trials = task.n_trials
        n_ports = self.n_ports
        input_size = n_ports + 1

        choices_sim = np.zeros(n_trials, dtype=int)
        rewards_sim = np.zeros(n_trials, dtype=int)
        if return_hidden:
            hidden_traj = np.zeros((n_trials, self.model.hidden_size))

        self.model.eval()
        hidden = None
        prev_choice = 0  # 0 = no previous trial (1-indexed choices)
        prev_reward = 0.0

        with torch.no_grad():
            for t in range(n_trials):
                if self._seg_mask[t]:
                    hidden = None
                    prev_choice = 0
                    prev_reward = 0.0

                # l_t = [one-hot(a_{t-1}), r_{t-1}]; l_0 = zeros
                x = torch.zeros(1, 1, input_size, device=self.device)
                if prev_choice > 0:
                    x[0, 0, prev_choice - 1] = 1.0  # one-hot (0-indexed)
                    x[0, 0, n_ports] = prev_reward

                logits, _, hidden = self.model(x, hidden)  # (1,1,n_actions)
                probs = F.softmax(logits[0, 0], dim=-1)  # (n_actions,)
                choice = torch.multinomial(probs, num_samples=1).item() + 1  # 1-indexed

                reward = int(rng.random() < task.probs[t, choice - 1])

                choices_sim[t] = choice
                rewards_sim[t] = reward
                if return_hidden:
                    hidden_traj[t] = hidden[0, 0].cpu().numpy()

                prev_choice = choice
                prev_reward = float(reward)

        sim_task = core.Bandit2Arm(
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
        if return_hidden:
            return sim_task, hidden_traj
        return sim_task

    def simulate(
        self,
        reward_schedule,
        min_trials_per_block: int = 100,
        prob_switch: float = 0.02,
        max_trials_per_block: int = 500,
        n_block_min: int = 4,
        n_block_max: int = 8,
        seed: int = None,
        return_hidden: bool = True,
    ):
        """Simulate the fitted RNN on a new reward schedule.

        The hidden state resets at window boundaries (random lengths drawn
        from ``[n_block_min, n_block_max]`` sessions), matching the convention
        used during training.

        Parameters
        ----------
        reward_schedule : array-like, shape (N, 2)
            Reward probabilities per arm for each of N sessions.
        min_trials_per_block : int
            Minimum trials before a session can end.
        prob_switch : float
            Per-trial probability of ending the session after
            ``min_trials_per_block``.
        max_trials_per_block : int
            Hard cap on session length.
        n_block_min, n_block_max : int
            Range of window lengths (in sessions) for hidden-state resets.
        seed : int, optional
            RNG seed for reproducibility.
        return_hidden : bool
            If True (default), also return the hidden-state trajectory.

        Returns
        -------
        Bandit2Arm
            Simulated task with probs, choices, rewards, and session/block/
            window IDs.
        np.ndarray, shape (n_trials, hidden_size)
            Hidden state at each trial (only when ``return_hidden=True``).
        """
        reward_schedule = np.asarray(reward_schedule)
        assert (
            reward_schedule.ndim == 2 and reward_schedule.shape[1] == 2
        ), "reward_schedule must be shape (N, 2)"

        rng = np.random.default_rng(seed)
        n_sessions = reward_schedule.shape[0]
        n_ports = self.n_ports
        input_size = n_ports + 1

        # Window boundaries: hidden state resets
        window_starts = set()
        idx = 0
        while idx < n_sessions:
            window_starts.add(idx)
            idx += int(rng.integers(n_block_min, n_block_max + 1))

        all_probs, all_choices, all_rewards = [], [], []
        all_session_ids, all_block_ids, all_window_ids = [], [], []
        if return_hidden:
            all_hidden = []

        self.model.eval()
        hidden = None
        window_id = 0
        block_id_in_window = 0

        with torch.no_grad():
            for session_idx in range(n_sessions):
                if session_idx in window_starts:
                    hidden = None
                    window_id += 1
                    block_id_in_window = 0

                block_probs = reward_schedule[session_idx]
                block_id_in_window += 1

                prev_choice = 0
                prev_reward = 0.0
                trial = 0

                while True:
                    x = torch.zeros(1, 1, input_size, device=self.device)
                    if prev_choice > 0:
                        x[0, 0, prev_choice - 1] = 1.0
                        x[0, 0, n_ports] = prev_reward

                    logits, _, hidden = self.model(x, hidden)
                    probs = F.softmax(logits[0, 0], dim=-1)
                    choice = torch.multinomial(probs, num_samples=1).item() + 1

                    reward = int(rng.random() < block_probs[choice - 1])

                    all_probs.append(block_probs.copy())
                    all_choices.append(choice)
                    all_rewards.append(reward)
                    all_session_ids.append(session_idx + 1)
                    all_block_ids.append(block_id_in_window)
                    all_window_ids.append(window_id)
                    if return_hidden:
                        all_hidden.append(hidden[0, 0].cpu().numpy())

                    prev_choice = choice
                    prev_reward = float(reward)
                    trial += 1

                    if trial >= min_trials_per_block and (
                        rng.random() < prob_switch or trial >= max_trials_per_block
                    ):
                        break

                hidden = hidden.detach()

        sim_task = core.Bandit2Arm(
            probs=np.array(all_probs),
            choices=np.array(all_choices, dtype=int),
            rewards=np.array(all_rewards, dtype=int),
            session_ids=np.array(all_session_ids, dtype=int),
            block_ids=np.array(all_block_ids, dtype=int),
            window_ids=np.array(all_window_ids, dtype=int),
        )
        if return_hidden:
            return sim_task, np.vstack(all_hidden)
        return sim_task

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, extra: dict = None):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "hidden_size": self.model.hidden_size,
            "input_size": self.model.input_size,
            "num_actions": self.model.num_actions,
            **self._common_checkpoint_fields(extra),
        }
        torch.save(checkpoint, path)
        print(f"Saved to {path}")

    @staticmethod
    def load(path: str, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Reconstruct task from saved arrays.
        # Older checkpoints did not save probs; fall back to dummy zeros.
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

        fitter = VanillaRNNFit2Arm(
            task=task,
            hidden_size=checkpoint["hidden_size"],
            segment_starts=checkpoint["seg_mask"],
            device=device,
        )
        fitter.model.load_state_dict(checkpoint["model_state_dict"])
        fitter.model.eval()
        fitter.nll_history = checkpoint.get("nll_history", [])
        # Restore cached outputs so callers don't need to recompute
        fitter._loaded_nll_per_trial = checkpoint.get("nll_per_trial", None)
        fitter._loaded_predict_proba = checkpoint.get("predict_proba", None)
        fitter.extra = checkpoint.get("extra", {})
        return fitter

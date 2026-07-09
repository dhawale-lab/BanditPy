import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .. import core
from .model import _get_slurm_cpus


def _run_fold(
    fold,
    train_task,
    test_task,
    test_windows,
    hidden_size,
    device_str,
    n_epochs,
    lr,
    lr_min,
    progress_bar,
):
    """Module-level worker so ProcessPoolExecutor can pickle it."""
    import torch  # re-import in subprocess

    torch.set_num_threads(1)  # prevent intra-op thread contention across workers

    fold_fitter = VanillaRNNFit2Arm(
        task=train_task,
        hidden_size=hidden_size,
        segment_starts="window",
        device=device_str,
    )
    fold_fitter.fit(n_epochs=n_epochs, lr=lr, lr_min=lr_min, progress_bar=progress_bar)
    train_nll = fold_fitter.nll_per_trial
    n_train = train_task.n_trials

    test_fitter = VanillaRNNFit2Arm(
        task=test_task,
        hidden_size=hidden_size,
        segment_starts="window",
        device=device_str,
    )
    test_fitter.model.load_state_dict(fold_fitter.model.state_dict())
    test_nll = test_fitter.nll_per_trial
    n_test = test_task.n_trials

    records = [
        {
            "fold": fold,
            "window_id": w,
            "train_nll": train_nll,
            "test_nll": test_nll,
            "n_train_trials": n_train,
            "n_test_trials": n_test,
        }
        for w in test_windows
    ]
    return fold, train_nll, test_nll, test_windows, records


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


class VanillaRNNTrainer2Arm:
    """Actor-critic trainer for VanillaRNNModel on a 2-arm bandit task.

    Mirrors BanditTrainer2Arm (rnn_models.py) but uses nn.RNN instead of LSTM.
    The hidden state is a single tensor — no cell state.
    """

    def __init__(
        self,
        input_size=3,
        hidden_size=48,
        num_actions=2,
        lr=0.0001,
        lr_min=1e-6,
        gamma=0.9,
        beta_entropy=0.045,
        beta_value=0.025,
        model_path="vanilla_rnn_2arm.pt",
        device=None,
    ):
        self.input_size = input_size
        self.num_actions = num_actions
        self.lr = lr
        self.lr_min = lr_min
        self.gamma = gamma
        self.beta_entropy = beta_entropy
        self.beta_value = beta_value
        self.model_path = model_path
        self.train_type = None

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = VanillaRNNModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_actions=num_actions,
        ).to(self.device)

        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=self.lr, alpha=0.99
        )

        self.policy_loss_history = []
        self.value_loss_history = []
        self.entropy_bonus_history = []
        self.training_loss_history = []

    def _validate_probs(self, reward_probs):
        if (
            not isinstance(reward_probs, np.ndarray)
            or reward_probs.ndim != 2
            or reward_probs.shape[1] != 2
        ):
            raise ValueError("reward_probs must be a numpy array of shape (N, 2).")
        if not (np.all(reward_probs >= 0) and np.all(reward_probs <= 1)):
            raise ValueError("All reward probabilities must be between 0 and 1.")
        frac_structured = np.mean(np.isclose(reward_probs.sum(axis=1), 1.0))
        self.train_type = "Structured" if frac_structured > 0.5 else "Unstructured"
        return reward_probs

    def _window_boundaries(self, n_sessions, n_block_min, n_block_max):
        starts = set()
        idx = 0
        while idx < n_sessions:
            starts.add(idx)
            idx += np.random.randint(n_block_min, n_block_max + 1)
        return starts

    def _make_input(self, env_action, reward):
        """Build the l_t = [a_{t-1} (one-hot), r_{t-1}] input tensor."""
        vec = torch.zeros(self.input_size, device=self.device)
        vec[env_action - 1] = 1.0  # one-hot action (1-indexed → 0-indexed)
        vec[self.num_actions] = reward  # scalar reward at last position
        return vec

    def _discounted_return(self, rewards):
        G, R = [], 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            G.insert(0, R)
        return torch.tensor(G, dtype=torch.float32, device=self.device)

    def _detach_hidden(self, h):
        return h.detach() if h is not None else None

    def _apply_lr(self, session_idx, n_sessions, warmup_steps):
        if session_idx < warmup_steps:
            new_lr = self.lr * (session_idx + 1) / warmup_steps
        else:
            progress = (session_idx - warmup_steps) / max(1, n_sessions - warmup_steps)
            new_lr = self.lr_min + 0.5 * (self.lr - self.lr_min) * (
                1.0 + math.cos(math.pi * progress)
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr

    def train(
        self,
        reward_probs,
        min_block_trials=100,
        p_switch=0.02,
        max_block_trials=500,
        n_block_min=4,
        n_block_max=8,
        clip_norm=1.0,
        lr_warmup_frac=0.02,
        return_df=False,
        save_model=False,
        progress_bar=True,
    ):
        """Train VanillaRNN on a block-structured 2-arm bandit task.

        Hidden state resets at window boundaries (random lengths drawn from
        [n_block_min, n_block_max] sessions). Optimizer steps every session
        (TBPTT — gradients truncated at session boundaries).

        Parameters
        ----------
        reward_probs : np.ndarray, shape (N, 2)
            Reward probabilities per arm for each of N sessions.
        min_block_trials : int
            Minimum trials before a session can end.
        p_switch : float
            Per-trial probability of ending the session after min_block_trials.
        max_block_trials : int
            Hard cap on session length.
        n_block_min, n_block_max : int
            Range of window lengths (in sessions) for hidden-state resets.
        clip_norm : float
            Gradient norm clipping threshold.
        lr_warmup_frac : float
            Fraction of sessions used for linear LR warmup before cosine decay.
        return_df : bool
            If True, return a DataFrame with per-trial training data.
        save_model : bool
            If True, save the model after training.
        progress_bar : bool
            Show tqdm progress bar.

        Returns
        -------
        pd.DataFrame or None
        """
        reward_probs = self._validate_probs(reward_probs)
        n_sessions = reward_probs.shape[0]
        warmup_steps = max(1, int(n_sessions * lr_warmup_frac))
        window_starts = self._window_boundaries(n_sessions, n_block_min, n_block_max)

        print(
            f"VanillaRNN training: {n_sessions} sessions, "
            f"window {n_block_min}-{n_block_max} blocks, "
            f"min {min_block_trials} trials/session, p_switch={p_switch} "
            f"[{self.train_type}], lr {self.lr:.2e}→{self.lr_min:.2e} "
            f"(warmup {warmup_steps} sessions)"
        )

        training_data = []
        rnn_hidden = None
        window_id = 0
        block_id_in_window = 0

        for session_idx in tqdm(range(n_sessions), disable=not progress_bar):
            if session_idx in window_starts:
                rnn_hidden = None
                window_id += 1
                block_id_in_window = 0

            block_reward_probs = reward_probs[session_idx]
            block_id_in_window += 1

            session_start_hidden = rnn_hidden

            input_tensors, model_actions, rewards_received = [], [], []
            current_input = torch.zeros((1, 1, self.input_size), device=self.device)
            trial = 0

            while True:
                policy_logits_step, _, rnn_hidden = self.model(
                    current_input, rnn_hidden
                )
                prob_step = F.softmax(policy_logits_step.squeeze(0), dim=-1)
                model_action = (
                    torch.distributions.Categorical(prob_step).sample().item()
                )
                env_action = model_action + 1
                reward = (
                    1.0 if random.random() < block_reward_probs[model_action] else 0.0
                )

                model_actions.append(model_action)
                rewards_received.append(reward)
                next_input = self._make_input(env_action, reward)
                input_tensors.append(next_input)
                current_input = next_input.unsqueeze(0).unsqueeze(0)

                if return_df:
                    training_data.append(
                        {
                            "session_id": session_idx + 1,
                            "window_id": window_id,
                            "block_id": block_id_in_window,
                            "block_trial": trial + 1,
                            "chosen_action": env_action,
                            "reward": reward,
                            "arm1_reward_prob": block_reward_probs[0],
                            "arm2_reward_prob": block_reward_probs[1],
                        }
                    )

                trial += 1
                if trial >= min_block_trials and (
                    random.random() < p_switch or trial >= max_block_trials
                ):
                    break

            rnn_hidden = self._detach_hidden(rnn_hidden)

            G = self._discounted_return(rewards_received)
            x_seq = torch.stack(input_tensors).unsqueeze(0)

            policy_logits_seq, value_estimates_seq, _ = self.model(
                x_seq, session_start_hidden
            )
            policy_logits_seq = policy_logits_seq.squeeze(0)
            value_estimates_seq = value_estimates_seq.squeeze(0)

            actions_tensor = torch.tensor(
                model_actions, dtype=torch.long, device=self.device
            )
            log_probs = torch.distributions.Categorical(
                logits=policy_logits_seq
            ).log_prob(actions_tensor)
            advantage = G - value_estimates_seq
            policy_loss = -(log_probs * advantage.detach()).mean()
            value_loss = self.beta_value * advantage.pow(2).mean()
            dist_entropy = (
                torch.distributions.Categorical(logits=policy_logits_seq)
                .entropy()
                .mean()
            )
            entropy_bonus = -self.beta_entropy * dist_entropy
            loss = policy_loss + value_loss + entropy_bonus

            self.policy_loss_history.append(policy_loss.item())
            self.value_loss_history.append(value_loss.item())
            self.entropy_bonus_history.append(entropy_bonus.item())
            self.training_loss_history.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
            self.optimizer.step()
            self._apply_lr(session_idx, n_sessions, warmup_steps)

        final_avg_loss = (
            np.mean(self.training_loss_history[-100:])
            if self.training_loss_history
            else float("nan")
        )
        print(f"Training complete. Final avg loss: {final_avg_loss:.4f}")
        self.model._is_trained = True

        if save_model:
            self.save_model()

        if return_df:
            return pd.DataFrame(training_data)
        return None

    def evaluate(
        self,
        reward_probs,
        min_block_trials=100,
        p_switch=0.02,
        max_block_trials=500,
        n_block_min=4,
        n_block_max=8,
        progress_bar=True,
    ):
        """Evaluate the trained model stochastically.

        Parameters
        ----------
        reward_probs : np.ndarray, shape (N, 2)

        Returns
        -------
        pd.DataFrame
        """
        if not getattr(self.model, "_is_trained", False):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model._is_trained = True
            except FileNotFoundError:
                print(f"Evaluation failed: model not found at {self.model_path}.")
                return pd.DataFrame()

        reward_probs = self._validate_probs(reward_probs)
        n_sessions = reward_probs.shape[0]
        window_starts = self._window_boundaries(n_sessions, n_block_min, n_block_max)

        evaluation_data = []
        rnn_hidden = None
        window_id = 0
        block_id_in_window = 0

        for session_idx in tqdm(range(n_sessions), disable=not progress_bar):
            if session_idx in window_starts:
                rnn_hidden = None
                window_id += 1
                block_id_in_window = 0

            block_reward_probs = reward_probs[session_idx]
            block_id_in_window += 1
            current_input = torch.zeros((1, 1, self.input_size), device=self.device)

            trial = 0
            while True:
                with torch.no_grad():
                    policy_logits_step, _, rnn_hidden = self.model(
                        current_input, rnn_hidden
                    )

                prob_step = F.softmax(policy_logits_step.squeeze(0), dim=-1)
                model_action = torch.multinomial(prob_step, num_samples=1).item()
                env_action = model_action + 1
                reward = (
                    1.0 if random.random() < block_reward_probs[model_action] else 0.0
                )

                evaluation_data.append(
                    {
                        "session_id": session_idx + 1,
                        "window_id": window_id,
                        "block_id": block_id_in_window,
                        "block_trial": trial + 1,
                        "chosen_action": env_action,
                        "reward": reward,
                        "arm1_reward_prob": block_reward_probs[0],
                        "arm2_reward_prob": block_reward_probs[1],
                    }
                )

                next_input = self._make_input(env_action, reward)
                current_input = next_input.unsqueeze(0).unsqueeze(0)

                trial += 1
                if trial >= min_block_trials and (
                    random.random() < p_switch or trial >= max_block_trials
                ):
                    break

        print("Evaluation complete.")
        return pd.DataFrame(evaluation_data)

    def save_model(self):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "policy_loss_history": self.policy_loss_history,
            "value_loss_history": self.value_loss_history,
            "entropy_bonus_history": self.entropy_bonus_history,
            "training_loss_history": self.training_loss_history,
            "input_size": self.input_size,
            "hidden_size": self.model.hidden_size,
            "lr": self.lr,
            "lr_min": self.lr_min,
            "gamma": self.gamma,
            "beta_entropy": self.beta_entropy,
            "beta_value": self.beta_value,
        }
        torch.save(checkpoint, self.model_path)
        print(f"Model saved to {self.model_path}")

    @staticmethod
    def load_model(model_path, device="cpu", verbose=True):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found at {model_path}")

        checkpoint = torch.load(model_path, map_location=device)

        trainer = VanillaRNNTrainer2Arm(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            lr=checkpoint.get("lr", 0.0001),
            lr_min=checkpoint.get("lr_min", 1e-6),
            gamma=checkpoint.get("gamma", 0.9),
            beta_entropy=checkpoint.get("beta_entropy", 0.045),
            beta_value=checkpoint.get("beta_value", 0.025),
            model_path=model_path,
            device=device,
        )

        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.model.eval()
        if "optimizer_state_dict" in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.policy_loss_history = checkpoint.get("policy_loss_history", [])
        trainer.value_loss_history = checkpoint.get("value_loss_history", [])
        trainer.entropy_bonus_history = checkpoint.get("entropy_bonus_history", [])
        trainer.training_loss_history = checkpoint.get("training_loss_history", [])
        trainer.model._is_trained = True

        if verbose:
            print(f"Restored VanillaRNNTrainer2Arm from {model_path}")
        return trainer


class VanillaRNNFit2Arm:
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
        assert isinstance(task, core.Bandit2Arm), "task must be a Bandit2Arm object"
        self.task = task
        self.n_ports = task.n_ports
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = VanillaRNNModel(
            input_size=self.n_ports + 1,  # one-hot action + scalar reward
            hidden_size=hidden_size,
            num_actions=self.n_ports,
        ).to(self.device)

        self._seg_mask = self._resolve_segment_starts(segment_starts)
        self.segments = self._build_segments()

        self.nll_history = []

    # ------------------------------------------------------------------

    def _resolve_segment_starts(self, segment_starts) -> np.ndarray:
        if isinstance(segment_starts, str):
            if segment_starts == "session":
                ids = self.task.session_ids
            elif segment_starts == "window":
                assert (
                    self.task.window_ids is not None
                ), "window_ids must be set for segment_starts='window'"
                ids = self.task.window_ids
            elif segment_starts == "block":
                assert (
                    self.task.block_ids is not None
                ), "block_ids must be set for segment_starts='block'"
                ids = self.task.block_ids
            else:
                raise ValueError(
                    f"Unknown segment_starts '{segment_starts}'. "
                    "Use 'session', 'window', 'block', or a boolean array."
                )
            mask = np.concatenate(([True], ids[1:] != ids[:-1]))
        else:
            mask = np.asarray(segment_starts, dtype=bool)
            assert len(mask) == self.task.n_trials
            if not mask[0]:
                mask = mask.copy()
                mask[0] = True
        return mask

    def _build_segments(self):
        """Pre-build (x_seq, y_seq) tensors for every segment."""
        choices = self.task.choices  # 1-indexed, shape (n_trials,)
        rewards = self.task.rewards  # 0 or 1,   shape (n_trials,)
        n_ports = self.n_ports

        boundaries = np.append(np.where(self._seg_mask)[0], self.task.n_trials)
        segments = []

        for seg_start, seg_stop in zip(boundaries[:-1], boundaries[1:]):
            seg_len = seg_stop - seg_start
            seg_choices = choices[seg_start:seg_stop]  # 1-indexed
            seg_rewards = rewards[seg_start:seg_stop]

            # l_t = [one_hot(a_{t-1}), r_{t-1}];  l_0 = zeros
            x = np.zeros((seg_len, n_ports + 1), dtype=np.float32)
            for t in range(1, seg_len):
                x[t, seg_choices[t - 1] - 1] = 1.0  # one-hot (0-indexed)
                x[t, n_ports] = float(seg_rewards[t - 1])

            y = (seg_choices - 1).astype(np.int64)  # 0-indexed targets

            x_t = torch.tensor(x, device=self.device).unsqueeze(0)  # (1, T, D)
            y_t = torch.tensor(y, device=self.device)  # (T,)
            segments.append((x_t, y_t))

        return segments

    def _total_nll(self):
        """Forward pass through all segments; return total NLL and trial count."""
        total_nll = torch.tensor(0.0, device=self.device)
        n_trials = 0
        for x_seq, y_seq in self.segments:
            logits, _, _ = self.model(x_seq)  # (1, T, n_actions)
            logits = logits.squeeze(0)  # (T, n_actions)
            total_nll = total_nll + F.cross_entropy(logits, y_seq, reduction="sum")
            n_trials += len(y_seq)
        return total_nll, n_trials

    def fit(
        self,
        n_epochs: int = 500,
        lr: float = 0.001,
        lr_min: float = 1e-5,
        progress_bar: bool = True,
    ):
        """Fit the RNN to the observed choice sequence.

        Uses Adam with cosine LR decay.  NLL per trial is tracked in
        ``self.nll_history``.

        Parameters
        ----------
        n_epochs : int
        lr : float
            Initial learning rate.
        lr_min : float
            Minimum learning rate at end of cosine decay.
        progress_bar : bool
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr_min
        )

        self.nll_history = []
        self.model.train()

        for _ in tqdm(range(n_epochs), disable=not progress_bar):
            optimizer.zero_grad()
            total_nll, n_trials = self._total_nll()
            nll_per_trial = total_nll / n_trials
            nll_per_trial.backward()
            optimizer.step()
            scheduler.step()
            self.nll_history.append(nll_per_trial.item())

        self.model.eval()
        print(
            f"Fit complete. Final NLL/trial: {self.nll_history[-1]:.4f}  "
            f"(n_trials={n_trials}, n_segments={len(self.segments)})"
        )

    @property
    def nll_per_trial(self) -> float:
        """NLL per trial with current model weights (no gradient)."""
        self.model.eval()
        with torch.no_grad():
            total_nll, n_trials = self._total_nll()
        return (total_nll / n_trials).item()

    def predict_proba(self) -> np.ndarray:
        """Choice probabilities for every trial in the original trial order.

        Returns
        -------
        np.ndarray, shape (n_trials, n_actions)
            Softmax choice probabilities.
        """
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for x_seq, _ in self.segments:
                logits, _, _ = self.model(x_seq)  # (1, T, n_actions)
                probs = F.softmax(logits.squeeze(0), dim=-1)  # (T, n_actions)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)

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

    def cross_validate(
        self,
        k: int = 5,
        stratify: bool = True,
        n_epochs: int = 500,
        lr: float = 0.001,
        lr_min: float = 1e-5,
        seed: int = None,
        progress_bar: bool = False,
        n_jobs: int = None,
    ) -> pd.DataFrame:
        """K-fold cross-validation using windows as the split unit.

        Windows are the natural unit because the hidden state already resets at
        window boundaries — there is no leakage between windows.  Block structure
        within each window is preserved intact.

        For each fold a fresh model is initialised and fitted on the training
        windows; the held-out windows are used only for evaluation.

        Parameters
        ----------
        k : int
            Number of folds.
        stratify : bool
            If True, windows are sorted by index before folding so that early
            and late windows are distributed evenly across folds.  Recommended
            when the animal shows a learning trend across windows.
        n_epochs, lr, lr_min : int / float
            Passed to ``fit()`` for each fold.
        seed : int, optional
            Random seed for reproducible fold assignment.
        progress_bar : bool
            Show per-epoch tqdm bar for each fold.
        n_jobs : int or None
            Number of parallel worker processes.  ``None`` (default) reads
            ``SLURM_CPUS_PER_TASK`` / ``SLURM_JOB_CPUS_PER_NODE`` and falls
            back to 1 when neither is set.  Values > 1 launch each fold in a
            separate process via ``ProcessPoolExecutor``; each worker sets
            ``torch.set_num_threads(1)`` to avoid intra-op thread contention.
            Workers are clamped to ``k`` so you never spawn more processes than
            folds.  Parallel execution is only beneficial for CPU — avoid with
            ``device="cuda"``.

        Returns
        -------
        pd.DataFrame with columns:
            fold          : fold index (0-based)
            window_id     : held-out window ID
            train_nll     : NLL/trial on training windows
            test_nll      : NLL/trial on held-out window
            n_train_trials: number of training trials
            n_test_trials : number of trials in the held-out window
        """
        assert (
            self.task.window_ids is not None
        ), "window_ids must be set on the task for cross-validation"

        rng = np.random.default_rng(seed)
        window_ids = np.unique(self.task.window_ids)
        n_windows = len(window_ids)

        assert k <= n_windows, f"k={k} exceeds number of windows ({n_windows})"

        # Assign windows to folds
        if stratify:
            # interleave by window order so each fold has early+late windows
            ordered = np.arange(n_windows)
            fold_assignments = ordered % k
        else:
            perm = rng.permutation(n_windows)
            fold_assignments = np.empty(n_windows, dtype=int)
            fold_assignments[perm] = np.arange(n_windows) % k

        # Pre-build per-fold task splits (cheap — numpy ops only)
        fold_args = []
        for fold in range(k):
            test_mask = fold_assignments == fold
            test_windows = window_ids[test_mask]
            train_windows = window_ids[~test_mask]
            train_task = self.task._filtered(
                np.isin(self.task.window_ids, train_windows)
            )
            test_task = self.task._filtered(np.isin(self.task.window_ids, test_windows))
            fold_args.append((fold, train_task, test_task, test_windows))

        device_str = str(self.device)
        hidden_size = self.model.hidden_size
        if n_jobs is None:
            n_jobs = _get_slurm_cpus(default=1)
        workers = max(1, min(n_jobs, k))
        print(f"Using {workers} worker(s) for {k}-fold CV")

        records = []
        if workers == 1:
            # Sequential path — identical behaviour to before
            for fold, train_task, test_task, test_windows in fold_args:
                _, train_nll, test_nll, test_windows, fold_records = _run_fold(
                    fold,
                    train_task,
                    test_task,
                    test_windows,
                    hidden_size,
                    device_str,
                    n_epochs,
                    lr,
                    lr_min,
                    progress_bar,
                )
                print(
                    f"  Fold {fold + 1}/{k} — "
                    f"train NLL/trial: {train_nll:.4f}, "
                    f"test NLL/trial: {test_nll:.4f} "
                    f"(test windows: {test_windows.tolist()})"
                )
                records.extend(fold_records)
        else:
            # Parallel path
            futures = {}
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for fold, train_task, test_task, test_windows in fold_args:
                    fut = pool.submit(
                        _run_fold,
                        fold,
                        train_task,
                        test_task,
                        test_windows,
                        hidden_size,
                        device_str,
                        n_epochs,
                        lr,
                        lr_min,
                        progress_bar,
                    )
                    futures[fut] = fold

                for fut in as_completed(futures):
                    fold, train_nll, test_nll, test_windows, fold_records = fut.result()
                    print(
                        f"  Fold {fold + 1}/{k} — "
                        f"train NLL/trial: {train_nll:.4f}, "
                        f"test NLL/trial: {test_nll:.4f} "
                        f"(test windows: {test_windows.tolist()})"
                    )
                    records.extend(fold_records)

        df = pd.DataFrame(records)
        mean_test = df["test_nll"].mean()
        std_test = df["test_nll"].std()
        print(f"\n{k}-fold CV — mean test NLL/trial: {mean_test:.4f} ± {std_test:.4f}")
        return df

    def save(self, path: str, extra: dict = None):
        checkpoint = {
            # --- model ---
            "model_state_dict": self.model.state_dict(),
            "hidden_size": self.model.hidden_size,
            "input_size": self.model.input_size,
            "num_actions": self.model.num_actions,
            # --- training diagnostics ---
            "nll_history": self.nll_history,
            "nll_per_trial": self.nll_per_trial,
            # --- segmentation ---
            "seg_mask": self._seg_mask,  # bool array, shape (n_trials,)
            # --- task related (for alignment in downstream analyses) ---
            "probs": self.task.probs,
            "choices": self.task.choices,
            "rewards": self.task.rewards,
            "session_ids": self.task.session_ids,
            "block_ids": self.task.block_ids,
            "window_ids": self.task.window_ids,
            # --- model output ---
            "predict_proba": self.predict_proba(),  # (n_trials, n_actions)
            # --- user-supplied extras (cv_results, metadata, etc.) ---
            "extra": extra if extra is not None else {},
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

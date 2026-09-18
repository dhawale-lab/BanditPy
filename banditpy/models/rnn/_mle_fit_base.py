"""Shared scaffolding for supervised-MLE RNN fitters on a Bandit2Arm task.

'VanillaRNNFit2Arm' (rnn_fit.py) and 'MemoryANNFit2Arm' (memory_ann.py) both
fit an nn.Module to observed choices by maximizing trial-by-trial choice
likelihood — they differ only in the underlying architecture and how a
segment's tensors are built / passed through it. Everything else (segment
resolution, the Adam+cosine fit loop, restarts, predict/accuracy/nll
properties, k-fold cross-validation with optional multiprocessing) is
identical between them and lives here.

Subclasses must:
  - call 'RNNFit2ArmBase.__init__' with a constructed 'model' and an
    'init_kwargs' dict of the architecture-specific constructor kwargs
    (e.g. {"hidden_size": 48}) needed to reconstruct a fresh instance of
    the same subclass — used by 'fit(n_restarts>1)' and 'cross_validate'.
  - implement '_segment_tensors' (build whatever tensors '_forward_segment'
    needs from one segment's 1-indexed choices / 0-1 rewards arrays) and
    '_forward_segment' (run 'self.model' on one such segment, returning
    '(logits, target_idx)').
"""

import math
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ... import core
from ..model import _get_slurm_cpus


def _run_restart(
    fitter_cls,
    init_kwargs,
    task,
    seg_mask,
    device_str,
    seed,
    n_epochs,
    lr,
    lr_min,
    progress_bar,
):
    """Module-level worker so ProcessPoolExecutor can pickle it."""
    import torch  # re-import in subprocess

    torch.set_num_threads(1)
    torch.manual_seed(seed)  # controls the model's random weight init

    fitter = fitter_cls(
        task=task, segment_starts=seg_mask, device=device_str, **init_kwargs
    )
    fitter.fit(n_epochs=n_epochs, lr=lr, lr_min=lr_min, progress_bar=progress_bar)
    return seed, fitter.nll_history[-1], fitter.model.state_dict(), fitter.nll_history


def _run_fold(
    fitter_cls,
    init_kwargs,
    fold,
    train_task,
    test_task,
    test_windows,
    device_str,
    n_epochs,
    lr,
    lr_min,
    progress_bar,
):
    """Module-level worker so ProcessPoolExecutor can pickle it."""
    import torch  # re-import in subprocess

    torch.set_num_threads(1)

    fold_fitter = fitter_cls(
        task=train_task, segment_starts="window", device=device_str, **init_kwargs
    )
    fold_fitter.fit(n_epochs=n_epochs, lr=lr, lr_min=lr_min, progress_bar=progress_bar)
    train_nll = fold_fitter.nll_per_trial
    n_train = train_task.n_trials

    test_fitter = fitter_cls(
        task=test_task, segment_starts="window", device=device_str, **init_kwargs
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


class RNNFit2ArmBase:
    """See module docstring. Not instantiated directly."""

    def __init__(self, task, model, init_kwargs, segment_starts="session", device=None):
        assert isinstance(task, core.Bandit2Arm), "task must be a Bandit2Arm object"
        self.task = task
        self.n_ports = task.n_ports
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self._init_kwargs = init_kwargs

        self._seg_mask = self._resolve_segment_starts(segment_starts)
        self.segments = self._build_segments()

        self.nll_history = []

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _segment_tensors(self, seg_choices, seg_rewards):
        """Build whatever tensors '_forward_segment' needs from one segment's
        1-indexed choices and 0/1 rewards arrays (both numpy, shape (T,))."""
        raise NotImplementedError

    def _forward_segment(self, segment):
        """Run 'self.model' on one pre-built segment. Returns (logits, target_idx)
        with logits shape (T, n_actions) and target_idx shape (T,), 0-indexed."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Segment resolution (identical across subclasses)
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
        choices = self.task.choices  # 1-indexed, shape (n_trials,)
        rewards = self.task.rewards  # 0 or 1,   shape (n_trials,)

        boundaries = np.append(np.where(self._seg_mask)[0], self.task.n_trials)
        segments = []
        for seg_start, seg_stop in zip(boundaries[:-1], boundaries[1:]):
            seg_choices = choices[seg_start:seg_stop]
            seg_rewards = rewards[seg_start:seg_stop]
            segments.append(self._segment_tensors(seg_choices, seg_rewards))
        return segments

    def _total_nll(self):
        total_nll = torch.tensor(0.0, device=self.device)
        n_trials = 0
        for segment in self.segments:
            logits, target = self._forward_segment(segment)
            total_nll = total_nll + F.cross_entropy(logits, target, reduction="sum")
            n_trials += len(target)
        return total_nll, n_trials

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit_single(self, n_epochs, lr, lr_min, progress_bar):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr_min
        )

        nll_history = []
        self.model.train()

        for _ in tqdm(range(n_epochs), disable=not progress_bar):
            optimizer.zero_grad()
            total_nll, n_trials = self._total_nll()
            nll_per_trial = total_nll / n_trials
            nll_per_trial.backward()
            optimizer.step()
            scheduler.step()
            nll_history.append(nll_per_trial.item())

        self.model.eval()
        return nll_history, n_trials

    def fit(
        self,
        n_epochs: int = 500,
        lr: float = 0.001,
        lr_min: float = 1e-5,
        progress_bar: bool = True,
        n_restarts: int = 1,
        seed: int = None,
        n_jobs: int = None,
    ):
        """Fit to the observed choice sequence. See subclass docstring for details.

        'n_restarts' > 1 trains that many independent weight initializations
        (optionally in parallel via 'n_jobs') and keeps the one with the
        lowest final training NLL/trial.
        """
        if n_restarts == 1:
            self.nll_history, n_trials = self._fit_single(
                n_epochs, lr, lr_min, progress_bar
            )
            print(
                f"Fit complete. Final NLL/trial: {self.nll_history[-1]:.4f}  "
                f"(n_trials={n_trials}, n_segments={len(self.segments)})"
            )
            return

        seeds = (
            np.random.default_rng(seed).integers(0, 2**31 - 1, size=n_restarts).tolist()
        )

        device_str = str(self.device)
        fitter_cls = type(self)
        if n_jobs is None:
            n_jobs = _get_slurm_cpus(default=1)
        workers = max(1, min(n_jobs, n_restarts))
        print(f"Using {workers} worker(s) for {n_restarts} restarts")

        results = []
        if workers == 1:
            for i, s in enumerate(seeds):
                _, final_nll, state_dict, history = _run_restart(
                    fitter_cls,
                    self._init_kwargs,
                    self.task,
                    self._seg_mask,
                    device_str,
                    s,
                    n_epochs,
                    lr,
                    lr_min,
                    progress_bar,
                )
                print(
                    f"  Restart {i + 1}/{n_restarts} (seed={s}) — "
                    f"final NLL/trial: {final_nll:.4f}"
                )
                results.append((s, final_nll, state_dict, history))
        else:
            futures = {}
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for s in seeds:
                    fut = pool.submit(
                        _run_restart,
                        fitter_cls,
                        self._init_kwargs,
                        self.task,
                        self._seg_mask,
                        device_str,
                        s,
                        n_epochs,
                        lr,
                        lr_min,
                        progress_bar,
                    )
                    futures[fut] = s
                for fut in as_completed(futures):
                    s, final_nll, state_dict, history = fut.result()
                    print(f"  Restart (seed={s}) — final NLL/trial: {final_nll:.4f}")
                    results.append((s, final_nll, state_dict, history))

        results.sort(key=lambda r: r[1])  # lowest final NLL/trial first
        best_seed, best_nll, best_state, best_history = results[0]

        self.model.load_state_dict(best_state)
        self.model.to(self.device)
        self.model.eval()
        self.nll_history = best_history
        self.restart_nlls = {s: nll for s, nll, _, _ in results}
        self.best_restart_seed = best_seed

        print(
            f"Fit complete ({n_restarts} restarts). Best seed={best_seed}, "
            f"final NLL/trial: {best_nll:.4f} (n_segments={len(self.segments)})"
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @property
    def nll_per_trial(self) -> float:
        """NLL per trial with current model weights (no gradient)."""
        self.model.eval()
        with torch.no_grad():
            total_nll, n_trials = self._total_nll()
        return (total_nll / n_trials).item()

    @property
    def predictive_accuracy(self) -> float:
        """Geometric mean probability assigned to the actual choice,
        i.e. 'exp(-nll_per_trial)'."""
        return float(np.exp(-self.nll_per_trial))

    def predict_proba(self) -> np.ndarray:
        """Choice probabilities for every trial in the original trial order.

        Returns
        -------
        np.ndarray, shape (n_trials, n_actions)
        """
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for segment in self.segments:
                logits, _ = self._forward_segment(segment)
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    def predict_choices(self, stochastic: bool = False) -> np.ndarray:
        """Predicted choice for every trial, 1-indexed to match 'task.choices'."""
        probs = self.predict_proba()
        if stochastic:
            rng = np.random.default_rng()
            return np.array([rng.choice(self.n_ports, p=p) + 1 for p in probs])
        return probs.argmax(axis=1) + 1

    def accuracy(self, stochastic: bool = False) -> float:
        """Fraction of trials where the predicted choice matches the actual choice."""
        predicted = self.predict_choices(stochastic=stochastic)
        return float((predicted == self.task.choices).mean())

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

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
        window boundaries — there is no leakage between windows. Block structure
        within each window is preserved intact.

        Parameters
        ----------
        k : int
        stratify : bool
            If True, windows are sorted by index before folding so early and
            late windows are distributed evenly across folds.
        n_epochs, lr, lr_min : passed to 'fit()' for each fold.
        seed : int, optional
        progress_bar : bool
        n_jobs : int or None
            None reads 'SLURM_CPUS_PER_TASK'/'SLURM_JOB_CPUS_PER_NODE', falls
            back to 1. Values > 1 launch each fold in a separate process.
            Only beneficial on CPU — avoid with 'device="cuda"'.

        Returns
        -------
        pd.DataFrame with columns:
            fold, window_id, train_nll, test_nll, n_train_trials, n_test_trials
        """
        assert (
            self.task.window_ids is not None
        ), "window_ids must be set on the task for cross-validation"

        rng = np.random.default_rng(seed)
        window_ids = np.unique(self.task.window_ids)
        n_windows = len(window_ids)
        assert k <= n_windows, f"k={k} exceeds number of windows ({n_windows})"

        if stratify:
            ordered = np.arange(n_windows)
            fold_assignments = ordered % k
        else:
            perm = rng.permutation(n_windows)
            fold_assignments = np.empty(n_windows, dtype=int)
            fold_assignments[perm] = np.arange(n_windows) % k

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
        fitter_cls = type(self)
        if n_jobs is None:
            n_jobs = _get_slurm_cpus(default=1)
        workers = max(1, min(n_jobs, k))
        print(f"Using {workers} worker(s) for {k}-fold CV")

        records = []
        if workers == 1:
            for fold, train_task, test_task, test_windows in fold_args:
                _, train_nll, test_nll, test_windows, fold_records = _run_fold(
                    fitter_cls,
                    self._init_kwargs,
                    fold,
                    train_task,
                    test_task,
                    test_windows,
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
            futures = {}
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for fold, train_task, test_task, test_windows in fold_args:
                    fut = pool.submit(
                        _run_fold,
                        fitter_cls,
                        self._init_kwargs,
                        fold,
                        train_task,
                        test_task,
                        test_windows,
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

    # ------------------------------------------------------------------
    # Persistence helper (subclasses add architecture-specific fields)
    # ------------------------------------------------------------------

    def _common_checkpoint_fields(self, extra=None) -> dict:
        return {
            "nll_history": self.nll_history,
            "nll_per_trial": self.nll_per_trial,
            "seg_mask": self._seg_mask,
            "probs": self.task.probs,
            "choices": self.task.choices,
            "rewards": self.task.rewards,
            "session_ids": self.task.session_ids,
            "block_ids": self.task.block_ids,
            "window_ids": self.task.window_ids,
            "predict_proba": self.predict_proba(),
            "extra": extra if extra is not None else {},
        }

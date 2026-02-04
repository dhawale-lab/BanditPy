import numpy as np
from scipy.special import logsumexp
from banditpy.core import Bandit2Arm
from .policy.base import BasePolicy
from tqdm import tqdm
import os
from .optim import resolve_optimizer


def softmax_loglik(logits, choice, beta):
    z = beta * logits
    return z[choice] - logsumexp(z)


def softmax_sample(logits, beta, rng):
    z = beta * logits
    p = np.exp(z - logsumexp(z))
    return rng.choice(len(p), p=p)


def _get_slurm_cpus(default=1):
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE"):
        if var in os.environ:
            try:
                return int(os.environ[var])
            except ValueError:
                pass
    return default


class DecisionModel:
    def __init__(self, task: Bandit2Arm, policy: BasePolicy, reset_mode="session"):
        # Allow passing either an instance or a policy class; normalize to an instance.
        if isinstance(policy, type) and issubclass(policy, BasePolicy):
            policy = policy()

        if not isinstance(policy, BasePolicy):
            raise TypeError("policy must be a BasePolicy instance or subclass")

        self.task = task
        self.policy = policy

        self.reset_mode = reset_mode
        self.resets = self._compute_resets(task, reset_mode)

        self.choices = np.asarray(task.choices, int) - 1  # Choices 0/1
        self.rewards = np.asarray(task.rewards, float)

        self.nll = None
        self.params = None
        self.fit_fvals = None
        self.fit_fval_mean = None
        self.fit_fval_std = None

    # -------------------- RESET MODE --------------------

    def _compute_resets(self, task, reset_mode):
        """
        reset_mode options:
            "session"  -> task.is_session_start
            "block"    -> task.is_block_start
            "window"   -> task.is_window_start
            ndarray    -> boolean/binary mask length == n_trials
        """

        # ---------- 1) ARRAY CASE FIRST ----------
        # (Supports bool, int {0,1}, numpy, list)
        if hasattr(reset_mode, "__array__") or isinstance(reset_mode, (list, tuple)):
            mask = np.asarray(reset_mode)

            if mask.shape[0] != task.n_trials:
                raise ValueError(
                    f"Custom reset mask must have length {task.n_trials}, "
                    f"got {mask.shape[0]}"
                )

            # allow bool or {0,1}
            if mask.dtype != bool:
                uniq = np.unique(mask)
                if not np.all(np.isin(uniq, (0, 1))):
                    raise ValueError(
                        "Custom reset mask must be boolean or contain only {0,1}"
                    )
                mask = mask.astype(bool)

            return mask

        # ---------- 2) SYMBOLIC RESET MODES ----------
        match reset_mode:
            case "session":
                return task.is_session_start
            case "block":
                return task.is_block_start
            case "window":
                return task.is_window_start
            case _:
                raise ValueError(
                    "reset_mode must be 'session', 'block', 'window', "
                    "or a boolean/0-1 mask array"
                )

    # -------------------- NLL --------------------

    def _nll(self, theta):
        names = self.policy.param_names()
        params = dict(zip(names, theta))

        self.policy.set_params(params)

        # reset after params are set so policies that read from self.params in reset don't KeyError
        self.policy.reset()

        beta = params["beta"]
        nll = 0.0

        for c, r, reset in zip(self.choices, self.rewards, self.resets):
            if reset:
                self.policy.reset()
            else:
                self.policy.forget()

            logits = self.policy.logits()
            nll -= softmax_loglik(logits, c, beta)
            self.policy.update(c, r)

        return nll

    # -------------------- FIT --------------------

    def fit(self, n_starts=10, seed=None, progress=False, n_jobs=None, optimizer=None):
        rng = np.random.default_rng(seed)

        if n_jobs is None:
            n_jobs = _get_slurm_cpus(default=1)
        n_jobs = max(1, min(n_jobs, n_starts))

        print(f"Using {n_jobs} workers")

        bounds_dict = self.policy.get_bounds()
        names = self.policy.param_names()
        bounds = [(n, bounds_dict[n]) for n in names]

        seeds = rng.integers(0, 2**32 - 1, size=n_starts)

        opt = resolve_optimizer(optimizer)

        best_fun, best_x, fvals = opt.fit(
            objective=self._nll,
            bounds=bounds,
            seeds=seeds,
            n_jobs=n_jobs,
            progress=progress,
        )

        self.params = dict(zip(names, best_x))
        self.nll = best_fun
        self.fit_fvals = fvals
        self.fit_fval_mean = float(fvals.mean())
        self.fit_fval_std = float(fvals.std())

        self.policy.set_params(self.params)

    # -------------------- POSTERIOR PREDICTIVE --------------------

    def simulate_posterior_predictive(self, seed=None) -> Bandit2Arm:
        if self.params is None:
            raise RuntimeError("Model must be fit before simulation.")

        rng = np.random.default_rng(seed)

        self.policy.set_params(self.params)
        self.policy.reset()

        task = self.task
        n_trials = task.n_trials

        choices = np.zeros(n_trials, dtype=int)
        rewards = np.zeros(n_trials, dtype=int)

        for t in range(n_trials):
            if self.resets[t]:
                self.policy.reset()
            else:
                self.policy.forget()

            logits = self.policy.logits()
            c = softmax_sample(logits, beta=self.params["beta"], rng=rng)

            p = task.probs[t, c]
            r = int(rng.random() < p)

            self.policy.update(c, r)

            choices[t] = c + 1
            rewards[t] = r

        return Bandit2Arm(
            probs=task.probs.copy(),
            choices=choices,
            rewards=rewards,
            session_ids=task.session_ids.copy(),
            block_ids=None if task.block_ids is None else task.block_ids.copy(),
            window_ids=None if task.window_ids is None else task.window_ids.copy(),
            starts=None if task.starts is None else task.starts.copy(),
            stops=None if task.stops is None else task.stops.copy(),
            datetime=None if task.datetime is None else task.datetime.copy(),
            metadata=task.metadata,
        )

    # -------------------- POLICY SIMULATION --------------------

    @classmethod
    def simulate_policy(
        cls,
        policy,
        reward_schedule,
        min_trials_per_block,
        params=None,
        prob_switch=1.0,
        seed=None,
        metadata=None,
    ):
        """
        Simulate a policy on a multi-block 2-armed bandit with optional variable block lengths.

        Args:
            policy: A `BasePolicy` instance (mutated in-place during simulation).
            reward_schedule: Sequence of `(p1, p2)` tuples; one per block specifying reward probs.
            min_trials_per_block: Int or sequence giving the minimum trials to run per block.
            params: Optional dict of policy parameters (must include `beta`).
                If None, assumes the policy was already configured via `policy.set_params()`.
            prob_switch: Float or sequence in (0, 1]; probability of switching after min trials.
                Example: `min_trials_per_block=100`, `prob_switch=0.02` yields median ~150 trials.
            seed: RNG seed for reproducibility.
            metadata: Optional metadata stored on the returned `Bandit2Arm` task.

        Returns:
            Bandit2Arm: Simulated task with probs, choices, rewards, and block/session ids.
        """
        rng = np.random.default_rng(seed)

        if params is not None:
            policy.set_params(params)
        elif not policy.params:
            raise ValueError(
                "params is None and policy has no parameters set; "
                "call policy.set_params(...) or provide params"
            )

        if "beta" not in policy.params:
            raise ValueError("policy parameters must include 'beta'")

        beta = policy.params["beta"]
        policy.reset()

        if isinstance(min_trials_per_block, int):
            min_trials_per_block = [min_trials_per_block] * len(reward_schedule)

        if isinstance(prob_switch, (int, float)):
            prob_switch = [prob_switch] * len(reward_schedule)

        assert len(min_trials_per_block) == len(reward_schedule)
        assert len(prob_switch) == len(reward_schedule)

        if not all(0 < p <= 1 for p in prob_switch):
            raise ValueError("prob_switch must be in (0, 1]")

        probs_list, choices, rewards = [], [], []
        session_ids, block_ids = [], []

        session_counter = 1
        block_counter = 1

        for (p1, p2), n_trials, p_switch in zip(
            reward_schedule, min_trials_per_block, prob_switch
        ):
            trials_in_block = 0

            while True:
                logits = policy.logits()
                c = softmax_sample(logits, beta=beta, rng=rng)
                r = int(rng.random() < [p1, p2][c])

                policy.update(c, r)

                probs_list.append([p1, p2])
                choices.append(c + 1)
                rewards.append(r)
                session_ids.append(session_counter)
                block_ids.append(block_counter)

                trials_in_block += 1

                if trials_in_block >= n_trials and rng.random() < p_switch:
                    break

            session_counter += 1
            block_counter += 1
            policy.reset()

        return Bandit2Arm(
            probs=np.asarray(probs_list),
            choices=np.asarray(choices),
            rewards=np.asarray(rewards),
            session_ids=np.asarray(session_ids),
            block_ids=np.asarray(block_ids),
            window_ids=None,
            starts=None,
            stops=None,
            datetime=None,
            metadata=metadata,
        )

    # -------------------- GREEDY SIM --------------------

    def simulate_greedy(self):
        self.policy.reset()
        choices = []

        for c, r, reset in zip(self.task.choices, self.task.rewards, self.resets):
            if reset:
                self.policy.reset()

            logits = self.policy.logits()
            choice = np.argmax(logits)
            choices.append(choice + 1)

            self.policy.update(choice, r)

        return np.array(choices)

    # -------------------- METRICS / OUTPUT --------------------

    def bic(self):
        if self.nll is None:
            raise RuntimeError("Model must be fit before computing BIC.")
        k = len(self.params)
        n = len(self.choices)
        return k * np.log(n) + 2.0 * self.nll

    def print_params(self):
        if self.params is None:
            print("Fit the model first.")
            return

        print("Fitted parameters:")
        for k, v in self.params.items():
            print(f"  {k}: {v:.4f}")
        print(f"NLL: {self.nll:.2f}")
        print(f"BIC: {self.bic():.2f}")
        if self.fit_fval_mean is not None:
            print(
                f"Restart NLL mean±SD: "
                f"{self.fit_fval_mean:.3f} ± {self.fit_fval_std:.3f}"
            )

    def describe(self):
        if self.params is None:
            print("Model is not fit yet. Call fit() first.")
            return

        specs = self.policy.parameter_specs()
        names = self.policy.param_names()

        print("\nParameters")
        print("-" * 80)

        # Column layout
        # name (14) | value (10) | bounds (18) | description (rest)
        header = f"{'name':<14} {'fitted':>10}   {'bounds':<18} description"
        print(header)
        print("-" * 80)

        for name in names:
            v = self.params[name]
            b = specs[name].bounds
            desc = specs[name].description or ""

            bounds_str = f"({b[0]:.4g}, {b[1]:.4g})"

            print(f"{name:<14} " f"{v:>10.4f}   " f"{bounds_str:<18}" f"{desc}")

        print("-" * 80)
        print(f"NLL: {self.nll:.3f}")
        print(f"BIC: {self.bic():.3f}")

    def to_dict(self):
        if self.params is None:
            raise RuntimeError("Model must be fit before calling to_dict().")

        out = dict(self.params)
        out.update(
            dict(
                nll=float(self.nll),
                bic=float(self.bic()),
                n_trials=int(len(self.choices)),
                fit_fval_mean=(
                    None if self.fit_fval_mean is None else float(self.fit_fval_mean)
                ),
                fit_fval_std=(
                    None if self.fit_fval_std is None else float(self.fit_fval_std)
                ),
                policy_type=self.policy.__class__.__name__,
            )
        )
        return out

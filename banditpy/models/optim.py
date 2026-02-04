import numpy as np
from scipy.optimize import minimize, differential_evolution
from joblib import Parallel, delayed
from tqdm import tqdm


class BaseOptimizer:
    """Protocol for optimizers used by DecisionModel.

    Subclasses must implement fit(objective, bounds, seeds, n_jobs, progress)
    and return (best_fun, best_x, fvals).
    """

    def fit(
        self, objective, bounds, seeds, n_jobs=1, progress=False
    ):  # pragma: no cover - interface
        raise NotImplementedError


class LBFGSOptimizer(BaseOptimizer):
    def fit(self, objective, bounds, seeds, n_jobs=1, progress=False):
        def _run(seed):
            rng = np.random.default_rng(seed)
            x0 = np.array([rng.uniform(*b) for b in bounds])
            res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
            return res.fun, res.x

        iterator = seeds
        if progress and n_jobs == 1:
            iterator = tqdm(iterator, desc="LBFGS starts")

        if n_jobs == 1:
            results = [_run(s) for s in iterator]
        else:
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_run)(s) for s in iterator
            )

        best_fun, best_x = min(results, key=lambda t: t[0])
        fvals = np.array([r[0] for r in results], dtype=float)
        return best_fun, best_x, fvals


class DEOptimizer(BaseOptimizer):
    def __init__(self, popsize=15, maxiter=200, tol=1e-6):
        self.popsize = popsize
        self.maxiter = maxiter
        self.tol = tol

    def fit(self, objective, bounds, seeds, n_jobs=1, progress=False):
        def _run(seed):
            res = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=self.maxiter,
                popsize=self.popsize,
                tol=self.tol,
                seed=int(seed),
                polish=True,
            )
            return res.fun, res.x

        iterator = seeds
        if progress and n_jobs == 1:
            iterator = tqdm(iterator, desc="DE starts")

        if n_jobs == 1:
            results = [_run(s) for s in iterator]
        else:
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_run)(s) for s in iterator
            )

        best_fun, best_x = min(results, key=lambda t: t[0])
        fvals = np.array([r[0] for r in results], dtype=float)
        return best_fun, best_x, fvals


class OptunaOptimizer(BaseOptimizer):
    def __init__(
        self, n_trials=100, timeout=None, sampler=None, pruner=None, show_progress=False
    ):
        self.n_trials = n_trials
        self.timeout = timeout
        self.sampler = sampler
        self.pruner = pruner
        self.show_progress = show_progress

    def fit(self, objective, bounds, seeds, n_jobs=1, progress=False):
        import optuna

        # Silence Optuna logs in worker processes.
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        optuna.logging.disable_default_handler()

        def _make_sampler(seed):
            if self.sampler is None:
                return optuna.samplers.TPESampler(seed=int(seed))
            if isinstance(self.sampler, str):
                name = self.sampler.lower()
                if name == "tpe":
                    return optuna.samplers.TPESampler(seed=int(seed))
                if name == "cma":
                    return optuna.samplers.CmaEsSampler(seed=int(seed))
                raise ValueError(f"Unknown Optuna sampler '{self.sampler}'")
            return self.sampler

        def _make_pruner():
            if self.pruner is None:
                return optuna.pruners.NopPruner()
            if isinstance(self.pruner, str):
                name = self.pruner.lower()
                if name == "nop":
                    return optuna.pruners.NopPruner()
                if name == "median":
                    return optuna.pruners.MedianPruner()
                return optuna.pruners.NopPruner()
            return self.pruner

        def _run(seed):
            sampler = _make_sampler(seed)
            pruner = _make_pruner()

            def _objective(trial):
                theta = [
                    trial.suggest_float(name, low, high) for name, (low, high) in bounds
                ]
                return objective(theta)

            study = optuna.create_study(
                direction="minimize", sampler=sampler, pruner=pruner
            )
            show_bar = self.show_progress and n_jobs == 1 and len(seeds) == 1
            study.optimize(
                _objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=1,
                show_progress_bar=show_bar,
            )

            best_trial = study.best_trial
            best_fun = float(best_trial.value)
            best_x = np.array(
                [best_trial.params[name] for name, _ in bounds], dtype=float
            )
            return best_fun, best_x

        iterator = seeds
        if progress and n_jobs == 1:
            iterator = tqdm(iterator, desc="Optuna starts")

        if n_jobs == 1:
            results = [_run(s) for s in iterator]
        else:
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_run)(s) for s in iterator
            )

        best_fun, best_x = min(results, key=lambda t: t[0])
        fvals = np.array([r[0] for r in results], dtype=float)
        return best_fun, best_x, fvals


def resolve_optimizer(optimizer=None):
    """Factory to resolve optimizer input (None, string, or BaseOptimizer)."""
    if optimizer is None or (
        isinstance(optimizer, str) and optimizer.lower() == "lbfgs"
    ):
        return LBFGSOptimizer()

    if isinstance(optimizer, str):
        name = optimizer.lower()
        if name == "de":
            return DEOptimizer()
        if name == "optuna":
            return OptunaOptimizer()
        raise ValueError(f"Unknown optimizer '{optimizer}'")

    if isinstance(optimizer, BaseOptimizer):
        return optimizer

    raise TypeError("optimizer must be None, a string, or a BaseOptimizer instance")

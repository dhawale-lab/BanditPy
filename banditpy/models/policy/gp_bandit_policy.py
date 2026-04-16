from __future__ import annotations

import numpy as np

from .base import BasePolicy, ParameterSpec


class GP2Arm(BasePolicy):
    """
    Two-armed bandit policy using independent Gaussian Processes (RBF kernel)
    with UCB action selection.
    """

    parameters = [
        ParameterSpec("length_scale", (0.01, 10.0), description="RBF length scale"),
        ParameterSpec("signal_var", (1e-4, 10.0), description="Kernel variance"),
        ParameterSpec("noise_var", (1e-6, 1.0), description="Observation noise"),
    ]

    def __init__(self):
        super().__init__()
        self.bounds["beta"] = (0.0, 10.0)

    def reset(self):
        self._xs = [[], []]
        self._ys = [[], []]

    def forget(self):
        return

    def logits(self):
        # For compatibility, return UCB scores as logits at a default context x=0
        return np.array(
            (self._ucb(0, 0.0), self._ucb(1, 0.0)),
            dtype=float,
        )

    def update(self, choice, reward, x=0.0):
        self._xs[choice].append(float(x))
        self._ys[choice].append(float(reward))

    def choose(self, x=0.0):
        scores = [self._ucb(0, x), self._ucb(1, x)]
        return int(np.argmax(scores))

    def _ucb(self, arm, x):
        mean, var = self._predict(arm, np.array([x], dtype=float))
        return float(mean[0] + self.params["beta"] * np.sqrt(max(var[0], 0.0)))

    def _predict(self, arm, x_star):
        xs = np.array(self._xs[arm], dtype=float)
        ys = np.array(self._ys[arm], dtype=float)

        if xs.size == 0:
            prior_mean = np.zeros_like(x_star, dtype=float)
            prior_var = np.full_like(x_star, self.params["signal_var"], dtype=float)
            return prior_mean, prior_var

        K = self._rbf(xs[:, None], xs[None, :])
        K += np.eye(len(xs)) * self.params["noise_var"]
        Ks = self._rbf(xs[:, None], x_star[None, :])
        Kss = self._rbf(x_star[:, None], x_star[None, :])

        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, ys))

        mean = Ks.T @ alpha
        v = np.linalg.solve(L, Ks)
        cov = Kss - v.T @ v
        var = np.clip(np.diag(cov), a_min=0.0, a_max=None)
        return mean, var

    def _rbf(self, x1, x2):
        ls = self.params["length_scale"]
        sv = self.params["signal_var"]
        sqdist = (x1 - x2) ** 2
        return sv * np.exp(-0.5 * sqdist / (ls**2))

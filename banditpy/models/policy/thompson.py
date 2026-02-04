import numpy as np
from dataclasses import dataclass
from typing import List
from scipy.special import betainc, beta as Bfn

from .base import BasePolicy, ParameterSpec


class BaseThompson2Arm(BasePolicy):
    """
    Shared posterior mechanics for Thompson sampling.
    Subclasses differ only in learning-rate structure.
    """

    def __init__(self, n_sim: int = 500, use_analytic: bool = False):
        super().__init__()
        self.n_sim = n_sim
        self.use_analytic = use_analytic

    # --- State lifecycle ---

    def reset(self):
        self.s = np.zeros(2)
        self.f = np.zeros(2)

    def forget(self):
        self.s *= self.params["tau"]
        self.f *= self.params["tau"]

    # --- Thompson logits ---

    def logits(self):
        p = self.params

        alpha = np.maximum(p["alpha0"] + self.s, 1e-6)
        beta = np.maximum(p["beta0"] + self.f, 1e-6)

        if self.use_analytic:
            # posterior mean → DecisionModel applies softmax(beta·logits)
            return alpha / (alpha + beta)

        samples = np.random.beta(
            alpha[:, None],
            beta[:, None],
            size=(2, self.n_sim),
        )
        return samples.mean(axis=1)

    # --- Diagnostics ---

    def posterior_trajectory(self, task):
        """Return posterior (alpha,beta,mean) trajectories."""
        self.reset()
        A, B = [], []

        for c, r, reset in zip(
            task.choices - 1,
            task.rewards,
            task.is_session_start,
        ):
            if reset:
                self.reset()

            A.append(self.params["alpha0"] + self.s)
            B.append(self.params["beta0"] + self.f)

            self.forget()
            self.update(c, r)

        A = np.vstack(A)
        B = np.vstack(B)
        mean = A / (A + B)
        return A, B, mean


class ThompsonShared2Arm(BaseThompson2Arm):
    """
    Thompson sampling:
    Shared learning rates for chosen / unchosen outcomes.
    """

    parameters: List[ParameterSpec] = [
        ParameterSpec("alpha0", (1e-3, 20.0), description="Prior alpha"),
        ParameterSpec("beta0", (1e-3, 20.0), description="Prior beta"),
        ParameterSpec("tau", (0.1, 0.999), description="Decay factor"),
        ParameterSpec("lr_chosen", (0.0, 1.0), description="LR (chosen arm)"),
        ParameterSpec("lr_unchosen", (0.0, 1.0), description="LR (unchosen arm)"),
        ParameterSpec("beta", (0.1, 20.0), description="Inverse temperature"),
    ]

    def update(self, choice, reward):
        p = self.params
        other = 1 - choice

        lr_c_pos = lr_c_neg = p["lr_chosen"]
        lr_u_pos = lr_u_neg = p["lr_unchosen"]

        if reward == 1:
            self.s[choice] += lr_c_pos
            self.f[other] += lr_u_neg
        else:
            self.f[choice] += lr_c_neg
            self.s[other] += lr_u_pos


class ThompsonSplit2Arm(BaseThompson2Arm):
    """
    Thompson sampling:
    Fully independent pos/neg learning rates for both arms.
    """

    parameters: List[ParameterSpec] = [
        ParameterSpec("alpha0", (1, 20.0), description="Prior alpha"),
        ParameterSpec("beta0", (1, 20.0), description="Prior beta"),
        ParameterSpec("tau", (0.1, 0.999), description="Decay factor"),
        ParameterSpec(
            "lr_c_pos", (0.0, 1.0), description="LR (chosen arm, positive reward)"
        ),
        ParameterSpec("lr_c_neg", (0.0, 1.0), description="LR (chosen arm, no reward)"),
        ParameterSpec(
            "lr_u_pos", (0.0, 1.0), description="LR (unchosen arm, positive reward)"
        ),
        ParameterSpec(
            "lr_u_neg", (0.0, 1.0), description="LR (unchosen arm, no reward)"
        ),
        ParameterSpec("beta", (0.1, 20.0), description="Inverse temperature"),
    ]

    def update(self, choice, reward):
        p = self.params
        other = 1 - choice

        if reward == 1:
            self.s[choice] += p["lr_c_pos"]
            self.f[other] += p["lr_u_neg"]
        else:
            self.f[choice] += p["lr_c_neg"]
            self.s[other] += p["lr_u_pos"]

import numpy as np
from .base import BasePolicy, ParameterSpec


class EmpiricalUCB(BasePolicy):
    parameters = [
        ParameterSpec("explore", (1e-3, 10.0)),
        ParameterSpec("tau", (0.5, 0.999)),
        ParameterSpec("q_init", (0.0, 1.0), default=0.5),
    ]

    def reset(self):
        q0 = self.params["q_init"]
        self.q = np.full(2, q0)
        self.n = np.zeros(2)
        self.t = 0.0

    def forget(self):
        tau = self.params["tau"]
        q0 = self.params["q_init"]
        self.q = q0 + tau * (self.q - q0)
        self.n *= tau
        self.t *= tau

    def logits(self):
        c = self.params["explore"]
        denom = np.maximum(self.n, 1e-9)
        bonus = c * np.sqrt(np.log(self.t + 1.0) / denom)
        return self.q + bonus

    def update(self, choice, reward):
        self.t += 1.0
        self.n[choice] += 1.0
        self.q[choice] += (reward - self.q[choice]) / self.n[choice]


class RLUCB(BasePolicy):
    parameters = [
        ParameterSpec("explore", (1e-3, 10.0)),
        ParameterSpec("tau", (0.5, 0.999)),
        ParameterSpec("q_init", (0.0, 1.0), default=0.5),
        ParameterSpec("lr_chosen", (-1.0, 1.0)),
        ParameterSpec("lr_unchosen", (-1.0, 1.0)),
    ]

    def reset(self):
        self.q = np.full(2, self.params["q_init"])
        self.n = np.zeros(2)
        self.t = 0.0

    def forget(self):
        tau = self.params["tau"]
        q0 = self.params["q_init"]
        self.q = q0 + tau * (self.q - q0)
        self.n *= tau
        self.t *= tau

    def logits(self):
        c = self.params["explore"]
        denom = np.maximum(self.n, 1e-9)
        bonus = c * np.sqrt(np.log(self.t + 1.0) / denom)
        return self.q + bonus

    def update(self, choice, reward):
        lr_c = self.params["lr_chosen"]
        lr_u = self.params["lr_unchosen"]

        self.q[choice] += lr_c * (reward - self.q[choice])
        other = 1 - choice
        self.q[other] += lr_u * (0.5 - self.q[other])

        self.n[choice] += 1.0
        self.t += 1.0


class BayesianUCB(BasePolicy):
    parameters = [
        ParameterSpec("explore", (1e-3, 10.0)),
        ParameterSpec("tau", (0.5, 0.999)),
        ParameterSpec("q_init", (0.0, 1.0), default=0.5),
    ]

    def __init__(self, prior_strength=2.0):
        super().__init__()
        self.prior_strength = prior_strength

    def reset(self):
        q0 = self.params["q_init"]
        s = self.prior_strength
        self.alpha = np.full(2, s * q0)
        self.beta = np.full(2, s * (1 - q0))

    def forget(self):
        tau = self.params["tau"]
        q0 = self.params["q_init"]
        s = self.prior_strength
        pa, pb = s * q0, s * (1 - q0)
        self.alpha = pa + tau * (self.alpha - pa)
        self.beta = pb + tau * (self.beta - pb)

    def logits(self):
        c = self.params["explore"]
        mean = self.alpha / (self.alpha + self.beta)
        var = (
            self.alpha
            * self.beta
            / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        )
        return mean + c * np.sqrt(np.maximum(var, 1e-9))

    def update(self, choice, reward):
        self.alpha[choice] += reward
        self.beta[choice] += 1 - reward


class RLBayesianUCB(BasePolicy):
    parameters = [
        ParameterSpec("explore", (1e-3, 10.0)),
        ParameterSpec("tau", (0.5, 0.999)),
        ParameterSpec("q_init", (0.0, 1.0), default=0.5),
        ParameterSpec("lr_chosen", (-1.0, 1.0)),
        ParameterSpec("lr_unchosen", (-1.0, 1.0)),
    ]

    def __init__(self, prior_strength=2.0):
        super().__init__()
        self.prior_strength = prior_strength

    def reset(self):
        q0 = self.params["q_init"]
        s = self.prior_strength
        self.alpha = np.full(2, s * q0)
        self.beta = np.full(2, s * (1 - q0))

    def forget(self):
        tau = self.params["tau"]
        q0 = self.params["q_init"]
        s = self.prior_strength
        pa, pb = s * q0, s * (1 - q0)
        self.alpha = pa + tau * (self.alpha - pa)
        self.beta = pb + tau * (self.beta - pb)

    def logits(self):
        c = self.params["explore"]
        mean = self.alpha / (self.alpha + self.beta)
        var = (
            self.alpha
            * self.beta
            / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        )
        return mean + c * np.sqrt(np.maximum(var, 1e-9))

    def update(self, choice, reward):
        lr_c = self.params["lr_chosen"]
        lr_u = self.params["lr_unchosen"]
        q0 = self.params["q_init"]

        # Chosen arm (RL-style update)
        self.alpha[choice] += lr_c * reward
        self.beta[choice] += lr_c * (1 - reward)

        # Unchosen arm (counterfactual)
        other = 1 - choice
        self.alpha[other] += lr_u * q0
        self.beta[other] += lr_u * (1 - q0)

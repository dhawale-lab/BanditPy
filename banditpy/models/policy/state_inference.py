import numpy as np
from .base import BasePolicy, ParameterSpec


class StateInference2Arm(BasePolicy):
    """
    Two-state Bayesian state-inference policy.

    Belief b_t(s) is updated from action/reward observations using the
    compatibility parameter ``c`` and symmetric stay bias ``y`` for the
    hidden state transition matrix [[0.5+0.5y, 0.5-0.5y], [0.5-0.5y, 0.5+0.5y]].
    Action logits are the current beliefs; the DecisionModel applies softmax
    with ``beta`` (so ``beta≈10`` reproduces the fixed gain in the reference).
    """

    parameters = [
        ParameterSpec("c", (0.0, 0.99), description="Observation compatibility"),
        ParameterSpec("y", (0.0, 0.99), description="Stay bias in state transitions"),
        ParameterSpec(
            "b0", (0.0, 1.0), default=0.5, description="Initial belief P(s=0)"
        ),
        ParameterSpec(
            "beta",
            (0.1, 20.0),
            default=10.0,
            description="Inverse temperature for softmax",
        ),
    ]

    def reset(self):
        p0 = np.clip(self.params["b0"], 1e-6, 1 - 1e-6)
        self.b = np.array([p0, 1.0 - p0], dtype=float)

    def forget(self):
        return  # no passive decay for this model

    def logits(self):
        return self.b.copy()

    def _likelihood(self, choice: int, reward: int):
        c = self.params["c"]
        like = np.empty(2, dtype=float)

        for s in (0, 1):
            same = choice == s
            compatible = (reward == 1 and same) or (reward == 0 and not same)
            sign = 1.0 if compatible else -1.0
            like[s] = 0.5 * (1.0 + sign * c)

        np.clip(like, 1e-6, 1.0, out=like)
        return like

    def update(self, choice, reward):
        like = self._likelihood(choice, reward)
        b_post = like * self.b
        norm = b_post.sum()
        if norm <= 0:
            b_post[:] = 0.5
            norm = 1.0
        b_post /= norm

        y = self.params["y"]
        stay = 0.5 * (1.0 + y)
        switch = 0.5 * (1.0 - y)
        T = np.array([[stay, switch], [switch, stay]])

        self.b = b_post @ T
        self.b /= self.b.sum()

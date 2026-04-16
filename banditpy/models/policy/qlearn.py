import numpy as np
from banditpy.models.policy.base import BasePolicy, ParameterSpec


def _softmax(x: np.ndarray, beta: float) -> np.ndarray:
    z = beta * x
    z -= z.max()
    e = np.exp(z)
    s = e.sum()
    if s <= 0:
        return np.full_like(x, 1.0 / len(x))
    return e / s


class Qlearn2Arm(BasePolicy):
    """
    Vanilla 2-arm Q-learning with counterfactual updates.

    Update rule:
    Q[choice] += alpha_c * (reward - Q[choice])
    Q[unchosen] += alpha_u * (reward - Q[choice])
    """

    parameters = [
        ParameterSpec(
            "alpha_c", (0.0, 1.0), description="Learning rate for chosen option"
        ),
        ParameterSpec(
            "alpha_u", (0.0, 1.0), description="Learning rate for unchosen option"
        ),
    ]

    def reset(self):
        self.q = np.full(2, 0.5)

    def forget(self):
        pass  # no forgetting in vanilla Q-learning

    def logits(self):
        return self.q.copy()

    def update(self, choice, reward):
        a_c = self.params["alpha_c"]
        a_u = self.params["alpha_u"]

        other = 1 - choice
        pe = reward - self.q[choice]

        self.q[choice] += a_c * pe
        self.q[other] += a_u * pe

        self.q[:] = np.clip(self.q, 0.0, 1.0)


class QlearnBias2Arm(BasePolicy):
    """
    2-arm Q-learning with counterfactual updates and a port bias term.

    Update rule:
    Q[choice] += alpha_c * (reward - Q[choice])
    Q[unchosen] += alpha_u * (reward - Q[choice])

    Choice logits:
    logit[0] = Q[0] + bias
    logit[1] = Q[1] - bias
    """

    parameters = [
        ParameterSpec(
            "alpha_c", (0.0, 1.0), description="Learning rate for chosen option"
        ),
        ParameterSpec(
            "alpha_u", (0.0, 1.0), description="Learning rate for unchosen option"
        ),
        ParameterSpec(
            "bias", (-2.0, 2.0), default=0.0, description="Bias toward port 0 vs port 1"
        ),
    ]

    def reset(self):
        self.q = np.full(2, 0.5)

    def forget(self):
        pass

    def logits(self):
        b = self.params["bias"]
        return np.array([self.q[0] + b, self.q[1] - b])

    def update(self, choice, reward):
        a_c = self.params["alpha_c"]
        a_u = self.params["alpha_u"]

        other = 1 - choice
        pe = reward - self.q[choice]

        self.q[choice] += a_c * pe
        self.q[other] += a_u * pe

        self.q[:] = np.clip(self.q, 0.0, 1.0)


class QlearnH2Arm(BasePolicy):

    parameters = [
        ParameterSpec("alpha_c", (-1.0, 1.0), description="Learning rate (chosen)"),
        ParameterSpec("alpha_u", (-1.0, 1.0), description="Learning rate (unchosen)"),
        ParameterSpec("alpha_h", (0.0, 1.0), description="Perseverance learning"),
        ParameterSpec("scaler", (1, 10.0), description="Perseverance scale"),
    ]

    def __init__(self):
        super().__init__()
        self.bounds["beta"] = (0.01, 20.0)

    def reset(self):
        self.q0 = 0.5
        self.q = np.array([self.q0, self.q0], dtype=float)
        self.h = 0.5

        p = self.params
        self._ac = p["alpha_c"]
        self._au = p["alpha_u"]
        self._ah = p["alpha_h"]
        self._sc = p["scaler"]

    def forget(self):
        return

    def logits(self):
        h = self.h
        bias0 = h - 0.5
        bias1 = 0.5 - h
        return np.array(
            (
                self.q[0] + self.params["scaler"] * bias0,
                self.q[1] + self.params["scaler"] * bias1,
            ),
            dtype=float,
        )

    def update(self, choice, reward):
        p = self.params
        other = 1 - choice

        rpe = reward - self.q[choice]

        self.q[choice] += p["alpha_c"] * rpe
        self.q[other] += p["alpha_u"] * rpe

        # fast in-place clamp
        np.minimum(self.q, 1.0, out=self.q)
        np.maximum(self.q, 0.0, out=self.q)

        self.h += p["alpha_h"] * (choice - self.h)


class QlearnHierarchical2Arm(BasePolicy):
    """
    Two-option hierarchical RL for a 2-armed bandit.

    A meta-controller mixes two option policies. Each option holds its own
    action values; the meta-controller maintains option values. Action
    probabilities are a mixture of option policies. Updates use soft
    responsibilities over options given the chosen action.
    """

    _disable_common = ["beta"]

    parameters = [
        ParameterSpec("alpha_q", (0.0, 1.0), description="LR for option Q-values"),
        ParameterSpec(
            "alpha_meta", (0.0, 1.0), description="LR for meta option values"
        ),
        ParameterSpec("tau", (0.5, 1.0), default=1.0, description="Forgetting factor"),
        ParameterSpec(
            "q_init", (0.0, 1.0), default=0.5, description="Initial action value"
        ),
        ParameterSpec(
            "m_init", (-1.0, 1.0), default=0.0, description="Initial meta value"
        ),
        ParameterSpec(
            "beta_meta", (0.1, 10.0), description="Inverse temp over options"
        ),
        ParameterSpec(
            "beta_option", (0.1, 10.0), description="Inverse temp within options"
        ),
    ]

    def __init__(self, n_options: int = 2):
        super().__init__()
        self.n_options = n_options

    def reset(self):
        q0 = self.params.get("q_init", 0.5)
        m0 = self.params.get("m_init", 0.0)
        self.q = np.full((self.n_options, 2), q0, dtype=float)
        self.m = np.full(self.n_options, m0, dtype=float)

    def forget(self):
        tau = self.params["tau"]
        q0 = self.params.get("q_init", 0.5)
        m0 = self.params.get("m_init", 0.0)
        self.q = q0 + tau * (self.q - q0)
        self.m = m0 + tau * (self.m - m0)

    def logits(self):
        p_meta = _softmax(self.m, self.params["beta_meta"])

        beta_opt = self.params["beta_option"]
        opt_probs = np.vstack(
            [_softmax(self.q[i], beta_opt) for i in range(self.n_options)]
        )

        p_action = p_meta @ opt_probs
        p_action = np.clip(p_action, 1e-9, 1.0)
        return np.log(p_action)

    def update(self, choice, reward):
        p_meta = _softmax(self.m, self.params["beta_meta"])
        beta_opt = self.params["beta_option"]
        opt_probs = np.vstack(
            [_softmax(self.q[i], beta_opt) for i in range(self.n_options)]
        )

        resp = p_meta * opt_probs[:, choice]
        resp_sum = resp.sum()
        if resp_sum <= 0:
            resp = np.full(self.n_options, 1.0 / self.n_options)
        else:
            resp /= resp_sum

        aq = self.params["alpha_q"]
        am = self.params["alpha_meta"]

        for k in range(self.n_options):
            pe = reward - self.q[k, choice]
            self.q[k, choice] += aq * resp[k] * pe
            np.minimum(self.q[k], 1.0, out=self.q[k])
            np.maximum(self.q[k], 0.0, out=self.q[k])

            m_pe = reward - self.m[k]
            self.m[k] += am * resp[k] * m_pe


class QlearnWM2Arm(BasePolicy):
    """
    RL + Working Memory model for 2-arm bandit.

    Adapted from Collins & Frank (2012). A model-free RL learner and a
    working memory (WM) system jointly drive action selection. The WM
    system encodes outcomes with a learning rate of 1 (perfect one-shot
    memory) but decays toward chance over time. The mixing weight between
    systems is updated via Bayesian model averaging based on each
    system's predictive accuracy.

    For the 2-arm case (set size n_s = 1), WM capacity is always
    sufficient, so the capacity parameter C is not included.

    Action probabilities:
        p(a) = (1 - w) * softmax(beta_rl * Q_RL) + w * softmax(beta_wm * Q_WM)

    Common beta is disabled; beta_rl and beta_wm control exploration independently.

    Reference
    ---------
    Collins, A. G. E. & Frank, M. J. (2012). How much of reinforcement
    learning is working memory, not reinforcement learning? European
    Journal of Neuroscience, 35(7), 1024-1035.
    """

    _disable_common = ["beta"]

    parameters = [
        ParameterSpec("alpha_rl", (0.0, 1.0), description="RL learning rate"),
        ParameterSpec("beta_rl", (0.1, 20.0), description="RL inverse temperature"),
        ParameterSpec("beta_wm", (0.1, 20.0), description="WM inverse temperature"),
        ParameterSpec("decay", (0.0, 1.0), description="Decay rate toward initial Q"),
        ParameterSpec("w0", (0.0, 1.0), default=0.5, description="Initial WM weight"),
    ]

    def reset(self):
        self.q_rl = np.full(2, 0.5)
        self.q_wm = np.full(2, 0.5)
        self.w = self.params.get("w0", 0.5)

    def forget(self):
        eps = self.params["decay"]
        self.q_rl += eps * (0.5 - self.q_rl)
        self.q_wm += eps * (0.5 - self.q_wm)

    def logits(self):
        p_rl = _softmax(self.q_rl, self.params["beta_rl"])
        p_wm = _softmax(self.q_wm, self.params["beta_wm"])
        p_mix = (1.0 - self.w) * p_rl + self.w * p_wm
        p_mix = np.clip(p_mix, 1e-9, 1.0)
        return np.log(p_mix)

    def update(self, choice, reward):
        # --- Bayesian update of mixture weight (using pre-update Q) ---
        q_rl_c = np.clip(self.q_rl[choice], 1e-6, 1.0 - 1e-6)
        q_wm_c = np.clip(self.q_wm[choice], 1e-6, 1.0 - 1e-6)

        p_rl_lik = q_rl_c if reward == 1 else (1.0 - q_rl_c)
        p_wm_lik = q_wm_c if reward == 1 else (1.0 - q_wm_c)

        num = p_wm_lik * self.w
        den = num + p_rl_lik * (1.0 - self.w)
        self.w = num / den if den > 1e-12 else 0.5
        self.w = np.clip(self.w, 1e-6, 1.0 - 1e-6)

        # --- RL update (chosen arm only, no counterfactual) ---
        pe = reward - self.q_rl[choice]
        self.q_rl[choice] += self.params["alpha_rl"] * pe
        np.clip(self.q_rl, 0.0, 1.0, out=self.q_rl)

        # --- WM update (perfect one-shot encoding, lr = 1) ---
        self.q_wm[choice] = float(reward)

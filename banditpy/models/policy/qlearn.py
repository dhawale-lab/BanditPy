import numpy as np
from banditpy.models.policy.base import BasePolicy, ParameterGroup, ParameterSpec
from .beta_schedule import NoBeta


def _softmax(x: np.ndarray, beta: float) -> np.ndarray:
    z = beta * x
    z -= z.max()
    e = np.exp(z)
    s = e.sum()
    if s <= 0:
        return np.full_like(x, 1.0 / len(x))
    return e / s


class Qlearn(BasePolicy):
    """
    2-arm Q-learning with counterfactual updates and a port bias term.

    Update rule:
    Q[choice] += alpha_c * (reward - Q[choice])
    Q[unchosen] += alpha_u * (reward - Q[choice])

    Choice logits:
    logit[0] = Q[0] + bias
    logit[1] = Q[1] - bias
    """

    class Params(ParameterGroup):
        alpha_c = ParameterSpec(
            "alpha_c", (0.0, 0.99), description="Learning rate for chosen option"
        )
        alpha_u = ParameterSpec(
            "alpha_u", (-0.99, 0.99), description="Learning rate for unchosen option"
        )
        bias = ParameterSpec(
            "bias", (-2.0, 2.0), default=0.0, description="Bias toward port 0 vs port 1"
        )

    params: Params

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


class QlearnSticky(BasePolicy):
    """Qlearn with a perseverance (sticky) term for the propensity to choose
    the same port irrespective of reward, plus a static port bias term."""

    class Params(ParameterGroup):
        alpha_c = ParameterSpec(
            "alpha_c", (-1.0, 1.0), description="Learning rate (chosen)"
        )
        alpha_u = ParameterSpec(
            "alpha_u", (-1.0, 1.0), description="Learning rate (unchosen)"
        )
        alpha_h = ParameterSpec(
            "alpha_h", (0.0, 1.0), description="Perseverance learning"
        )
        scaler = ParameterSpec("scaler", (1, 10.0), description="Perseverance scale")
        bias = ParameterSpec(
            "bias", (-2.0, 2.0), default=0.0, description="Bias toward port 0 vs port 1"
        )

    params: Params

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        stick0 = h - 0.5
        stick1 = 0.5 - h
        b = self.params["bias"]
        return np.array(
            (
                self.q[0] + self.params["scaler"] * stick0 + b,
                self.q[1] + self.params["scaler"] * stick1 - b,
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


class QlearnHierarchical(BasePolicy):
    """
    Two-option hierarchical RL for a 2-armed bandit.

    A meta-controller mixes two option policies. Each option holds its own
    action values; the meta-controller maintains option values. Action
    probabilities are a mixture of option policies, with 'beta_meta' and
    'beta_option' controlling exploration at each level. 'logits()'
    returns log-probabilities, so this policy uses 'NoBeta' by default.
    Updates use soft responsibilities over options given the chosen action.
    """

    # _disable_common = [""]

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha_q = ParameterSpec(
            "alpha_q", (0.0, 1.0), description="LR for option Q-values"
        )
        alpha_meta = ParameterSpec(
            "alpha_meta", (0.0, 1.0), description="LR for meta option values"
        )
        tau = ParameterSpec(
            "tau", (0.5, 1.0), default=1.0, description="Forgetting factor"
        )
        q_init = ParameterSpec(
            "q_init", (0.0, 1.0), default=0.5, description="Initial action value"
        )
        m_init = ParameterSpec(
            "m_init", (-1.0, 1.0), default=0.0, description="Initial meta value"
        )
        beta_meta = ParameterSpec(
            "beta_meta", (0.1, 10.0), description="Inverse temp over options"
        )
        beta_option = ParameterSpec(
            "beta_option", (0.1, 10.0), description="Inverse temp within options"
        )

    params: Params

    def __init__(self, n_options: int = 2, **kwargs):
        super().__init__(**kwargs)
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


class QlearnWM(BasePolicy):
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

    'beta_rl' and 'beta_wm' are internal parameters; the outer softmax
    in 'DecisionModel' should be neutralised by pairing with 'NoBeta()'.

    Reference
    ---------
    Collins, A. G. E. & Frank, M. J. (2012). How much of reinforcement
    learning is working memory, not reinforcement learning? European
    Journal of Neuroscience, 35(7), 1024-1035.
    """

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha_rl = ParameterSpec("alpha_rl", (0.0, 1.0), description="RL learning rate")
        beta_rl = ParameterSpec(
            "beta_rl", (0.1, 20.0), description="RL inverse temperature"
        )
        beta_wm = ParameterSpec(
            "beta_wm", (0.1, 20.0), description="WM inverse temperature"
        )
        decay = ParameterSpec(
            "decay", (0.0, 1.0), description="Decay rate toward initial Q"
        )
        w0 = ParameterSpec(
            "w0", (0.0, 1.0), default=0.5, description="Initial WM weight"
        )

    params: Params

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


class QlearnDynamicLR(BasePolicy):
    """
    2-arm Q-learning with dynamic learning rate.

    The learning rate is updated based on the prediction error magnitude.
    """

    class Params(ParameterGroup):
        alpha_c = ParameterSpec(
            "alpha_c", (0.0, 1.0), description="Learning rate for chosen option"
        )
        alpha_u = ParameterSpec(
            "alpha_u", (0.0, 1.0), description="Learning rate for unchosen option"
        )
        w_c = ParameterSpec(
            "w_c",
            (0.0, 1.0),
            description="Weight for chosen option learning rate update",
        )
        w_u = ParameterSpec(
            "w_u",
            (0.0, 1.0),
            description="Weight for unchosen option learning rate update",
        )

    params: Params

    def reset(self):
        self.q = np.full(2, 0.5)
        self.alpha_c = self.params["alpha_c"]
        self.alpha_u = self.params["alpha_u"]

    def forget(self):
        pass

    def logits(self):
        return self.q.copy()

    def update(self, choice, reward):
        other = 1 - choice
        pe = reward - self.q[choice]

        # Update learning rates based on prediction error (EWMA toward |pe|)
        self.alpha_c = (
            self.params["w_c"] * abs(pe) + (1 - self.params["w_c"]) * self.alpha_c
        )
        self.alpha_u = (
            self.params["w_u"] * abs(pe) + (1 - self.params["w_u"]) * self.alpha_u
        )

        # Update Q-values
        self.q[choice] += self.alpha_c * pe
        self.q[other] += self.alpha_u * pe

        # Clamp Q-values to [0, 1]
        np.clip(self.q, 0.0, 1.0, out=self.q)


class QlearnAdaptiveLR(BasePolicy):
    """
    2-arm Q-learning with a reward-rate dependent adaptive learning rate.

    The learning rate decreases as the recent (EWMA) reward rate rises,
    capturing the idea that once reward has become reliably predictable
    (e.g. the animal has settled on the better port), further updates
    should shrink; a drop in reward rate raises the learning rate back up.
    This is a simple scalar heuristic, not a reimplementation of any
    specific published model.

    Note that reward rate is a lagging indicator: right after a reversal
    it stays high for a few trials before dropping, so the learning rate
    is briefly slow to recover exactly when fast relearning matters most.

    Update rule:
    r_bar <- r_bar + w_r * (reward - r_bar)
    alpha_c <- clip(alpha_c0 - kappa_c * r_bar, 0, 1)
    alpha_u <- clip(alpha_u0 - kappa_u * r_bar, 0, 1)
    pe = reward - Q[choice]
    Q[choice] += alpha_c * pe
    Q[unchosen] += alpha_u * pe
    """

    class Params(ParameterGroup):
        alpha_c0 = ParameterSpec(
            "alpha_c0",
            (0.0, 0.99),
            description="Learning rate for chosen option at zero reward rate",
        )
        alpha_u0 = ParameterSpec(
            "alpha_u0",
            (-0.99, 0.99),
            description="Learning rate for unchosen option at zero reward rate",
        )
        kappa_c = ParameterSpec(
            "kappa_c",
            (0.0, 2.0),
            description="Drop in chosen learning rate per unit reward rate",
        )
        kappa_u = ParameterSpec(
            "kappa_u",
            (0.0, 2.0),
            description="Drop in unchosen learning rate per unit reward rate",
        )
        w_r = ParameterSpec(
            "w_r",
            (0.02, 0.5),
            description="EWMA weight for the reward-rate trace",
        )
        bias = ParameterSpec(
            "bias", (-2.0, 2.0), default=0.0, description="Bias toward port 0 vs port 1"
        )

    params: Params

    def reset(self):
        self.q = np.full(2, 0.5)
        self.r_bar = 0.5
        self.alpha_c = self.params["alpha_c0"]
        self.alpha_u = self.params["alpha_u0"]

    def forget(self):
        pass

    def logits(self):
        b = self.params["bias"]
        return np.array([self.q[0] + b, self.q[1] - b])

    def update(self, choice, reward):
        other = 1 - choice
        pe = reward - self.q[choice]

        self.r_bar += self.params["w_r"] * (reward - self.r_bar)

        self.alpha_c = np.clip(
            self.params["alpha_c0"] - self.params["kappa_c"] * self.r_bar, 0.0, 1.0
        )
        self.alpha_u = np.clip(
            self.params["alpha_u0"] - self.params["kappa_u"] * self.r_bar, -1.0, 1.0
        )

        self.q[choice] += self.alpha_c * pe
        self.q[other] += self.alpha_u * pe
        np.clip(self.q, 0.0, 1.0, out=self.q)


class QlearnDiff(BasePolicy):
    """
    2-arm Q-learning driven by the chosen-vs-unchosen value difference.

    Instead of separate learning rates for the chosen and unchosen options,
    a single alpha scales the update to the difference between them. The
    difference ranges from -1 to 1, and the reward prediction error compares
    reward to the absolute value of that difference (how confidently the
    chosen option was already favoured).

    Update rule:
    diff = Q[choice] - Q[unchosen]
    rpe = reward - abs(diff)
    Q[choice] += alpha * rpe
    Q[unchosen] -= alpha * rpe

    Choice logits:
    logit[0] = Q[0] + bias
    logit[1] = Q[1] - bias
    """

    class Params(ParameterGroup):
        alpha = ParameterSpec(
            "alpha", (0.0, 0.99), description="Learning rate for the chosen/unchosen difference"
        )

        bias = ParameterSpec(
            "bias", (-2.0, 2.0), default=0.0, description="Bias toward port 0 vs port 1"
        )

    params: Params

    def reset(self):
        self.q = np.full(2, 0.5)

    def forget(self):
        pass

    def logits(self):
        b = self.params["bias"]
        return np.array([self.q[0] + b, self.q[1] - b])

    def update(self, choice, reward):
        a = self.params["alpha"]
        other = 1 - choice

        diff = self.q[choice] - self.q[other]
        rpe = reward - abs(diff)

        delta = a * rpe
        self.q[choice] += delta
        self.q[other] -= delta

        self.q[:] = np.clip(self.q, 0.0, 1.0)

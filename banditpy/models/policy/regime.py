import numpy as np
from .base import BasePolicy, ParameterGroup, ParameterSpec
from .beta_schedule import NoBeta

N_REGIMES = 3


def _softmax(x: np.ndarray, beta: float) -> np.ndarray:
    z = beta * x
    z -= z.max()
    e = np.exp(z)
    s = e.sum()
    if s <= 0:
        return np.full_like(x, 1.0 / len(x))
    return e / s


class Qlearn3Regime(BasePolicy):
    """
    3-regime unsupervised HMM over latent behavioral regimes, each with its
    own Q-learning sub-policy: chosen/unchosen learning rates
    'alpha_c_k'/'alpha_u_k' (same rule as 'Qlearn') and inverse temp
    'beta_k'. No block-type label is given — regime is inferred purely
    from the choice/reward sequence. Belief 'b' propagates via a symmetric
    transition matrix with shared persistence 'stay'; with 'stay'->1 a
    single regime reduces to 'Qlearn'.

    'logits()' returns the log of the belief-weighted mixture of each
    regime's choice probs (paired with 'NoBeta'). Because the belief
    update is standard HMM forward-filtering, per-trial log 'logits()'
    summed by 'DecisionModel' is exactly the HMM marginal log-likelihood,
    so this fits directly into 'fit()'/'cross_validate()', no EM needed.

    Note: shared 'stay' forces occupancy to be exactly uniform (1/3 each)
    regardless of its value — see 'QlearnDiff3StayRegime' for per-regime
    'stay' and a closed-form occupancy.

    Per trial: lik[k] = P(choice | regime k); resp = normalize(b * lik);
    q[k,c] += alpha_c_k * resp[k] * (reward - q[k,c]); q[k,~c] the same
    with alpha_u_k; b <- resp @ T(stay).
    """

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha_c_0 = ParameterSpec(
            "alpha_c_0", (0.0, 0.99), description="Learning rate (chosen), regime 0"
        )
        alpha_u_0 = ParameterSpec(
            "alpha_u_0", (-0.99, 0.99), description="Learning rate (unchosen), regime 0"
        )
        alpha_c_1 = ParameterSpec(
            "alpha_c_1", (0.0, 0.99), description="Learning rate (chosen), regime 1"
        )
        alpha_u_1 = ParameterSpec(
            "alpha_u_1", (-0.99, 0.99), description="Learning rate (unchosen), regime 1"
        )
        alpha_c_2 = ParameterSpec(
            "alpha_c_2", (0.0, 0.99), description="Learning rate (chosen), regime 2"
        )
        alpha_u_2 = ParameterSpec(
            "alpha_u_2", (-0.99, 0.99), description="Learning rate (unchosen), regime 2"
        )
        beta_0 = ParameterSpec(
            "beta_0", (0.1, 20.0), description="Inverse temp, regime 0"
        )
        beta_1 = ParameterSpec(
            "beta_1", (0.1, 20.0), description="Inverse temp, regime 1"
        )
        beta_2 = ParameterSpec(
            "beta_2", (0.1, 20.0), description="Inverse temp, regime 2"
        )
        stay = ParameterSpec(
            "stay",
            (0.0, 0.99),
            description="Probability of remaining in the same regime",
        )

    params: Params

    def reset(self):
        self.q = np.full((N_REGIMES, 2), 0.5)
        self.b = np.full(N_REGIMES, 1.0 / N_REGIMES)

    def forget(self):
        pass

    def get_state(self):
        return self.b.copy()

    def _regime_choice_probs(self):
        p = self.params
        betas = (p["beta_0"], p["beta_1"], p["beta_2"])
        return np.vstack([_softmax(self.q[k], betas[k]) for k in range(N_REGIMES)])

    def logits(self):
        opt_probs = self._regime_choice_probs()
        p_action = self.b @ opt_probs
        p_action = np.clip(p_action, 1e-9, 1.0)
        return np.log(p_action)

    def update(self, choice, reward):
        opt_probs = self._regime_choice_probs()
        lik = opt_probs[:, choice]

        resp = self.b * lik
        resp_sum = resp.sum()
        if resp_sum <= 0:
            resp = np.full(N_REGIMES, 1.0 / N_REGIMES)
        else:
            resp /= resp_sum

        p = self.params
        alphas_c = (p["alpha_c_0"], p["alpha_c_1"], p["alpha_c_2"])
        alphas_u = (p["alpha_u_0"], p["alpha_u_1"], p["alpha_u_2"])
        other = 1 - choice
        for k in range(N_REGIMES):
            pe = reward - self.q[k, choice]
            self.q[k, choice] += alphas_c[k] * resp[k] * pe
            self.q[k, other] += alphas_u[k] * resp[k] * pe
        np.clip(self.q, 0.0, 1.0, out=self.q)

        stay = p["stay"]
        switch = (1.0 - stay) / (N_REGIMES - 1)
        T = np.full((N_REGIMES, N_REGIMES), switch)
        np.fill_diagonal(T, stay)

        self.b = resp @ T
        self.b /= self.b.sum()


class QlearnDiff1StayRegime(BasePolicy):
    """
    'Qlearn3Regime' with 'QlearnDiff's difference-based update in place of
    separate chosen/unchosen rates: one 'alpha_k' per regime, still moving
    both options each trial (7 free params vs. 10). With 'stay'->1, a
    single regime reduces to 'QlearnDiff'.

    Note: shared 'stay' forces occupancy to be exactly uniform (1/3 each)
    — see 'QlearnDiff3StayRegime' for per-regime 'stay'.

    Per trial: lik[k] = P(choice | regime k); resp = normalize(b * lik);
    diff_k = q[k,c] - q[k,~c]; rpe_k = reward - abs(diff_k);
    delta_k = alpha_k * resp[k] * rpe_k; q[k,c] += delta_k;
    q[k,~c] -= delta_k; b <- resp @ T(stay).
    """

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha_0 = ParameterSpec(
            "alpha_0", (0.0, 0.99), description="Learning rate, regime 0"
        )
        alpha_1 = ParameterSpec(
            "alpha_1", (0.0, 0.99), description="Learning rate, regime 1"
        )
        alpha_2 = ParameterSpec(
            "alpha_2", (0.0, 0.99), description="Learning rate, regime 2"
        )
        beta_0 = ParameterSpec(
            "beta_0", (0.1, 20.0), description="Inverse temp, regime 0"
        )
        beta_1 = ParameterSpec(
            "beta_1", (0.1, 20.0), description="Inverse temp, regime 1"
        )
        beta_2 = ParameterSpec(
            "beta_2", (0.1, 20.0), description="Inverse temp, regime 2"
        )
        stay = ParameterSpec(
            "stay",
            (0.0, 0.99),
            description="Probability of remaining in the same regime",
        )

    params: Params

    def reset(self):
        self.q = np.full((N_REGIMES, 2), 0.5)
        self.b = np.full(N_REGIMES, 1.0 / N_REGIMES)

    def forget(self):
        pass

    def get_state(self):
        return self.b.copy()

    def _regime_choice_probs(self):
        p = self.params
        betas = (p["beta_0"], p["beta_1"], p["beta_2"])
        return np.vstack([_softmax(self.q[k], betas[k]) for k in range(N_REGIMES)])

    def logits(self):
        opt_probs = self._regime_choice_probs()
        p_action = self.b @ opt_probs
        p_action = np.clip(p_action, 1e-9, 1.0)
        return np.log(p_action)

    def update(self, choice, reward):
        opt_probs = self._regime_choice_probs()
        lik = opt_probs[:, choice]

        resp = self.b * lik
        resp_sum = resp.sum()
        if resp_sum <= 0:
            resp = np.full(N_REGIMES, 1.0 / N_REGIMES)
        else:
            resp /= resp_sum

        p = self.params
        alphas = (p["alpha_0"], p["alpha_1"], p["alpha_2"])
        other = 1 - choice
        for k in range(N_REGIMES):
            diff = self.q[k, choice] - self.q[k, other]
            rpe = reward - abs(diff)
            delta = alphas[k] * resp[k] * rpe
            self.q[k, choice] += delta
            self.q[k, other] -= delta
        np.clip(self.q, 0.0, 1.0, out=self.q)

        stay = p["stay"]
        switch = (1.0 - stay) / (N_REGIMES - 1)
        T = np.full((N_REGIMES, N_REGIMES), switch)
        np.fill_diagonal(T, stay)

        self.b = resp @ T
        self.b /= self.b.sum()


class QlearnDiff3StayRegime(BasePolicy):
    """
    'QlearnDiff1StayRegime' with per-regime persistence 'stay_0'/'stay_1'/
    'stay_2' instead of one shared 'stay'. A shared 'stay' forces uniform
    (1/3, 1/3, 1/3) occupancy regardless of its value; per-regime
    'stay_k' lifts that constraint. For this transition design
    (off-diagonal mass split uniformly), stationary occupancy has a
    closed form via flux balance:

        pi_k proportional to 1 / (1 - stay_k)     (mean dwell time)

    so a lopsided occupancy split (e.g. one dominant regime) is something
    the fit can discover without being told the task's true proportions.
    'occupancy()' returns this distribution.

    Per trial: same as 'QlearnDiff1StayRegime', but
    b <- resp @ T(stay_0, stay_1, stay_2).
    """

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha_0 = ParameterSpec(
            "alpha_0", (0.0, 0.99), description="Learning rate, regime 0"
        )
        alpha_1 = ParameterSpec(
            "alpha_1", (0.0, 0.99), description="Learning rate, regime 1"
        )
        alpha_2 = ParameterSpec(
            "alpha_2", (0.0, 0.99), description="Learning rate, regime 2"
        )
        beta_0 = ParameterSpec(
            "beta_0", (0.1, 20.0), description="Inverse temp, regime 0"
        )
        beta_1 = ParameterSpec(
            "beta_1", (0.1, 20.0), description="Inverse temp, regime 1"
        )
        beta_2 = ParameterSpec(
            "beta_2", (0.1, 20.0), description="Inverse temp, regime 2"
        )
        stay_0 = ParameterSpec(
            "stay_0", (0.0, 0.99), description="Probability of remaining in regime 0"
        )
        stay_1 = ParameterSpec(
            "stay_1", (0.0, 0.99), description="Probability of remaining in regime 1"
        )
        stay_2 = ParameterSpec(
            "stay_2", (0.0, 0.99), description="Probability of remaining in regime 2"
        )

    params: Params

    def reset(self):
        self.q = np.full((N_REGIMES, 2), 0.5)
        self.b = np.full(N_REGIMES, 1.0 / N_REGIMES)

    def forget(self):
        pass

    def get_state(self):
        return self.b.copy()

    def _regime_choice_probs(self):
        p = self.params
        betas = (p["beta_0"], p["beta_1"], p["beta_2"])
        return np.vstack([_softmax(self.q[k], betas[k]) for k in range(N_REGIMES)])

    def logits(self):
        opt_probs = self._regime_choice_probs()
        p_action = self.b @ opt_probs
        p_action = np.clip(p_action, 1e-9, 1.0)
        return np.log(p_action)

    def _transition_matrix(self):
        p = self.params
        stays = (p["stay_0"], p["stay_1"], p["stay_2"])
        T = np.empty((N_REGIMES, N_REGIMES))
        for k in range(N_REGIMES):
            switch_k = (1.0 - stays[k]) / (N_REGIMES - 1)
            T[k, :] = switch_k
            T[k, k] = stays[k]
        return T

    def update(self, choice, reward):
        opt_probs = self._regime_choice_probs()
        lik = opt_probs[:, choice]

        resp = self.b * lik
        resp_sum = resp.sum()
        if resp_sum <= 0:
            resp = np.full(N_REGIMES, 1.0 / N_REGIMES)
        else:
            resp /= resp_sum

        p = self.params
        alphas = (p["alpha_0"], p["alpha_1"], p["alpha_2"])
        other = 1 - choice
        for k in range(N_REGIMES):
            diff = self.q[k, choice] - self.q[k, other]
            rpe = reward - abs(diff)
            delta = alphas[k] * resp[k] * rpe
            self.q[k, choice] += delta
            self.q[k, other] -= delta
        np.clip(self.q, 0.0, 1.0, out=self.q)

        self.b = resp @ self._transition_matrix()
        self.b /= self.b.sum()

    def occupancy(self):
        """Closed-form stationary occupancy, 'pi_k = 1/(1-stay_k)'
        normalized — long-run fraction of trials in each regime."""
        p = self.params
        inv_leave = np.array(
            [
                1.0 / (1.0 - p["stay_0"]),
                1.0 / (1.0 - p["stay_1"]),
                1.0 / (1.0 - p["stay_2"]),
            ]
        )
        return inv_leave / inv_leave.sum()


class MoARegime(BasePolicy):
    """
    Mixture-of-agents HMM (MoA-HMM), after Venditto, Miller, Brody & Daw
    ("Dynamic reinforcement learning reveals time-dependent shifts in
    strategy during reward learning"). Unlike the other classes here,
    regimes don't own private Q-values — a fixed set of agents (mbr, mbc,
    mfr, mfc, bias) updates *unconditionally* every trial regardless of
    current regime belief; only the weights combining them into a choice
    are regime-specific:

        V_k(y) = sum_A beta_A_k * Q_A(y);   p(y | regime k) = softmax(V_k)

    Agents:
    - mfr: Q *= (1-alpha_mfr); Q[c] += alpha_mfr * reward.
    - mfc: Q *= (1-alpha_mfc); Q[c] += alpha_mfc (ignores reward).
    - bias: fixed [+1, -1], never updated; weight 'beta_bias' is shared
      across regimes (a side preference is a stable trait, not something
      that should flip with strategy).
    - mbr/mbc: the paper's versions credit-assign via a "common vs. rare
      transition" flag from the two-step task, which has no analogue in a
      2-armed bandit (no intermediate state). Reinterpreted here with the
      coupled chosen/unchosen update from 'QlearnDiff', exploiting the
      structured task's known arm anti-correlation instead:

        pe_mbr = reward - Q_mbr[c]; Q_mbr[c] += alpha_mbr*pe_mbr; Q_mbr[~c] -= alpha_mbr*pe_mbr
        pe_mbc = 1 - Q_mbc[c];      Q_mbc[c] += alpha_mbc*pe_mbc; Q_mbc[~c] -= alpha_mbc*pe_mbc

      (mbc derived from mbr by swapping reward for a constant, the same
      way the paper derives mfc from mfr.) A deliberate reinterpretation,
      not a literal port.

    Transitions use a full 3x3 matrix — 'softmax([0, trans_k_1, trans_k_2])'
    per row k, regime 0 as reference — rather than a 'stay' parameter,
    matching the paper's asymmetric transition dynamics. 'occupancy()'
    solves for the stationary distribution numerically (no closed form for
    a general matrix).

    23 free params: 4 agent alphas (shared across regimes) + 4x3 regime
    combination weights + 1 shared beta_bias + 6 transition logits.
    """

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha_mbr = ParameterSpec(
            "alpha_mbr",
            (0.0, 0.99),
            description="Learning rate, model-based reward agent",
        )
        alpha_mbc = ParameterSpec(
            "alpha_mbc",
            (0.0, 0.99),
            description="Learning rate, model-based choice agent",
        )
        alpha_mfr = ParameterSpec(
            "alpha_mfr",
            (0.0, 0.99),
            description="Learning rate, model-free reward agent",
        )
        alpha_mfc = ParameterSpec(
            "alpha_mfc", (0.0, 0.99), description="Learning rate, choice-kernel agent"
        )
        beta_mbr_0 = ParameterSpec(
            "beta_mbr_0", (-10.0, 10.0), description="Regime 0 weight on mbr agent"
        )
        beta_mbc_0 = ParameterSpec(
            "beta_mbc_0", (-10.0, 10.0), description="Regime 0 weight on mbc agent"
        )
        beta_mfr_0 = ParameterSpec(
            "beta_mfr_0", (-10.0, 10.0), description="Regime 0 weight on mfr agent"
        )
        beta_mfc_0 = ParameterSpec(
            "beta_mfc_0", (-10.0, 10.0), description="Regime 0 weight on mfc agent"
        )
        beta_mbr_1 = ParameterSpec(
            "beta_mbr_1", (-10.0, 10.0), description="Regime 1 weight on mbr agent"
        )
        beta_mbc_1 = ParameterSpec(
            "beta_mbc_1", (-10.0, 10.0), description="Regime 1 weight on mbc agent"
        )
        beta_mfr_1 = ParameterSpec(
            "beta_mfr_1", (-10.0, 10.0), description="Regime 1 weight on mfr agent"
        )
        beta_mfc_1 = ParameterSpec(
            "beta_mfc_1", (-10.0, 10.0), description="Regime 1 weight on mfc agent"
        )
        beta_mbr_2 = ParameterSpec(
            "beta_mbr_2", (-10.0, 10.0), description="Regime 2 weight on mbr agent"
        )
        beta_mbc_2 = ParameterSpec(
            "beta_mbc_2", (-10.0, 10.0), description="Regime 2 weight on mbc agent"
        )
        beta_mfr_2 = ParameterSpec(
            "beta_mfr_2", (-10.0, 10.0), description="Regime 2 weight on mfr agent"
        )
        beta_mfc_2 = ParameterSpec(
            "beta_mfc_2", (-10.0, 10.0), description="Regime 2 weight on mfc agent"
        )
        beta_bias = ParameterSpec(
            "beta_bias",
            (-10.0, 10.0),
            description="Weight on bias agent (shared across regimes)",
        )
        trans_0_1 = ParameterSpec(
            "trans_0_1",
            (-10.0, 10.0),
            description="Logit, regime 0 -> regime 1 (ref: regime 0)",
        )
        trans_0_2 = ParameterSpec(
            "trans_0_2",
            (-10.0, 10.0),
            description="Logit, regime 0 -> regime 2 (ref: regime 0)",
        )
        trans_1_1 = ParameterSpec(
            "trans_1_1",
            (-10.0, 10.0),
            description="Logit, regime 1 -> regime 1 (ref: regime 0)",
        )
        trans_1_2 = ParameterSpec(
            "trans_1_2",
            (-10.0, 10.0),
            description="Logit, regime 1 -> regime 2 (ref: regime 0)",
        )
        trans_2_1 = ParameterSpec(
            "trans_2_1",
            (-10.0, 10.0),
            description="Logit, regime 2 -> regime 1 (ref: regime 0)",
        )
        trans_2_2 = ParameterSpec(
            "trans_2_2",
            (-10.0, 10.0),
            description="Logit, regime 2 -> regime 2 (ref: regime 0)",
        )

    params: Params

    def reset(self):
        self.q_mbr = np.full(2, 0.5)
        self.q_mbc = np.full(2, 0.5)
        self.q_mfr = np.zeros(2)
        self.q_mfc = np.zeros(2)
        self.q_bias = np.array([1.0, -1.0])
        self.b = np.full(N_REGIMES, 1.0 / N_REGIMES)

    def forget(self):
        pass

    def get_state(self):
        return self.b.copy()

    def _agent_values(self):
        return np.vstack([self.q_mbr, self.q_mbc, self.q_mfr, self.q_mfc])

    def _regime_betas(self):
        p = self.params
        return np.array(
            [
                [p["beta_mbr_0"], p["beta_mbc_0"], p["beta_mfr_0"], p["beta_mfc_0"]],
                [p["beta_mbr_1"], p["beta_mbc_1"], p["beta_mfr_1"], p["beta_mfc_1"]],
                [p["beta_mbr_2"], p["beta_mbc_2"], p["beta_mfr_2"], p["beta_mfc_2"]],
            ]
        )

    def _regime_choice_probs(self):
        betas = self._regime_betas()  # (N_REGIMES, N_AGENTS)
        agents = self._agent_values()  # (N_AGENTS, 2)
        # Broadcast-accumulate instead of `betas @ agents` (2D-by-2D `@`):
        # this environment's numpy/BLAS build hard-crashes (no traceback)
        # on small 2D matmuls, while 1D-by-2D products and elementwise ops
        # are unaffected — see `self.b @ opt_probs` below, which is safe.
        V = np.zeros((N_REGIMES, 2))
        for a in range(agents.shape[0]):
            V += betas[:, a : a + 1] * agents[a]
        V += self.params["beta_bias"] * self.q_bias  # shared across regimes
        return np.vstack([_softmax(V[k], 1.0) for k in range(N_REGIMES)])

    def logits(self):
        opt_probs = self._regime_choice_probs()
        p_action = self.b @ opt_probs
        p_action = np.clip(p_action, 1e-9, 1.0)
        return np.log(p_action)

    def _transition_matrix(self):
        p = self.params
        # Row k = softmax([0, trans_k_1, trans_k_2]): regime 0 is a fixed
        # logit-0 reference destination, so this covers the full simplex
        # per row (a general, possibly asymmetric, transition matrix) with
        # only 2 free parameters per row.
        row_logits = np.array(
            [
                [0.0, p["trans_0_1"], p["trans_0_2"]],
                [0.0, p["trans_1_1"], p["trans_1_2"]],
                [0.0, p["trans_2_1"], p["trans_2_2"]],
            ]
        )
        return np.vstack([_softmax(row_logits[k], 1.0) for k in range(N_REGIMES)])

    def update(self, choice, reward):
        opt_probs = self._regime_choice_probs()
        lik = opt_probs[:, choice]

        resp = self.b * lik
        resp_sum = resp.sum()
        if resp_sum <= 0:
            resp = np.full(N_REGIMES, 1.0 / N_REGIMES)
        else:
            resp /= resp_sum

        p = self.params
        other = 1 - choice

        # unconditional agent updates — not scaled by resp
        pe_mbr = reward - self.q_mbr[choice]
        self.q_mbr[choice] += p["alpha_mbr"] * pe_mbr
        self.q_mbr[other] -= p["alpha_mbr"] * pe_mbr
        np.clip(self.q_mbr, 0.0, 1.0, out=self.q_mbr)

        pe_mbc = 1.0 - self.q_mbc[choice]
        self.q_mbc[choice] += p["alpha_mbc"] * pe_mbc
        self.q_mbc[other] -= p["alpha_mbc"] * pe_mbc
        np.clip(self.q_mbc, 0.0, 1.0, out=self.q_mbc)

        self.q_mfr *= 1.0 - p["alpha_mfr"]
        self.q_mfr[choice] += p["alpha_mfr"] * reward

        self.q_mfc *= 1.0 - p["alpha_mfc"]
        self.q_mfc[choice] += p["alpha_mfc"]

        self.b = resp @ self._transition_matrix()
        self.b /= self.b.sum()

    def occupancy(self):
        """Long-run regime occupancy — stationary distribution of the
        fitted transition matrix, solved numerically (no closed form for
        a general matrix)."""
        T = self._transition_matrix()
        eigvals, eigvecs = np.linalg.eig(T.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        stat = np.real(eigvecs[:, idx])
        stat = np.clip(stat, 0.0, None)
        return stat / stat.sum()


class Qlearn2Regime(BasePolicy):
    """
    2-regime mixture-of-agents HMM — 'MoARegime's mechanism (fixed, shared
    agents that update unconditionally every trial; only the weights
    combining them into a choice are regime-specific), scaled down to 2
    agents and 2 regimes instead of 4 and 3.

    Agents: 'q1'/'q2' are two independent, symmetric 'Qlearn'-style
    submodels (own chosen/unchosen rates 'alpha_c_1'/'alpha_u_1' and
    'alpha_c_2'/'alpha_u_2'; standard counterfactual update, not the
    diff-based one) — unlike 'MoARegime's agents there's no built-in
    qualitative difference between them, that's left for the fit to
    discover via their regime-specific weights. 'bias' is a fixed
    [+1, -1] contrast, never updated, with a single weight 'beta_bias'
    shared across regimes (same mechanic as 'MoARegime's bias agent).

        V_k(y) = beta_q1_k*Q1(y) + beta_q2_k*Q2(y) + beta_bias*Q_bias(y)
        p(y | regime k) = softmax(V_k)

    With only 2 regimes, per-regime persistence 'stay_0'/'stay_1' is
    already the fully general transition matrix (nothing left to split
    when there's only one other regime to switch to), so the same
    closed-form occupancy as 'QlearnDiff3StayRegime' applies:
    'pi_k = 1/(1-stay_k)' normalized.

    Per trial: lik[k] = P(choice | regime k) from pre-update agent
    values; resp = normalize(b * lik); q1, q2 updated unconditionally
    (NOT scaled by resp — the key difference from 'Qlearn3Regime'-style
    classes) via pe = reward - q[c], q[c] += alpha_c*pe, q[~c] += alpha_u*pe;
    b <- resp @ [[stay_0, 1-stay_0], [1-stay_1, stay_1]].
    """

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha_c_1 = ParameterSpec(
            "alpha_c_1", (0.0, 0.99), description="Learning rate (chosen), agent 1"
        )
        alpha_u_1 = ParameterSpec(
            "alpha_u_1", (-0.99, 0.99), description="Learning rate (unchosen), agent 1"
        )
        alpha_c_2 = ParameterSpec(
            "alpha_c_2", (0.0, 0.99), description="Learning rate (chosen), agent 2"
        )
        alpha_u_2 = ParameterSpec(
            "alpha_u_2", (-0.99, 0.99), description="Learning rate (unchosen), agent 2"
        )
        beta_q1_0 = ParameterSpec(
            "beta_q1_0", (-10.0, 10.0), description="Regime 0 weight on agent 1"
        )
        beta_q2_0 = ParameterSpec(
            "beta_q2_0", (-10.0, 10.0), description="Regime 0 weight on agent 2"
        )
        beta_q1_1 = ParameterSpec(
            "beta_q1_1", (-10.0, 10.0), description="Regime 1 weight on agent 1"
        )
        beta_q2_1 = ParameterSpec(
            "beta_q2_1", (-10.0, 10.0), description="Regime 1 weight on agent 2"
        )
        beta_bias = ParameterSpec(
            "beta_bias",
            (-10.0, 10.0),
            description="Weight on bias agent (shared across regimes)",
        )
        stay_0 = ParameterSpec(
            "stay_0", (0.0, 0.99), description="Probability of remaining in regime 0"
        )
        stay_1 = ParameterSpec(
            "stay_1", (0.0, 0.99), description="Probability of remaining in regime 1"
        )

    params: Params

    def reset(self):
        self.q1 = np.full(2, 0.5)
        self.q2 = np.full(2, 0.5)
        self.q_bias = np.array([1.0, -1.0])
        self.b = np.full(2, 0.5)

    def forget(self):
        pass

    def get_state(self):
        return self.b.copy()

    def _agent_values(self):
        return np.vstack([self.q1, self.q2])

    def _regime_betas(self):
        p = self.params
        return np.array(
            [
                [p["beta_q1_0"], p["beta_q2_0"]],
                [p["beta_q1_1"], p["beta_q2_1"]],
            ]
        )

    def _regime_choice_probs(self):
        betas = self._regime_betas()  # (2, 2)
        agents = self._agent_values()  # (2, 2)
        # Broadcast-accumulate instead of `betas @ agents` — see
        # 'MoARegime._regime_choice_probs' for why (2D `@` crashes here).
        V = np.zeros((2, 2))
        for a in range(agents.shape[0]):
            V += betas[:, a : a + 1] * agents[a]
        V += self.params["beta_bias"] * self.q_bias
        return np.vstack([_softmax(V[k], 1.0) for k in range(2)])

    def logits(self):
        opt_probs = self._regime_choice_probs()
        p_action = self.b @ opt_probs
        p_action = np.clip(p_action, 1e-9, 1.0)
        return np.log(p_action)

    def _transition_matrix(self):
        p = self.params
        s0, s1 = p["stay_0"], p["stay_1"]
        return np.array([[s0, 1.0 - s0], [1.0 - s1, s1]])

    def update(self, choice, reward):
        opt_probs = self._regime_choice_probs()
        lik = opt_probs[:, choice]

        resp = self.b * lik
        resp_sum = resp.sum()
        if resp_sum <= 0:
            resp = np.full(2, 0.5)
        else:
            resp /= resp_sum

        p = self.params
        other = 1 - choice

        # unconditional agent updates — not scaled by resp
        pe1 = reward - self.q1[choice]
        self.q1[choice] += p["alpha_c_1"] * pe1
        self.q1[other] += p["alpha_u_1"] * pe1
        np.clip(self.q1, 0.0, 1.0, out=self.q1)

        pe2 = reward - self.q2[choice]
        self.q2[choice] += p["alpha_c_2"] * pe2
        self.q2[other] += p["alpha_u_2"] * pe2
        np.clip(self.q2, 0.0, 1.0, out=self.q2)

        self.b = resp @ self._transition_matrix()
        self.b /= self.b.sum()

    def occupancy(self):
        """Closed-form stationary occupancy, 'pi_k = 1/(1-stay_k)'
        normalized — long-run fraction of trials in each regime."""
        p = self.params
        inv_leave = np.array([1.0 / (1.0 - p["stay_0"]), 1.0 / (1.0 - p["stay_1"])])
        return inv_leave / inv_leave.sum()

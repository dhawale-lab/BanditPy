import numpy as np


def _sample_pairs(p, n, rng, structured):
    """Sample probability pairs; structured pairs sum to 1, unstructured pairs do not."""
    p1 = rng.choice(p, size=n, replace=True)
    p2 = np.empty_like(p1, dtype=float)

    for i, val in enumerate(p1):
        if structured:
            p2[i] = np.round(1 - val, 1)
        else:
            # pick any probability that is neither equal to p1 nor the structured complement
            options = np.setdiff1d(p, [val, np.round(1 - val, 1)])
            p2[i] = rng.choice(options)

        # if equality sneaks in, resample that element
        if p2[i] == val:
            options = np.setdiff1d(p, [val]) if structured else np.setdiff1d(p, [val, p2[i]])
            p2[i] = rng.choice(options)

    return np.column_stack((p1, p2))


def generate_probs_2arm(p=None, N=1000, frac_impurity=0.16):
    """Generate structured (sum to 1) and unstructured (sum != 1) 2-arm probabilities.

    Returns a tuple `(pu, ps)` each of shape (N, 2), with a fraction `frac_impurity`
    of cross-contamination between structured and unstructured sets. Equal-arm pairs
    are excluded.
    """
    if p is None:
        p = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])

    if not (0 <= frac_impurity < 0.5):
        raise ValueError("frac_impurity must be in [0, 0.5)")

    rng = np.random.default_rng()

    n_impure = int(N * frac_impurity)
    n_pure = N - n_impure

    # Structured set: mostly sums to 1, with impure unstructured pairs mixed in
    ps_struct = _sample_pairs(p, n_pure, rng, structured=True)
    ps_impure = _sample_pairs(p, n_impure, rng, structured=False)
    ps = np.vstack((ps_struct, ps_impure))
    rng.shuffle(ps)

    # Unstructured set: mostly not summing to 1, with impure structured pairs mixed in
    pu_unstruct = _sample_pairs(p, n_pure, rng, structured=False)
    pu_impure = _sample_pairs(p, n_impure, rng, structured=True)
    pu = np.vstack((pu_unstruct, pu_impure))
    rng.shuffle(pu)

    # Sanity checks with tolerance for floating rounding
    ps_sum = ps.sum(axis=1)
    pu_sum = pu.sum(axis=1)
    frac_u_in_s = np.mean(~np.isclose(ps_sum, 1.0))
    frac_s_in_u = np.mean(np.isclose(pu_sum, 1.0))

    if frac_u_in_s > frac_impurity + 1e-6 or frac_s_in_u > frac_impurity + 1e-6:
        raise AssertionError("frac_impurity violated")

    if np.any(np.isclose(ps[:, 0], ps[:, 1])) or np.any(np.isclose(pu[:, 0], pu[:, 1])):
        raise AssertionError("Equal probabilities detected")

    return pu, ps

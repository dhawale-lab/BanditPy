import numpy as np


def _center_columns(X: np.ndarray) -> np.ndarray:
    """Subtract the column (feature) mean, i.e. center samples along axis 0."""
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X, Y) -> float:
    """Linear Centered Kernel Alignment (CKA) between two activation matrices.

    CKA measures how similar the representational geometry of two sets of
    activations is, and is invariant to orthogonal transformations (rotations/
    reflections) and isotropic scaling of each activation matrix - i.e. it does
    not care which direction in hidden-unit space corresponds to "left" or
    "right" in a given network. This makes it well suited for comparing RNN
    hidden states across independently trained networks whose unit ordering
    and axis directions are arbitrary.

    ``X`` and ``Y`` must have the same number of rows (samples), and rows must
    correspond to the same conditions/trials across the two matrices (e.g. both
    obtained by simulating each RNN on the same ``reward_schedule``), since CKA
    compares the pairwise similarity structure across matched samples.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features_x)
        Activations from the first network (e.g. hidden states).
    Y : array-like, shape (n_samples, n_features_y)
        Activations from the second network. ``n_features_y`` need not equal
        ``n_features_x``.

    Returns
    -------
    float
        CKA similarity in [0, 1], where 1 means identical representational
        geometry (up to rotation/reflection/isotropic scaling).

    References
    ----------
    Kornblith et al. (2019), "Similarity of Neural Network Representations
    Revisited", ICML.
    """
    X = _center_columns(np.asarray(X, dtype=float))
    Y = _center_columns(np.asarray(Y, dtype=float))

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples (rows), got {X.shape[0]} and {Y.shape[0]}"
        )

    cross_term = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    norm_x = np.linalg.norm(X.T @ X, ord="fro")
    norm_y = np.linalg.norm(Y.T @ Y, ord="fro")

    denom = norm_x * norm_y
    if denom == 0:
        return np.nan

    return cross_term / denom


def pairwise_cka(activations) -> np.ndarray:
    """Pairwise linear CKA matrix across a list of activation matrices.

    Parameters
    ----------
    activations : sequence of array-like, each shape (n_samples, n_features_i)
        Activation matrices (e.g. RNN hidden states) from ``n`` networks,
        each obtained on the same set of matched samples/conditions (same
        ``n_samples`` across all entries), so rows correspond across networks.
        ``n_features_i`` can differ across networks.

    Returns
    -------
    np.ndarray, shape (n, n)
        Symmetric matrix of pairwise linear CKA values (diagonal = 1).
    """
    n = len(activations)
    cka_matrix = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            cka_matrix[i, j] = cka_matrix[j, i] = linear_cka(
                activations[i], activations[j]
            )

    return cka_matrix

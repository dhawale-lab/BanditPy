import numpy as np


def orthogonal_procrustes_align(source, target, scale: bool = False):
    """Align ``source`` onto ``target`` via orthogonal Procrustes.

    Finds the rotation/reflection (and optional isotropic scale) that best
    superimposes ``source`` onto ``target``, using matched samples (rows)
    across both matrices. Unlike CKA (see ``linear_cka``), which only scores
    representational similarity, this returns an explicit transform that can
    be applied to project one network's activations/coordinates (e.g. PCA
    scores) into another network's coordinate frame - useful for overlaying
    or averaging point clouds across networks whose axis directions (e.g.
    sign of "left" vs "right" choice) are arbitrary.

    Parameters
    ----------
    source : array-like, shape (n_samples, p)
        Activations/coordinates to be rotated (e.g. one network's PCA scores).
    target : array-like, shape (n_samples, p)
        Reference activations/coordinates, matched sample-for-sample with
        ``source`` (same conditions/trials), and with the same shape.
    scale : bool, optional
        If True, also solve for an isotropic scale factor (full Procrustes).
        Default is False (rotation/reflection only).

    Returns
    -------
    aligned : np.ndarray, shape (n_samples, p)
        ``source``, rotated (and optionally scaled) and re-centered onto
        ``target``'s mean.
    R : np.ndarray, shape (p, p)
        Orthogonal rotation/reflection matrix.
    scale_factor : float
        Isotropic scale factor applied (1.0 if `scale=False`).
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    if source.shape != target.shape:
        raise ValueError(
            f"source and target must have matching shape, got {source.shape} and {target.shape}"
        )

    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    source_c = source - source_mean
    target_c = target - target_mean

    U, S, Vt = np.linalg.svd(source_c.T @ target_c)
    R = U @ Vt

    scale_factor = 1.0
    if scale:
        denom = np.linalg.norm(source_c, ord="fro") ** 2
        if denom > 0:
            scale_factor = S.sum() / denom

    aligned = scale_factor * (source_c @ R) + target_mean

    return aligned, R, scale_factor


def align_to_common_space(activations, reference_idx: int = 0, scale: bool = False):
    """Align a list of matched activation matrices into a common coordinate frame.

    Each entry in ``activations`` is aligned onto ``activations[reference_idx]``
    via ``orthogonal_procrustes_align``, so that e.g. PCA trajectories from
    multiple independently trained RNNs can be overlaid/averaged despite
    arbitrary axis directions.

    Parameters
    ----------
    activations : sequence of array-like, each shape (n_samples, p)
        Matched activations/coordinates (e.g. per-network PCA scores), with
        the same number of samples (matched conditions/trials) and the same
        number of columns across all entries.
    reference_idx : int, optional
        Index of the entry to treat as the fixed reference frame. Default 0.
    scale : bool, optional
        Passed to ``orthogonal_procrustes_align``. Default False.

    Returns
    -------
    list of np.ndarray
        Aligned versions of `activations`, in the same order. The reference
        entry is returned centered but otherwise unrotated.
    """
    reference = np.asarray(activations[reference_idx], dtype=float)
    aligned = []

    for i, arr in enumerate(activations):
        if i == reference_idx:
            aligned.append(np.asarray(arr, dtype=float))
            continue
        arr_aligned, _, _ = orthogonal_procrustes_align(arr, reference, scale=scale)
        aligned.append(arr_aligned)

    return aligned

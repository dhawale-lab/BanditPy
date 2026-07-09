import numpy as np
from ..core import Bandit2Arm


def lzw_compressed_length(sequence) -> int:
    """Return the number of codewords in the LZW-compressed sequence.

    The LZW algorithm builds a dictionary of encountered sub-sequences and
    emits a codeword each time a new entry is added.  The compressed length
    (number of codewords) is a proxy for the information content of the
    sequence: shorter compressed lengths indicate more internal repetition.

    Parameters
    ----------
    sequence : array-like
        Sequence of discrete symbols (e.g. choice labels).

    Returns
    -------
    int
        Number of LZW codewords in the compressed output.
    """
    sequence = list(sequence)
    n = len(sequence)
    if n == 0:
        return 0

    symbols = sorted(set(sequence))
    dictionary = {(s,): i for i, s in enumerate(symbols)}
    dict_size = len(symbols)

    w = (sequence[0],)
    n_codes = 0

    for c in sequence[1:]:
        wc = w + (c,)
        if wc in dictionary:
            w = wc
        else:
            n_codes += 1
            dictionary[wc] = dict_size
            dict_size += 1
            w = (c,)

    n_codes += 1  # emit the final buffered codeword
    return n_codes


class CompressibilityRatio2Arm:
    """Compressibility ratio for Bandit2Arm choice sequences using LZW.

    Estimates how structured (compressible) a participant's choice sequence is
    relative to random sequences of the same length and alphabet.

    Method (Findling et al. / Koechlin lab):
        1. Compress each segment's choice sequence with LZW -> lLZW.
        2. Draw ``n_random`` random sequences of the same length using the same
           unique elements (uniform sampling).
        3. Compress each random sequence -> bLZW (mean compressed length).
        4. Compressibility ratio = bLZW / lLZW.

    A ratio > 1 means the original sequence is *more* compressible (more
    structured) than chance.

    Parameters
    ----------
    task : Bandit2Arm
        The two-armed bandit task object.
    n_random : int, optional
        Number of random sequences drawn to estimate the baseline bLZW.
        Default is 150.
    rng : int or np.random.Generator, optional
        Seed or generator for reproducibility.  Default is None (unseeded).

    Examples
    --------
    >>> cr = CompressibilityRatio2Arm(task)
    >>> ratios = cr.compute()                          # segments by window (default)
    >>> ratios = cr.compute("session")                 # segments by session
    >>> ratios = cr.compute("block")                   # segments by block
    >>> ratios = cr.compute(task.is_session_start)     # custom boolean array
    """

    def __init__(self, task: Bandit2Arm, n_random: int = 150, rng=None):
        assert isinstance(task, Bandit2Arm), "task must be a Bandit2Arm object"
        self.task = task
        self.n_random = n_random
        self.rng = np.random.default_rng(rng)

    def compute(self, segment_starts="window"):
        """Compute compressibility ratio per segment.

        Parameters
        ----------
        segment_starts : str or array-like of bool/int, optional
            Defines how trials are divided into segments.  Each contiguous run
            of trials between segment-start positions is analysed independently.

            - ``"window"`` *(default)*: segment boundaries taken from
              ``task.window_ids`` (requires ``window_ids`` to be set).
            - ``"session"``: boundaries from ``task.session_ids``.
            - ``"block"``: boundaries from ``task.block_ids`` (requires
              ``block_ids`` to be set).
            - array-like of bool/int with length ``n_trials``: ``True``/``1``
              marks the first trial of a new segment.

        Returns
        -------
        ratios : np.ndarray, shape (n_segments,)
            Compressibility ratio (bLZW / lLZW) per segment.
        """
        starts_mask = self._resolve_segment_starts(segment_starts)
        segment_indices = np.where(starts_mask)[0]
        # Append sentinel for the final segment
        boundaries = np.append(segment_indices, self.task.n_trials)

        ratios = []
        for seg_start, seg_stop in zip(boundaries[:-1], boundaries[1:]):
            choices = self.task.choices[seg_start:seg_stop]
            ratios.append(self._compute_ratio(choices))

        return np.array(ratios)

    def _resolve_segment_starts(self, segment_starts) -> np.ndarray:
        """Return a boolean array (length n_trials) marking segment starts."""
        if isinstance(segment_starts, str):
            if segment_starts == "window":
                assert (
                    self.task.window_ids is not None
                ), "window_ids must be set on the task for segment_starts='window'"
                ids = self.task.window_ids
            elif segment_starts == "session":
                ids = self.task.session_ids
            elif segment_starts == "block":
                assert (
                    self.task.block_ids is not None
                ), "block_ids must be set on the task for segment_starts='block'"
                ids = self.task.block_ids
            else:
                raise ValueError(
                    f"Unknown segment_starts string '{segment_starts}'. "
                    "Use 'window', 'session', or 'block'."
                )
            mask = np.concatenate(([True], ids[1:] != ids[:-1]))
            return mask.astype(bool)

        # Array-like path
        mask = np.asarray(segment_starts, dtype=bool)
        assert mask.shape == (self.task.n_trials,), (
            f"segment_starts array must have length {self.task.n_trials}, "
            f"got {mask.shape}"
        )
        if not mask[0]:
            mask = mask.copy()
            mask[0] = True  # ensure the full sequence is covered
        return mask

    def _compute_ratio(self, choices) -> float:
        """Compute bLZW / lLZW for a single choice sequence.

        Parameters
        ----------
        choices : np.ndarray
            1-D array of discrete choice labels.

        Returns
        -------
        float
            Compressibility ratio.  Returns NaN for sequences of length < 2.
        """
        if len(choices) < 2:
            return np.nan

        elements = np.unique(choices)

        # --- original sequence compressed length --------------------------
        l_lzw = lzw_compressed_length(choices)

        # --- baseline: mean compressed length of random sequences ---------
        seq_len = len(choices)
        baseline_lengths = np.empty(self.n_random, dtype=float)

        for i in range(self.n_random):
            rand_seq = self.rng.choice(elements, size=seq_len)
            baseline_lengths[i] = lzw_compressed_length(rand_seq)

        b_lzw = baseline_lengths.mean()

        return b_lzw / l_lzw

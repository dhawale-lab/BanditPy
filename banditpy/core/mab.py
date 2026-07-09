import numpy as np
from .data_manager import DataManager
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from scipy.optimize import curve_fit


class BanditTask(DataManager):
    """
    Base class for multi-armed bandit task data.

    Structure:
    ----------
    - Each trial contains: reward probabilities (probs), choices, rewards, session_ids, and optionally block_ids, window_ids, start/stop indices, and datetimes.
    - Trials are grouped into sessions (session_ids), blocks (block_ids), and windows (window_ids) for flexible analysis.

    Parameters
    ----------
    probs : np.ndarray
        Array of shape (n_trials, n_arms) with reward probabilities for each arm at each trial.
    choices : np.ndarray
        Array of shape (n_trials,) with chosen arm indices (1-based).
    rewards : np.ndarray
        Array of shape (n_trials,) with reward outcomes for each trial.
    session_ids : np.ndarray
        Array of shape (n_trials,) with session identifiers (monotonically increasing, unique for each reward probability combination).
    block_ids : np.ndarray, optional
        Array of shape (n_trials,) with block identifiers (reset to 1 at the start of each experiment window, increment for each uncued reward probability change).
    window_ids : np.ndarray, optional
        Array of shape (n_trials,) with window identifiers.
    starts : np.ndarray, optional
        Array of shape (n_trials,) indicating trial start indices.
    stops : np.ndarray, optional
        Array of shape (n_trials,) indicating trial stop indices.
    datetime : np.ndarray, optional
        Array of shape (n_trials,) with datetime information for each trial.
    metadata : dict, optional
        Dictionary of additional metadata.

    Key Properties
    --------------
    n_ports : int
        Number of arms in the bandit task.
    n_trials : int
        Number of trials in the dataset.
    mean_ntrials, min_ntrials, max_ntrials : float/int
        Statistics on number of trials per session.
    n_sessions : int
        Number of unique sessions.
    sessions : np.ndarray
        Unique session IDs.
    ntrials_session : np.ndarray
        Number of trials per session.
    is_session_start : np.ndarray
        Boolean array, True at the start of each session.
    is_window_start : np.ndarray
        Boolean array, True at the start of each window (requires window_ids).

    Methods
    -------
    filter_by_trials(min_trials=100, clip_max=None)
        Filter sessions by minimum number of trials and optionally clip to max trials per session.
    filter_by_session_id(ids)
        Filter trials by session IDs.
    to_df()
        Convert the bandit task data to a pandas DataFrame.
    auto_block_window_ids(time_window_min=40)
        Auto-generate block_ids and window_ids using datetime and reward probability changes.
    _filtered(mask)
        Return a new BanditTask instance with filtered data.

    Usage Example
    -------------
    >>> task = BanditTask(probs, choices, rewards, session_ids, datetime=datetimes)
    >>> block_ids, window_ids = task.auto_block_window_ids(time_window_min=40)
    >>> df = task.to_df()
    >>> filtered_task = task.filter_by_trials(min_trials=50)

    Notes
    -----
    - session_ids should be unique for each reward probability combination and increment globally.
    - block_ids reset at the start of each window and increment for each uncued reward probability change.
    - window_ids reset at the start of each experiment window (e.g., every 40 minutes).
    - Use is_session_start, is_window_start for custom resetting logic in models.
    """

    def __init__(
        self,
        probs,
        choices,
        rewards,
        session_ids,
        block_ids=None,
        window_ids=None,
        starts=None,
        stops=None,
        datetime=None,
        metadata=None,
    ):
        super().__init__(metadata=metadata)
        assert probs.ndim == 2, "probs must be (n_trials, n_arms)"
        assert probs.shape[0] > probs.shape[1], "n_arms can't be greater than n_trials"
        assert (
            probs.shape[0] == len(choices) == len(rewards) == len(session_ids)
        ), "Mismatch in trials length"

        self.probs = self._fix_probs(probs)
        self.choices = self._fix_choices(choices.astype(int))
        self.rewards = self._fix_rewards(rewards.astype(int))
        self.session_ids = self._fix_session_ids(session_ids)

        if starts is not None:
            assert starts.shape[0] == probs.shape[0], "starts should be of same length"
            self.starts = starts
        else:
            self.starts = None

        if stops is not None:
            assert stops.shape[0] == probs.shape[0], "stops should be of same length"
            self.stops = stops
        else:
            self.stops = None

        self.window_ids = self._fix_window_ids(window_ids)
        self.block_ids = self._fix_block_ids(block_ids, self.window_ids)
        self.datetime = self._fix_datetime(datetime)

        self.sessions, self.ntrials_session = np.unique(
            self.session_ids, return_counts=True
        )

    @staticmethod
    def _fix_probs(probs):
        return probs / 100 if probs.max() > 1 else probs

    @staticmethod
    def _fix_choices(choices):
        choices = np.squeeze(choices)
        return choices - choices.min() + 1

    @staticmethod
    def _fix_rewards(rewards):
        rewards = np.squeeze(rewards)
        return rewards - rewards.min()

    @staticmethod
    def _fix_datetime(datetime):
        if datetime is None:
            return None

        dt = np.asarray(datetime)

        # Normalize shape to 1D
        dt = np.squeeze(dt)
        if dt.ndim == 0:
            dt = dt.reshape(1)
        if dt.ndim > 1:
            raise ValueError("datetime must be 1D or squeezable to 1D")

        # Already datetime-like numpy array
        if np.issubdtype(dt.dtype, np.datetime64):
            return dt.astype("datetime64[s]")

        # Numeric input: interpreted as unix epoch seconds
        if np.issubdtype(dt.dtype, np.number):
            return dt.astype("datetime64[s]")

        # String/object input: parse via pandas
        parsed = pd.to_datetime(dt, errors="coerce")
        if np.any(pd.isna(parsed)):
            raise ValueError("datetime contains unparseable values")

        return np.asarray(parsed.to_numpy()).astype("datetime64[s]")

    @staticmethod
    def _fix_session_ids(session_ids):
        session_ids = np.squeeze(session_ids)
        sessdiff = np.diff(session_ids, prepend=session_ids[0])
        neg_bool = np.where(sessdiff < 0, 1, 0)
        session_starts = np.insert(np.where(sessdiff != 0)[0], 0, 0)
        session_start_ids = np.arange(len(session_starts)) + 1

        if np.sum(neg_bool) > 0:
            session_ids = np.repeat(
                session_start_ids,
                np.diff(np.append(session_starts, len(session_ids))),
            )

        # Shift to 1-based only if 0-based
        if session_ids.min() == 0:
            session_ids = session_ids + 1

        return session_ids

    @staticmethod
    def _fix_window_ids(window_ids):
        """Shift window_ids to 1-based if 0-based."""
        if window_ids is None:
            return None
        window_ids = np.asarray(window_ids)
        if window_ids.min() == 0:
            window_ids = window_ids + 1
        return window_ids

    @staticmethod
    def _fix_block_ids(block_ids, window_ids):
        """Shift block_ids to 1-based if 0-based, checked per window."""
        if block_ids is None:
            return None
        block_ids = np.asarray(block_ids, dtype=int)
        if window_ids is not None:
            window_ids = np.asarray(window_ids)
            for w in np.unique(window_ids):
                mask = window_ids == w
                if block_ids[mask].min() == 0:
                    block_ids[mask] = block_ids[mask] + 1
        else:
            if block_ids.min() == 0:
                block_ids = block_ids + 1
        return block_ids

    @property
    def n_days(self):
        if self.datetime is None:
            raise ValueError("datetime must be set to compute n_days")
        first_date = self.datetime[0]
        last_date = self.datetime[-1]
        return int(np.ceil((last_date - first_date) / np.timedelta64(1, "D")))

    @property
    def n_ports(self):
        return self.probs.shape[1]

    @property
    def n_trials(self):
        return self.probs.shape[0]

    @property
    def mean_ntrials(self):
        return np.mean(self.ntrials_session)

    @property
    def min_ntrials(self):
        return np.min(self.ntrials_session)

    @property
    def max_ntrials(self):
        return np.max(self.ntrials_session)

    @property
    def n_sessions(self):
        return len(self.sessions)

    @property
    def is_session_start(self):
        session_starts = np.diff(self.session_ids, prepend=self.session_ids[0])
        session_starts = np.clip(session_starts, 0, 1)
        session_starts[0] = 1
        return session_starts.astype(bool)

    @property
    def is_window_start(self):
        assert self.window_ids is not None, "window_ids must be set"
        window_starts = np.diff(self.window_ids, prepend=self.window_ids[0])
        window_starts = np.clip(window_starts, 0, 1)
        window_starts[0] = 1
        return window_starts.astype(bool)

    def trial_slice(self, start, stop):
        """
        Slice trials [start:stop] within each session.

        Parameters
        ----------
        start : int
            Start trial index within session (0-based, inclusive)
        stop : int
            Stop trial index within session (0-based, exclusive)

        Returns
        -------
        BanditTask
            New task containing sliced trials.

        Raises
        ------
        ValueError
            If any session has fewer than `stop` trials.
        """

        if start < 0 or stop <= start:
            raise ValueError("Invalid slice bounds: require 0 <= start < stop")

        # Check session lengths
        too_short = self.sessions[self.ntrials_session < stop]
        if len(too_short) > 0:
            raise ValueError(
                f"Sessions {too_short.tolist()} have fewer than {stop} trials"
            )

        # Build mask
        mask = np.zeros(self.n_trials, dtype=bool)

        for sess in self.sessions:
            sess_idx = np.where(self.session_ids == sess)[0]
            slice_idx = sess_idx[start:stop]
            mask[slice_idx] = True

        return self._filtered(mask)

    def filter_by_datetime(self, start=None, stop=None):
        """Filter trials by datetime range.

        Parameters
        ----------
        start : str or np.datetime64, optional
            Start datetime (inclusive). Can be a string parseable by np.datetime64 or a np.datetime64 object.
        stop : str or np.datetime64, optional
            Stop datetime (inclusive). Can be a string parseable by np.datetime64 or a np.datetime64 object.

        Returns
        -------
        BanditTask
            New task containing only trials within the specified datetime range.

        Raises
        ------
        ValueError
            If datetime is not set or if start/stop are invalid.
        """
        if self.datetime is None:
            raise ValueError("datetime must be set to filter by datetime")

        dt = self.datetime.astype("datetime64[s]")

        if start is not None:
            start_dt = np.datetime64(start)
            dt_mask = dt >= start_dt
        else:
            dt_mask = np.ones_like(dt, dtype=bool)

        if stop is not None:
            stop_dt = np.datetime64(stop)
            dt_mask &= dt <= stop_dt

        return self._filtered(dt_mask)

    def filter_by_trials(self, min_trials=100, clip_max=None):
        valid_sessions = self.sessions[self.ntrials_session >= min_trials]
        mask = np.isin(self.session_ids, valid_sessions)

        if clip_max is not None:
            clipped_mask = []
            for session in valid_sessions:
                session_mask = self.session_ids == session
                session_indices = np.where(session_mask)[0][:clip_max]
                clipped_mask.extend(session_indices)
            mask = np.zeros_like(self.session_ids, dtype=bool)
            mask[clipped_mask] = True

        return self._filtered(mask)

    def filter_by_session_id(self, start=None, stop=None, ids=None):
        """Filter trials by session id(s).

        Priority: explicit ``ids`` overrides ``start``/``stop`` bounds.
        """

        session_ids = np.asarray(self.session_ids)

        if ids is not None:
            target_ids = np.asarray(list(np.atleast_1d(ids)))
            mask = np.isin(session_ids, target_ids)
        else:
            if start is None and stop is None:
                raise ValueError("Provide start/stop or ids to filter sessions")

            if start is None:
                start = session_ids.min()
            if stop is None:
                stop = session_ids.max()

            mask = (session_ids >= start) & (session_ids <= stop)

        return self._filtered(mask)

    def filter_by_block_id(self, start=None, stop=None, ids=None):
        """
        Filter trials by block_id.

        Parameters
        ----------
        start : int, optional
            Inclusive lower bound for block_id (e.g., 2 for "2 onward").
        stop : int, optional
            Inclusive upper bound for block_id (e.g., 3 for "1 to 3").
        ids : iterable of int, optional
            Explicit set/list of block_ids to keep (overrides start/end if provided).

        Examples
        --------
        - Blocks 1 only: filter_by_block_id(start=1, stop=1)
        - Blocks 1-3:    filter_by_block_id(start=1, stop=3)
        - Blocks 2+:     filter_by_block_id(start=2)
        """
        if self.block_ids is None:
            raise ValueError("block_ids must be set to filter by block")

        block_ids = np.asarray(self.block_ids)

        if ids is not None:
            ids = np.asarray(list(ids))
            mask = np.isin(block_ids, ids)
        else:
            if start is None and stop is None:
                raise ValueError("Provide start/stop or ids to filter blocks")

            if start is None:
                start = block_ids.min()
            if stop is None:
                stop = block_ids.max()

            mask = (block_ids >= start) & (block_ids <= stop)

        return self._filtered(mask)

    def filter_by_days(self, start: int = None, stop: int = None):
        """Filter by day offset from the first trial's date.

        Parameters
        ----------
        start : int, optional
            Skip the first `start` days (inclusive offset from day 0).
        stop : int, optional
            Exclude trials after `stop` days from the first trial.

        Returns
        -------
        BanditTask
            Filtered task containing only trials within the specified day range.
        """

        if self.datetime is None:
            raise ValueError("datetime is not set, cannot filter by days.")

        first_date = self.datetime[0]
        total_days = (self.datetime[-1] - first_date) / np.timedelta64(1, "D")

        if start is not None and start > total_days:
            raise ValueError(f"start ({start}) exceeds total days ({total_days:.0f}).")
        if stop is not None and stop > total_days:
            raise ValueError(f"stop ({stop}) exceeds total days ({total_days:.0f}).")
        if start is not None and stop is not None and stop <= start:
            raise ValueError(f"stop ({stop}) must be greater than start ({start}).")

        start_dt = first_date + pd.Timedelta(days=start) if start is not None else None
        stop_dt = first_date + pd.Timedelta(days=stop) if stop is not None else None

        return self.filter_by_datetime(start=start_dt, stop=stop_dt)

    def get_block_start_mask(self, start=None, stop=None, ids=None):
        """
        Boolean mask marking the first trial of specified blocks.

        Parameters
        ----------
        start : int, optional
            Inclusive lower bound for block_id (e.g., 2 for "2 onward").
        stop : int, optional
            Inclusive upper bound for block_id.
        ids : iterable of int, optional
            Explicit block_ids to mark; overrides start/stop if provided.

        Returns
        -------
        np.ndarray
            Boolean array of shape (n_trials,) with True at the first trial of
            the selected blocks and False elsewhere.
        """
        if self.block_ids is None:
            raise ValueError("block_ids must be set to compute block starts")

        block_ids = np.asarray(self.block_ids)

        if ids is not None:
            target_ids = set(np.asarray(list(ids)).tolist())
        else:
            if start is None and stop is None:
                raise ValueError("Provide start/stop or ids to select block starts")
            if start is None:
                start = block_ids.min()
            if stop is None:
                stop = block_ids.max()
            target_ids = set(range(int(start), int(stop) + 1))

        # Identify block start indices
        block_start_bool = np.concatenate(([True], block_ids[1:] != block_ids[:-1]))
        start_indices = np.where(block_start_bool)[0]

        mask = np.zeros_like(block_ids, dtype=bool)
        for idx in start_indices:
            if block_ids[idx] in target_ids:
                mask[idx] = True

        return mask

    def _filtered(self, mask):
        """Return a new instance of the same class with filtered data."""
        return self.__class__(
            probs=self.probs[mask],
            choices=self.choices[mask],
            rewards=self.rewards[mask],
            session_ids=self.session_ids[mask],
            block_ids=None if self.block_ids is None else self.block_ids[mask],
            window_ids=None if self.window_ids is None else self.window_ids[mask],
            starts=None if self.starts is None else self.starts[mask],
            stops=None if self.stops is None else self.stops[mask],
            datetime=None if self.datetime is None else self.datetime[mask],
            metadata=self.metadata,
        )

    @classmethod
    def from_csv(cls):
        pass

    def to_df(self):
        """Convert the bandit task data to a pandas DataFrame."""
        data = pd.DataFrame(
            {
                "choices": self.choices,
                "rewards": self.rewards,
                "session_ids": self.session_ids,
                "block_ids": self.block_ids,
                "window_ids": self.window_ids,
                "starts": self.starts,
                "stops": self.stops,
                "datetime": self.datetime,
            }
        )

        for i in range(self.probs.shape[1]):
            data.insert(i + 2, f"probs_{i+1}", self.probs[:, i])

        return data

    def auto_block_window_ids(self, time_window_min=40):
        """
        Auto-generate window_ids and block_ids for each trial.

        - A block is defined as a contiguous set of trials with fixed reward probabilities.
        - block_ids reset to 1 at the start of each experiment window (e.g., 40-min period).
        - session_ids are global and unique for each reward probability combination (do not reset).

        Parameters
        ----------
        time_window_min : int
            Time window in minutes for splitting blocks by datetime.

        Returns
        -------
        block_ids : np.ndarray
            Array of block IDs for each trial (starts at 1 for each experiment window).
        window_ids : np.ndarray
            Array of window IDs for each trial (starts at 1 for first window).
        """
        n_trials = len(self.probs)
        session_ids = self.session_ids
        block_ids = np.zeros(n_trials, dtype=int)
        window_ids = np.zeros(n_trials, dtype=int)

        # If datetime is provided, split windows by time gaps > time_window_min
        assert (
            self.datetime is not None
        ), "Datetime must be provided for window splitting."
        dt = self.datetime.astype("datetime64[s]")
        gap = np.diff(dt, prepend=dt[0]).astype("timedelta64[s]").astype(int) / 60
        window_starts_bool = gap > time_window_min
        window_starts_bool[0] = 1  # Ensure first trial is a window start
        window_ids = np.cumsum(window_starts_bool)

        _, counts = np.unique(window_ids, return_counts=True)
        chunks = np.split(session_ids, np.cumsum(counts)[:-1])
        block_ids = np.concatenate([chunk - chunk[0] + 1 for chunk in chunks])

        self.block_ids = block_ids
        self.window_ids = window_ids


class Bandit2Arm(BanditTask):
    """
    A class to hold a two armed bandit task data.
    """

    def __init__(
        self,
        probs,
        choices,
        rewards,
        session_ids,
        block_ids=None,
        window_ids=None,
        starts=None,
        stops=None,
        datetime=None,
        metadata=None,
    ):
        super().__init__(
            probs=probs,
            choices=choices,
            rewards=rewards,
            session_ids=session_ids,
            block_ids=block_ids,
            window_ids=window_ids,
            starts=starts,
            stops=stops,
            datetime=datetime,
            metadata=metadata,
        )
        assert self.n_ports == 2, "TwoArmedBandit requires exactly 2 arms"

    @property
    def is_block_start(self):
        """Boolean array, True at the start of each block. Requires block_ids."""
        assert self.block_ids is not None, "block_ids must be set"
        block_starts = np.diff(self.block_ids, prepend=self.block_ids[0])
        block_starts = np.clip(block_starts, 0, 1)
        block_starts[0] = 1
        return block_starts.astype(bool)

    @property
    def is_choice_high(self):
        # return (np.argmax(self.probs, axis=1) + 1) == self.choices
        return (
            np.max(self.probs, axis=1)
            == self.probs[np.arange(self.n_trials), self.choices - 1]
        )

    @property
    def probs_corr(self):
        return stats.pearsonr(self.probs[:, 0], self.probs[:, 1])[0]

    @property
    def is_structured(self):
        # return np.abs(self.probs_corr) > 0.9
        return self.probs_corr < -0.9

    @property
    def session_probs(self):
        """Get the probabilities for each session.

        Returns
        -------
        _type_
            _description_
        """
        return self.probs[self.is_session_start, :]

    def get_binarized_choices(self):
        """get choices coded as 0 and 1

        Returns
        -------
        _type_
            _description_
        """
        assert self.n_ports == 2, "Only implemented for 2AB task"
        return np.where(self.choices == 2, 1, 0)  # Port 1: 0 and Port 2: 1

    def get_port_bias(self, kind="linear"):
        """Get the port bias as a function of delta probability.

        Returns
        -------
        fitparams : LinregressResult or np.ndarray
            The result of linear regression or logistic fit between delta probability and choice bias.
        unique_prob_diff : np.ndarray
            Unique delta probabilities.
        choice_diff : np.ndarray
            Mean choice bias for each unique delta probability.
        """

        prob_diff = np.diff(self.probs, axis=1).squeeze().round(2)
        unique_prob_diff = np.unique(prob_diff)

        choices = self.choices
        choices = np.where(choices == 2, 1, -1)

        choice_diff = np.array(
            [np.mean(choices[prob_diff == pd]) for pd in unique_prob_diff]
        )
        good_idx = ~np.isnan(choice_diff)

        if kind == "linear":
            fitparams = stats.linregress(
                unique_prob_diff[good_idx], choice_diff[good_idx]
            )
            y_est = fitparams.slope * unique_prob_diff + fitparams.intercept
        elif kind == "logistic":

            def logistic(x, ymin, ymax, k, x0):
                return ymin + (ymax - ymin) / (1 + np.exp(-k * (x - x0)))

            p0 = [
                np.min(choice_diff),
                np.max(choice_diff),
                1.0,
                np.median(unique_prob_diff),
            ]
            popt, _ = curve_fit(
                logistic,
                unique_prob_diff[good_idx],
                choice_diff[good_idx],
                p0=p0,
                maxfev=10000,
                method="trf",
            )

            fitparams = popt  # L, x0, k
            y_est = logistic(unique_prob_diff, *popt)
        elif kind == "tanh":

            def tanh_model(x, A, k, x0):
                return A * np.tanh(k * (x - x0))

            p0 = [np.max(np.abs(choice_diff)), 1.0, np.median(unique_prob_diff)]
            fitparams, _ = curve_fit(
                tanh_model,
                unique_prob_diff[good_idx],
                choice_diff[good_idx],
                p0=p0,
                maxfev=10000,
                method="trf",
            )
            y_est = tanh_model(unique_prob_diff, *fitparams)
        else:
            raise ValueError("kind must be 'linear', 'logistic' or 'tanh'")

        return fitparams, unique_prob_diff, choice_diff, y_est

    @staticmethod
    def from_csv(
        fp,
        probs,
        choices,
        rewards,
        session_ids,
        block_ids=None,
        window_ids=None,
        starts=None,
        stops=None,
        datetime=None,
    ):
        """Build Bandit2Arm from a CSV file.

        probs: list[str] column names for per-arm probabilities.
        choices, rewards, session_ids, block_ids, window_ids, starts, stops, datetime: column names (str).
        """
        df = pd.read_csv(fp)
        return Bandit2Arm(
            probs=df.loc[:, probs].to_numpy(),
            choices=df[choices].to_numpy(),
            rewards=df[rewards].to_numpy(),
            session_ids=df[session_ids].to_numpy(),
            block_ids=None if block_ids is None else df[block_ids].to_numpy(),
            window_ids=None if window_ids is None else df[window_ids].to_numpy(),
            starts=None if starts is None else df[starts].to_numpy(),
            stops=None if stops is None else df[stops].to_numpy(),
            datetime=None if datetime is None else df[datetime].to_numpy(),
            metadata=None,
        )

    @staticmethod
    def from_df(
        df,
        probs,
        choices,
        rewards,
        session_ids,
        block_ids=None,
        window_ids=None,
        starts=None,
        stops=None,
        datetime=None,
    ):
        """Build Bandit2Arm from an existing DataFrame."""
        return Bandit2Arm(
            probs=df.loc[:, probs].to_numpy(),
            choices=df[choices].to_numpy(),
            rewards=df[rewards].to_numpy(),
            session_ids=df[session_ids].to_numpy(),
            block_ids=None if block_ids is None else df[block_ids].to_numpy(),
            window_ids=None if window_ids is None else df[window_ids].to_numpy(),
            starts=None if starts is None else df[starts].to_numpy(),
            stops=None if stops is None else df[stops].to_numpy(),
            datetime=None if datetime is None else df[datetime].to_numpy(),
            metadata=None,
        )

    def filter_by_probs(self, probs):
        """Keep only sessions with probabilities that match the given probabilities.

        Parameters
        ----------
        probs : list of length or 2D numpy array, optional
            Probabilities to match. If a list, it should contain two elements representing the probabilities for each port. If a 2D numpy array, it should have shape (n, 2) where each row represents the probabilities for each port.

        Returns
        -------
        Bandit2Arm with filtered data
        """
        probs = np.array(probs).reshape(-1, 2)
        assert probs.shape[1] == 2, "This method is only implemented for 2AB task"
        mask = (self.probs[:, None, :] == probs[None, :, :]).all(axis=2).any(axis=1)
        return self._filtered(mask)

    def filter_by_deltaprob(self, delta_min, delta_max=None):
        """Keep only sessions with delta probabilities between min and max.

        Note: Only implemented for 2AB task

        Parameters
        ----------
        delta_min : float
            Minimum delta probability to keep
        delta_max : float, optional
            Maximum delta probability to keep, by default None

        Returns
        -------
        Multi
        """
        assert self.n_ports == 2, "This method is only implemented for 2AB task"

        if delta_max is not None:
            assert delta_min <= delta_max, "min should be less than max"
        else:
            delta_max = 1

        # Calculate the absolute difference between probabilities of the two ports
        prob_diff = np.abs(np.diff(self.probs, axis=1).flatten().round(2))

        # Identify sessions where the probability difference exceeds the threshold
        delta_bool = (prob_diff >= delta_min) & (prob_diff <= delta_max)
        # valid_sessions = np.unique(self.session_ids[delta_bool])

        return self._filtered(delta_bool)

    def get_optimal_choice_probability(self, bin_size=None):
        """Get probability of choosing high arm on two armed bandit task

        Parameters
        ----------
        bin_size : int, optional
            no.of sessions over which performance is calculated, by default None
        roll_step : int, optional
            _description_, by default 40
        delta_prob : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """

        assert self.n_ports == 2, "This method is only implemented for 2AB task"

        # converting into n_sessions x n_trials dataframe and then into numpy array, converting to dataframe automatically handles the NaN values for sessions which has fewer trials
        is_choice_high_per_session = pd.DataFrame(
            np.split(
                self.is_choice_high.astype(int), np.cumsum(self.ntrials_session)[:-1]
            )
        ).to_numpy()

        assert (
            is_choice_high_per_session.shape[0] == self.n_sessions
        ), f"Number of sessions not matching"

        if bin_size is not None:

            sess_div_perf = np.array_split(
                is_choice_high_per_session, self.n_sessions // bin_size, axis=0
            )
            sess_div_perf = np.array([np.nanmean(_, axis=0) for _ in sess_div_perf])
        else:
            sess_div_perf = np.nanmean(is_choice_high_per_session, axis=0)

        return sess_div_perf

    def get_reward_probability(self):
        """Get the reward probabilities across sessions.

        Returns
        -------
        array-like
            The reward probabilities for each session.
        """
        session_rewards = pd.DataFrame(
            np.hsplit(self.rewards, np.cumsum(self.ntrials_session)[:-1])
        ).to_numpy()

        return np.nanmean(session_rewards, axis=0)

    def get_cummulative_reward(self):
        """Get the cumulative rewards for each session.

        Returns
        -------
        array-like
            The cumulative rewards for each session.
        """
        return np.array(
            [np.cumsum(session_rewards) for session_rewards in self.rewards]
        )

    def get_choice_entropy(self):
        """Calculate the entropy of the probabilities for each trial.

        Returns
        -------
        array-like
            The entropy values for each trial.
        """
        choices = self.choices - 1
        session_choices = pd.DataFrame(
            np.hsplit(choices, np.cumsum(self.ntrials_session)[:-1])
        ).to_numpy()

        port2_prob = np.nanmean(session_choices, axis=0)
        port1_prob = 1 - port2_prob
        choices_prob = np.vstack((port1_prob, port2_prob))

        return stats.entropy(choices_prob, base=2, axis=0)

    def get_prob_hist2d(self, stat="count"):
        """Get the probability matrix for each session. Calculates a 2D histogram of the reward probabilities.

        Returns
        -------
        2d array-like
            The probability matrix for each session.
        """
        probs_combinations = self.probs[self.is_session_start, :]
        p1_bins = np.linspace(0, 0.9, 10) + 0.05
        p2_bins = np.linspace(0, 0.9, 10) + 0.05
        H, xedges, yedges, _ = stats.binned_statistic_2d(
            probs_combinations[:, 0].astype(float),
            probs_combinations[:, 1].astype(float),
            values=probs_combinations[:, 0],
            statistic="count",
            bins=[p1_bins, p2_bins],
        )

        if stat == "prop":
            H = H / probs_combinations.shape[0]

        return H.T, xedges, yedges

    def get_trials_hist(self, bin_size=10):
        """Get histogram of number of trials per session."""

        ntrials_per_session = self.ntrials_session
        min_trials = ntrials_per_session.min()
        max_trials = ntrials_per_session.max()

        h, bins = np.histogram(
            ntrials_per_session,
            bins=np.arange(min_trials, max_trials + bin_size, bin_size),
        )

        return h, bins[:-1] + bin_size / 2

    def get_performance_prob_grid(
        self, n_last_trials=5, performance_metric="optimal_choice"
    ):
        """Get performance grid based on reward probabilities.

        Parameters
        ----------
        n_last_trials : int, optional
            Number of last trials to consider for performance calculation, by default 5
        performance_metric : str, optional
            Metric to use for performance calculation, by default "optimal_choice"

        Returns
        -------
        performance_grid : 2D array
            Grid of performance values.
        xedges : 1D array
            Edges of the bins along the x-axis.
        yedges : 1D array
            Edges of the bins along the y-axis.
        """
        probs = self.probs
        unique_probs = np.unique(probs.flatten())

        perf_mat = np.zeros((len(unique_probs), len(unique_probs)))

        for i1, p1 in enumerate(unique_probs):
            for i2, p2 in enumerate(unique_probs):
                p1p2_mask = (probs[:, 0] == p1) & (probs[:, 1] == p2)
                p2p1_mask = (probs[:, 0] == p2) & (probs[:, 1] == p1)
                mask = p1p2_mask | p2p1_mask

                if mask.sum() > 100:
                    task_p1p2 = self._filtered(mask)
                    if performance_metric == "optimal_choice":
                        perf_p1p2 = task_p1p2.get_optimal_choice_probability()
                    if performance_metric == "reward_rate":
                        perf_p1p2 = task_p1p2.get_reward_probability()

                    perf_mat[i1, i2] = perf_p1p2[-n_last_trials:].mean()

        return perf_mat, unique_probs

    def compare_reward_alignment(self) -> pd.DataFrame:
        """
        Compare the empirical reward rate of the majority-chosen arm in each session
        against the programmed reward probability for that arm.

        Returns
        -------
        pd.DataFrame
            Columns:
                - session_id: session identifier.
                - prog_rew: programmed reward probability for the dominant arm.
                - emp_rew: empirical mean reward when that arm was chosen.
                - session_trials: total trials in that session.
                - dominant_trials: number of times the dominant arm was chosen.
                - reward_diff: absolute difference between emp_rew and prog_rew.
        """
        prog_rew = self.probs[np.arange(self.n_trials), self.choices - 1]
        df = pd.DataFrame(
            {
                "session_id": self.session_ids,
                "choice": self.choices,
                "prog_rew": prog_rew,
                "reward": self.rewards,
            }
        )

        session_counts = df.groupby("session_id").size()
        df["session_trials"] = df["session_id"].map(session_counts)

        majority_choice = (
            df.groupby(["session_id", "choice"])
            .size()
            .unstack(fill_value=0)
            .idxmax(axis=1)
        )
        df = df[df["choice"] == df["session_id"].map(majority_choice)]

        dominant_counts = df.groupby("session_id").size()
        df["dominant_trials"] = df["session_id"].map(dominant_counts)

        out = df.groupby(["session_id", "prog_rew"], as_index=False).agg(
            emp_rew=("reward", "mean"),
            session_trials=("session_trials", "first"),
            dominant_trials=("dominant_trials", "first"),
        )
        out["reward_diff"] = np.abs(out["emp_rew"] - out["prog_rew"])

        return out

    def get_perf_per_day(
        self,
        windows_per_day: int = 3,
        by: str = "window",
        day_start_hour: int = 0,
        fill_missing: bool = False,
        min_trials_per_day: int = 0,
        _return_dates: bool = False,
    ):
        """Per-day mean P(high-reward choice), averaged over all windows and blocks.

        Parameters
        ----------
        windows_per_day : int, optional
            Number of time windows per day. Used when by='window'. Default is 3.
        by : {'window', 'datetime'}, optional
            How to group trials into days.

            - 'window': group `window_ids` in consecutive chunks of
              `windows_per_day`. Requires `window_ids` to be set.
            - 'datetime': group by calendar date extracted from
              `self.datetime`. Requires `datetime` to be set.

            Default is 'window'.
        day_start_hour : int, optional
            Hour (0-23) at which a new experimental day begins. Used only
            when by='datetime'. Timestamps are shifted back by this many
            hours before grouping, so that e.g. day_start_hour=19 treats
            7 PM as the start of a new day. Default is 0 (midnight).
        fill_missing : bool, optional
            Only used when by='datetime'. If True, the returned array
            spans every calendar day from the first to the last experimental
            day; days with no data (or fewer than min_trials_per_day
            trials) are filled with np.nan. If False (default),
            only days that have sufficient data are included and the array
            is compact (no NaNs).
        min_trials_per_day : int, optional
            Minimum number of trials a day must have to be included. Days
            with fewer trials are treated as missing (np.nan when
            fill_missing=True, silently skipped otherwise). Default is 0
            (include all days that have any data).

        Returns
        -------
        np.ndarray
            Array of shape (n_days,) with mean P(High) for each day.
            When fill_missing=True, missing days are np.nan.
        """
        if by == "datetime":
            assert self.datetime is not None, "datetime must be set on the task"
            assert 0 <= day_start_hour <= 23, "day_start_hour must be between 0 and 23"
            shifted = pd.DatetimeIndex(self.datetime) - pd.Timedelta(
                hours=day_start_hour
            )
            dates = shifted.normalize()
            unique_dates = sorted(dates.unique())

            if fill_missing:
                # Build a full range from first to last date
                all_dates = pd.date_range(unique_dates[0], unique_dates[-1], freq="D")
                perf_map = {}
                for date in unique_dates:
                    mask = dates == date
                    if mask.sum() >= max(min_trials_per_day, 1):
                        perf_map[date] = self.is_choice_high[mask].mean()
                perf_arr = np.array([perf_map.get(d, np.nan) for d in all_dates])
                if _return_dates:
                    return perf_arr, list(all_dates)
                return perf_arr
            else:
                daily_perf = []
                dates_list = []
                for date in unique_dates:
                    mask = dates == date
                    if mask.sum() >= max(min_trials_per_day, 1):
                        daily_perf.append(self.is_choice_high[mask].mean())
                        dates_list.append(date)
                perf_arr = np.array(daily_perf)
                if _return_dates:
                    return perf_arr, dates_list
                return perf_arr
        else:
            assert self.window_ids is not None, "window_ids must be set on the task"
            unique_windows = np.unique(self.window_ids)
            n_days = len(unique_windows) // windows_per_day
            daily_perf = []
            dates_list = []
            for d in range(n_days):
                day_windows = unique_windows[
                    d * windows_per_day : (d + 1) * windows_per_day
                ]
                mask = np.isin(self.window_ids, day_windows)
                dt = (
                    self.datetime[mask][0]
                    if (self.datetime is not None and mask.any())
                    else None
                )
                if mask.sum() >= max(min_trials_per_day, 1):
                    daily_perf.append(self.is_choice_high[mask].mean())
                    dates_list.append(dt)
                elif fill_missing:
                    daily_perf.append(np.nan)
                    dates_list.append(None)
            perf_arr = np.array(daily_perf)
            if _return_dates:
                return perf_arr, dates_list
            return perf_arr

    def get_expertise_day(
        self,
        windows_per_day: int = 3,
        by: str = "window",
        day_start_hour: int = 0,
        fill_missing: bool = False,
        min_trials_per_day: int = 0,
        n_consecutive: int = 3,
        threshold_pct: float = 50,
        baseline_frac: float = 1.0,
    ):
        """Find the first day the animal became an expert.

        Pipeline:
        1. Compute raw per-day performance via get_perf_per_day.
        2. Apply a trailing rolling mean of n_consecutive non-missing days to
           produce smoothed_perf. This reduces noise so that single bad days
           do not obscure genuine expertise.
        3. Compute the threshold as threshold_pct percentile of
           smoothed_perf over the first baseline_frac fraction of days. Using
           only the early phase (baseline_frac < 1.0) prevents the threshold
           from being inflated by expert-phase performance.
        4. Return the first day where smoothed_perf >= threshold.

        Because the rolling mean already encodes a "sustained" requirement,
        no additional consecutive-day streak logic is needed.

        Missing days (np.nan) are skipped when building the rolling mean and
        excluded from the baseline percentile calculation.

        Parameters
        ----------
        windows_per_day : int, optional
            Number of time windows per day. Used when by='window'. Default is 3.
        by : {'window', 'datetime'}, optional
            How to group trials into days. Passed directly to
            get_perf_per_day. Default is 'window'.
        day_start_hour : int, optional
            Hour (0-23) at which a new experimental day begins. Passed
            directly to get_perf_per_day. Only used when by='datetime'.
            Default is 0 (midnight).
        fill_missing : bool, optional
            Passed directly to get_perf_per_day. Default is False.
        min_trials_per_day : int, optional
            Passed directly to get_perf_per_day. Default is 0.
        n_consecutive : int, optional
            Rolling window size in non-missing days. Controls how much
            smoothing is applied before threshold detection. Default is 3.
        threshold_pct : float, optional
            Percentile of the smoothed baseline performance used as the
            expertise threshold. Default is 50 (median).
        baseline_frac : float, optional
            Fraction of total days (0 < baseline_frac <= 1.0) from the start
            of training used to derive the threshold. 1.0 uses all days
            (default). Set to e.g. 0.3 to anchor the threshold to the early
            learning phase and avoid inflation by expert-phase performance.

        Returns
        -------
        expertise_day : int or None
            0-based index into perf_per_day of the first day where the
            n_consecutive-day trailing mean of performance >= threshold, or
            None if the criterion is never met.
        expertise_datetime : datetime-like or None
            Datetime corresponding to expertise_day, or None.
        threshold : float
            The threshold value derived from the smoothed baseline.
        perf_per_day : np.ndarray
            Raw per-day performance array (unsmoothed).
        """
        perf_per_day, dates_array = self.get_perf_per_day(
            windows_per_day=windows_per_day,
            by=by,
            day_start_hour=day_start_hour,
            fill_missing=fill_missing,
            min_trials_per_day=min_trials_per_day,
            _return_dates=True,
        )

        # Step 1: rolling mean over non-missing days, mapped back to full array
        valid_idx = [i for i, v in enumerate(perf_per_day) if not np.isnan(v)]
        valid_vals = np.array([perf_per_day[i] for i in valid_idx])

        smoothed_perf = np.full(len(perf_per_day), np.nan)
        kernel = np.ones(n_consecutive) / n_consecutive
        # mode='valid' yields len(valid_vals) - n_consecutive + 1 trailing means;
        # result[j] = mean(valid_vals[j : j+n_consecutive]), mapped to valid_idx[j+n_consecutive-1]
        for j, val in enumerate(np.convolve(valid_vals, kernel, mode="valid")):
            smoothed_perf[valid_idx[j + n_consecutive - 1]] = val

        # Step 2: threshold from smoothed baseline phase
        assert 0 < baseline_frac <= 1.0, "baseline_frac must be in (0, 1]"
        n_baseline = max(int(len(smoothed_perf) * baseline_frac), 1)
        threshold = float(np.nanpercentile(smoothed_perf[:n_baseline], threshold_pct))

        # Step 3: first day where smoothed performance >= threshold
        for i, v in enumerate(smoothed_perf):
            if not np.isnan(v) and v >= threshold:
                expertise_datetime = dates_array[i] if i < len(dates_array) else None
                return i, expertise_datetime, threshold, perf_per_day

        return None, None, threshold, perf_per_day


class Bandit4Arm(BanditTask):
    """
    A class to hold a multi-armed bandit task data.
    """

    def __init__(
        self,
        probs,
        choices,
        rewards,
        session_ids,
        starts=None,
        stops=None,
        datetime=None,
        metadata=None,
    ):
        super().__init__(
            probs=probs,
            choices=choices,
            rewards=rewards,
            session_ids=session_ids,
            starts=starts,
            stops=stops,
            datetime=datetime,
            metadata=metadata,
        )
        assert self.n_ports == 4, "Bandit4Arm requires exactly 4 arms"

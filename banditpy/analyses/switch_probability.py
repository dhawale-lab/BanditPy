import numpy as np
import pandas as pd
from ..core import Bandit2Arm
from numpy.lib.stride_tricks import sliding_window_view


class SwitchProb2Arm:
    def __init__(self, task: Bandit2Arm):
        assert isinstance(task, Bandit2Arm), "task must be a Bandit2Arm object"
        self.task = task

    def by_session(self, session_id=None):
        """Get the probability of switching between two ports in a session.

        Parameters
        ----------
        session_id : int,array-like, optional
            The session IDs to analyze. If None, all sessions are analyzed.

        Returns
        -------
        float
            The probability of switching between two ports in the specified session.
        """

        if session_id is None:
            print("Calculating switch probability for all sessions")
            session_id = self.task.sessions
            # Ignore the first trial of each session
            valid_indices = ~self.task.is_session_start

            # Calculate switches (change in choices)
            switches = np.diff(self.task.choices, prepend=self.task.choices[0]) != 0

            # Calculate switching probability
            switch_probability = np.mean(switches[valid_indices])

        elif isinstance(session_id, (list, np.ndarray)):
            print(f"Calculating switch probability for sessions: {session_id}")
            mask = self.task.session_ids == session_id
            choices = self.task.choices[mask]

            # Calculate the number of switches and total trials in the session
            switches = np.sum(np.diff(choices) != 0)
            total_trials = len(choices) - 1

            # Calculate the switch probability
            switch_probability = switches / total_trials if total_trials > 0 else 0

        return switch_probability

    def by_trial(self):
        """Get the probability of switching between ports as a function of trials.

        Returns
        -------
        array-like
            The probability of switching between two ports in the specified session.
        """

        # Calculate switches (change in choices)
        switches = np.diff(self.task.choices, prepend=self.task.choices[0]) != 0

        # convert to 2D array of shape (n_sessions, n_trials) and calculate switch probability across sessions and ignore the first trial
        switch_prob = (
            pd.DataFrame(np.split(switches, np.cumsum(self.task.ntrials_session)[:-1]))
            .mean(axis=0)
            .to_numpy()[1:]
        )

        # Calculate switch probability across session and ignore the first trial
        # switch_prob = np.nanmean(switches, axis=0)[1:]

        return switch_prob

    def by_history(self, n_past):
        """Get the probability of switching between ports as a function of history.

        References
        ----------
        Beron et al. 2022

        Parameters
        ----------
        n_past : int, optional
            History length

        Returns
        -------
        array
            The probability of switching on next action. If your n_past is 3, then probability of switching on next choice given unique sequence of 3 past actions/rewards.
        array_like
            Unique sequences. Coded as a,A,b,B representing same-unrewarded, same-rewarded, switched-unrewarded, switched-rewarded repectively.


        """

        assert self.task.n_ports == 2, "Only implemented for 2AB task"
        # Calculate switches (change in choices)
        choices = self.task.choices.copy()
        rewards = self.task.rewards.copy()

        switches = np.diff(choices)  # length is n_trials - 1
        switches_bool = switches != 0
        rewards = rewards[1:]  # length is n_trials - 1

        letter_code = np.empty_like(switches, dtype="<U1")
        letter_code[(switches == 0) & (rewards == 0)] = "a"
        letter_code[(switches == 0) & (rewards == 1)] = "A"
        letter_code[(switches != 0) & (rewards == 0)] = "b"
        letter_code[(switches != 0) & (rewards == 1)] = "B"

        # converting into n_past slices
        letter_code = sliding_window_view(letter_code, n_past)[:-1]
        # merging into single string
        letter_code = np.array(["".join(_) for _ in letter_code])
        seq_switches = switches_bool[n_past : len(letter_code) + n_past]

        unq_history = np.unique(letter_code)
        unq_indx = [np.where(letter_code == _)[0] for _ in unq_history]
        switch_prob = np.array([np.mean(seq_switches[_]) for _ in unq_indx])

        def sort_key(row):
            return (
                row[2].upper(),
                row[2].islower(),
                row[1].upper(),
                row[1].islower(),
            )

        sorted_seq = np.array(sorted(unq_history, key=sort_key))
        sort_indx = np.array([np.where(unq_history == _)[0][0] for _ in sorted_seq])
        switch_prob = switch_prob[sort_indx]

        return switch_prob, sorted_seq

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

    def by_trial(
        self,
        trial_window=None,
        split_by_reward=False,
        equalize_by=None,
        tier_threshold=0.5,
    ):
        """Get the probability of switching between ports as a function of trials.

        Captures oscillation/exploration behavior (e.g. sticking with an
        arm, then quickly switching after a low reward) at each trial
        position within a session.

        Parameters
        ----------
        trial_window : int, optional
            Number of consecutive trials to average within each session
            before averaging across sessions, by default None (one value
            per trial position). E.g. trial_window=10 gives one value per
            10-trial window instead of per trial — useful for running
            stats on coarser bins. Not to be confused with `window_ids`
            (experimental time-block) on the task — this windows over
            trial count, not recording time.
        split_by_reward : bool, optional
            If True, compute switch probability separately conditioned on
            whether the previous trial was rewarded or not (i.e.
            win-switch vs. lose-switch), returning a tuple
            `(switch_after_reward, switch_after_noreward)` instead of a
            single curve. Trials where the previous outcome doesn't match
            the condition (and each session's first trial, which has no
            previous trial) are NaN. Default False.
        equalize_by : {None, "combo", "deltap", "tier"}, optional
            Give every probability condition equal weight instead of
            letting more-sampled conditions dominate the average.

            - "combo": computed separately for each unique
              (order-independent) probability pair, then those curves are
              averaged with equal weight.
            - "deltap": same, but grouped by unique |p1 - p2| instead of
              the exact pair.
            - "tier": same, but grouped into three coarse tiers by how
              many arms are at/above `tier_threshold`: "low-low" (0),
              "high-low" (1), "high-high" (2).

            Default None (pool all trials/sessions as usual). Composable
            with `trial_window`.
        tier_threshold : float, optional
            Only used when `equalize_by="tier"`. Default 0.5.

        Returns
        -------
        array-like or tuple of array-like
            If `split_by_reward` is False: probability of switching at
            each trial position within a session. Shape (n_trials,), or
            (n_windows,) if `trial_window` is given. Each session's first
            trial has no previous choice to compare against, so it is NaN
            and excluded from the average.
            If True: tuple `(switch_after_reward, switch_after_noreward)`,
            each with the same shape.
        """
        task = self.task
        assert task.n_ports == 2, "Only implemented for 2AB task"

        def switch_metric(t):
            session_choices = np.split(t.choices, np.cumsum(t.ntrials_session)[:-1])
            return np.concatenate(
                [
                    np.concatenate(([np.nan], (sess[1:] != sess[:-1]).astype(float)))
                    for sess in session_choices
                ]
            )

        def reward_split_switch_metrics(t):
            session_choices = np.split(t.choices, np.cumsum(t.ntrials_session)[:-1])
            session_rewards = np.split(t.rewards, np.cumsum(t.ntrials_session)[:-1])

            after_reward_list = []
            after_noreward_list = []
            for ch, rw in zip(session_choices, session_rewards):
                switch = np.concatenate(([np.nan], (ch[1:] != ch[:-1]).astype(float)))
                prev_reward = np.concatenate(([np.nan], rw[:-1].astype(float)))

                after_reward_list.append(np.where(prev_reward == 1, switch, np.nan))
                after_noreward_list.append(np.where(prev_reward == 0, switch, np.nan))

            return np.concatenate(after_reward_list), np.concatenate(after_noreward_list)

        if split_by_reward:
            if equalize_by is not None:
                after_reward_curve = task._equalized_curve(
                    lambda t: t._session_curve(
                        reward_split_switch_metrics(t)[0], trial_window
                    ),
                    equalize_by,
                    tier_threshold,
                )
                after_noreward_curve = task._equalized_curve(
                    lambda t: t._session_curve(
                        reward_split_switch_metrics(t)[1], trial_window
                    ),
                    equalize_by,
                    tier_threshold,
                )
                return after_reward_curve, after_noreward_curve

            after_reward, after_noreward = reward_split_switch_metrics(task)
            return (
                task._session_curve(after_reward, trial_window),
                task._session_curve(after_noreward, trial_window),
            )

        if equalize_by is not None:
            return task._equalized_curve(
                lambda t: t._session_curve(switch_metric(t), trial_window),
                equalize_by,
                tier_threshold,
            )

        return task._session_curve(switch_metric(task), trial_window)

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

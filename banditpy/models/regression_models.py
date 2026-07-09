import numpy as np
import pandas as pd
from .. import core

from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LogisticRegression
from pathlib import Path


class Logistic2Arm:
    """
    Logistic regression model for a 2-armed bandit task.

    Based on Miller et al. 2021, "From predictive models to cognitive models:
    separating predictions from explanations in human behavioral data".

    The model predicts the current trial's choice from the past `n_past` trials
    using three sets of lagged features:
      - C * R  : interaction (outcome-weighted choice, "effect of outcome")
      - C      : previous choices alone ("choice preservation")
      - R      : previous rewards alone ("reward seeking")

    Parameters
    ----------
    task : core.Bandit2Arm
        A 2-armed bandit task object containing choices, rewards, and metadata.
    n_past : int, optional
        Number of past trials to use as lagged features. Default is 5.
    reset_at : array-like of int, optional
        Binary array of length n_trials. A 1 marks the first trial of a new
        epoch (e.g., a new experimental window or session). Trials whose
        lookback window crosses an epoch boundary are excluded from fitting
        and prediction. If None, no boundaries are enforced.

    Attributes
    ----------
    choices : np.ndarray
        Reformatted choices: -1 for port 1 (left), +1 for port 2 (right).
    rewards : np.ndarray
        Reformatted rewards: -1 for no reward, +1 for reward.
    model : LogisticRegression
        Underlying sklearn logistic regression model.
    """

    def __init__(self, task: core.Bandit2Arm, n_past=5, reset_at=None):
        """
        Initialize the Logistic2Arm model.

        Parameters
        ----------
        task : core.Bandit2Arm
            A 2-armed bandit task object.
        n_past : int, optional
            Number of past trials used as lagged features. Default is 5.
        reset_at : array-like of int, optional
            Binary array of length n_trials. A 1 at position i means trial i
            starts a new epoch; any feature window spanning an epoch boundary
            is dropped. If None, all windows are used. Example: set to 1 at
            each window_id or session_id change to prevent cross-boundary
            history leakage.
        """
        assert task.n_ports == 2, "Only 2-armed bandit task is supported"
        self.choices, self.rewards = self._reformat_choices_rewards(
            task.choices.copy(), task.rewards.copy()
        )
        self.rewards = task.rewards.copy()
        self.n_past = n_past
        if reset_at is not None:
            reset_at = np.asarray(reset_at)
            assert (
                reset_at.shape == task.choices.shape
            ), "reset_at must have the same length as trials"
        self.reset_at = reset_at
        self.model = LogisticRegression(solver="lbfgs")
        self.feature_names = []

    @property
    def coef(self):
        return self.model.coef_.squeeze().reshape(3, -1)

    @property
    def coef_names(self):
        return ["reward seeking", "choice preservation", "effect of outcome"]

    def get_coef_df(self, form="wide"):
        """Get the coefficients and their names as a DataFrame."""
        df = pd.DataFrame(data=self.coef.T, columns=self.coef_names)
        df["past_id"] = np.arange(1, self.n_past + 1)

        if form == "long":
            df = df.melt(id_vars="past_id", var_name="coef_name", value_name="coef")
        return df

    def _reformat_choices_rewards(self, choices, rewards):
        """
        Convert choices to -1 for left, 1 for right.
        Convert rewards to -1 for no reward, 1 for reward.
        """
        choices[choices == 1] = -1
        choices[choices == 2] = 1

        rewards[rewards == 0] = -1
        rewards[rewards == 1] = 1
        return choices, rewards

    def _prepare_features(self, choices, rewards):
        """
        Build the feature matrix X and target vector y from lagged choices and rewards.

        For each target trial t, features are the n_past preceding trials:
          - C_windows  : lagged choices (most recent first)
          - R_windows  : lagged rewards (most recent first)
          - CxR_windows: element-wise product (interaction term)

        If reset_at is set, trials whose lookback window spans an epoch boundary
        are removed: reset_at is cumsum'd into epoch ids, and a sliding window of
        size n_past+1 (lookback + target) is used to detect boundary crossings.
        Because the cumsum is monotonically non-decreasing, a window is valid iff
        its first and last epoch id are equal.

        Parameters
        ----------
        choices : np.ndarray
            Reformatted choices array (-1 / +1).
        rewards : np.ndarray
            Reformatted rewards array (-1 / +1).

        Returns
        -------
        X : np.ndarray, shape (n_valid_samples, n_past * 3)
            Feature matrix. Columns are ordered [CxR_lag1..n, C_lag1..n, R_lag1..n].
        y : np.ndarray, shape (n_valid_samples,)
            Target choices (-1 or +1).
        """
        # Generate sliding windows and flipping arrays from left to right so that the first column is the most recent choice
        C_windows = sliding_window_view(choices, window_shape=self.n_past)[:-1][:, ::-1]
        R_windows = sliding_window_view(rewards, window_shape=self.n_past)[:-1][:, ::-1]

        actual_choices = choices[self.n_past :]

        # Interaction term
        CxR_windows = C_windows * R_windows

        # Stack features: shape (n_samples, n_lags * 3)
        X = np.hstack([CxR_windows, C_windows, R_windows])
        y = actual_choices.astype(int)  # -1 for left , 1 for right

        if self.reset_at is not None:
            # Convert boundary flags to epoch ids: [1,0,0,1,0,...] → [1,1,1,2,2,...]
            epoch_ids = np.cumsum(self.reset_at)
            # Window of size n_past+1 spans [lookback... target]; since cumsum is
            # monotonically non-decreasing, first==last iff all elements are equal
            epoch_windows = sliding_window_view(epoch_ids, window_shape=self.n_past + 1)
            valid_mask = epoch_windows[:, 0] == epoch_windows[:, -1]
            X, y = X[valid_mask], y[valid_mask]

        return X, y

    def fit(self):
        """
        Fit the logistic regression model on all valid trials.

        Returns
        -------
        self : Logistic2Arm
            The fitted model instance (for method chaining).
        """
        X, y = self._prepare_features(self.choices, self.rewards)
        self.model.fit(X, y)
        return self

    # def predict_proba(self, choices, rewards):
    #     X, _ = self._prepare_features(choices, rewards)
    #     return self.model.predict_proba(X)[:, 1]  # Prob of choosing right

    def predict(self, stochastic=True):
        """
        Predict choices for all valid trials.

        Parameters
        ----------
        stochastic : bool, optional
            If True, sample choices from the predicted probability distribution.
            If False, take the argmax (deterministic). Default is True.

        Returns
        -------
        predicted_choices : np.ndarray
            Predicted choices encoded as 1 (left) or 2 (right).
        """
        X, _ = self._prepare_features(self.choices, self.rewards)

        if stochastic:
            probs = self.model.predict_proba(X)
            predicted_choices = np.array([np.random.choice([1, 2], p=p) for p in probs])
        else:
            predicted_choices = self.model.predict(X)
            predicted_choices[predicted_choices == 1] = 2
            predicted_choices[predicted_choices == -1] = 1

        return predicted_choices

    def get_coef_dict(self):
        """
        Return model coefficients as a dictionary keyed by coefficient name.

        Returns
        -------
        dict
            Keys are coef_names; values are arrays of length n_past.
        """
        coef_dict = dict(zip(self.coef_names, self.coef))
        coef_dict.update({"past_id": np.arange(1, self.n_past + 1)})
        return coef_dict

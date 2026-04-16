import numpy as np
import pandas as pd
from .. import core

from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LogisticRegression
from pathlib import Path


class Logistic2Arm:
    """Based on Miller et al. 2021, "From predictive models to cognitive models....." """

    def __init__(self, task: core.Bandit2Arm, n_past=5):
        assert task.n_ports == 2, "Only 2-armed bandit task is supported"
        self.choices, self.rewards = self._reformat_choices_rewards(
            task.choices.copy(), task.rewards.copy()
        )
        self.rewards = task.rewards.copy()
        self.n_past = n_past
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
        Prepare lagged features using sliding_window_view.
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

        return X, y

    def fit(self):
        X, y = self._prepare_features(self.choices, self.rewards)
        self.model.fit(X, y)
        return self

    # def predict_proba(self, choices, rewards):
    #     X, _ = self._prepare_features(choices, rewards)
    #     return self.model.predict_proba(X)[:, 1]  # Prob of choosing right

    def predict(self, stochastic=True):
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
        return dict(zip(self.coef_names, self.coef))

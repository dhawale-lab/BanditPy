import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from .. import core


def plot_trial_by_trial_2Arm(
    task: core.Bandit2Arm, sort_by_deltaprob=False, ax=None
) -> None:
    """
    Plot the trial-by-trial performance of a 2-armed bandit problem.
    """
    if not isinstance(task, core.Bandit2Arm):
        raise TypeError("task must be an instance of Bandit2Arm")

    choice_bool = task.is_choice_high
    ntrials_session = task.ntrials_session

    choice_bool_session = pd.DataFrame(
        np.split(choice_bool, np.cumsum(ntrials_session)[:-1])
    ).to_numpy()

    if sort_by_deltaprob:
        probs = task.session_probs
        deltaprob = np.abs(probs[:, 0] - probs[:, 1])
        sort_indx = np.argsort(deltaprob)
        choice_bool_session = choice_bool_session[sort_indx, :]

    # Plotting
    if ax is None:
        _, ax = plt.subplots()

    # cmap = sns.color_palette("light:b_r", n_colors=2, as_cmap=True)
    # cmap = sns.color_palette("ch:start=.2,rot=-.3_r", as_cmap=True)

    cmap = ListedColormap(["#1f78b4", "#a6cee3"])
    ax.pcolormesh(choice_bool_session, cmap=cmap)
    ax.set_title("Trial-by-Trial behavior of 2-Armed Bandit")
    ax.set_xlabel("Trials")
    ax.set_ylabel("Sessions #")

    if sort_by_deltaprob:
        ax.spines["right"].set_visible(True)
        ax2 = ax.twinx()
        sorted_deltaprob = deltaprob[sort_indx].round(2)

        if task.n_sessions >= 3:
            low, mid, high = 0, task.n_sessions // 2, task.n_sessions - 1
            ax2.set_yticks([low, mid, high], sorted_deltaprob[[low, mid, high]])
        else:
            ticks = list(range(task.n_sessions))
            ax2.set_yticks(ticks, sorted_deltaprob[: len(ticks)])

        ax2.set_ylabel("Delta probabilities")

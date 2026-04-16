import numpy as np


# Single exponential rise to asymptote
def fit_single_exp(t, P0, P_inf, tau):
    return P0 + (P_inf - P0) * (1 - np.exp(-t / tau))


# Double exponential rise to asymptote
def fit_double_exp(t, P0, P_inf, A1, tau1, tau2):
    A2 = 1 - A1  # constrain weights to sum to 1
    return P0 + (P_inf - P0) * (1 - A1 * np.exp(-t / tau1) - A2 * np.exp(-t / tau2))

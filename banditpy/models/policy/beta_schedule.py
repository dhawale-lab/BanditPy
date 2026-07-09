import numpy as np
from typing import Dict, List, Tuple

from .base import ParameterGroup, ParameterSpec


class BetaSchedule:
    """
    Base class for softmax inverse-temperature (beta) schedules.

    Separates the decision stage (softmax) from the learning policy.
    ``DecisionModel`` holds a ``BetaSchedule`` instance and calls:

        - ``get_beta()``    : current beta value for this trial
        - ``get_epsilon()`` : current lapse rate (default 0)
        - ``update()``     : advance internal state after each ``policy.update()``
        - ``reset()``       : reset to initial state at session/block boundaries
        - ``set_params()``  : receive fitted parameter values

    Subclasses declare their free parameters via a nested ``Params(ParameterGroup)``
    class (same convention as ``BasePolicy``).
    Bounds are accessible and editable via ``self.params``.

    Examples
    --------
    model = DecisionModel(task, Qlearn2Arm())
    model = DecisionModel(task, Qlearn2Arm(), beta_schedule=ExponentialBeta())
    model.beta_schedule.params.beta_0.set_bounds(0.5, 20.0)
    """

    class Params(ParameterGroup):
        """Default empty params; subclasses override with their ParameterSpec attrs."""

        pass

    def __init__(self):
        self.params = type(self).Params()
        self._param_specs: Dict[str, ParameterSpec] = self.params.specs_dict()

    # --- Parameter API (mirrors BasePolicy) ---

    def active_parameter_names(self) -> List[str]:
        return self.params.active_names()

    def get_bounds(self) -> Dict[str, Tuple]:
        return self.params.bounds_dict()

    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return self.params.specs_dict()

    def set_bounds(self, bounds: Dict[str, Tuple] = None, **kwargs):
        """Set parameter bounds. Accepts a dict or keyword arguments.

        Examples
        --------
        beta_schedule.set_bounds({"beta_rate": (0.0, 0.1)})
        beta_schedule.set_bounds(beta_rate=(0.0, 0.1))
        """
        combined = {**(bounds or {}), **kwargs}
        for k, v in combined.items():
            if k not in self._param_specs:
                raise ValueError(f"Unknown parameter '{k}'")
            self._param_specs[k].bounds = tuple(v)

    def set_params(self, params: Dict[str, float]):
        for k, v in params.items():
            if k in self.params:
                self.params[k] = float(v)

    def get_param_defaults(self) -> Dict[str, float]:
        return {k: p.default for k, p in self._param_specs.items()}

    # --- Schedule interface ---

    def get_beta(self) -> float:
        raise NotImplementedError

    def get_epsilon(self) -> float:
        return float(self.params.get("epsilon", 0.0))

    def update(self):
        """Advance internal trial counter. Called after each ``policy.update()``."""
        pass

    def reset(self):
        """Reset to initial state. Called at session/block boundaries."""
        pass


class StaticBeta(BetaSchedule):
    """
    Constant inverse temperature across all trials.

    Parameters: ``beta``, ``epsilon`` (inactive by default).
    """

    class Params(ParameterGroup):
        beta = ParameterSpec(
            "beta",
            (0.1, 10.0),
            description="Inverse temperature for softmax",
        )
        epsilon = ParameterSpec(
            "epsilon",
            (0.0, 0.3),
            default=0.0,
            active=False,
            description="Lapse rate (probability of random choice)",
        )

    def get_beta(self) -> float:
        return float(self.params["beta"])


class ExponentialBeta(BetaSchedule):
    """
    Exponentially growing beta: beta_0 * exp(beta_rate * t).

    The counter t resets to 0 at each reset() call (session boundary),
    so beta returns to beta_0 at the start of each session.
    Output is clipped to [0.01, 100.0].

    Parameters: beta_0, beta_rate, epsilon (inactive by default).

    Notes
    -----
    Positive beta_rate → beta grows (increasingly greedy/exploitative over a session).
    Default bounds (0.0, 0.05) enforce exploration-decreasing behaviour;
    widen with model.beta_schedule.params.beta_rate.set_bounds(0.0, 0.1).
    """

    class Params(ParameterGroup):
        beta_0 = ParameterSpec(
            "beta_0",
            (0.1, 10.0),
            description="Initial inverse temperature",
        )
        beta_rate = ParameterSpec(
            "beta_rate",
            (0.0, 0.05),
            description="Exponential growth rate of beta per trial (positive → less exploration over time)",
        )
        epsilon = ParameterSpec(
            "epsilon",
            (0.0, 0.3),
            default=0.0,
            active=False,
            description="Lapse rate (probability of random choice)",
        )

    def __init__(self):
        super().__init__()
        self._t: int = 0

    def get_beta(self) -> float:
        beta = self.params["beta_0"] * np.exp(self.params["beta_rate"] * self._t)
        return float(np.clip(beta, 0.01, 100.0))

    def update(self):
        self._t += 1

    def reset(self):
        self._t = 0


class LinearBeta(BetaSchedule):
    """
    Linearly changing beta: ``beta_0 + beta_rate * t``.

    The counter ``t`` resets to 0 at each ``reset()`` call.
    Output is clipped to ``[0.01, 100.0]``.

    Parameters: ``beta_0``, ``beta_rate``, ``epsilon`` (inactive by default).
    """

    class Params(ParameterGroup):
        beta_0 = ParameterSpec(
            "beta_0",
            (0.1, 10.0),
            description="Initial inverse temperature",
        )
        beta_rate = ParameterSpec(
            "beta_rate",
            (-0.1, 0.1),
            description="Linear rate of change of beta per trial",
        )
        epsilon = ParameterSpec(
            "epsilon",
            (0.0, 0.3),
            default=0.0,
            active=False,
            description="Lapse rate (probability of random choice)",
        )

    def __init__(self):
        super().__init__()
        self._t: int = 0

    def get_beta(self) -> float:
        beta = self.params["beta_0"] + self.params["beta_rate"] * self._t
        return float(np.clip(beta, 0.01, 100.0))

    def update(self):
        self._t += 1

    def reset(self):
        self._t = 0


class PowerLawBeta(BetaSchedule):
    """
    Power-law changing beta: ``beta_0 * (t + 1) ** beta_exp``.

    The counter ``t`` resets to 0 at each ``reset()`` call.
    Output is clipped to ``[0.01, 100.0]``.

    Parameters: ``beta_0``, ``beta_exp``, ``epsilon`` (inactive by default).
    """

    class Params(ParameterGroup):
        beta_0 = ParameterSpec(
            "beta_0",
            (0.1, 10.0),
            description="Initial inverse temperature",
        )
        beta_exp = ParameterSpec(
            "beta_exp",
            (-1.0, 1.0),
            description="Power-law exponent for beta schedule",
        )
        epsilon = ParameterSpec(
            "epsilon",
            (0.0, 0.3),
            default=0.0,
            active=False,
            description="Lapse rate (probability of random choice)",
        )

    def __init__(self):
        super().__init__()
        self._t: int = 0

    def get_beta(self) -> float:
        beta = self.params["beta_0"] * ((self._t + 1) ** self.params["beta_exp"])
        return float(np.clip(beta, 0.01, 100.0))

    def update(self):
        self._t += 1

    def reset(self):
        self._t = 0


class PowerLaw10Beta(BetaSchedule):
    """
    Power-law beta anchored at trial 10: ``theta * (t / 10) ** c``.

    At trial 10 the schedule always returns ``theta``, making ``theta``
    directly interpretable as the inverse temperature mid-session.
    ``c`` (the response-consistency parameter) controls the shape:

    * ``c > 0`` — beta grows over time (increasingly greedy)
    * ``c < 0`` — beta shrinks over time (increasingly exploratory)
    * ``c = 0`` — beta is constant at ``theta`` for all trials

    The counter ``t`` resets to 1 at each ``reset()`` call.
    Output is clipped to ``[0.01, 100.0]``.

    Reference: Equation 14 in Wilson & Niv (2012) or equivalent.

    Parameters: ``theta``, ``c``, ``epsilon`` (inactive by default).
    """

    class Params(ParameterGroup):
        theta = ParameterSpec(
            "theta",
            (0.1, 20.0),
            description="Inverse temperature at trial 10 (anchor point)",
        )
        c = ParameterSpec(
            "c",
            (-3.0, 3.0),
            description="Response-consistency exponent (+ grows, - shrinks)",
        )
        epsilon = ParameterSpec(
            "epsilon",
            (0.0, 0.3),
            default=0.0,
            active=False,
            description="Lapse rate (probability of random choice)",
        )

    def __init__(self):
        super().__init__()
        self._t: int = 1

    def get_beta(self) -> float:
        beta = self.params["theta"] * ((self._t / 10.0) ** self.params["c"])
        return float(np.clip(beta, 0.01, 100.0))

    def update(self):
        self._t += 1

    def reset(self):
        self._t = 1


class NoBeta(BetaSchedule):
    """
    Fixed ``beta=1.0``, ``epsilon=0.0`` — no free parameters.

    Use for policies that handle their own softmax internally
    (e.g. ``QlearnWM2Arm``, ``QlearnHierarchical2Arm``). Passing
    their log-probability logits through softmax with ``beta=1`` is a
    mathematical identity, so no distortion is introduced.
    """

    class Params(ParameterGroup):
        pass

    def get_beta(self) -> float:
        return 1.0

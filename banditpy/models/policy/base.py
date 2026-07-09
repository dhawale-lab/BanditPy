import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List


@dataclass
class ParameterSpec:
    """
    Specification and current value for a single model parameter.

    Holds bounds, default, active flag, description, and the current
    fitted/set value in one object.

    Examples
    --------
    policy.params.alpha_c.set_bounds(0.0, 0.5)
    policy.params.alpha_c.value       # current fitted value
    policy.params.alpha_c.enable()    # include in optimisation
    """

    name: str
    bounds: Tuple[float, float]
    default: Optional[float] = None
    active: bool = True
    description: str = ""
    value: Optional[float] = None  # current fitted/set value

    def copy(self) -> "ParameterSpec":
        return ParameterSpec(
            name=self.name,
            bounds=tuple(self.bounds),
            default=self.default,
            active=self.active,
            description=self.description,
            value=self.value,
        )

    def set_bounds(self, lo: float, hi: float) -> "ParameterSpec":
        """Set bounds in-place. Returns self for chaining."""
        self.bounds = (lo, hi)
        return self

    def set_value(self, value: float) -> "ParameterSpec":
        """Set current value in-place. Returns self for chaining."""
        self.value = value
        return self

    def enable(self) -> "ParameterSpec":
        """Mark as active (will be optimised). Returns self for chaining."""
        self.active = True
        return self

    def disable(self) -> "ParameterSpec":
        """Mark as inactive (use default value). Returns self for chaining."""
        self.active = False
        return self


class ParameterGroup:
    """
    Typed container of ParameterSpec instances for a policy or beta schedule.

    Subclasses declare parameters as class-level ParameterSpec attributes::

        class Params(ParameterGroup):
            alpha_c = ParameterSpec("alpha_c", (0.0, 1.0), description="...")
            alpha_u = ParameterSpec("alpha_u", (0.0, 1.0), description="...")

    Per-instance isolation: ``__init__`` copies each class-level ParameterSpec
    to the instance so that modifying one instance's bounds never affects others.

    Both attribute-style (autocomplete) and dict-style (backward compat) access
    are supported::

        policy.params.alpha_c.set_bounds(0.0, 0.5)   # attribute-style
        policy.params["alpha_c"]                       # dict-style (returns value)
        policy.params["alpha_c"] = 0.3                 # dict-style (sets value)
    """

    _spec_names: List[str] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Collect ParameterSpec attrs defined directly on this class, in order
        cls._spec_names = [
            name for name, val in cls.__dict__.items() if isinstance(val, ParameterSpec)
        ]

    def __init__(self):
        # Create per-instance copies so instances don't share ParameterSpec objects
        for name in type(self)._spec_names:
            class_spec = type(self).__dict__[name]
            setattr(self, name, class_spec.copy())

    # --- Dict-style access (backward compat with self.params["key"]) ---

    def __getitem__(self, key: str) -> float:
        """Return current value; fall back to spec.default if not yet set."""
        spec = getattr(self, key, None)
        if not isinstance(spec, ParameterSpec):
            raise KeyError(key)
        if spec.value is not None:
            return spec.value
        if spec.default is not None:
            return spec.default
        raise KeyError(f"Parameter '{key}' has no value set yet")

    def __setitem__(self, key: str, value: float):
        """Set current value of parameter ``key``."""
        spec = getattr(self, key, None)
        if not isinstance(spec, ParameterSpec):
            raise KeyError(key)
        spec.value = float(value)

    def get(self, key: str, default=None):
        """Return current value; fall back to spec.default then provided default."""
        spec = getattr(self, key, None)
        if not isinstance(spec, ParameterSpec):
            return default
        if spec.value is not None:
            return spec.value
        if spec.default is not None:
            return spec.default
        return default

    def __contains__(self, key: str) -> bool:
        return isinstance(getattr(self, key, None), ParameterSpec)

    # --- Iteration and introspection ---

    def __iter__(self):
        for name in type(self)._spec_names:
            yield name, getattr(self, name)

    def specs_dict(self) -> Dict[str, "ParameterSpec"]:
        """Return {name: ParameterSpec} for all parameters."""
        return {name: getattr(self, name) for name in type(self)._spec_names}

    def active_names(self) -> List[str]:
        """Return names of active (to-be-fitted) parameters."""
        return [name for name, spec in self if spec.active]

    def bounds_dict(self) -> Dict[str, Tuple]:
        """Return {name: bounds} for all parameters."""
        return {name: spec.bounds for name, spec in self}

    def set_bounds_for(self, name: str, lo: float, hi: float) -> "ParameterGroup":
        """Set bounds by string name. Returns self for chaining."""
        getattr(self, name).set_bounds(lo, hi)
        return self

    def to_values_dict(self) -> Dict[str, float]:
        """Return {name: value} for all parameters."""
        return {name: spec.value for name, spec in self}


class BasePolicy:
    """
    Subclasses define ``parameters = [ParameterSpec(...), ...]``.

    Each policy owns its ``beta_schedule`` (a ``BetaSchedule`` instance that
    controls the softmax inverse temperature used by ``DecisionModel``). The
    default is ``StaticBeta()``; override via the ``beta_schedule`` constructor
    argument or by setting ``default_beta_schedule`` as a class variable.

    Lifecycle:
        __init__()  → reset()  (policy starts in a valid state)
        reset()     → initialize state for a new session
        forget()    → optional decay step between trials
        logits()    → compute choice values
        update()    → apply learning update
    """

    _disable_common: List[str] = []
    default_beta_schedule = None  # BetaSchedule subclass; None → StaticBeta

    class Params(ParameterGroup):
        """Default empty params; subclasses override with their ParameterSpec attrs."""

        pass

    def __init__(self, beta_schedule=None, **kwargs):
        # Instantiate per-class Params, creating per-instance ParameterSpec copies
        self.params = type(self).Params()

        # Convenience reference for legacy/internal code (same ParameterSpec objects)
        self._param_specs: Dict[str, ParameterSpec] = self.params.specs_dict()

        # Disable parameters declared in _disable_common
        for name in self._disable_common:
            if name in self._param_specs:
                spec = self._param_specs[name]
                spec.active = False
                if spec.default is None:
                    spec.default = 0.0

        # Resolve beta_schedule: explicit arg > subclass default > StaticBeta.
        # StaticBeta is imported lazily to avoid a circular import
        # (beta_schedule.py already imports from base.py).
        if beta_schedule is None:
            from .beta_schedule import StaticBeta  # noqa: PLC0415

            beta_schedule = (self.default_beta_schedule or StaticBeta)()
        self.beta_schedule = beta_schedule

    # ---------------- Parameter API ----------------

    def param_names(self):
        return list(self._param_specs.keys())

    def active_parameter_names(self):
        return self.params.active_names() + self.beta_schedule.active_parameter_names()

    def get_bounds(self):
        d = self.params.bounds_dict()
        d.update(self.beta_schedule.get_bounds())
        return d

    def set_bounds(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self._param_specs:
                raise ValueError(f"Unknown parameter '{k}'")
            self._param_specs[k].bounds = tuple(v)

    def set_params(self, params: Dict[str, float]):
        beta_names = set(self.beta_schedule.parameter_specs())
        for k, v in params.items():
            if k in beta_names:
                self.beta_schedule.params[k] = float(v)
            elif k in self.params:
                self.params[k] = float(v)
            else:
                raise ValueError(f"Unknown parameter '{k}'")

    def get_param_defaults(self):
        return {k: p.default for k, p in self._param_specs.items()}

    def parameter_specs(self):
        combined = self.params.specs_dict()
        combined.update(self.beta_schedule.parameter_specs())
        return combined

    def describe(self, as_markdown: bool = False):
        """
        Print (and optionally return) a table of parameter metadata.

        Shows policy parameters and beta schedule parameters in separate sections.

        Parameters
        ----------
        as_markdown : bool, optional
            If True, returns a Markdown table string instead of printing.
        """
        headers = ["Parameter", "Bounds", "Default", "Active", "Description"]

        def _make_rows(specs):
            rows = []
            for p in specs.values():
                default_str = "—" if p.default is None else f"{p.default}"
                rows.append(
                    [
                        p.name,
                        f"{p.bounds}",
                        default_str,
                        "True" if p.active else "False",
                        p.description or "",
                    ]
                )
            return rows

        sched_specs = self.beta_schedule.parameter_specs()
        sched_name = self.beta_schedule.__class__.__name__

        if not as_markdown:
            col_hdr = (
                f"{headers[0]:<14}{headers[1]:<18}{headers[2]:<8}"
                f"{headers[3]:<8}{headers[4]}"
            )
            sep = "-" * 90
            print("Policy parameters")
            print(sep)
            print(col_hdr)
            print(sep)
            for r in _make_rows(self._param_specs):
                print(f"{r[0]:<14}{r[1]:<18}{r[2]:<8}{r[3]:<8}{r[4]}")
            print(f"\nBeta schedule ({sched_name})")
            print(sep)
            if sched_specs:
                print(col_hdr)
                print(sep)
                for r in _make_rows(sched_specs):
                    print(f"{r[0]:<14}{r[1]:<18}{r[2]:<8}{r[3]:<8}{r[4]}")
            else:
                print("  (no parameters)")
            return

        # ---- Markdown export ----
        md = ["### Policy parameters\n"]
        md += [
            "| Parameter | Bounds | Default | Active | Description |",
            "|----------:|--------|--------:|:------:|-------------|",
        ]
        for r in _make_rows(self._param_specs):
            md.append(f"| {r[0]} | `{r[1]}` | {r[2]} | {r[3]} | {r[4]} |")
        md.append(f"\n### Beta schedule ({sched_name})\n")
        if sched_specs:
            md += [
                "| Parameter | Bounds | Default | Active | Description |",
                "|----------:|--------|--------:|:------:|-------------|",
            ]
            for r in _make_rows(sched_specs):
                md.append(f"| {r[0]} | `{r[1]}` | {r[2]} | {r[3]} | {r[4]} |")
        else:
            md.append("*(no parameters)*")
        return "\n".join(md)

    # ---------------- Lifecycle hooks ----------------

    def reset(self):
        """Initialize internal state (must be implemented by subclasses)."""
        raise NotImplementedError

    def forget(self):
        """Optional decay step; default is a no-op."""
        pass

    def logits(self):
        raise NotImplementedError

    def update(self, choice, reward):
        raise NotImplementedError

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List


@dataclass
class ParameterSpec:
    name: str
    bounds: Tuple[float, float]
    default: Optional[float] = None
    active: bool = True
    description: str = ""

    def copy(self):
        return ParameterSpec(
            name=self.name,
            bounds=tuple(self.bounds),
            default=self.default,
            active=self.active,
            description=self.description,
        )


class BoundsRegistry:
    """
    Dict-style bounds editor.
    Safe for pickling and parallel execution.

    Examples
    --------
    policy.bounds["beta"] = (0.1, 20)
    policy.bounds.enable("tau")
    policy.bounds.disable("lr_unchosen")
    """

    def __init__(self, specs: Dict[str, ParameterSpec]):
        self._specs = specs

    # --- dict-like access ---

    def __getitem__(self, name):
        if name not in self._specs:
            raise KeyError(f"No such parameter '{name}'")
        return self._specs[name].bounds

    def __setitem__(self, name, value):
        if name not in self._specs:
            raise KeyError(f"No such parameter '{name}'")
        self._specs[name].bounds = tuple(value)

    # --- enable / disable parameters for fitting ---

    def enable(self, name):
        self._specs[name].active = True

    def disable(self, name):
        self._specs[name].active = False

    # --- helpers ---

    def as_list(self, names: List[str]):
        return [self._specs[n].bounds for n in names]

    def to_dict(self):
        return {k: p.bounds for k, p in self._specs.items()}


class BasePolicy:
    """
    Subclasses define `parameters = [ParameterSpec(...), ...]`.

    Lifecycle:
        __init__()  → reset()  (policy starts in a valid state)
        reset()     → initialize state for a new session
        forget()    → optional decay step between trials
        logits()    → compute choice values
        update()    → apply learning update
    """

    parameters: List[ParameterSpec] = []

    def __init__(self):
        self._param_specs: Dict[str, ParameterSpec] = {
            p.name: p.copy() for p in self.parameters
        }

        # Runtime parameter values after fit() or set_params()
        self.params: Dict[str, float] = {}

        # Dict-style bounds API
        self.bounds = BoundsRegistry(self._param_specs)

    # ---------------- Parameter API ----------------

    def param_names(self):
        return list(self._param_specs.keys())

    def active_parameter_names(self):
        return [k for k, p in self._param_specs.items() if p.active]

    def get_bounds(self):
        return self.bounds.to_dict()

    def set_bounds(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self._param_specs:
                raise ValueError(f"Unknown parameter '{k}'")
            self._param_specs[k].bounds = tuple(v)

    def set_params(self, params: Dict[str, float]):
        for k, v in params.items():
            if k not in self._param_specs:
                raise ValueError(f"Unknown parameter '{k}'")
            self.params[k] = float(v)

    def get_param_defaults(self):
        return {k: p.default for k, p in self._param_specs.items()}

    def parameter_specs(self):
        return self._param_specs

    def describe(self, as_markdown: bool = False):
        """
        Print (and optionally return) a table of parameter metadata.

        Parameters
        ----------
        as_markdown : bool, optional
            If True, returns a Markdown table string instead of printing.
        """

        specs = self._param_specs.values()
        headers = ["Parameter", "Bounds", "Default", "Active", "Description"]

        rows = []
        for p in specs:
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

        # ---- Console table ----
        if not as_markdown:
            print(
                f"{headers[0]:<14}{headers[1]:<18}{headers[2]:<8}"
                f"{headers[3]:<8}{headers[4]}"
            )
            print("-" * 90)
            for r in rows:
                print(f"{r[0]:<14}{r[1]:<18}{r[2]:<8}{r[3]:<8}{r[4]}")
            return

        # ---- Markdown export ----
        md = [
            "| Parameter | Bounds | Default | Active | Description |",
            "|----------:|--------|--------:|:------:|-------------|",
        ]
        for r in rows:
            md.append(f"| {r[0]} | `{r[1]}` | {r[2]} | {r[3]} | {r[4]} |")
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

"""Scenario registry -- auto-discovers Simulation subclasses from this package."""

from __future__ import annotations

import importlib
import pkgutil

from harness.evals.base import Simulation

ALL_SIMULATIONS: dict[str, type[Simulation]] = {}

for _finder, _module_name, _is_pkg in pkgutil.iter_modules(__path__):
    _module = importlib.import_module(f"{__name__}.{_module_name}")
    for _attr_name in dir(_module):
        _attr = getattr(_module, _attr_name)
        if (
            isinstance(_attr, type)
            and issubclass(_attr, Simulation)
            and _attr is not Simulation
            and getattr(_attr, "name", "")
        ):
            ALL_SIMULATIONS[_attr.name] = _attr


def get_simulation_cls(name: str) -> type[Simulation]:
    """Look up a simulation class by name. Raises KeyError if not found."""
    return ALL_SIMULATIONS[name]

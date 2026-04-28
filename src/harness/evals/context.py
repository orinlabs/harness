"""ContextVar for the active simulation instance.

Tool handlers that need to advance simulated time or access simulation
state call ``get_simulation()`` during an eval run.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Simulation

_active_simulation: contextvars.ContextVar[Simulation | None] = contextvars.ContextVar(
    "active_simulation", default=None
)


def get_simulation() -> Simulation:
    sim = _active_simulation.get()
    if sim is None:
        raise RuntimeError("No active simulation in this context")
    return sim


def set_simulation(sim: Simulation | None) -> None:
    _active_simulation.set(sim)

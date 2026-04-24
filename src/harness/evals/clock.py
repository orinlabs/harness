"""Offset-based simulated clock for eval runs.

The clock stores a global offset (as a ``contextvars.ContextVar``) between
wall-clock time and simulated time. Callers that need simulation-aware
"now" should use ``SimulatedClock.now()`` or the module-level
``_simulated_now`` helper. Between scheduled events the runner jumps
forward via ``advance_to()``; during agent execution wall-clock ticks
pass through naturally (a 45-second LLM call shows up as 45 simulated
seconds).

The bedrock version of this file monkey-patched ``django.utils.timezone.now``
so that every call site in the Django codebase transparently saw
simulated time. The harness has no analogous global "now" shim, so this
module only exposes the offset — callers that care about simulated time
explicitly route through ``SimulatedClock``. If we later add tool handlers
that read wall-clock time directly, this module can grow a monkey-patch
hook for that specific source.
"""

from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)

_sim_clock_offset: contextvars.ContextVar[timedelta | None] = contextvars.ContextVar(
    "sim_clock_offset", default=None
)


def _original_now() -> datetime:
    """Wall-clock now (UTC, timezone-aware)."""
    return datetime.now(tz=UTC)


def _simulated_now() -> datetime:
    """Return wall-clock-plus-offset when inside a simulated clock context."""
    offset = _sim_clock_offset.get()
    if offset is None:
        return _original_now()
    return _original_now() + offset


class SimulatedClock:
    """Manages simulated time via a global offset from wall-clock time."""

    def __init__(self, start_time: datetime):
        self.start_time = start_time
        self._base_wall_time = _original_now()
        self._base_sim_time = start_time

    def activate(self):
        offset = self._base_sim_time - self._base_wall_time
        _sim_clock_offset.set(offset)

    def deactivate(self):
        _sim_clock_offset.set(None)

    def advance_to(self, target_time: datetime) -> dict:
        """Jump simulated time forward to *target_time*.

        Returns a log entry dict describing the advance.
        """
        prev_sim = self.now()
        wall_now = _original_now()
        self._base_wall_time = wall_now
        self._base_sim_time = target_time
        offset = target_time - wall_now
        _sim_clock_offset.set(offset)

        entry = {
            "event": "clock_advance",
            "from_sim_time": prev_sim.isoformat(),
            "to_sim_time": target_time.isoformat(),
            "wall_time": wall_now.isoformat(),
            "offset_seconds": offset.total_seconds(),
        }
        logger.debug("Clock advance: %s -> %s", prev_sim, target_time)
        return entry

    def now(self) -> datetime:
        offset = _sim_clock_offset.get()
        if offset is None:
            return _original_now()
        return _original_now() + offset

    @property
    def elapsed_sim_time(self) -> timedelta:
        return self.now() - self.start_time


@contextmanager
def simulated_clock_context(start_time: datetime):
    """Context manager that installs and tears down a simulated clock."""
    clock = SimulatedClock(start_time)
    clock.activate()
    try:
        yield clock
    finally:
        clock.deactivate()

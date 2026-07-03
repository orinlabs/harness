"""Offset-based simulated clock for eval runs.

Thin wrapper around :mod:`harness.core.clock`, which owns the process-wide
offset between wall-clock time and simulated time. Between scheduled events
the runner jumps forward via ``advance_to()``; during agent execution
wall-clock ticks pass through naturally (a 45-second LLM call shows up as
45 simulated seconds).

Because the offset lives in the core clock, *every* production call site
that reads time through ``harness.core.clock`` (memory timestamps, the
summarizer's "now", tracer span times, ...) transparently sees simulated
time while a ``SimulatedClock`` is active -- no monkey-patching needed.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta

from harness.core import clock as core_clock

logger = logging.getLogger(__name__)


def _original_now() -> datetime:
    """Wall-clock now (UTC, timezone-aware)."""
    return core_clock.wall_now()


def _simulated_now() -> datetime:
    """Return wall-clock-plus-offset (wall-clock when no offset is set)."""
    return core_clock.now()


class SimulatedClock:
    """Manages simulated time via a global offset from wall-clock time."""

    def __init__(self, start_time: datetime):
        self.start_time = start_time
        self._base_wall_time = _original_now()
        self._base_sim_time = start_time
        self._prior_offset = core_clock.get_offset()

    def activate(self):
        self._prior_offset = core_clock.get_offset()
        core_clock.set_offset(self._base_sim_time - self._base_wall_time)

    def deactivate(self):
        # Restore whatever offset was active before (e.g. one installed by
        # `--sim-start-time`) instead of unconditionally reverting to wall.
        core_clock.set_offset(self._prior_offset)

    def advance_to(self, target_time: datetime) -> dict:
        """Jump simulated time forward to *target_time*.

        Returns a log entry dict describing the advance.
        """
        prev_sim = self.now()
        wall_now = _original_now()
        self._base_wall_time = wall_now
        self._base_sim_time = target_time
        offset = target_time - wall_now
        core_clock.set_offset(offset)

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
        return core_clock.now()

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

"""Process-wide simulated clock.

All production "what time is it" reads route through this module so a run
can be started at an arbitrary simulated moment. The clock stores a single
offset between wall-clock time and simulated time:

    simulated_now = wall_now + offset

Wall-clock keeps ticking underneath, so simulated time always equals the
configured start time plus real elapsed run time (a 45-second LLM call
shows up as 45 simulated seconds). When no offset is set, every helper
returns plain wall-clock time.

Two writers exist:

  * ``harness --sim-start-time <epoch>`` (or ``$HARNESS_SIM_START_TIME``)
    sets the offset once at process start via :func:`set_start_epoch`.
  * ``harness.evals.clock.SimulatedClock`` sets and re-sets the offset as
    the eval runner jumps between scheduled events.

The offset is a module-level global rather than a ``contextvars.ContextVar``
on purpose: worker threads (LLM timeout watchers, etc.) must see the same
simulated time as the main loop, and contextvars don't propagate into
already-running threads.

Durations (``time.monotonic()`` deltas) are intentionally *not* routed
through here -- elapsed wall time is elapsed wall time.
"""

from __future__ import annotations

import time as _time
from datetime import UTC, datetime, timedelta

_offset: timedelta = timedelta(0)


def wall_now() -> datetime:
    """True wall-clock now (UTC, timezone-aware). Ignores the offset."""
    return datetime.now(tz=UTC)


def now() -> datetime:
    """Simulated now (UTC, timezone-aware): wall-clock plus the offset."""
    return wall_now() + _offset


def now_iso() -> str:
    """Simulated now as an ISO-8601 string."""
    return now().isoformat()


def time_ns() -> int:
    """Simulated epoch nanoseconds (``time.time_ns()`` plus the offset)."""
    return _time.time_ns() + int(_offset.total_seconds() * 1_000_000_000)


def get_offset() -> timedelta:
    return _offset


def set_offset(offset: timedelta) -> None:
    global _offset
    _offset = offset


def set_start(start: datetime) -> None:
    """Anchor simulated time so that "now" equals *start* at this instant.

    From here on, simulated time = start + elapsed wall time.
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    set_offset(start - wall_now())


def set_start_epoch(epoch_seconds: float) -> None:
    """:func:`set_start` from epoch seconds (what the CLI flag accepts)."""
    set_start(datetime.fromtimestamp(epoch_seconds, tz=UTC))


def reset() -> None:
    """Clear the offset (back to wall-clock). Used by teardown paths/tests."""
    set_offset(timedelta(0))

"""Time-windowed memory deletion ("forget the last N minutes").

``storage.reset_agent_memory`` wipes an agent's entire memory DB.
``forget_since`` is the surgical variant: it deletes the raw ``messages``
logged at/after a cutoff and removes the tiered-summary buckets that cover
the affected window, so the summarizer regenerates them from the
(now-truncated) message log on its next run. Everything strictly before the
cutoff is preserved.

Summary buckets are keyed in the agent's local timezone (see
``summarizer.py``), so the cutoff is converted to that zone the *same* way
the summarizer converts message timestamps -- ``datetime.fromtimestamp``
yields naive wall-clock and ``force_timezone`` then labels it. Production
runs the summarizer with the default ``timezone_name="UTC"``.
"""

from __future__ import annotations

import logging
from datetime import datetime

from harness.core import clock, storage
from harness.memory.bucketing import floor_to_5_minutes, hour_start
from harness.memory.marks import force_timezone, week_start_sunday

logger = logging.getLogger(__name__)

_NS_PER_MINUTE = 60 * 1_000_000_000


def forget_since(cutoff_ns: int, *, timezone_name: str = "UTC") -> dict[str, int]:
    """Delete memory at/after ``cutoff_ns`` for the currently-loaded agent DB.

    Removes raw ``messages`` rows with ``ts_ns >= cutoff_ns`` and every
    tiered-summary bucket whose period overlaps ``[cutoff, now]``. The
    boundary 5-minute bucket (the one containing ``cutoff``) is deleted too;
    the summarizer rebuilds it from any surviving pre-cutoff messages on its
    next pass, so retained history is never lost.

    Returns a per-table count of deleted rows. ``storage.load`` must have
    been called first.
    """
    db = storage.db
    if db is None:
        raise RuntimeError("storage.load must be called before forget_since")

    cutoff_local = force_timezone(datetime.fromtimestamp(cutoff_ns / 1_000_000_000), timezone_name)

    counts: dict[str, int] = {}

    counts["messages"] = db.execute("DELETE FROM messages WHERE ts_ns >= ?", (cutoff_ns,)).rowcount

    # five_minute_summaries: (date, hour, minute) >= floored 5-minute boundary.
    five = floor_to_5_minutes(cutoff_local)
    five_date = five.date().isoformat()
    counts["five_minute_summaries"] = db.execute(
        "DELETE FROM five_minute_summaries "
        "WHERE date > ? "
        "   OR (date = ? AND hour > ?) "
        "   OR (date = ? AND hour = ? AND minute >= ?)",
        (five_date, five_date, five.hour, five_date, five.hour, five.minute),
    ).rowcount

    # hourly_summaries: (date, hour) >= cutoff hour.
    hr = hour_start(cutoff_local)
    hr_date = hr.date().isoformat()
    counts["hourly_summaries"] = db.execute(
        "DELETE FROM hourly_summaries WHERE date > ? OR (date = ? AND hour >= ?)",
        (hr_date, hr_date, hr.hour),
    ).rowcount

    # daily_summaries: date >= cutoff day.
    counts["daily_summaries"] = db.execute(
        "DELETE FROM daily_summaries WHERE date >= ?",
        (cutoff_local.date().isoformat(),),
    ).rowcount

    # weekly_summaries: week_start_date >= cutoff week (Sunday-based).
    wk = week_start_sunday(cutoff_local)
    counts["weekly_summaries"] = db.execute(
        "DELETE FROM weekly_summaries WHERE week_start_date >= ?",
        (wk.date().isoformat(),),
    ).rowcount

    # monthly_summaries: (year, month) >= cutoff month.
    counts["monthly_summaries"] = db.execute(
        "DELETE FROM monthly_summaries WHERE year > ? OR (year = ? AND month >= ?)",
        (cutoff_local.year, cutoff_local.year, cutoff_local.month),
    ).rowcount

    storage.flush()
    logger.info(
        "forget_since: cutoff_ns=%s local=%s deleted=%s",
        cutoff_ns,
        cutoff_local.isoformat(),
        counts,
    )
    return counts


def forget_recent_minutes(
    minutes: int,
    *,
    timezone_name: str = "UTC",
    now_ns: int | None = None,
) -> dict[str, int]:
    """Forget everything logged in the last ``minutes`` minutes.

    Computes the cutoff from the harness clock (the same clock that stamps
    ``ts_ns``) and delegates to :func:`forget_since`.
    """
    if minutes <= 0:
        raise ValueError(f"minutes must be positive, got {minutes}")
    base = now_ns if now_ns is not None else clock.time_ns()
    cutoff_ns = base - minutes * _NS_PER_MINUTE
    return forget_since(cutoff_ns, timezone_name=timezone_name)

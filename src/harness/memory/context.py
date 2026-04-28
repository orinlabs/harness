"""Memory context builder.

`fetch_data(current_time, min_resolution)` queries each summary tier over a
half-open mark-boundary range and returns a `MemoryData` bundle of typed rows.
`render(data)` formats those rows into the tier-labeled summary block that
gets appended to the user's system prompt.

`min_resolution` controls how deep we go:
  - None: include raw messages (finest)
  - FIVE_MINUTE: down to 5-minute summaries, no raw messages
  - HOURLY: down to hourly, etc.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta

from harness.core import storage
from harness.memory.marks import compute_windows, force_timezone
from harness.memory.rows import (
    DailySummary,
    FiveMinuteSummary,
    HourlySummary,
    MessageRow,
    MonthlySummary,
    WeeklySummary,
)
from harness.memory.types import PeriodType

logger = logging.getLogger(__name__)


RESOLUTION_LEVEL: dict[PeriodType, int] = {
    PeriodType.FIVE_MINUTE: 0,
    PeriodType.HOURLY: 1,
    PeriodType.DAILY: 2,
    PeriodType.WEEKLY: 3,
    PeriodType.MONTHLY: 4,
}


@dataclass
class MemoryData:
    """Raw data fetched from the database for building memory context.

    Callers may inspect or remove items before handing the remainder to the
    renderer (e.g. pop `messages` into the LLM messages array so the model
    sees the conversation directly).
    """

    monthly_summaries: list[MonthlySummary] = field(default_factory=list)
    weekly_summaries: list[WeeklySummary] = field(default_factory=list)
    daily_summaries: list[DailySummary] = field(default_factory=list)
    hourly_summaries: list[HourlySummary] = field(default_factory=list)
    five_minute_summaries: list[FiveMinuteSummary] = field(default_factory=list)
    messages: list[MessageRow] = field(default_factory=list)
    last_summarized_time: datetime | None = None


def _dt_to_ns(dt: datetime) -> int:
    """Convert a timezone-aware or naive datetime to nanoseconds since epoch.

    Naive datetimes are treated as UTC (consistent with `marks.py` conventions).
    """
    if dt.tzinfo is None:

        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1_000_000_000)


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


class MemoryContextBuilder:
    """Constructs a time-aware memory context using stratified summaries.

    Scoping to a single agent is implicit via the per-agent sqlite file
    opened by `storage.load(agent_id)`.
    """

    def __init__(self, timezone: str = "UTC", time_offset: int = 0):
        self.timezone = timezone
        self.time_offset = time_offset

    # ------------------------------------------------------------------
    # Step 1: Fetch all the data we need
    # ------------------------------------------------------------------

    def fetch_data(
        self,
        current_time: datetime,
        min_resolution: PeriodType | None = None,
    ) -> MemoryData:
        """Fetch summaries/messages strictly before *current_time* using mark boundaries."""
        assert storage.db is not None, "storage.load must be called first"
        db = storage.db

        local_now = force_timezone(current_time, self.timezone)
        windows = compute_windows(local_now, min_resolution)
        min_level = RESOLUTION_LEVEL.get(min_resolution, 0) if min_resolution else -1

        # -- Messages: (five_min_cursor, now] --
        messages: list[MessageRow] = []
        if min_resolution is None:
            assert windows.message_start is not None and windows.message_end is not None
            lower_bound = windows.message_start

            # Fall back to the start of the current hour if no 5-minute
            # summaries exist yet. Preserves conversation continuity
            # before summarization has produced any 5m rows (e.g. on
            # the first turn of a brand-new agent).
            has_5m = db.execute(
                "SELECT 1 FROM five_minute_summaries LIMIT 1"
            ).fetchone()
            if not has_5m:
                lower_bound = windows.message_end.replace(
                    minute=0, second=0, microsecond=0
                )

            rows = db.execute(
                """
                SELECT id, ts_ns, role, content_json
                FROM messages
                WHERE ts_ns > ? AND ts_ns <= ?
                ORDER BY ts_ns
                LIMIT 200
                """,
                (_dt_to_ns(lower_bound), _dt_to_ns(windows.message_end)),
            ).fetchall()
            messages = [
                MessageRow(
                    id=r["id"],
                    ts_ns=r["ts_ns"],
                    role=r["role"],
                    content=json.loads(r["content_json"]),
                )
                for r in rows
            ]

        # -- FiveMinuteSummary: [five_min_start, five_min_end) --
        five_minute_summaries: list[FiveMinuteSummary] = []
        if min_level <= RESOLUTION_LEVEL[PeriodType.FIVE_MINUTE]:
            assert windows.five_min_start is not None and windows.five_min_end is not None
            rows = _select_dhm_range(
                db,
                table="five_minute_summaries",
                start=windows.five_min_start,
                end=windows.five_min_end,
                columns="id, date, hour, minute, summary, message_count",
                order="date, hour, minute",
            )
            five_minute_summaries = [
                FiveMinuteSummary(
                    id=r["id"],
                    date=_parse_date(r["date"]),
                    hour=r["hour"],
                    minute=r["minute"],
                    summary=r["summary"],
                    message_count=r["message_count"],
                )
                for r in rows
            ]

        # -- HourlySummary: [hourly_start, hourly_end) --
        hourly_summaries: list[HourlySummary] = []
        if min_level <= RESOLUTION_LEVEL[PeriodType.HOURLY]:
            assert windows.hourly_start is not None and windows.hourly_end is not None
            rows = _select_dh_range(
                db,
                table="hourly_summaries",
                start=windows.hourly_start,
                end=windows.hourly_end,
                columns="id, date, hour, summary, message_count",
                order="date, hour",
            )
            hourly_summaries = [
                HourlySummary(
                    id=r["id"],
                    date=_parse_date(r["date"]),
                    hour=r["hour"],
                    summary=r["summary"],
                    message_count=r["message_count"],
                )
                for r in rows
            ]

        # -- DailySummary: [daily_start.date(), daily_end.date()) --
        daily_summaries: list[DailySummary] = []
        if min_level <= RESOLUTION_LEVEL[PeriodType.DAILY]:
            assert windows.daily_start is not None and windows.daily_end is not None
            start_d = windows.daily_start.date().isoformat()
            end_d = windows.daily_end.date().isoformat()
            rows = db.execute(
                """
                SELECT id, date, summary, message_count
                FROM daily_summaries
                WHERE date >= ? AND date < ?
                ORDER BY date
                """,
                (start_d, end_d),
            ).fetchall()
            daily_summaries = [
                DailySummary(
                    id=r["id"],
                    date=_parse_date(r["date"]),
                    summary=r["summary"],
                    message_count=r["message_count"],
                )
                for r in rows
            ]

        # -- WeeklySummary: [weekly_start.date(), weekly_end.date()) --
        weekly_summaries: list[WeeklySummary] = []
        if min_level <= RESOLUTION_LEVEL[PeriodType.WEEKLY]:
            assert windows.weekly_start is not None and windows.weekly_end is not None
            start_d = windows.weekly_start.date().isoformat()
            end_d = windows.weekly_end.date().isoformat()
            rows = db.execute(
                """
                SELECT id, week_start_date, summary, message_count
                FROM weekly_summaries
                WHERE week_start_date >= ? AND week_start_date < ?
                ORDER BY week_start_date
                """,
                (start_d, end_d),
            ).fetchall()
            weekly_summaries = [
                WeeklySummary(
                    id=r["id"],
                    week_start_date=_parse_date(r["week_start_date"]),
                    summary=r["summary"],
                    message_count=r["message_count"],
                )
                for r in rows
            ]

        # -- MonthlySummary: strictly before (boundary.year, boundary.month), last 24 --
        monthly_summaries: list[MonthlySummary] = []
        if min_level <= RESOLUTION_LEVEL[PeriodType.MONTHLY]:
            boundary = windows.monthly_end or local_now
            rows = db.execute(
                """
                SELECT id, year, month, summary, message_count
                FROM monthly_summaries
                WHERE year < ? OR (year = ? AND month < ?)
                ORDER BY year DESC, month DESC
                LIMIT 24
                """,
                (boundary.year, boundary.year, boundary.month),
            ).fetchall()
            # Reverse so the oldest month is first in the returned list.
            monthly_summaries = [
                MonthlySummary(
                    id=r["id"],
                    year=r["year"],
                    month=r["month"],
                    summary=r["summary"],
                    message_count=r["message_count"],
                )
                for r in reversed(rows)
            ]

        return MemoryData(
            monthly_summaries=monthly_summaries,
            weekly_summaries=weekly_summaries,
            daily_summaries=daily_summaries,
            hourly_summaries=hourly_summaries,
            five_minute_summaries=five_minute_summaries,
            messages=messages,
            last_summarized_time=windows.message_start or windows.five_min_end,
        )

    # ------------------------------------------------------------------
    # Step 2: Render summaries into text (no message replay — harness
    # returns those separately from build_llm_inputs).
    # ------------------------------------------------------------------

    def render(self, data: MemoryData, max_tokens: int = 50_000) -> str:
        parts: list[str] = []

        if data.monthly_summaries:
            parts.append("=== MONTHLY SUMMARIES ===")
            for m in data.monthly_summaries:
                parts.append(f"\n{m.year}-{m.month:02d}: {m.summary}")

        if data.weekly_summaries:
            parts.append("\n\n=== RECENT WEEKLY SUMMARIES ===")
            for w in data.weekly_summaries:
                parts.append(f"\nWeek of {w.week_start_date}: {w.summary}")

        if data.daily_summaries:
            parts.append("\n\n=== RECENT DAILY SUMMARIES ===")
            for d in data.daily_summaries:
                parts.append(f"\n{d.date}: {d.summary}")

        if data.hourly_summaries:
            parts.append("=== HOURLY SUMMARIES ===")
            for h in data.hourly_summaries:
                dt = datetime.combine(h.date, time(h.hour, 0, 0))
                dt = force_timezone(dt, self.timezone) + timedelta(
                    minutes=self.time_offset
                )
                parts.append(f"\n{dt.strftime('%H:%M:%S')}: {h.summary}")

        if data.five_minute_summaries:
            parts.append("\n\n=== 5-MINUTE SUMMARIES ===")
            for s in data.five_minute_summaries:
                start_dt = datetime.combine(s.date, time(s.hour, s.minute, 0))
                start_dt = force_timezone(start_dt, self.timezone)
                end_dt = start_dt + timedelta(minutes=5)
                if self.time_offset:
                    start_dt += timedelta(minutes=self.time_offset)
                    end_dt += timedelta(minutes=self.time_offset)
                parts.append(
                    f"\n{start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')}: {s.summary}"
                )

        return "\n".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# SQL helpers for (date, hour[, minute]) half-open range queries.
# ---------------------------------------------------------------------------


def _select_dh_range(
    db,
    *,
    table: str,
    start: datetime,
    end: datetime,
    columns: str,
    order: str,
) -> list:
    """SELECT rows in [start, end) on (date, hour) tuple ordering."""
    s_date, s_hour = start.date().isoformat(), start.hour
    e_date, e_hour = end.date().isoformat(), end.hour

    if (s_date, s_hour) >= (e_date, e_hour):
        return []

    sql = f"""
        SELECT {columns} FROM {table}
        WHERE (
            (date > ?) OR (date = ? AND hour >= ?)
        ) AND (
            (date < ?) OR (date = ? AND hour < ?)
        )
        ORDER BY {order}
    """
    return db.execute(
        sql,
        (s_date, s_date, s_hour, e_date, e_date, e_hour),
    ).fetchall()


def _select_dhm_range(
    db,
    *,
    table: str,
    start: datetime,
    end: datetime,
    columns: str,
    order: str,
) -> list:
    """SELECT rows in [start, end) on (date, hour, minute) tuple ordering."""
    s_date, s_hour, s_min = start.date().isoformat(), start.hour, start.minute
    e_date, e_hour, e_min = end.date().isoformat(), end.hour, end.minute

    if (s_date, s_hour, s_min) >= (e_date, e_hour, e_min):
        return []

    sql = f"""
        SELECT {columns} FROM {table}
        WHERE (
            (date > ?)
            OR (date = ? AND hour > ?)
            OR (date = ? AND hour = ? AND minute >= ?)
        ) AND (
            (date < ?)
            OR (date = ? AND hour < ?)
            OR (date = ? AND hour = ? AND minute < ?)
        )
        ORDER BY {order}
    """
    return db.execute(
        sql,
        (
            s_date,
            s_date,
            s_hour,
            s_date,
            s_hour,
            s_min,
            e_date,
            e_date,
            e_hour,
            e_date,
            e_hour,
            e_min,
        ),
    ).fetchall()

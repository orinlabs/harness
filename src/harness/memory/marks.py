from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from harness.memory.types import PeriodType


@dataclass(frozen=True)
class MemoryMarks:
    """Boundary marks for stratified memory export.

    All marks are timezone-aware if the input is timezone-aware.
    Intervals are intended to be half-open: [start, end).
    """

    now: datetime

    # Most recent completed 1-minute boundary at/before now.
    one_min_cursor: datetime

    # Most recent completed 5-minute boundary at/before now.
    five_min_cursor: datetime

    # Start of the 5-minute-summary window (hour_start(five_min_cursor) - 1h).
    five_min_window_start: datetime

    # Start of the hourly-summary window (day_start(five_min_window_start) - 1d).
    hourly_window_start: datetime

    # Start of the daily-summary window (week_start_sunday(hourly_window_start) - 1w).
    daily_window_start: datetime

    # Start of the weekly-summary window (month_start(daily_window_start) - 1 month).
    weekly_window_start: datetime

    # Start of the monthly-summary window (epoch sentinel).
    monthly_window_start: datetime


@dataclass(frozen=True)
class MemoryWindows:
    """Ranges derived from the cursor algorithm for a min_resolution.

    Summary ranges are half-open: [start, end).
    Messages are (start, end] to mean strictly after the last 5-minute cursor.
    """

    now: datetime
    min_resolution: PeriodType | None

    message_start: datetime | None
    message_end: datetime | None

    one_min_start: datetime | None
    one_min_end: datetime | None

    five_min_start: datetime | None
    five_min_end: datetime | None

    hourly_start: datetime | None
    hourly_end: datetime | None

    daily_start: datetime | None
    daily_end: datetime | None

    weekly_start: datetime | None
    weekly_end: datetime | None

    monthly_end: datetime | None  # month-start boundary; include months strictly before this


def force_timezone(dt: datetime, tz_name: str) -> datetime:
    """Attach `tz_name` to a naive datetime, or convert an aware one to that zone."""
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(tz_name)
    except Exception:
        tz = UTC
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def floor_to_1_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def floor_to_5_minutes(dt: datetime) -> datetime:
    dt = dt.replace(second=0, microsecond=0)
    return dt.replace(minute=(dt.minute // 5) * 5)


def hour_start(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def day_start(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def week_start_sunday(dt: datetime) -> datetime:
    """Return Sunday 00:00:00 of the week containing dt."""
    d0 = day_start(dt)
    # Python weekday(): Monday=0..Sunday=6; convert to Sunday=0..Saturday=6.
    sunday_based = (d0.weekday() + 1) % 7
    return d0 - timedelta(days=sunday_based)


def month_start(dt: datetime) -> datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def prev_month_start(dt: datetime) -> datetime:
    """Return the start of the month immediately before dt's month."""
    ms = month_start(dt)
    if ms.month == 1:
        return ms.replace(year=ms.year - 1, month=12)
    return ms.replace(month=ms.month - 1)


def epoch_start_like(dt: datetime) -> datetime:
    """Epoch sentinel preserving tzinfo when possible."""
    # If input is naive, keep the sentinel naive.
    if dt.tzinfo is None:
        return datetime(1970, 1, 1, 0, 0, 0)
    return datetime(1970, 1, 1, 0, 0, 0, tzinfo=dt.tzinfo)


def shift_timeline(dt: datetime, delta: timedelta) -> datetime:
    """Shift a datetime by a real-time delta, robust across DST transitions.

    For timezone-aware datetimes, shift in UTC and convert back to the original tz.
    For naive datetimes, this is the same as dt + delta.
    """
    if dt.tzinfo is None:
        return dt + delta
    return (dt.astimezone(UTC) + delta).astimezone(dt.tzinfo)


def day_start_prev(dt: datetime) -> datetime:
    """Return the previous calendar day at 00:00 in dt's tz."""
    d0 = day_start(dt)
    prev = d0.date() - timedelta(days=1)
    return datetime(prev.year, prev.month, prev.day, 0, 0, 0, tzinfo=d0.tzinfo)


def week_start_sunday_prev(dt: datetime) -> datetime:
    """Return the Sunday 00:00 of the previous week (calendar), in dt's tz."""
    ws = week_start_sunday(dt)
    prev = ws.date() - timedelta(days=7)
    return datetime(prev.year, prev.month, prev.day, 0, 0, 0, tzinfo=ws.tzinfo)


def compute_marks(now: datetime) -> MemoryMarks:
    """Compute stratified boundary marks from a single datetime.

    Algorithm (cursor-style):
    - five_min_cursor = floor_to_5_minutes(now)
    - five_min_window_start = hour_start(five_min_cursor) - 1 hour
    - hourly_window_start = day_start(five_min_window_start) - 1 day
    - daily_window_start = week_start_sunday(hourly_window_start) - 1 week
    - weekly_window_start = prev_month_start(daily_window_start)
    - monthly_window_start = epoch sentinel
    """
    one_min_cursor = floor_to_1_minute(now)
    five_min_cursor = floor_to_5_minutes(now)
    five_min_window_start = shift_timeline(hour_start(five_min_cursor), -timedelta(hours=1))
    hourly_window_start = day_start_prev(five_min_window_start)
    daily_window_start = week_start_sunday_prev(hourly_window_start)
    weekly_window_start = prev_month_start(daily_window_start)
    monthly_window_start = epoch_start_like(now)

    return MemoryMarks(
        now=now,
        one_min_cursor=one_min_cursor,
        five_min_cursor=five_min_cursor,
        five_min_window_start=five_min_window_start,
        hourly_window_start=hourly_window_start,
        daily_window_start=daily_window_start,
        weekly_window_start=weekly_window_start,
        monthly_window_start=monthly_window_start,
    )


def compute_windows(now: datetime, min_resolution: PeriodType | None) -> MemoryWindows:
    """Compute query windows for a given min_resolution using the cursor algorithm.

    This is the resolution-aware version of compute_marks(): we only advance the
    cursor through steps that correspond to layers we are actually including.
    """
    # Snap the cursor to the boundary of the requested resolution, then walk outward.
    cursor = now

    message_start = None
    message_end = None

    if min_resolution is None:
        message_end = cursor
        cursor = floor_to_1_minute(cursor)
        message_start = cursor
    elif min_resolution == PeriodType.ONE_MINUTE:
        cursor = floor_to_1_minute(cursor)
    elif min_resolution == PeriodType.FIVE_MINUTE:
        cursor = floor_to_5_minutes(cursor)
    elif min_resolution == PeriodType.HOURLY:
        cursor = hour_start(cursor)
    elif min_resolution == PeriodType.DAILY:
        cursor = day_start(cursor)
    elif min_resolution == PeriodType.WEEKLY:
        cursor = week_start_sunday(cursor)
    elif min_resolution == PeriodType.MONTHLY:
        cursor = month_start(cursor)

    one_min_start = None
    one_min_end = None
    if min_resolution in (None, PeriodType.ONE_MINUTE):
        one_min_end = cursor
        one_min_start = floor_to_5_minutes(one_min_end)
        cursor = one_min_start

    five_min_start = None
    five_min_end = None
    if min_resolution in (None, PeriodType.ONE_MINUTE, PeriodType.FIVE_MINUTE):
        five_min_end = cursor
        five_min_start = shift_timeline(hour_start(five_min_end), -timedelta(hours=1))
        cursor = five_min_start

    hourly_start = None
    hourly_end = None
    if min_resolution in (
        None,
        PeriodType.ONE_MINUTE,
        PeriodType.FIVE_MINUTE,
        PeriodType.HOURLY,
    ):
        hourly_end = cursor
        hourly_start = day_start_prev(hourly_end)
        cursor = hourly_start

    daily_start = None
    daily_end = None
    if min_resolution in (
        None,
        PeriodType.ONE_MINUTE,
        PeriodType.FIVE_MINUTE,
        PeriodType.HOURLY,
        PeriodType.DAILY,
    ):
        daily_end = cursor
        daily_start = week_start_sunday_prev(daily_end)
        cursor = daily_start

    weekly_start = None
    weekly_end = None
    if min_resolution in (
        None,
        PeriodType.ONE_MINUTE,
        PeriodType.FIVE_MINUTE,
        PeriodType.HOURLY,
        PeriodType.DAILY,
        PeriodType.WEEKLY,
    ):
        weekly_end = cursor
        weekly_start = prev_month_start(weekly_end)
        cursor = weekly_start

    monthly_end = cursor if min_resolution == PeriodType.MONTHLY else month_start(cursor)

    return MemoryWindows(
        now=now,
        min_resolution=min_resolution,
        message_start=message_start,
        message_end=message_end,
        one_min_start=one_min_start,
        one_min_end=one_min_end,
        five_min_start=five_min_start,
        five_min_end=five_min_end,
        hourly_start=hourly_start,
        hourly_end=hourly_end,
        daily_start=daily_start,
        daily_end=daily_end,
        weekly_start=weekly_start,
        weekly_end=weekly_end,
        monthly_end=monthly_end,
    )

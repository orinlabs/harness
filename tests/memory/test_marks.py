from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from unittest import TestCase as SimpleTestCase

from harness.memory.marks import compute_marks


class ComputeMarksTests(SimpleTestCase):
    def test_marks_mar4_0934(self):
        now = datetime(2026, 3, 4, 9, 34, 0, tzinfo=UTC)
        m = compute_marks(now)

        assert m.five_min_cursor == datetime(2026, 3, 4, 9, 30, 0, tzinfo=UTC)
        assert m.five_min_window_start == datetime(2026, 3, 4, 8, 0, 0, tzinfo=UTC)
        assert m.hourly_window_start == datetime(2026, 3, 3, 0, 0, 0, tzinfo=UTC)
        assert m.daily_window_start == datetime(2026, 2, 22, 0, 0, 0, tzinfo=UTC)
        assert m.weekly_window_start == datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert m.monthly_window_start == datetime(1970, 1, 1, 0, 0, 0, tzinfo=UTC)

    def test_marks_mar4_1033(self):
        now = datetime(2026, 3, 4, 10, 33, 0, tzinfo=UTC)
        m = compute_marks(now)

        assert m.five_min_cursor == datetime(2026, 3, 4, 10, 30, 0, tzinfo=UTC)
        assert m.five_min_window_start == datetime(2026, 3, 4, 9, 0, 0, tzinfo=UTC)
        assert m.hourly_window_start == datetime(2026, 3, 3, 0, 0, 0, tzinfo=UTC)
        assert m.daily_window_start == datetime(2026, 2, 22, 0, 0, 0, tzinfo=UTC)
        assert m.weekly_window_start == datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

    def test_marks_mar4_0034_crosses_to_previous_day(self):
        now = datetime(2026, 3, 4, 0, 34, 0, tzinfo=UTC)
        m = compute_marks(now)

        assert m.five_min_cursor == datetime(2026, 3, 4, 0, 30, 0, tzinfo=UTC)
        assert m.five_min_window_start == datetime(2026, 3, 3, 23, 0, 0, tzinfo=UTC)
        assert m.hourly_window_start == datetime(2026, 3, 2, 0, 0, 0, tzinfo=UTC)
        assert m.daily_window_start == datetime(2026, 2, 22, 0, 0, 0, tzinfo=UTC)
        assert m.weekly_window_start == datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

    def test_marks_mar1_0004_crosses_month_boundary(self):
        now = datetime(2026, 3, 1, 0, 4, 0, tzinfo=UTC)
        m = compute_marks(now)

        assert m.five_min_cursor == datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC)
        assert m.five_min_window_start == datetime(2026, 2, 28, 23, 0, 0, tzinfo=UTC)
        assert m.hourly_window_start == datetime(2026, 2, 27, 0, 0, 0, tzinfo=UTC)
        assert m.daily_window_start == datetime(2026, 2, 15, 0, 0, 0, tzinfo=UTC)
        assert m.weekly_window_start == datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

    def test_marks_jan1_1234_crosses_year_boundary(self):
        now = datetime(2026, 1, 1, 12, 34, 0, tzinfo=UTC)
        m = compute_marks(now)

        assert m.five_min_cursor == datetime(2026, 1, 1, 12, 30, 0, tzinfo=UTC)
        assert m.five_min_window_start == datetime(2026, 1, 1, 11, 0, 0, tzinfo=UTC)
        assert m.hourly_window_start == datetime(2025, 12, 31, 0, 0, 0, tzinfo=UTC)
        assert m.daily_window_start == datetime(2025, 12, 21, 0, 0, 0, tzinfo=UTC)
        assert m.weekly_window_start == datetime(2025, 11, 1, 0, 0, 0, tzinfo=UTC)
        assert m.monthly_window_start == datetime(1970, 1, 1, 0, 0, 0, tzinfo=UTC)

    def test_marks_preserve_non_utc_tzinfo(self):
        tz = timezone(timedelta(hours=-8))
        now = datetime(2026, 3, 4, 9, 34, 0, tzinfo=tz)
        m = compute_marks(now)

        assert m.now.tzinfo == tz
        assert m.five_min_cursor.tzinfo == tz
        assert m.five_min_window_start.tzinfo == tz
        assert m.hourly_window_start.tzinfo == tz
        assert m.daily_window_start.tzinfo == tz
        assert m.weekly_window_start.tzinfo == tz
        assert m.monthly_window_start.tzinfo == tz

        # Spot-check the wall-time values match the UTC version, but in this tz.
        assert m.five_min_cursor == datetime(2026, 3, 4, 9, 30, 0, tzinfo=tz)
        assert m.five_min_window_start == datetime(2026, 3, 4, 8, 0, 0, tzinfo=tz)
        assert m.hourly_window_start == datetime(2026, 3, 3, 0, 0, 0, tzinfo=tz)

    def test_marks_exactly_on_five_minute_boundary(self):
        now = datetime(2026, 3, 4, 10, 35, 0, tzinfo=UTC)
        m = compute_marks(now)
        assert m.five_min_cursor == datetime(2026, 3, 4, 10, 35, 0, tzinfo=UTC)
        assert m.five_min_window_start == datetime(2026, 3, 4, 9, 0, 0, tzinfo=UTC)

    def test_marks_exactly_on_hour_boundary(self):
        now = datetime(2026, 3, 4, 10, 0, 0, tzinfo=UTC)
        m = compute_marks(now)
        assert m.five_min_cursor == datetime(2026, 3, 4, 10, 0, 0, tzinfo=UTC)
        assert m.five_min_window_start == datetime(2026, 3, 4, 9, 0, 0, tzinfo=UTC)
        assert m.hourly_window_start == datetime(2026, 3, 3, 0, 0, 0, tzinfo=UTC)

    def test_marks_exactly_on_midnight(self):
        now = datetime(2026, 3, 4, 0, 0, 0, tzinfo=UTC)
        m = compute_marks(now)
        assert m.five_min_cursor == datetime(2026, 3, 4, 0, 0, 0, tzinfo=UTC)
        assert m.five_min_window_start == datetime(2026, 3, 3, 23, 0, 0, tzinfo=UTC)
        assert m.hourly_window_start == datetime(2026, 3, 2, 0, 0, 0, tzinfo=UTC)

    def test_marks_naive_datetime_stays_naive(self):
        now = datetime(2026, 3, 4, 9, 34, 0)
        m = compute_marks(now)

        assert m.now.tzinfo is None
        assert m.five_min_cursor.tzinfo is None
        assert m.five_min_window_start.tzinfo is None
        assert m.hourly_window_start.tzinfo is None
        assert m.daily_window_start.tzinfo is None
        assert m.weekly_window_start.tzinfo is None
        assert m.monthly_window_start.tzinfo is None

        assert m.monthly_window_start == datetime(1970, 1, 1, 0, 0, 0)

    def test_marks_dst_spring_forward_america_los_angeles(self):
        # America/Los_Angeles DST starts Mar 8, 2026 at 02:00 -> 03:00.
        # Pick a time just after the jump.
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/Los_Angeles")
        now = datetime(2026, 3, 8, 3, 4, 0, tzinfo=tz)
        m = compute_marks(now)

        # five_min_cursor floors within the post-jump hour (PDT, UTC-7).
        assert m.five_min_cursor == datetime(2026, 3, 8, 3, 0, 0, tzinfo=tz)
        assert m.five_min_cursor.utcoffset() == timedelta(hours=-7)

        # five_min_window_start is one real hour earlier than hour_start(03:00 PDT),
        # which lands at 01:00 PST (UTC-8), skipping the nonexistent 02:00.
        assert m.five_min_window_start == datetime(2026, 3, 8, 1, 0, 0, tzinfo=tz)
        assert m.five_min_window_start.utcoffset() == timedelta(hours=-8)

        # hourly_window_start uses the day start from five_min_window_start, then -1 day.
        assert m.hourly_window_start == datetime(2026, 3, 7, 0, 0, 0, tzinfo=tz)
        assert m.hourly_window_start.utcoffset() == timedelta(hours=-8)

    def test_marks_dst_fall_back_fold0_america_los_angeles(self):
        # America/Los_Angeles DST ends Nov 1, 2026 at 02:00 -> 01:00 (hour repeats).
        # fold=0 is the first occurrence (still PDT, UTC-7).
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/Los_Angeles")
        now = datetime(2026, 11, 1, 1, 34, 0, tzinfo=tz, fold=0)
        m = compute_marks(now)

        assert m.five_min_cursor == datetime(2026, 11, 1, 1, 30, 0, tzinfo=tz, fold=0)
        assert m.five_min_cursor.utcoffset() == timedelta(hours=-7)

        # hour_start(01:30 PDT) - 1 hour => 00:00 PDT
        assert m.five_min_window_start == datetime(2026, 11, 1, 0, 0, 0, tzinfo=tz)
        assert m.five_min_window_start.utcoffset() == timedelta(hours=-7)

    def test_marks_dst_fall_back_fold1_america_los_angeles(self):
        # fold=1 is the second occurrence (PST, UTC-8).
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/Los_Angeles")
        now = datetime(2026, 11, 1, 1, 34, 0, tzinfo=tz, fold=1)
        m = compute_marks(now)

        assert m.five_min_cursor == datetime(2026, 11, 1, 1, 30, 0, tzinfo=tz, fold=1)
        assert m.five_min_cursor.utcoffset() == timedelta(hours=-8)

        # One real hour earlier than 01:00 PST is 01:00 PDT (the earlier, first 1am).
        assert m.five_min_window_start == datetime(2026, 11, 1, 1, 0, 0, tzinfo=tz, fold=0)
        assert m.five_min_window_start.utcoffset() == timedelta(hours=-7)

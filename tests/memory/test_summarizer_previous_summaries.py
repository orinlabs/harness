"""Verify that `fetch_data` for a given PeriodType returns the previous
adjacent bucket in the same tier (so a new summary pass can see the bucket
immediately before the one it's about to write).
"""

from __future__ import annotations

from datetime import date, datetime

from harness.memory.types import PeriodType
from tests.memory.conftest import (
    insert_daily,
    insert_five_min,
    insert_hourly,
    insert_monthly,
    insert_weekly,
)


def test_five_minute_includes_previous_across_hour_boundary(builder, storage_env):
    insert_five_min(storage_env, date(2024, 1, 15), hour=10, minute=55, summary="prev-5m")

    # New summary starts at 11:00; should be able to see the 10:55 window.
    reference_time = datetime(2024, 1, 15, 11, 0, 0)
    data = builder.fetch_data(reference_time, PeriodType.FIVE_MINUTE)

    summaries = [s.summary for s in data.five_minute_summaries]
    assert "prev-5m" in summaries


def test_hourly_includes_previous_across_day_boundary(builder, storage_env):
    insert_hourly(storage_env, date(2024, 1, 14), hour=23, summary="prev-hour")

    reference_time = datetime(2024, 1, 15, 0, 0, 0)
    data = builder.fetch_data(reference_time, PeriodType.HOURLY)

    summaries = [s.summary for s in data.hourly_summaries]
    assert "prev-hour" in summaries


def test_daily_includes_previous_days(builder, storage_env):
    insert_daily(storage_env, date(2024, 1, 12), summary="d12")
    insert_daily(storage_env, date(2024, 1, 13), summary="d13")

    reference_time = datetime(2024, 1, 14, 0, 0, 0)
    data = builder.fetch_data(reference_time, PeriodType.DAILY)

    summaries = [s.summary for s in data.daily_summaries]
    assert "d12" in summaries
    assert "d13" in summaries


def test_weekly_includes_previous_weeks(builder, storage_env):
    insert_weekly(storage_env, date(2024, 1, 8), summary="w-jan8")

    reference_time = datetime(2024, 1, 15, 0, 0, 0)  # week starting Jan 15
    data = builder.fetch_data(reference_time, PeriodType.WEEKLY)

    summaries = [s.summary for s in data.weekly_summaries]
    assert "w-jan8" in summaries


def test_monthly_includes_previous_months(builder, storage_env):
    insert_monthly(storage_env, year=2023, month=12, summary="m-dec")

    reference_time = datetime(2024, 1, 1, 0, 0, 0)
    data = builder.fetch_data(reference_time, PeriodType.MONTHLY)

    summaries = [s.summary for s in data.monthly_summaries]
    assert "m-dec" in summaries

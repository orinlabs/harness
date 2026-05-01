"""Integration tests for MemoryContextBuilder.fetch_data mark-boundary queries.

The mark computation itself is tested in `test_marks.py`. These tests verify
that `fetch_data` uses those marks as DB query boundaries and respects
`min_resolution` gating.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from harness.memory.marks import compute_marks
from harness.memory.types import PeriodType
from tests.memory.conftest import (
    insert_five_min,
    insert_hourly,
    insert_message,
)


def test_messages_window_is_after_five_min_cursor(builder, storage_env):
    current_time = datetime(2024, 1, 10, 10, 12, 30, tzinfo=UTC)
    marks = compute_marks(current_time)

    # Anchor a 5-minute summary so fetch_data uses the cursor-based
    # message window. Without one, fetch_data falls back to start-of-hour
    # for conversation continuity (tested implicitly elsewhere).
    insert_five_min(
        storage_env,
        marks.five_min_cursor.date(),
        marks.five_min_cursor.hour,
        marks.five_min_cursor.minute,
        "anchor",
    )

    insert_message(storage_env, marks.five_min_cursor - timedelta(seconds=30), "before-cursor")
    insert_message(storage_env, marks.five_min_cursor + timedelta(seconds=15), "after-cursor")
    insert_message(storage_env, current_time + timedelta(minutes=1), "future")

    data = builder.fetch_data(current_time)
    msgs = [m.content["content"] for m in data.messages]

    assert "after-cursor" in msgs
    assert "before-cursor" not in msgs
    assert "future" not in msgs


def test_five_minute_summaries_use_marks_range(builder, storage_env):
    current_time = datetime(2024, 1, 10, 10, 33, 0, tzinfo=UTC)
    marks = compute_marks(current_time)

    # Inside [five_min_window_start, five_min_cursor)
    in_start = marks.five_min_window_start
    in_mid = marks.five_min_window_start + timedelta(minutes=30)
    in_end_minus = marks.five_min_cursor - timedelta(minutes=5)

    insert_five_min(storage_env, in_start.date(), in_start.hour, in_start.minute, "in-start")
    insert_five_min(storage_env, in_mid.date(), in_mid.hour, in_mid.minute, "in-mid")
    insert_five_min(
        storage_env, in_end_minus.date(), in_end_minus.hour, in_end_minus.minute, "in-end-minus"
    )

    # Outside the range
    out_before = marks.five_min_window_start - timedelta(minutes=5)
    out_at_end = marks.five_min_cursor  # end-exclusive
    insert_five_min(
        storage_env, out_before.date(), out_before.hour, out_before.minute, "out-before"
    )
    insert_five_min(
        storage_env, out_at_end.date(), out_at_end.hour, out_at_end.minute, "out-at-end"
    )

    data = builder.fetch_data(current_time)
    summaries = [s.summary for s in data.five_minute_summaries]

    assert "in-start" in summaries
    assert "in-mid" in summaries
    assert "in-end-minus" in summaries
    assert "out-before" not in summaries
    assert "out-at-end" not in summaries


def test_hourly_summaries_use_marks_range(builder, storage_env):
    current_time = datetime(2024, 1, 10, 10, 33, 0, tzinfo=UTC)
    marks = compute_marks(current_time)

    # Inside [hourly_window_start, five_min_window_start)
    in_start = marks.hourly_window_start
    in_mid = marks.hourly_window_start + timedelta(hours=12)
    in_end_minus = marks.five_min_window_start - timedelta(hours=1)

    insert_hourly(storage_env, in_start.date(), in_start.hour, "in-start")
    insert_hourly(storage_env, in_mid.date(), in_mid.hour, "in-mid")
    insert_hourly(storage_env, in_end_minus.date(), in_end_minus.hour, "in-end-minus")

    # Outside the range
    out_before = marks.hourly_window_start - timedelta(hours=1)
    out_at_end = marks.five_min_window_start  # end-exclusive
    insert_hourly(storage_env, out_before.date(), out_before.hour, "out-before")
    insert_hourly(storage_env, out_at_end.date(), out_at_end.hour, "out-at-end")

    data = builder.fetch_data(current_time)
    summaries = [s.summary for s in data.hourly_summaries]

    assert "in-start" in summaries
    assert "in-mid" in summaries
    assert "in-end-minus" in summaries
    assert "out-before" not in summaries
    assert "out-at-end" not in summaries


def test_min_resolution_five_minute_excludes_raw_messages(builder, storage_env):
    current_time = datetime(2024, 1, 10, 10, 12, 0, tzinfo=UTC)
    marks = compute_marks(current_time)
    insert_message(storage_env, marks.five_min_cursor + timedelta(minutes=1), "msg")

    data = builder.fetch_data(current_time, min_resolution=PeriodType.FIVE_MINUTE)
    assert len(data.messages) == 0

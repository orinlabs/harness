"""Tests for time-windowed memory deletion (`forget_since` + CLI)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pytest

from harness.core import storage as storage_module
from harness.memory import forget_recent_minutes, forget_since
from harness.memory.bucketing import floor_to_5_minutes, hour_start
from harness.memory.marks import force_timezone, week_start_sunday

_MIN_NS = 60 * 1_000_000_000


@pytest.fixture
def storage_env(tmp_path, monkeypatch):
    """Point storage at a tmp dir using the real production migrations.

    Deliberately does NOT reload the storage module: ``forget.py`` binds
    ``storage`` at import time, so a reload would leave it pointing at a
    stale module object whose ``db`` the test never sets.
    """
    monkeypatch.setattr(storage_module, "_STORAGE_ROOT", tmp_path)
    storage_module.close()
    yield storage_module
    storage_module.close()


def _local(ts_ns: int, tz: str = "UTC") -> datetime:
    """Reproduce the summarizer's ts_ns -> local-bucket conversion."""
    return force_timezone(datetime.fromtimestamp(ts_ns / 1_000_000_000), tz)


def _insert_message(conn, msg_id: str, ts_ns: int) -> None:
    conn.execute(
        "INSERT INTO messages (id, ts_ns, role, content_json) VALUES (?, ?, ?, ?)",
        (msg_id, ts_ns, "user", "{}"),
    )


def _insert_5m(conn, key, *, summary: str) -> None:
    d, h, m = key
    conn.execute(
        "INSERT INTO five_minute_summaries "
        "(id, date, hour, minute, summary, message_count, created_at_ns) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (f"5m-{d}-{h}-{m}", d, h, m, summary, 1, 1),
    )


def test_forget_since_keeps_older_drops_newer_messages(storage_env):
    storage = storage_env
    conn = storage.load("agent-forget-msgs")
    now_ns = time.time_ns()
    _insert_message(conn, "old", now_ns - 90 * _MIN_NS)
    _insert_message(conn, "recent", now_ns - 5 * _MIN_NS)
    storage.flush()

    counts = forget_since(now_ns - 30 * _MIN_NS)

    assert counts["messages"] == 1
    remaining = [r["id"] for r in conn.execute("SELECT id FROM messages")]
    assert remaining == ["old"]


def test_forget_since_reconciles_summary_tiers(storage_env):
    storage = storage_env
    conn = storage.load("agent-forget-sum")
    now_ns = time.time_ns()
    cutoff_ns = now_ns - 30 * _MIN_NS
    cutoff_local = _local(cutoff_ns)

    # 5-minute tier: the bucket containing the cutoff is dropped; one two
    # hours earlier survives.
    recent5 = floor_to_5_minutes(cutoff_local)
    old5 = floor_to_5_minutes(cutoff_local - timedelta(hours=2))
    _insert_5m(conn, (recent5.date().isoformat(), recent5.hour, recent5.minute), summary="recent")
    _insert_5m(conn, (old5.date().isoformat(), old5.hour, old5.minute), summary="old")

    # Hourly tier.
    recent_hr = hour_start(cutoff_local)
    old_hr = hour_start(cutoff_local - timedelta(hours=2))
    for hr, label in ((recent_hr, "recent"), (old_hr, "old")):
        conn.execute(
            "INSERT INTO hourly_summaries "
            "(id, date, hour, summary, message_count, created_at_ns) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (f"h-{label}", hr.date().isoformat(), hr.hour, label, 1, 1),
        )

    # Daily tier.
    for day, label in ((cutoff_local, "recent"), (cutoff_local - timedelta(days=2), "old")):
        conn.execute(
            "INSERT INTO daily_summaries (id, date, summary, message_count, created_at_ns) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"d-{label}", day.date().isoformat(), label, 1, 1),
        )

    # Weekly tier.
    recent_wk = week_start_sunday(cutoff_local)
    old_wk = week_start_sunday(cutoff_local - timedelta(weeks=2))
    for wk, label in ((recent_wk, "recent"), (old_wk, "old")):
        conn.execute(
            "INSERT INTO weekly_summaries "
            "(id, week_start_date, summary, message_count, created_at_ns) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"w-{label}", wk.date().isoformat(), label, 1, 1),
        )

    # Monthly tier.
    prev_month = cutoff_local.replace(day=1) - timedelta(days=1)
    for y, mo, label in (
        (cutoff_local.year, cutoff_local.month, "recent"),
        (prev_month.year, prev_month.month, "old"),
    ):
        conn.execute(
            "INSERT INTO monthly_summaries "
            "(id, year, month, summary, message_count, created_at_ns) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (f"m-{label}", y, mo, label, 1, 1),
        )
    storage.flush()

    forget_since(cutoff_ns)

    def _labels(table):
        return sorted(r["summary"] for r in conn.execute(f"SELECT summary FROM {table}"))

    assert _labels("five_minute_summaries") == ["old"]
    assert _labels("hourly_summaries") == ["old"]
    assert _labels("daily_summaries") == ["old"]
    assert _labels("weekly_summaries") == ["old"]
    assert _labels("monthly_summaries") == ["old"]


def test_forget_recent_minutes_uses_supplied_now(storage_env):
    storage = storage_env
    conn = storage.load("agent-forget-now")
    anchor_ns = time.time_ns()
    _insert_message(conn, "old", anchor_ns - 50 * _MIN_NS)
    _insert_message(conn, "recent", anchor_ns - 3 * _MIN_NS)
    storage.flush()

    counts = forget_recent_minutes(10, now_ns=anchor_ns)

    assert counts["messages"] == 1
    assert [r["id"] for r in conn.execute("SELECT id FROM messages")] == ["old"]


def test_forget_recent_minutes_rejects_nonpositive(storage_env):
    storage_env.load("agent-forget-bad")
    with pytest.raises(ValueError):
        forget_recent_minutes(0)


def test_forget_since_on_empty_db_is_noop(storage_env):
    storage = storage_env
    storage.load("agent-forget-empty")
    counts = forget_since(time.time_ns() - 30 * _MIN_NS)
    assert counts["messages"] == 0
    assert counts["five_minute_summaries"] == 0


def test_forget_memory_cli(tmp_path, monkeypatch, capsys):
    storage_root = tmp_path / "storage"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(storage_module, "_STORAGE_ROOT", storage_root)
    storage_module.close()

    conn = storage_module.load("agent-cli-forget")
    now_ns = time.time_ns()
    _insert_message(conn, "old", now_ns - 90 * _MIN_NS)
    _insert_message(conn, "recent", now_ns - 5 * _MIN_NS)
    storage_module.flush()

    from harness.cli import main

    rc = main(["forget-memory", "agent-cli-forget", "--minutes", "30", "--log-level", "ERROR"])
    assert rc == 0

    captured = capsys.readouterr()
    assert "id=agent-cli-forget" in captured.err
    assert "minutes=30" in captured.err

    conn = storage_module.load("agent-cli-forget")
    remaining = [r["id"] for r in conn.execute("SELECT id FROM messages")]
    assert remaining == ["old"]
    storage_module.close()

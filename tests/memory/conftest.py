"""Shared fixtures + seed helpers for the tiered memory tests.

`storage_env` opens a fresh sqlite file and applies all migrations. `builder`
returns a `MemoryContextBuilder` bound to that file. The `insert_*` helpers
write rows directly via SQL so tests can seed specific bucket keys without
going through the summarizer.
"""
from __future__ import annotations

import importlib
import json
import time
import uuid
from datetime import UTC, datetime
from datetime import date as _date
from pathlib import Path

import pytest


@pytest.fixture
def storage_env(tmp_path, monkeypatch):
    """Fresh sqlite + applied migrations, scoped to 'agent-test'."""
    monkeypatch.setenv("HARNESS_STORAGE_ROOT", str(tmp_path))
    mig_dir = Path(__file__).parent.parent.parent / "src/harness/memory/migrations"
    monkeypatch.setenv("HARNESS_MIGRATIONS_DIR", str(mig_dir))

    from harness.core import storage as storage_module

    importlib.reload(storage_module)
    storage_module.load("agent-test")
    try:
        yield storage_module
    finally:
        storage_module.close()


@pytest.fixture
def builder(storage_env):
    from harness.memory.context import MemoryContextBuilder

    return MemoryContextBuilder(timezone="UTC")


# ---------------------------------------------------------------------------
# Direct SQL seed helpers.
# ---------------------------------------------------------------------------


def _now_ns() -> int:
    return time.time_ns()


def insert_five_min(storage, d: _date, hour: int, minute: int, summary: str = "5min"):
    storage.db.execute(
        "INSERT INTO five_minute_summaries "
        "(id, date, hour, minute, summary, message_count, created_at_ns) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), d.isoformat(), hour, minute, summary, 1, _now_ns()),
    )


def insert_hourly(storage, d: _date, hour: int, summary: str = "hourly"):
    storage.db.execute(
        "INSERT INTO hourly_summaries "
        "(id, date, hour, summary, message_count, created_at_ns) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), d.isoformat(), hour, summary, 1, _now_ns()),
    )


def insert_daily(storage, d: _date, summary: str = "daily"):
    storage.db.execute(
        "INSERT INTO daily_summaries "
        "(id, date, summary, message_count, created_at_ns) "
        "VALUES (?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), d.isoformat(), summary, 1, _now_ns()),
    )


def insert_weekly(storage, week_start: _date, summary: str = "weekly"):
    storage.db.execute(
        "INSERT INTO weekly_summaries "
        "(id, week_start_date, summary, message_count, created_at_ns) "
        "VALUES (?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), week_start.isoformat(), summary, 1, _now_ns()),
    )


def insert_monthly(storage, year: int, month: int, summary: str = "monthly"):
    storage.db.execute(
        "INSERT INTO monthly_summaries "
        "(id, year, month, summary, message_count, created_at_ns) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), year, month, summary, 1, _now_ns()),
    )


def insert_message(storage, ts: datetime, content: str = "msg"):
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    ts_ns = int(ts.timestamp() * 1_000_000_000)
    msg = {"role": "user", "content": content}
    storage.db.execute(
        "INSERT INTO messages (id, ts_ns, role, content_json) VALUES (?, ?, ?, ?)",
        (str(uuid.uuid4()), ts_ns, "user", json.dumps(msg)),
    )

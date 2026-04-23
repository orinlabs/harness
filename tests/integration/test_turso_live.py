"""Live Turso integration test. Opt-in only.

This test provisions a real Turso database via the Platform API, applies
migrations, round-trips a row, and deletes the database on teardown. It's
excluded from the default `pytest` run because it hits the public internet
and burns a real DB slot.

Enable with:
    RUN_LIVE_TURSO=1 uv run pytest tests/integration/test_turso_live.py -o "addopts="

Required env (populated by `.env` at repo root):
    HARNESS_TURSO_ORG
    HARNESS_TURSO_PLATFORM_TOKEN
    HARNESS_DATABASE_TOKEN  (group-scoped)
    HARNESS_TURSO_GROUP     (optional, defaults to 'default')
"""
from __future__ import annotations

import importlib
import os
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv

# `tests/conftest.py` strips Turso env vars from all tests for safety; here we
# explicitly opt back in by reloading `.env` on top.
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)


LIVE = os.environ.get("RUN_LIVE_TURSO") == "1"


pytestmark = pytest.mark.skipif(
    not LIVE, reason="RUN_LIVE_TURSO=1 required; hits real Turso"
)


@pytest.fixture
def live_agent_id():
    """A unique agent id per test run; DB gets auto-provisioned and deleted."""
    missing = [
        v
        for v in (
            "HARNESS_TURSO_ORG",
            "HARNESS_TURSO_PLATFORM_TOKEN",
            "HARNESS_DATABASE_TOKEN",
        )
        if not os.environ.get(v)
    ]
    if missing:
        pytest.fail(
            f"Live Turso test needs: {', '.join(missing)}. Populate .env and rerun."
        )

    from harness.core import storage

    importlib.reload(storage)
    agent_id = f"harness-ci-{uuid.uuid4().hex[:12]}"
    try:
        yield agent_id
    finally:
        try:
            storage.close()
        except Exception:
            pass
        storage.delete_agent_db(agent_id)


def test_live_turso_roundtrip(live_agent_id):
    """Provision DB, apply migrations, insert+read, teardown."""
    from harness.core import storage

    storage.load(live_agent_id)

    migrations = [r["name"] for r in storage.db.execute(
        "SELECT name FROM applied_migrations ORDER BY name"
    )]
    assert "0001_initial" in migrations

    storage.db.execute(
        "INSERT OR REPLACE INTO one_minute_summaries "
        "(id, date, hour, minute, summary, message_count, created_at_ns) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("s1", "2024-01-15", 10, 30, "live turso round-trip", 1, 1),
    )
    storage.flush()

    row = storage.db.execute(
        "SELECT summary, message_count FROM one_minute_summaries WHERE id = ?",
        ("s1",),
    ).fetchone()
    assert row["summary"] == "live turso round-trip"
    assert row["message_count"] == 1

    storage.close()


def test_live_turso_reopen_survives(live_agent_id):
    """Close + reopen hits the same DB; migrations don't re-apply."""
    from harness.core import storage

    storage.load(live_agent_id)
    storage.db.execute(
        "INSERT OR REPLACE INTO daily_summaries "
        "(id, date, summary, message_count, created_at_ns) "
        "VALUES (?, ?, ?, ?, ?)",
        ("d1", "2024-01-15", "persisted", 3, 1),
    )
    storage.flush()
    storage.close()

    storage.load(live_agent_id)
    row = storage.db.execute(
        "SELECT summary FROM daily_summaries WHERE id = ?", ("d1",)
    ).fetchone()
    assert row["summary"] == "persisted"

    applied = list(storage.db.execute("SELECT name FROM applied_migrations"))
    assert len(applied) == 1, "migrations should not re-run on reopen"

    storage.close()

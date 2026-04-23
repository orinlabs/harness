"""Smoke test the libsql/Turso code path without needing real Turso creds.

Uses libsql-experimental's in-process `:memory:` mode to exercise the
`_LibsqlWrapper` + `_LibsqlCursor` + `_LibsqlRow` adapter layer against a real
libsql connection. Ensures memory code's dict-style row access works with the
remote driver the same as it does with sqlite3.Row.
"""
from __future__ import annotations

import importlib

import libsql_experimental as libsql
import pytest


@pytest.fixture
def libsql_storage(tmp_path, monkeypatch):
    """Intercept `libsql.connect` to return an in-memory db, and point storage
    at a dummy HARNESS_DATABASE_URL so the remote branch is taken."""
    from pathlib import Path

    monkeypatch.setenv("HARNESS_DATABASE_URL", "libsql://unused-for-this-test")
    monkeypatch.delenv("HARNESS_DATABASE_TOKEN", raising=False)
    monkeypatch.delenv("HARNESS_TURSO_ORG", raising=False)
    mig_dir = Path(__file__).parent.parent / "src/harness/memory/migrations"
    monkeypatch.setenv("HARNESS_MIGRATIONS_DIR", str(mig_dir))

    real_connect = libsql.connect

    def fake_connect(*args, **kwargs):
        return real_connect(":memory:")

    monkeypatch.setattr(libsql, "connect", fake_connect)

    from harness.core import storage as storage_module

    importlib.reload(storage_module)
    yield storage_module
    storage_module.close()


def test_libsql_wrapper_load_applies_migrations(libsql_storage):
    libsql_storage.load("agent-libsql-1")

    rows = list(
        libsql_storage.db.execute("SELECT name FROM applied_migrations ORDER BY name")
    )
    assert [r["name"] for r in rows] == ["0001_initial"]


def test_libsql_wrapper_row_dict_access(libsql_storage):
    """Memory code relies on row['col']; verify the wrapper supplies it."""
    libsql_storage.load("agent-libsql-2")

    libsql_storage.db.execute(
        "INSERT INTO one_minute_summaries "
        "(id, date, hour, minute, summary, message_count, created_at_ns) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("x1", "2024-01-15", 10, 30, "hello", 2, 1),
    )

    row = libsql_storage.db.execute(
        "SELECT date, hour, minute, summary FROM one_minute_summaries WHERE id = ?",
        ("x1",),
    ).fetchone()

    assert row["date"] == "2024-01-15"
    assert row["hour"] == 10
    assert row["minute"] == 30
    assert row["summary"] == "hello"
    assert row[0] == "2024-01-15"  # int index still works
    assert dict(zip(row.keys(), row)) == {
        "date": "2024-01-15",
        "hour": 10,
        "minute": 30,
        "summary": "hello",
    }


def test_libsql_wrapper_memory_service_round_trip(libsql_storage):
    """End-to-end: log messages + build_llm_inputs works over the libsql adapter."""
    libsql_storage.load("agent-libsql-3")

    # Skip summarisation (requires LLM creds) by inserting messages directly.
    libsql_storage.db.execute(
        "INSERT INTO messages (id, ts_ns, role, content_json) VALUES (?, ?, ?, ?)",
        ("m1", 1_700_000_000_000_000_000, "user", '{"role": "user", "content": "hi"}'),
    )

    rows = libsql_storage.db.execute(
        "SELECT id, role, content_json FROM messages ORDER BY ts_ns"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["role"] == "user"
    assert rows[0]["content_json"] == '{"role": "user", "content": "hi"}'

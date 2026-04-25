"""Chunk 2 verification: SQLite storage + migrations, real filesystem only."""
from __future__ import annotations

import time
from pathlib import Path

import pytest


@pytest.fixture
def storage_env(tmp_path, monkeypatch):
    """Point storage at a tmp dir and reload the module so env vars take effect."""
    monkeypatch.setenv("HARNESS_STORAGE_ROOT", str(tmp_path))

    import importlib

    from harness.core import storage as storage_module

    importlib.reload(storage_module)
    yield storage_module
    storage_module.close()


@pytest.fixture
def custom_migrations(tmp_path, monkeypatch):
    """A writable migrations directory that tests can drop files into."""
    mig_dir = tmp_path / "migrations"
    mig_dir.mkdir()

    initial = Path(__file__).parent.parent / "src/harness/memory/migrations/0001_initial.sql"
    (mig_dir / "0001_initial.sql").write_text(initial.read_text())

    monkeypatch.setenv("HARNESS_MIGRATIONS_DIR", str(mig_dir))
    return mig_dir


def test_fresh_db_applies_initial_migration(storage_env, custom_migrations):
    storage = storage_env
    conn = storage.load("agent-1")

    rows = list(conn.execute("SELECT name FROM applied_migrations ORDER BY name"))
    assert [r["name"] for r in rows] == ["0001_initial"]

    tables = {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "messages" in tables
    assert "one_minute_summaries" in tables
    assert "monthly_summaries" in tables


def test_messages_survive_reopen(storage_env, custom_migrations):
    storage = storage_env
    conn = storage.load("agent-1")
    now = time.time_ns()
    for i in range(100):
        conn.execute(
            "INSERT INTO messages (id, ts_ns, role, content_json) VALUES (?, ?, ?, ?)",
            (f"m{i}", now + i, "user", f'{{"i": {i}}}'),
        )
    storage.flush()
    storage.close()

    conn2 = storage.load("agent-1")
    count = conn2.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"]
    assert count == 100

    applied = list(conn2.execute("SELECT name FROM applied_migrations"))
    assert len(applied) == 1, "migrations must not re-apply on reopen"


def test_new_migration_applied_on_next_load(storage_env, custom_migrations):
    storage = storage_env
    storage.load("agent-1")
    storage.close()

    (custom_migrations / "0002_add_tags.sql").write_text(
        "CREATE TABLE IF NOT EXISTS tags (id TEXT PRIMARY KEY, label TEXT NOT NULL);"
    )

    conn = storage.load("agent-1")
    applied = [
        r["name"]
        for r in conn.execute("SELECT name FROM applied_migrations ORDER BY name")
    ]
    assert applied == ["0001_initial", "0002_add_tags"]

    conn.execute("INSERT INTO tags (id, label) VALUES (?, ?)", ("t1", "hello"))
    got = conn.execute("SELECT label FROM tags WHERE id='t1'").fetchone()["label"]
    assert got == "hello"


def test_idempotent_second_load(storage_env, custom_migrations):
    storage = storage_env
    conn1 = storage.load("agent-1")
    first = conn1.execute(
        "SELECT applied_at_ns FROM applied_migrations WHERE name='0001_initial'"
    ).fetchone()["applied_at_ns"]
    storage.close()

    time.sleep(0.01)
    conn2 = storage.load("agent-1")
    second = conn2.execute(
        "SELECT applied_at_ns FROM applied_migrations WHERE name='0001_initial'"
    ).fetchone()["applied_at_ns"]

    assert first == second, "applied_at_ns must not change on re-load"


def test_fresh_load_time_under_250ms(storage_env, custom_migrations):
    storage = storage_env
    t = time.perf_counter()
    storage.load("agent-fresh")
    elapsed = time.perf_counter() - t
    assert elapsed < 0.25, f"fresh load took {elapsed * 1000:.1f}ms, budget 250ms"


def test_populated_load_time_under_250ms(storage_env, custom_migrations):
    """Load against a ~5MB sqlite file and assert <250ms."""
    storage = storage_env
    conn = storage.load("agent-big")

    blob = "x" * 500
    now = time.time_ns()
    for i in range(10_000):
        conn.execute(
            "INSERT INTO messages (id, ts_ns, role, content_json) VALUES (?, ?, ?, ?)",
            (f"m{i}", now + i, "user", f'{{"b": "{blob}"}}'),
        )
    storage.flush()
    storage.close()

    t = time.perf_counter()
    storage.load("agent-big")
    elapsed = time.perf_counter() - t
    assert elapsed < 0.25, f"populated load took {elapsed * 1000:.1f}ms, budget 250ms"


def test_per_agent_isolation(storage_env, custom_migrations):
    storage = storage_env

    conn_a = storage.load("agent-A")
    conn_a.execute(
        "INSERT INTO messages (id, ts_ns, role, content_json) VALUES (?, ?, ?, ?)",
        ("a1", 1, "user", "{}"),
    )
    storage.flush()
    storage.close()

    conn_b = storage.load("agent-B")
    count = conn_b.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"]
    assert count == 0, "agent-B must not see agent-A's messages"


def test_delete_local_agent_db_removes_sqlite_and_sidecars(storage_env, custom_migrations):
    storage = storage_env
    conn = storage.load("agent-delete")
    conn.execute(
        "INSERT INTO messages (id, ts_ns, role, content_json) VALUES (?, ?, ?, ?)",
        ("m1", 1, "user", "{}"),
    )
    storage.flush()

    db_path = storage._db_path("agent-delete")
    sidecars = [Path(f"{db_path}-wal"), Path(f"{db_path}-shm")]
    for path in sidecars:
        path.touch(exist_ok=True)

    assert db_path.exists()
    assert storage.delete_local_agent_db("agent-delete") is True

    assert storage.db is None
    assert not db_path.exists()
    assert all(not path.exists() for path in sidecars)


def test_delete_agent_db_requires_turso_env_when_requested(storage_env, monkeypatch):
    storage = storage_env
    monkeypatch.delenv("HARNESS_TURSO_ORG", raising=False)
    monkeypatch.delenv("HARNESS_TURSO_PLATFORM_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="HARNESS_TURSO_ORG"):
        storage.delete_agent_db("agent-delete", require_config=True)


def test_reset_agent_memory_discards_existing_local_memory(storage_env, custom_migrations):
    storage = storage_env
    conn = storage.load("agent-reset")
    conn.execute(
        "INSERT INTO messages (id, ts_ns, role, content_json) VALUES (?, ?, ?, ?)",
        ("m1", 1, "user", "{}"),
    )
    conn.execute(
        "INSERT INTO daily_summaries "
        "(id, date, summary, message_count, created_at_ns) "
        "VALUES (?, ?, ?, ?, ?)",
        ("d1", "2026-04-24", "remembered", 1, 1),
    )
    storage.flush()

    result = storage.reset_agent_memory("agent-reset")
    assert result == {"local": True, "remote": False}

    conn = storage.load("agent-reset")
    assert conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"] == 0
    assert conn.execute("SELECT COUNT(*) AS c FROM daily_summaries").fetchone()["c"] == 0
    applied = [r["name"] for r in conn.execute("SELECT name FROM applied_migrations")]
    assert applied == ["0001_initial"]

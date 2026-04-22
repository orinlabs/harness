"""SQLite-backed storage for harness memory.

Lifecycle:
  - `load(agent_id)` at the top of Harness.run() — opens the per-agent sqlite file,
    creates the applied_migrations table if missing, and applies any pending
    migrations in order.
  - `flush()` before process exit — no-op for the local-file MVP; future remote
    backends (snapshot-to-S3, Turso sync, etc.) hook here.

Module-level `db` is the open sqlite3 connection. Memory code reads/writes directly.

Swap point: replacing this file is how we move to Turso, Postgres, or snapshot-on-exit
without touching memory code.
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

_DEFAULT_STORAGE_ROOT = "/tmp/harness"
_MIGRATIONS_DIR = Path(__file__).parent.parent / "memory" / "migrations"

db: sqlite3.Connection | None = None
_loaded_agent_id: str | None = None


def _storage_root() -> Path:
    root = os.environ.get("HARNESS_STORAGE_ROOT", _DEFAULT_STORAGE_ROOT)
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _migrations_dir() -> Path:
    override = os.environ.get("HARNESS_MIGRATIONS_DIR")
    return Path(override) if override else _MIGRATIONS_DIR


def _db_path(agent_id: str) -> Path:
    safe = agent_id.replace("/", "_")
    return _storage_root() / f"{safe}.sqlite"


def load(agent_id: str) -> sqlite3.Connection:
    """Open the per-agent sqlite file and apply any pending migrations."""
    global db, _loaded_agent_id

    path = _db_path(agent_id)
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS applied_migrations (
            name TEXT PRIMARY KEY,
            applied_at_ns INTEGER NOT NULL
        )
        """
    )

    applied = {
        row["name"] for row in conn.execute("SELECT name FROM applied_migrations")
    }
    pending = _pending_migrations(applied)
    for name, sql in pending:
        _apply_migration(conn, name, sql)

    db = conn
    _loaded_agent_id = agent_id
    return conn


def flush() -> None:
    """Hook for remote-sync backends. No-op on local sqlite."""
    global db
    if db is not None:
        db.commit()


def close() -> None:
    """Close the connection. Used in tests to simulate process exit."""
    global db, _loaded_agent_id
    if db is not None:
        db.close()
    db = None
    _loaded_agent_id = None


def _pending_migrations(applied: set[str]) -> list[tuple[str, str]]:
    """Return [(name, sql)] for migrations not yet applied, sorted by filename."""
    d = _migrations_dir()
    if not d.exists():
        return []
    out: list[tuple[str, str]] = []
    for path in sorted(d.glob("*.sql")):
        name = path.stem
        if name in applied:
            continue
        out.append((name, path.read_text()))
    return out


def _apply_migration(conn: sqlite3.Connection, name: str, sql: str) -> None:
    """Apply one migration and record it.

    sqlite3's `executescript` issues its own COMMITs, so we can't wrap this in a
    manual transaction. Migrations must therefore be idempotent (use
    `IF NOT EXISTS` on DDL, etc.) so that re-applying a partially-applied
    migration is safe.
    """
    conn.executescript(sql)
    conn.execute(
        "INSERT INTO applied_migrations (name, applied_at_ns) VALUES (?, ?)",
        (name, time.time_ns()),
    )

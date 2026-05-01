"""Per-agent sqlite storage, local filesystem only.

Each agent gets its own sqlite file at ``<storage_root>/<agent_id>.sqlite``.
Storage is plain WAL-mode sqlite — no remote backend, no upload/download
roundtrip on the hot path. The harness itself can be deployed onto fast-
start infra (e.g. Daytona) without storage having to know.

Storage root precedence:

1. ``HARNESS_STORAGE_ROOT`` env var (used by tests and ops who want to
   stash agent DBs somewhere specific).
2. ``~/.harness/agents`` (default for interactive dev / CLI use).

Lifecycle:

- ``load(agent_id)``        — migrate any legacy ``~/harness.sqlite`` file
                               into the new per-agent path, open/create the
                               sqlite file, apply pending migrations, and
                               install it as the module-level ``db``
                               connection.
- ``flush()``                — commit any in-flight writes.
- ``close()``                — close the connection (used by tests to
                               simulate process exit).
- ``reset_agent_memory()``   — delete the on-disk file (plus WAL/SHM
                               sidecars). The next ``load`` starts from an
                               empty schema-applied DB.
- ``fetch_agent_db(agent_id)`` — return the local path without opening the
                                  file (used by inspection tools).

Module-level ``db`` is the currently open DB-API 2.0 connection. Memory
code calls ``db.execute(...)``, ``db.executemany(...)``, etc. directly
against this real ``sqlite3.Connection`` — no adapter layer.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_STORAGE_ROOT = Path.home() / ".harness" / "agents"
_MIGRATIONS_DIR = Path(__file__).parent.parent / "memory" / "migrations"
_SAFE_ID_RE = re.compile(r"[^a-z0-9-]+")

db: Any = None
_loaded_agent_id: str | None = None


# ---------------------------------------------------------------------------
# Public lifecycle
# ---------------------------------------------------------------------------


def load(agent_id: str) -> Any:
    """Open the per-agent database and apply any pending migrations."""
    global db, _loaded_agent_id

    _migrate_legacy_db(agent_id)
    conn = _open_local(agent_id)
    _apply_migrations(conn)

    db = conn
    _loaded_agent_id = agent_id
    return conn


def flush() -> None:
    """Commit any pending writes."""
    if db is None:
        return
    try:
        db.commit()
    except Exception as e:
        logger.warning("storage.flush: commit failed: %s", e)


def close() -> None:
    """Close the connection. Used in tests to simulate process exit."""
    global db, _loaded_agent_id
    if db is not None:
        try:
            db.close()
        except Exception as e:
            logger.warning("storage.close: close failed: %s", e)
    db = None
    _loaded_agent_id = None


def reset_agent_memory(agent_id: str) -> dict[str, bool]:
    """Reset an agent's memory so the next run starts from an empty DB.

    Deletes the on-disk sqlite file plus its WAL/SHM sidecars. The next
    ``storage.load(agent_id)`` recreates the file and reapplies
    migrations.

    Returns ``{"local": <bool>}`` for callers that want to log the
    outcome. The dict shape is intentionally extensible.
    """
    return {"local": delete_local_agent_db(agent_id)}


def fetch_agent_db(agent_id: str) -> Path:
    """Return the local path to an agent's sqlite file without opening it.

    Read-only counterpart to ``load()`` — no migrations, no connection,
    no global state. Intended for inspection tooling.

    Raises ``FileNotFoundError`` if the agent has never run (or if
    ``HARNESS_STORAGE_ROOT`` points somewhere else than the run that
    wrote the file).
    """
    p = _db_path(agent_id)
    if not p.exists():
        raise FileNotFoundError(
            f"No local sqlite at {p}. Agent {agent_id!r} has never run, "
            "or HARNESS_STORAGE_ROOT points elsewhere."
        )
    return p


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------


def _storage_root() -> Path:
    raw = os.environ.get("HARNESS_STORAGE_ROOT")
    p = Path(raw).expanduser() if raw else _DEFAULT_STORAGE_ROOT
    p.mkdir(parents=True, exist_ok=True)
    return p


def _db_path(agent_id: str) -> Path:
    return _storage_root() / f"{_sanitize(agent_id)}.sqlite"


def _open_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _open_local(agent_id: str) -> sqlite3.Connection:
    return _open_sqlite(_db_path(agent_id))


# ---------------------------------------------------------------------------
# Legacy layout migration
# ---------------------------------------------------------------------------


_LEGACY_DB_NAME = "harness.sqlite"


def _legacy_db_path() -> Path:
    """Pre-PR-#8 location: a single ``~/harness.sqlite`` per Daytona sandbox.

    The old harness ran on the bedrock host and uploaded one sqlite per
    agent into a per-agent sandbox, so it didn't need to multiplex by
    agent_id inside the sandbox -- there was always exactly one DB at
    ``~/harness.sqlite``. The new harness runs *inside* the sandbox and
    keeps storage under ``~/.harness/agents/<agent_id>.sqlite`` so the
    layout matches local-dev and tests. This helper points at the old
    spot so we can rename in place.
    """
    return Path.home() / _LEGACY_DB_NAME


def _migrate_legacy_db(agent_id: str) -> None:
    """Move pre-PR-#8 sqlite into the new per-agent path, if present.

    Schema is identical (same migrations dir, same ``applied_migrations``
    rows); only the path moved. Idempotent: after the first successful
    rename the legacy file no longer exists, and subsequent loads pay the
    cost of one ``Path.exists()`` check.

    Bails (logs + leaves both files in place) if both the legacy file and
    a new-layout file already exist for this agent. That should never
    happen in practice -- the legacy layout was one DB per sandbox -- but
    if it does, refusing to clobber is the conservative move.
    """
    legacy = _legacy_db_path()
    if not legacy.exists():
        return

    new = _db_path(agent_id)
    if new.exists():
        logger.warning(
            "storage: both legacy %s and new %s exist for agent %s; "
            "leaving legacy in place. Resolve manually if intended.",
            legacy,
            new,
            agent_id,
        )
        return

    new.parent.mkdir(parents=True, exist_ok=True)
    legacy.rename(new)
    logger.info(
        "storage: migrated legacy sqlite for agent %s: %s -> %s",
        agent_id,
        legacy,
        new,
    )

    # Move WAL/SHM sidecars too. A live WAL holds frames sqlite hasn't
    # checkpointed into the main file yet; leaving it behind would make
    # the renamed DB look like it's missing the most recent writes.
    for ext in ("-wal", "-shm"):
        sidecar = Path(f"{legacy}{ext}")
        if not sidecar.exists():
            continue
        target = Path(f"{new}{ext}")
        sidecar.rename(target)
        logger.info("storage: migrated legacy sidecar %s -> %s", sidecar, target)


def delete_local_agent_db(agent_id: str) -> bool:
    """Delete the agent's sqlite file (plus -wal/-shm sidecars).

    Returns True if anything was removed. Closes the active connection
    first if it points at this agent.
    """
    if _loaded_agent_id == agent_id:
        close()

    deleted = False
    path = _db_path(agent_id)
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        try:
            candidate.unlink()
            deleted = True
        except FileNotFoundError:
            continue
        except OSError as e:
            raise RuntimeError(f"Could not delete local storage file {candidate}: {e}") from e
    return deleted


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize(agent_id: str) -> str:
    """Coerce an agent id into a safe slug (lowercase alnum + hyphens)."""
    s = _SAFE_ID_RE.sub("-", agent_id.lower()).strip("-")
    return s or "agent"


def _migrations_dir() -> Path:
    override = os.environ.get("HARNESS_MIGRATIONS_DIR")
    return Path(override) if override else _MIGRATIONS_DIR


def _apply_migrations(conn) -> None:
    """Apply any migrations in `migrations_dir` not already recorded."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS applied_migrations ("
        "name TEXT PRIMARY KEY, applied_at_ns INTEGER NOT NULL)"
    )

    applied = {row["name"] for row in conn.execute("SELECT name FROM applied_migrations")}
    pending = _pending_migrations(applied)
    for name, sql in pending:
        _apply_migration(conn, name, sql)


def _pending_migrations(applied: set[str]) -> list[tuple[str, str]]:
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


def _apply_migration(conn, name: str, sql: str) -> None:
    """Apply one migration and record it.

    Migration SQL must be idempotent (use `IF NOT EXISTS` on DDL). sqlite's
    `executescript` issues its own COMMITs, so a partially-applied migration
    can leave DDL in place without a matching audit row; the next load will
    try to re-apply it.
    """
    conn.executescript(sql)
    conn.execute(
        "INSERT INTO applied_migrations (name, applied_at_ns) VALUES (?, ?)",
        (name, time.time_ns()),
    )

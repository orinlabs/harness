"""Shared utilities for the in-process fake adapters.

The fakes live alongside eval scenarios and persist their state in the
agent's sqlite DB (``harness.core.storage.db``). They are **not** wired
into the production cold-start path -- scenarios (or the eval runner)
explicitly call :func:`apply_migrations` after :func:`harness.core.storage.load`
to create the extra tables.

Why a separate migration runner?
    :func:`harness.core.storage._apply_migrations` is private and only
    walks the ``harness/memory/migrations`` directory. We intentionally do
    not pollute production agent databases with fake-adapter tables, so we
    use a parallel mechanism keyed off ``applied_migrations`` with a
    distinct migration name prefix.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from harness.core import storage

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


# ---------------------------------------------------------------------------
# Migration runner
# ---------------------------------------------------------------------------


def apply_migrations() -> None:
    """Apply any fake-adapter migrations that haven't been applied yet.

    Idempotent. Reuses the ``applied_migrations`` audit table that
    :mod:`harness.core.storage` set up -- migration file names are
    namespaced by directory so collisions are impossible in practice.

    Raises:
        RuntimeError: if no agent DB is currently loaded.
    """
    db = storage.db
    if db is None:
        raise RuntimeError(
            "apply_migrations() called before storage.load(agent_id). "
            "Scenarios must open the agent DB before wiring up fake adapters."
        )

    db.execute(
        "CREATE TABLE IF NOT EXISTS applied_migrations ("
        "name TEXT PRIMARY KEY, applied_at_ns INTEGER NOT NULL)"
    )
    applied = {row["name"] for row in db.execute("SELECT name FROM applied_migrations")}

    for path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        name = f"fakes/{path.stem}"
        if name in applied:
            continue
        db.executescript(path.read_text())
        db.execute(
            "INSERT INTO applied_migrations (name, applied_at_ns) VALUES (?, ?)",
            (name, time.time_ns()),
        )


# ---------------------------------------------------------------------------
# Utilities shared across the four fakes
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """Return the current time as an ISO-8601 UTC string (``...Z``)."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    """Stable-ish id generator mirroring bedrock's ``sim_*`` convention."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def require_db():
    """Return ``storage.db`` or raise a descriptive error."""
    db = storage.db
    if db is None:
        raise RuntimeError(
            "fake adapter invoked before storage.load(agent_id). "
            "Open the agent DB first (harness.core.storage.load)."
        )
    return db

"""SQL-backed storage for harness memory.

Two modes, selected by the `HARNESS_DATABASE_URL` env var:

1. **Local sqlite** (default, when `HARNESS_DATABASE_URL` is unset).
   Opens `/tmp/harness/{agent_id}.sqlite`. Simple and good for local dev/demo.

2. **Turso / libSQL** (when `HARNESS_TURSO_ORG` or `HARNESS_DATABASE_URL` is set).
   One database per agent.

   - `HARNESS_TURSO_ORG` (required for Turso mode). DB URLs are built as
     `libsql://agent-{agent_id}-{org}.turso.io` by default.
   - `HARNESS_TURSO_GROUP` — defaults to `"default"`.
   - `HARNESS_DATABASE_URL` — optional override. If set, this string is used
     verbatim, with `{agent_id}` and `{org}` substituted. Useful for
     self-hosted libSQL or custom edge proxies.
   - `HARNESS_DATABASE_TOKEN` — the **group-scoped** libSQL auth token used
     for data connections. Generate via the Platform API:
       POST /v1/organizations/{org}/groups/{group}/auth/tokens
   - `HARNESS_TURSO_PLATFORM_TOKEN` — the org-scoped **Platform API** token.
     Only needed if you want harness to auto-provision agent databases on load
     (via `POST /v1/organizations/{org}/databases`). Skip it if DBs are
     provisioned out-of-band.

Lifecycle:
  - `load(agent_id)` — open a connection and apply any pending migrations.
  - `flush()`        — commit (and, in remote mode, close; local sqlite is
                       opened in autocommit so flush() is a no-op until close).
  - `close()`        — close the connection. Tests use this to simulate
                       process exit.

Module-level `db` is the currently open DB-API 2.0 connection. Memory code
calls `db.execute(...)`, `db.executemany(...)`, etc. — nothing sqlite3-specific
leaks into memory module internals.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_STORAGE_ROOT = "/tmp/harness"
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

    url_template = os.environ.get("HARNESS_DATABASE_URL")
    org = os.environ.get("HARNESS_TURSO_ORG")
    if url_template or org:
        conn = _open_remote(agent_id, url_template, org)
    else:
        conn = _open_local(agent_id)

    _apply_migrations(conn)

    db = conn
    _loaded_agent_id = agent_id
    return conn


def flush() -> None:
    """Commit any pending writes. Hook for future remote-sync backends."""
    if db is not None:
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


def delete_agent_storage(agent_id: str, *, require_remote: bool = False) -> dict[str, bool]:
    """Delete all harness-owned storage for an agent.

    This is the production deletion hook Bedrock calls when an agent is being
    deleted. Local sqlite files are always removed; the Turso database is
    removed when Turso env is configured.
    """
    return {
        "local": delete_local_agent_db(agent_id),
        "remote": delete_agent_db(agent_id, require_config=require_remote),
    }


def reset_agent_memory(agent_id: str, *, require_remote: bool = False) -> dict[str, bool]:
    """Reset an agent's memory so the next run starts from an empty DB."""
    return {
        "local": delete_local_agent_db(agent_id),
        "remote": delete_agent_db(agent_id, require_config=require_remote),
    }


# ---------------------------------------------------------------------------
# Local sqlite backend (default)
# ---------------------------------------------------------------------------


def _storage_root() -> Path:
    root = os.environ.get("HARNESS_STORAGE_ROOT", _DEFAULT_STORAGE_ROOT)
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _db_path(agent_id: str) -> Path:
    return _storage_root() / f"{_sanitize(agent_id)}.sqlite"


def _open_local(agent_id: str) -> sqlite3.Connection:
    path = _db_path(agent_id)
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def delete_local_agent_db(agent_id: str) -> bool:
    """Delete the local sqlite database and WAL sidecars for `agent_id`."""
    if _loaded_agent_id == agent_id:
        close()

    path = _db_path(agent_id)
    deleted = False
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
# Turso / libSQL backend
# ---------------------------------------------------------------------------


def _open_remote(agent_id: str, url_template: str | None, org: str | None) -> Any:
    try:
        import libsql_experimental as libsql  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "Turso mode is enabled but `libsql-experimental` is not installed. "
            "Add it (uv add libsql-experimental) or unset HARNESS_TURSO_ORG / "
            "HARNESS_DATABASE_URL to fall back to local sqlite."
        ) from e

    safe_id = _sanitize(agent_id)
    db_token = os.environ.get("HARNESS_DATABASE_TOKEN") or None
    platform_token = os.environ.get("HARNESS_TURSO_PLATFORM_TOKEN") or None
    group = os.environ.get("HARNESS_TURSO_GROUP", "default")

    if org and platform_token:
        _ensure_turso_db(
            org=org, group=group, name=f"agent-{safe_id}", token=platform_token
        )

    url = _build_db_url(safe_id=safe_id, url_template=url_template, org=org)
    kwargs: dict[str, Any] = {"database": url}
    if db_token:
        kwargs["auth_token"] = db_token
    raw = libsql.connect(**kwargs)
    return _LibsqlWrapper(raw)


def _build_db_url(*, safe_id: str, url_template: str | None, org: str | None) -> str:
    """Compose the per-agent libSQL URL.

    Preference order:
      1. `HARNESS_DATABASE_URL` with `{agent_id}` / `{org}` substitutions.
      2. Standard Turso shape `libsql://agent-{agent_id}-{org}.turso.io` when
         only `HARNESS_TURSO_ORG` is set.
    """
    if url_template:
        url = url_template
        if org:
            url = url.replace("{org}", org)
        url = url.replace("{agent_id}", safe_id)
        return url
    if org:
        return f"libsql://agent-{safe_id}-{org}.turso.io"
    raise RuntimeError(
        "Cannot build a Turso URL: set either HARNESS_DATABASE_URL or "
        "HARNESS_TURSO_ORG."
    )


def delete_agent_db(agent_id: str, *, require_config: bool = False) -> bool:
    """Delete the Turso database for `agent_id` via the Platform API.

    No-op if the DB doesn't exist (404). Used by integration tests to tear
    down their provisioned DBs. Needs `HARNESS_TURSO_ORG` and
    `HARNESS_TURSO_PLATFORM_TOKEN` in the environment.
    """
    org = os.environ.get("HARNESS_TURSO_ORG")
    token = os.environ.get("HARNESS_TURSO_PLATFORM_TOKEN")
    if not (org and token):
        if require_config:
            missing = [
                name
                for name, value in (
                    ("HARNESS_TURSO_ORG", org),
                    ("HARNESS_TURSO_PLATFORM_TOKEN", token),
                )
                if not value
            ]
            raise RuntimeError(
                "Cannot delete remote agent storage; missing "
                f"{', '.join(missing)}."
            )
        return False

    name = f"agent-{_sanitize(agent_id)}"
    url = f"https://api.turso.tech/v1/organizations/{org}/databases/{name}"
    try:
        resp = httpx.delete(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        )
    except httpx.HTTPError as e:
        if require_config:
            raise RuntimeError(
                f"Turso Platform API unreachable while deleting {name!r}: {e}"
            ) from e
        logger.warning("storage: DELETE %s failed: %s", name, e)
        return False
    if resp.status_code in (200, 204, 404):
        return True
    if require_config:
        raise RuntimeError(
            f"Turso Platform API rejected deletion for {name!r}: "
            f"{resp.status_code} {resp.text!r}"
        )
    logger.warning(
        "storage: DELETE %s rejected: %s %s", name, resp.status_code, resp.text
    )
    return False


def _ensure_turso_db(*, org: str, group: str, name: str, token: str) -> None:
    """Create a Turso database via the Platform API. Idempotent via 409.

    Platform API docs: https://docs.turso.tech/api-reference/databases/create
    """
    url = f"https://api.turso.tech/v1/organizations/{org}/databases"
    try:
        resp = httpx.post(
            url,
            json={"name": name, "group": group},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
    except httpx.HTTPError as e:
        raise RuntimeError(
            f"Turso Platform API unreachable while provisioning {name!r}: {e}"
        ) from e

    if resp.status_code in (200, 201):
        logger.info("storage: provisioned Turso db %s in org %s", name, org)
        return
    if resp.status_code == 409:
        return
    raise RuntimeError(
        f"Turso Platform API rejected provisioning for {name!r}: "
        f"{resp.status_code} {resp.text!r}"
    )


class _LibsqlWrapper:
    """Thin adapter so `libsql_experimental`'s tuple-returning cursors look
    like `sqlite3` row-factory cursors to the rest of the codebase.

    Memory code does `row["col_name"]` *and* iterates tuple-style. Upstream
    libsql returns bare tuples; we wrap each cursor so `fetchone()/fetchall()`
    emit `_LibsqlRow` objects that support both.
    """

    def __init__(self, inner):
        self._inner = inner

    def execute(self, *args, **kwargs):
        cur = self._inner.execute(*args, **kwargs)
        return _LibsqlCursor(cur)

    def executemany(self, *args, **kwargs):
        cur = self._inner.executemany(*args, **kwargs)
        return _LibsqlCursor(cur)

    def executescript(self, sql: str):
        return self._inner.executescript(sql)

    def commit(self):
        return self._inner.commit()

    def rollback(self):
        return self._inner.rollback()

    def close(self):
        return self._inner.close()

    def cursor(self):
        return _LibsqlCursor(self._inner.cursor())


class _LibsqlCursor:
    def __init__(self, inner):
        self._inner = inner

    def __iter__(self):
        # libsql_experimental cursors don't support native iteration, so pull
        # everything via fetchall. Memory queries we run here return <=200 rows.
        cols = self._column_names()
        for row in self._inner.fetchall():
            yield _LibsqlRow(row, cols)

    def fetchone(self):
        row = self._inner.fetchone()
        if row is None:
            return None
        return _LibsqlRow(row, self._column_names())

    def fetchall(self):
        cols = self._column_names()
        return [_LibsqlRow(r, cols) for r in self._inner.fetchall()]

    def _column_names(self) -> tuple[str, ...]:
        desc = self._inner.description or ()
        return tuple(c[0] for c in desc)


class _LibsqlRow:
    """Behaves like `sqlite3.Row`: indexable by int or column name, iterable,
    convertible via `dict(row)`.
    """

    __slots__ = ("_values", "_cols")

    def __init__(self, values: tuple, cols: tuple[str, ...]):
        self._values = values
        self._cols = cols

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key]
        try:
            idx = self._cols.index(key)
        except ValueError:
            raise KeyError(key) from None
        return self._values[idx]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def keys(self):
        return list(self._cols)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize(agent_id: str) -> str:
    """Coerce an agent id into a DB-name-safe slug (lowercase alnum + hyphens).

    Turso enforces this strictly; local sqlite file names benefit from it too.
    """
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

    applied = {
        row["name"] for row in conn.execute("SELECT name FROM applied_migrations")
    }
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

"""Per-agent sqlite storage, optionally backed by a Daytona sandbox.

Two backends, selected by env:

1. **Daytona** (when ``DAYTONA_API_KEY`` is set — production mode).
   One Daytona sandbox per agent, identified by label
   ``harness.agent_id=<sanitized-id>``. The canonical sqlite file lives in the
   sandbox at ``~/harness.sqlite``. harness opens a local working copy at
   ``<storage_root>/daytona-cache/<agent_id>.sqlite``:

   - ``load(agent_id)``        — find/create+start the sandbox (including
                                  transparently resuming an archived one),
                                  download the sqlite file, open it locally,
                                  apply pending migrations.
   - ``flush()``                — commit locally, checkpoint WAL, upload the
                                  working copy back to the sandbox so the
                                  sandbox remains the source of truth.
   - ``close()``                — flush, then close the local connection.
   - ``delete_agent_storage()`` — archive the sandbox (preserves state in
                                  cheap object storage; resumable by the next
                                  ``load``) and delete the local cache. Pass
                                  ``purge=True`` to fully delete instead.
   - ``reset_agent_memory()``   — drop just ``harness.sqlite`` inside the
                                  sandbox so the next ``load`` starts fresh,
                                  without losing the sandbox itself.

   Config env:

   - ``DAYTONA_API_KEY`` (required to enable this mode).
   - ``DAYTONA_API_URL`` (optional; defaults to ``https://app.daytona.io/api``).
   - ``DAYTONA_TARGET``  (optional; SDK default region otherwise).
   - ``HARNESS_DAYTONA_AUTO_STOP_MINUTES`` (optional; passed as
      ``auto_stop_interval``. Default leaves SDK's 15-minute default in place.)

2. **Local sqlite** (default when ``DAYTONA_API_KEY`` is unset).
   Opens ``<storage_root>/<agent_id>.sqlite``. Simple and fast; used by tests
   and local dev when Daytona isn't available.

Module-level ``db`` is the currently open DB-API 2.0 connection. Memory code
calls ``db.execute(...)``, ``db.executemany(...)``, etc. Both backends are
real ``sqlite3.Connection`` objects — there's no adapter layer.
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

_DEFAULT_STORAGE_ROOT = "/tmp/harness"
_MIGRATIONS_DIR = Path(__file__).parent.parent / "memory" / "migrations"
_SAFE_ID_RE = re.compile(r"[^a-z0-9-]+")

# Path inside the Daytona sandbox (sandbox working dir is ~daytona/ by
# default). Keeping this relative means it resolves correctly whether the
# sandbox user is `daytona`, `root`, or anything else.
_SANDBOX_DB_PATH = "harness.sqlite"
_SANDBOX_LABEL_KEY = "harness.agent_id"

db: Any = None
_loaded_agent_id: str | None = None
# When in Daytona mode, we hold the live Sandbox handle so flush() can
# re-upload the working copy and delete_agent_storage() can tear the
# sandbox down.
_daytona_sandbox: Any = None
_daytona_local_path: Path | None = None


# ---------------------------------------------------------------------------
# Public lifecycle
# ---------------------------------------------------------------------------


def load(agent_id: str) -> Any:
    """Open the per-agent database and apply any pending migrations."""
    global db, _loaded_agent_id, _daytona_sandbox, _daytona_local_path

    if os.environ.get("DAYTONA_API_KEY"):
        conn, sandbox, local_path = _open_daytona(agent_id)
        _daytona_sandbox = sandbox
        _daytona_local_path = local_path
    else:
        conn = _open_local(agent_id)
        _daytona_sandbox = None
        _daytona_local_path = None

    _apply_migrations(conn)

    db = conn
    _loaded_agent_id = agent_id
    return conn


def flush() -> None:
    """Commit any pending writes.

    In Daytona mode, this also checkpoints WAL into the main sqlite file and
    uploads the working copy back to the sandbox so the sandbox remains the
    source of truth between runs.
    """
    if db is None:
        return
    try:
        db.commit()
    except Exception as e:
        logger.warning("storage.flush: commit failed: %s", e)
        return

    if _daytona_sandbox is not None and _daytona_local_path is not None:
        try:
            # Force WAL frames into the main DB file so the bytes we upload
            # are actually the latest state.
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception as e:
            logger.warning("storage.flush: wal_checkpoint failed: %s", e)
        try:
            _upload_db_to_sandbox(_daytona_sandbox, _daytona_local_path)
        except Exception as e:
            logger.exception(
                "storage.flush: failed to upload sqlite to Daytona sandbox %s: %s",
                getattr(_daytona_sandbox, "id", "?"),
                e,
            )


def close() -> None:
    """Close the connection. Used in tests to simulate process exit."""
    global db, _loaded_agent_id, _daytona_sandbox, _daytona_local_path
    if db is not None:
        try:
            db.close()
        except Exception as e:
            logger.warning("storage.close: close failed: %s", e)
    db = None
    _loaded_agent_id = None
    _daytona_sandbox = None
    _daytona_local_path = None


def delete_agent_storage(
    agent_id: str, *, require_remote: bool = False, purge: bool = False
) -> dict[str, bool]:
    """Tear down harness-owned storage for an agent.

    Always removes the local working copy. In Daytona mode:

    - Default (``purge=False``): archives the sandbox. State is preserved in
      cheap object storage and ``storage.load(agent_id)`` can resurrect it
      later. This is the right default for "Bedrock deleted the agent" — we
      keep the door open in case the agent is recreated.
    - ``purge=True``: fully deletes the sandbox. No recovery.
    """
    remote = (
        purge_agent_sandbox(agent_id, require_config=require_remote)
        if purge
        else archive_agent_sandbox(agent_id, require_config=require_remote)
    )
    return {
        "local": delete_local_agent_db(agent_id),
        "remote": remote,
    }


def reset_agent_memory(
    agent_id: str, *, require_remote: bool = False
) -> dict[str, bool]:
    """Reset an agent's memory so the next run starts from an empty DB.

    Leaves the Daytona sandbox in place (cheap to keep) but wipes the sqlite
    file inside it plus the local working copy.
    """
    return {
        "local": delete_local_agent_db(agent_id),
        "remote": clear_agent_sandbox_db(agent_id, require_config=require_remote),
    }


# ---------------------------------------------------------------------------
# Local sqlite backend
# ---------------------------------------------------------------------------


def _storage_root() -> Path:
    root = os.environ.get("HARNESS_STORAGE_ROOT", _DEFAULT_STORAGE_ROOT)
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _db_path(agent_id: str) -> Path:
    return _storage_root() / f"{_sanitize(agent_id)}.sqlite"


def _daytona_cache_path(agent_id: str) -> Path:
    """Local cache path for Daytona-mode working copy."""
    root = _storage_root() / "daytona-cache"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{_sanitize(agent_id)}.sqlite"


def _open_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _open_local(agent_id: str) -> sqlite3.Connection:
    return _open_sqlite(_db_path(agent_id))


def delete_local_agent_db(agent_id: str) -> bool:
    """Delete local sqlite file(s) for ``agent_id`` (both default and cache roots)."""
    if _loaded_agent_id == agent_id:
        close()

    deleted = False
    for path in (_db_path(agent_id), _daytona_cache_path(agent_id)):
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
            try:
                candidate.unlink()
                deleted = True
            except FileNotFoundError:
                continue
            except OSError as e:
                raise RuntimeError(
                    f"Could not delete local storage file {candidate}: {e}"
                ) from e
    return deleted


# ---------------------------------------------------------------------------
# Daytona backend
# ---------------------------------------------------------------------------


def _daytona_client() -> Any:
    """Build a Daytona client from env. Imported lazily so tests never touch it."""
    try:
        from daytona import Daytona, DaytonaConfig
    except ImportError as e:
        raise RuntimeError(
            "DAYTONA_API_KEY is set but the `daytona` package is not installed. "
            "Run `uv add daytona` or unset DAYTONA_API_KEY."
        ) from e

    api_key = os.environ.get("DAYTONA_API_KEY")
    if not api_key:
        raise RuntimeError("DAYTONA_API_KEY is required for Daytona storage mode.")

    kwargs: dict[str, Any] = {"api_key": api_key}
    if url := os.environ.get("DAYTONA_API_URL"):
        kwargs["api_url"] = url
    if target := os.environ.get("DAYTONA_TARGET"):
        kwargs["target"] = target
    return Daytona(DaytonaConfig(**kwargs))


def _find_agent_sandbox(client: Any, safe_id: str) -> Any | None:
    """Look up the agent's sandbox by harness label. Returns None if missing."""
    try:
        result = client.list(labels={_SANDBOX_LABEL_KEY: safe_id}, limit=1)
    except Exception as e:
        raise RuntimeError(
            f"Daytona list() failed while looking up sandbox for agent "
            f"{safe_id!r}: {e}"
        ) from e

    items = getattr(result, "items", None)
    if items is None:
        # Older/newer SDK variants: list() may return a plain list.
        items = result if isinstance(result, list) else []
    return items[0] if items else None


def _ensure_sandbox_started(client: Any, sandbox: Any) -> None:
    """Idempotently ensure the sandbox is in STARTED state.

    For archived sandboxes the SDK's ``start()`` handles restoration from
    object storage transparently, so the same call covers both
    stopped-from-idle and archived-then-resumed flows.
    """
    state = getattr(sandbox, "state", None)
    state_value = getattr(state, "value", state)
    if state_value == "started":
        return
    if state_value in ("starting", "creating", "restoring"):
        # SDK's create() already waits; for other transitional states we
        # don't have a nice wait primitive — attempting start() when the
        # sandbox is already transitioning raises a clear error upstream.
        return
    logger.info(
        "storage: starting Daytona sandbox id=%s (state=%s)",
        getattr(sandbox, "id", "?"),
        state_value,
    )
    client.start(sandbox)


def _create_agent_sandbox(client: Any, safe_id: str) -> Any:
    """Provision a fresh Daytona sandbox labeled for this agent."""
    from daytona import CreateSandboxFromImageParams, Image, Resources

    # We only use the sandbox to persist ~/harness.sqlite, so run on the
    # smallest footprint Daytona allows (1 vCPU / 1 GiB RAM / 1 GiB disk).
    # Resource overrides aren't supported on snapshot-based creates today, so
    # we build from a minimal Python image instead of the default snapshot.
    labels = {_SANDBOX_LABEL_KEY: safe_id}
    params_kwargs: dict[str, Any] = {
        "language": "python",
        "labels": labels,
        "image": Image.debian_slim("3.12"),
        "resources": Resources(cpu=1, memory=1, disk=1),
    }
    if raw := os.environ.get("HARNESS_DAYTONA_AUTO_STOP_MINUTES"):
        try:
            params_kwargs["auto_stop_interval"] = int(raw)
        except ValueError:
            logger.warning(
                "HARNESS_DAYTONA_AUTO_STOP_MINUTES=%r is not an int; ignoring.", raw
            )

    logger.info("storage: creating Daytona sandbox for agent %s", safe_id)
    sandbox = client.create(CreateSandboxFromImageParams(**params_kwargs))
    logger.info(
        "storage: Daytona sandbox ready id=%s state=%s",
        getattr(sandbox, "id", "?"),
        getattr(getattr(sandbox, "state", None), "value", "?"),
    )
    return sandbox


def _download_db_from_sandbox(sandbox: Any, local_path: Path) -> None:
    """Download the sandbox's canonical sqlite file into ``local_path``.

    A missing remote file is expected on the very first run — we treat it as
    "start from empty" and leave ``local_path`` absent so sqlite3.connect
    creates a fresh DB.
    """
    try:
        data = sandbox.fs.download_file(_SANDBOX_DB_PATH)
    except Exception as e:
        msg = str(e).lower()
        if "not found" in msg or "no such" in msg or "404" in msg:
            # Clean up any stale cache so migrations run against a fresh DB.
            for sidecar in (local_path, Path(f"{local_path}-wal"), Path(f"{local_path}-shm")):
                try:
                    sidecar.unlink()
                except FileNotFoundError:
                    pass
            return
        raise RuntimeError(
            f"Daytona download_file({_SANDBOX_DB_PATH!r}) failed: {e}"
        ) from e

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)
    # Invalidate any stale WAL/SHM from a previous load so sqlite sees the
    # downloaded file as the authoritative state.
    for sidecar in (Path(f"{local_path}-wal"), Path(f"{local_path}-shm")):
        try:
            sidecar.unlink()
        except FileNotFoundError:
            pass


def _upload_db_to_sandbox(sandbox: Any, local_path: Path) -> None:
    if not local_path.exists():
        return
    data = local_path.read_bytes()
    sandbox.fs.upload_file(data, _SANDBOX_DB_PATH)


def _open_daytona(agent_id: str) -> tuple[sqlite3.Connection, Any, Path]:
    """Open a per-agent sqlite connection backed by a Daytona sandbox.

    Returns ``(connection, sandbox, local_cache_path)``.
    """
    safe_id = _sanitize(agent_id)
    client = _daytona_client()

    sandbox = _find_agent_sandbox(client, safe_id)
    if sandbox is None:
        sandbox = _create_agent_sandbox(client, safe_id)
    else:
        _ensure_sandbox_started(client, sandbox)

    local_path = _daytona_cache_path(agent_id)
    _download_db_from_sandbox(sandbox, local_path)
    conn = _open_sqlite(local_path)
    return conn, sandbox, local_path


def _resolve_sandbox_for_teardown(
    agent_id: str, *, require_config: bool
) -> tuple[Any, Any] | None:
    """Return ``(client, sandbox)`` for an agent, or None when nothing to do.

    Returns None when Daytona isn't configured (and ``require_config`` is
    False) or when no sandbox exists for this agent. Raises when
    ``require_config`` is True and env is missing.
    """
    if not os.environ.get("DAYTONA_API_KEY"):
        if require_config:
            raise RuntimeError(
                "Cannot touch remote agent storage; DAYTONA_API_KEY is not set."
            )
        return None

    safe_id = _sanitize(agent_id)
    try:
        client = _daytona_client()
        sandbox = _find_agent_sandbox(client, safe_id)
    except RuntimeError:
        if require_config:
            raise
        return None
    if sandbox is None:
        return None
    return client, sandbox


def archive_agent_sandbox(
    agent_id: str, *, require_config: bool = False
) -> bool:
    """Archive the agent's Daytona sandbox. The sandbox is stopped first if
    needed (archive() requires a stopped sandbox).

    Archiving moves the sandbox filesystem to cheap object storage. The
    sandbox can be resurrected later via ``client.start(sandbox)`` (the SDK
    handles the recover step internally for archived sandboxes).

    Returns ``True`` when the archive succeeded OR no sandbox existed for
    this agent (the post-condition — "no live sandbox running" — holds
    either way). Returns ``False`` when Daytona isn't configured or the
    archive call errored.
    """
    if not os.environ.get("DAYTONA_API_KEY"):
        if require_config:
            raise RuntimeError(
                "Cannot archive remote agent storage; DAYTONA_API_KEY is not set."
            )
        return False

    resolved = _resolve_sandbox_for_teardown(agent_id, require_config=require_config)
    if resolved is None:
        return True  # no sandbox to archive; desired post-condition holds
    client, sandbox = resolved

    state = getattr(getattr(sandbox, "state", None), "value", None)
    if state == "archived":
        return True
    try:
        if state not in ("stopped", "stopping"):
            # archive() requires the sandbox be stopped first.
            try:
                client.stop(sandbox)
            except Exception as e:
                logger.warning(
                    "storage: stop before archive failed for sandbox id=%s: %s",
                    getattr(sandbox, "id", "?"),
                    e,
                )
        sandbox.archive()
    except Exception as e:
        if require_config:
            raise RuntimeError(
                f"Daytona archive failed for sandbox id={getattr(sandbox, 'id', '?')}: {e}"
            ) from e
        logger.warning(
            "storage: Daytona archive failed for sandbox id=%s: %s",
            getattr(sandbox, "id", "?"),
            e,
        )
        return False
    return True


def purge_agent_sandbox(
    agent_id: str, *, require_config: bool = False
) -> bool:
    """Fully delete the agent's Daytona sandbox. No recovery afterwards."""
    if not os.environ.get("DAYTONA_API_KEY"):
        if require_config:
            raise RuntimeError(
                "Cannot purge remote agent storage; DAYTONA_API_KEY is not set."
            )
        return False

    resolved = _resolve_sandbox_for_teardown(agent_id, require_config=require_config)
    if resolved is None:
        return True  # already gone
    client, sandbox = resolved

    try:
        client.delete(sandbox)
    except Exception as e:
        if require_config:
            raise RuntimeError(
                f"Daytona delete failed for sandbox id={getattr(sandbox, 'id', '?')}: {e}"
            ) from e
        logger.warning(
            "storage: Daytona delete failed for sandbox id=%s: %s",
            getattr(sandbox, "id", "?"),
            e,
        )
        return False
    return True


def clear_agent_sandbox_db(
    agent_id: str, *, require_config: bool = False
) -> bool:
    """Remove just ``harness.sqlite`` inside the agent's sandbox.

    Leaves the sandbox running so the next ``storage.load(agent_id)`` starts
    from an empty sqlite without paying the create-sandbox round-trip.
    """
    if not os.environ.get("DAYTONA_API_KEY"):
        if require_config:
            raise RuntimeError(
                "Cannot clear remote agent storage; DAYTONA_API_KEY is not set."
            )
        return False

    resolved = _resolve_sandbox_for_teardown(agent_id, require_config=require_config)
    if resolved is None:
        return True  # no sandbox => nothing to clear; fresh load will start empty
    client, sandbox = resolved

    # File delete needs the sandbox running.
    try:
        _ensure_sandbox_started(client, sandbox)
    except Exception as e:
        if require_config:
            raise RuntimeError(
                f"Daytona start failed before clearing db for sandbox id="
                f"{getattr(sandbox, 'id', '?')}: {e}"
            ) from e
        logger.warning(
            "storage: Daytona start-before-clear failed for sandbox id=%s: %s",
            getattr(sandbox, "id", "?"),
            e,
        )
        return False

    try:
        sandbox.fs.delete_file(_SANDBOX_DB_PATH)
    except Exception as e:
        msg = str(e).lower()
        if "not found" in msg or "no such" in msg or "404" in msg:
            return True
        if require_config:
            raise RuntimeError(
                f"Daytona delete_file({_SANDBOX_DB_PATH!r}) failed for sandbox id="
                f"{getattr(sandbox, 'id', '?')}: {e}"
            ) from e
        logger.warning(
            "storage: delete_file(%s) failed for sandbox id=%s: %s",
            _SANDBOX_DB_PATH,
            getattr(sandbox, "id", "?"),
            e,
        )
        return False
    return True


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

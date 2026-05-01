"""One-shot migration: every Turso agent DB -> its own Daytona sandbox.

For each ``agent-<uuid>`` database in your Turso org, this script:

1. Looks up the agent_id from the DB name.
2. Calls ``harness.core.storage.load(agent_id)``, which either finds the
   existing Daytona sandbox (labeled ``harness.agent_id=<sanitized-id>``) or
   provisions a fresh one, then opens a local sqlite working copy and applies
   harness migrations.
3. Best-effort installs the ``sqlite3`` CLI inside the sandbox (useful for
   later debugging; the file is always readable from Python regardless).
4. Reads each known table from the Turso DB and ``INSERT OR REPLACE``\\ s the
   rows into the local sqlite copy.
5. Flushes (WAL-checkpoint + upload to sandbox) and closes.

Run with:

    uv run --with libsql-experimental python scripts/migrate_turso_to_daytona.py

Required env (populate ``.env``):

    DAYTONA_API_KEY
    HARNESS_TURSO_ORG
    HARNESS_TURSO_PLATFORM_TOKEN   (for GET /v1/organizations/{org}/databases)
    HARNESS_DATABASE_TOKEN         (group-scoped libSQL auth token, for reads)

Flags:
    --limit N       only migrate the first N agent DBs (for testing)
    --only UUID     only migrate this specific agent_id (repeatable)
    --skip-empty    skip Turso DBs that have zero messages (still creates no
                    Daytona sandbox for them)
    --dry-run       list what would be migrated without touching Daytona
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Tables we migrate, in the order they should be written (no FK deps currently,
# but keeping a stable order makes logs easier to read).
TABLES: list[tuple[str, tuple[str, ...]]] = [
    ("messages", ("id", "ts_ns", "role", "content_json")),
    (
        "one_minute_summaries",
        ("id", "date", "hour", "minute", "summary", "message_count", "created_at_ns"),
    ),
    (
        "five_minute_summaries",
        ("id", "date", "hour", "minute", "summary", "message_count", "created_at_ns"),
    ),
    (
        "hourly_summaries",
        ("id", "date", "hour", "summary", "message_count", "created_at_ns"),
    ),
    ("daily_summaries", ("id", "date", "summary", "message_count", "created_at_ns")),
    (
        "weekly_summaries",
        ("id", "week_start_date", "summary", "message_count", "created_at_ns"),
    ),
    (
        "monthly_summaries",
        ("id", "year", "month", "summary", "message_count", "created_at_ns"),
    ),
]

AGENT_DB_PREFIX = "agent-"


def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def list_turso_dbs(org: str, platform_token: str) -> list[dict]:
    url = f"https://api.turso.tech/v1/organizations/{org}/databases"
    r = httpx.get(url, headers={"Authorization": f"Bearer {platform_token}"}, timeout=30.0)
    r.raise_for_status()
    return r.json().get("databases", [])


def agent_id_from_db_name(name: str) -> str | None:
    """``agent-<uuid>`` -> ``<uuid>``. Anything else returns None."""
    if not name.startswith(AGENT_DB_PREFIX):
        return None
    return name[len(AGENT_DB_PREFIX) :]


def open_turso(agent_id: str, org: str, db_token: str):
    import libsql_experimental as libsql

    url = f"libsql://agent-{agent_id}-{org}.turso.io"
    return libsql.connect(database=url, auth_token=db_token)


def install_sqlite_cli(sandbox) -> None:
    """Best-effort install of the sqlite3 CLI inside the sandbox.

    Daytona's default python snapshot ships with libsqlite3 (Python needs it),
    but the ``sqlite3`` binary isn't always present. Failures are logged and
    swallowed — the harness runtime doesn't actually need the CLI, this is a
    convenience for humans SSH-ing into the sandbox later.
    """
    try:
        check = sandbox.process.exec("which sqlite3 || true", timeout=30)
        stdout = (check.result or "").strip()
        if stdout and "sqlite3" in stdout:
            return
        sandbox.process.exec(
            "sudo apt-get update -y >/dev/null 2>&1 && "
            "sudo apt-get install -y sqlite3 >/dev/null 2>&1 || "
            "apt-get install -y sqlite3 >/dev/null 2>&1",
            timeout=180,
        )
    except Exception as e:
        print(f"    [sqlite3 install] skipped: {e}")


def copy_table(turso_conn, local_conn, table: str, cols: tuple[str, ...]) -> int:
    """Copy all rows of ``table`` from Turso -> local sqlite. Returns row count."""
    col_list = ", ".join(cols)
    try:
        rows = turso_conn.execute(f"SELECT {col_list} FROM {table}").fetchall()
    except Exception as e:
        # Older DBs may predate certain tables (e.g. five_minute_summaries was
        # added in a later migration). Treat "no such table" as empty.
        msg = str(e).lower()
        if "no such table" in msg or "not found" in msg:
            return 0
        raise

    if not rows:
        return 0

    placeholders = ", ".join(["?"] * len(cols))
    sql = f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})"
    # Normalize each row tuple — libsql sometimes returns its own Row object.
    local_conn.executemany(sql, [tuple(r) for r in rows])
    return len(rows)


def migrate_agent(
    agent_id: str,
    org: str,
    db_token: str,
    *,
    skip_empty: bool,
) -> dict[str, int] | None:
    """Migrate one agent. Returns per-table counts or None when skipped."""
    # 1. Pull a row count from Turso first so we can cheaply skip empty DBs.
    turso_conn = open_turso(agent_id, org, db_token)
    try:
        msg_count = turso_conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    except Exception as e:
        msg = str(e).lower()
        if "no such table" in msg:
            msg_count = 0
        else:
            raise

    if skip_empty and msg_count == 0:
        print("    empty Turso DB (0 messages) - skipping")
        try:
            turso_conn.close()
        except Exception:
            pass
        return None

    # 2. Open the Daytona-backed sqlite copy. This will create+start the
    #    sandbox if it doesn't exist, download the current ``harness.sqlite``
    #    (empty on first run), and apply migrations.
    from harness.core import storage

    storage.load(agent_id)
    sandbox = storage._daytona_sandbox
    if sandbox is not None:
        install_sqlite_cli(sandbox)

    # 3. Copy each table.
    counts: dict[str, int] = {}
    try:
        for table, cols in TABLES:
            counts[table] = copy_table(turso_conn, storage.db, table, cols)
    finally:
        try:
            turso_conn.close()
        except Exception:
            pass

    # 4. Checkpoint + upload + close.
    storage.flush()
    storage.close()
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only migrate first N agents")
    ap.add_argument(
        "--only", action="append", default=[], help="migrate only these agent_ids (repeatable)"
    )
    ap.add_argument("--skip-empty", action="store_true", help="skip Turso DBs with zero messages")
    ap.add_argument("--dry-run", action="store_true", help="list and exit")
    args = ap.parse_args()

    org = os.environ.get("HARNESS_TURSO_ORG")
    platform_token = os.environ.get("HARNESS_TURSO_PLATFORM_TOKEN")
    db_token = os.environ.get("HARNESS_DATABASE_TOKEN")
    daytona_key = os.environ.get("DAYTONA_API_KEY")

    missing = [
        k
        for k, v in (
            ("HARNESS_TURSO_ORG", org),
            ("HARNESS_TURSO_PLATFORM_TOKEN", platform_token),
            ("HARNESS_DATABASE_TOKEN", db_token),
            ("DAYTONA_API_KEY", daytona_key),
        )
        if not v
    ]
    if missing:
        print(f"missing env: {', '.join(missing)}")
        return 2

    try:
        import libsql_experimental  # noqa: F401
    except ImportError:
        print(
            "libsql-experimental is not installed. Run with:\n"
            "  uv run --with libsql-experimental python scripts/migrate_turso_to_daytona.py"
        )
        return 2

    banner(f"List Turso databases in org '{org}'")
    dbs = list_turso_dbs(org, platform_token)
    agent_ids: list[str] = []
    for db in dbs:
        name = db.get("Name") or db.get("name") or ""
        aid = agent_id_from_db_name(name)
        if aid is None:
            print(f"  [skip non-agent] {name}")
            continue
        agent_ids.append(aid)
    print(f"  found {len(agent_ids)} agent DB(s)")

    if args.only:
        wanted = set(args.only)
        agent_ids = [a for a in agent_ids if a in wanted]
        missing_only = wanted - set(agent_ids)
        if missing_only:
            print(f"  [warn] --only ids not present in Turso: {sorted(missing_only)}")

    if args.limit:
        agent_ids = agent_ids[: args.limit]

    print(f"  migrating {len(agent_ids)} agent(s)")

    if args.dry_run:
        for aid in agent_ids:
            print(f"  would migrate: {aid}")
        return 0

    successes: list[tuple[str, dict[str, int], float]] = []
    skipped: list[str] = []
    failures: list[tuple[str, str]] = []

    for i, aid in enumerate(agent_ids, 1):
        banner(f"[{i}/{len(agent_ids)}] agent={aid}")
        t0 = time.perf_counter()
        try:
            counts = migrate_agent(aid, org, db_token, skip_empty=args.skip_empty)
        except Exception as e:
            traceback.print_exc()
            failures.append((aid, str(e)))
            continue
        elapsed = time.perf_counter() - t0
        if counts is None:
            skipped.append(aid)
            continue
        total = sum(counts.values())
        pretty = ", ".join(f"{k}={v}" for k, v in counts.items() if v > 0) or "(all 0)"
        print(f"    copied {total} rows ({pretty}) in {elapsed:.1f}s")
        successes.append((aid, counts, elapsed))

    banner("DONE")
    print(f"  migrated: {len(successes)}")
    print(f"  skipped:  {len(skipped)}")
    print(f"  failed:   {len(failures)}")
    if failures:
        print("\n  failures:")
        for aid, err in failures:
            print(f"    {aid}: {err}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

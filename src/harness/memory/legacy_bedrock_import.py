"""Import legacy Bedrock memory exports into harness-owned storage."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.core import storage

SUMMARY_TABLES = (
    "one_minute_summaries",
    "five_minute_summaries",
    "hourly_summaries",
    "daily_summaries",
    "weekly_summaries",
    "monthly_summaries",
)

# Multi-row INSERT batch size. The legacy per-row ``executemany()`` pattern
# fanned out one round-trip per row against remote libSQL (Turso), which made
# large-agent imports take tens of minutes over WAN latency. Packing many
# tuples into one ``INSERT ... VALUES (...), (...), ...`` statement collapses
# each batch into a single wire round-trip. 500 rows × the widest schema
# (7 columns) = 3500 parameters, safely under SQLite/libSQL's 32766 bound-
# parameter cap (see SQLITE_MAX_VARIABLE_NUMBER on modern builds).
BULK_INSERT_BATCH_ROWS = 500


@dataclass(frozen=True)
class ImportCounts:
    messages: int
    one_minute_summaries: int
    five_minute_summaries: int
    hourly_summaries: int
    daily_summaries: int
    weekly_summaries: int
    monthly_summaries: int

    def as_dict(self) -> dict[str, int]:
        return {
            "messages": self.messages,
            "one_minute_summaries": self.one_minute_summaries,
            "five_minute_summaries": self.five_minute_summaries,
            "hourly_summaries": self.hourly_summaries,
            "daily_summaries": self.daily_summaries,
            "weekly_summaries": self.weekly_summaries,
            "monthly_summaries": self.monthly_summaries,
        }


def import_legacy_bedrock_memory(agent_id: str, payload_path: str | Path) -> ImportCounts:
    """Load one agent's legacy Bedrock memory export into the current harness DB.

    The payload is intentionally already grouped by agent so the CLI can be run
    independently per agent from a Bedrock migration. Imports are idempotent:
    legacy ids are stable and every insert uses SQLite/libSQL upsert semantics.
    """
    payload = _read_payload(payload_path)
    payload_agent_id = str(payload.get("agent_id") or "")
    if payload_agent_id != str(agent_id):
        raise ValueError(
            f"payload agent_id {payload_agent_id!r} does not match CLI agent_id {agent_id!r}"
        )

    conn = storage.load(agent_id)
    try:
        counts = ImportCounts(
            messages=_import_messages(conn, payload.get("messages") or []),
            one_minute_summaries=_import_one_minute(
                conn, payload.get("one_minute_summaries") or []
            ),
            five_minute_summaries=_import_five_minute(
                conn, payload.get("five_minute_summaries") or []
            ),
            hourly_summaries=_import_hourly(conn, payload.get("hourly_summaries") or []),
            daily_summaries=_import_daily(conn, payload.get("daily_summaries") or []),
            weekly_summaries=_import_weekly(conn, payload.get("weekly_summaries") or []),
            monthly_summaries=_import_monthly(conn, payload.get("monthly_summaries") or []),
        )
        storage.flush()
        return counts
    finally:
        storage.close()


def _read_payload(payload_path: str | Path) -> dict[str, Any]:
    with Path(payload_path).open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("legacy memory payload must be a JSON object")
    return payload


def _bulk_upsert(
    conn,
    *,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
    batch_size: int = BULK_INSERT_BATCH_ROWS,
) -> int:
    """Upsert ``rows`` into ``table`` using chunked multi-row VALUES statements.

    Every chunk is executed as a single SQL statement, which maps to exactly
    one wire round-trip on remote libSQL / Turso backends. This is materially
    faster than the per-row ``executemany()`` path the legacy code used when
    the connection is not local.

    Returns the total number of rows inserted across all batches.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    col_list = ", ".join(columns)
    row_placeholder = "(" + ", ".join(["?"] * len(columns)) + ")"

    total = 0
    batch: list[Sequence[Any]] = []
    for row in rows:
        if len(row) != len(columns):
            raise ValueError(
                f"row length {len(row)} does not match column count {len(columns)} "
                f"for table {table!r}"
            )
        batch.append(row)
        if len(batch) >= batch_size:
            _flush_batch(conn, table=table, col_list=col_list,
                         row_placeholder=row_placeholder, batch=batch)
            total += len(batch)
            batch = []

    if batch:
        _flush_batch(conn, table=table, col_list=col_list,
                     row_placeholder=row_placeholder, batch=batch)
        total += len(batch)

    return total


def _flush_batch(
    conn,
    *,
    table: str,
    col_list: str,
    row_placeholder: str,
    batch: list[Sequence[Any]],
) -> None:
    values_clause = ", ".join([row_placeholder] * len(batch))
    sql = (
        f"INSERT OR REPLACE INTO {table} ({col_list}) "
        f"VALUES {values_clause}"
    )
    # libsql_experimental only accepts tuples for ``parameters``; sqlite3
    # accepts any sequence. Normalise to tuple for backend portability.
    flat_params: list[Any] = []
    for row in batch:
        flat_params.extend(row)
    conn.execute(sql, tuple(flat_params))


def _import_messages(conn, rows: list[dict[str, Any]]) -> int:
    return _bulk_upsert(
        conn,
        table="messages",
        columns=("id", "ts_ns", "role", "content_json"),
        rows=(
            (
                str(row["id"]),
                int(row["ts_ns"]),
                str(row["role"]),
                json.dumps(row["content"], separators=(",", ":")),
            )
            for row in rows
        ),
    )


def _import_one_minute(conn, rows: list[dict[str, Any]]) -> int:
    return _bulk_upsert(
        conn,
        table="one_minute_summaries",
        columns=("id", "date", "hour", "minute", "summary", "message_count", "created_at_ns"),
        rows=(_dhm_values(row) for row in rows),
    )


def _import_five_minute(conn, rows: list[dict[str, Any]]) -> int:
    return _bulk_upsert(
        conn,
        table="five_minute_summaries",
        columns=("id", "date", "hour", "minute", "summary", "message_count", "created_at_ns"),
        rows=(_dhm_values(row) for row in rows),
    )


def _import_hourly(conn, rows: list[dict[str, Any]]) -> int:
    return _bulk_upsert(
        conn,
        table="hourly_summaries",
        columns=("id", "date", "hour", "summary", "message_count", "created_at_ns"),
        rows=(
            (
                str(row["id"]),
                str(row["date"]),
                int(row["hour"]),
                str(row["summary"]),
                int(row["message_count"]),
                int(row["created_at_ns"]),
            )
            for row in rows
        ),
    )


def _import_daily(conn, rows: list[dict[str, Any]]) -> int:
    return _bulk_upsert(
        conn,
        table="daily_summaries",
        columns=("id", "date", "summary", "message_count", "created_at_ns"),
        rows=(_date_values(row) for row in rows),
    )


def _import_weekly(conn, rows: list[dict[str, Any]]) -> int:
    return _bulk_upsert(
        conn,
        table="weekly_summaries",
        columns=("id", "week_start_date", "summary", "message_count", "created_at_ns"),
        rows=(
            (
                str(row["id"]),
                str(row["week_start_date"]),
                str(row["summary"]),
                int(row["message_count"]),
                int(row["created_at_ns"]),
            )
            for row in rows
        ),
    )


def _import_monthly(conn, rows: list[dict[str, Any]]) -> int:
    return _bulk_upsert(
        conn,
        table="monthly_summaries",
        columns=("id", "year", "month", "summary", "message_count", "created_at_ns"),
        rows=(
            (
                str(row["id"]),
                int(row["year"]),
                int(row["month"]),
                str(row["summary"]),
                int(row["message_count"]),
                int(row["created_at_ns"]),
            )
            for row in rows
        ),
    )


def _dhm_values(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["id"]),
        str(row["date"]),
        int(row["hour"]),
        int(row["minute"]),
        str(row["summary"]),
        int(row["message_count"]),
        int(row["created_at_ns"]),
    )


def _date_values(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["id"]),
        str(row["date"]),
        str(row["summary"]),
        int(row["message_count"]),
        int(row["created_at_ns"]),
    )

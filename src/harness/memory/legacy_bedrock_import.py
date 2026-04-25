"""Import legacy Bedrock memory exports into harness-owned storage."""

from __future__ import annotations

import json
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


def _import_messages(conn, rows: list[dict[str, Any]]) -> int:
    conn.executemany(
        """
        INSERT OR REPLACE INTO messages (id, ts_ns, role, content_json)
        VALUES (?, ?, ?, ?)
        """,
        [
            (
                str(row["id"]),
                int(row["ts_ns"]),
                str(row["role"]),
                json.dumps(row["content"], separators=(",", ":")),
            )
            for row in rows
        ],
    )
    return len(rows)


def _import_one_minute(conn, rows: list[dict[str, Any]]) -> int:
    conn.executemany(
        """
        INSERT OR REPLACE INTO one_minute_summaries
        (id, date, hour, minute, summary, message_count, created_at_ns)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [_dhm_values(row) for row in rows],
    )
    return len(rows)


def _import_five_minute(conn, rows: list[dict[str, Any]]) -> int:
    conn.executemany(
        """
        INSERT OR REPLACE INTO five_minute_summaries
        (id, date, hour, minute, summary, message_count, created_at_ns)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [_dhm_values(row) for row in rows],
    )
    return len(rows)


def _import_hourly(conn, rows: list[dict[str, Any]]) -> int:
    conn.executemany(
        """
        INSERT OR REPLACE INTO hourly_summaries
        (id, date, hour, summary, message_count, created_at_ns)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                str(row["id"]),
                str(row["date"]),
                int(row["hour"]),
                str(row["summary"]),
                int(row["message_count"]),
                int(row["created_at_ns"]),
            )
            for row in rows
        ],
    )
    return len(rows)


def _import_daily(conn, rows: list[dict[str, Any]]) -> int:
    conn.executemany(
        """
        INSERT OR REPLACE INTO daily_summaries
        (id, date, summary, message_count, created_at_ns)
        VALUES (?, ?, ?, ?, ?)
        """,
        [_date_values(row) for row in rows],
    )
    return len(rows)


def _import_weekly(conn, rows: list[dict[str, Any]]) -> int:
    conn.executemany(
        """
        INSERT OR REPLACE INTO weekly_summaries
        (id, week_start_date, summary, message_count, created_at_ns)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                str(row["id"]),
                str(row["week_start_date"]),
                str(row["summary"]),
                int(row["message_count"]),
                int(row["created_at_ns"]),
            )
            for row in rows
        ],
    )
    return len(rows)


def _import_monthly(conn, rows: list[dict[str, Any]]) -> int:
    conn.executemany(
        """
        INSERT OR REPLACE INTO monthly_summaries
        (id, year, month, summary, message_count, created_at_ns)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                str(row["id"]),
                int(row["year"]),
                int(row["month"]),
                str(row["summary"]),
                int(row["message_count"]),
                int(row["created_at_ns"]),
            )
            for row in rows
        ],
    )
    return len(rows)


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

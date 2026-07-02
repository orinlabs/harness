"""Tests for `harness export-memory`.

Production use: Bedrock's VAPI voice webhook execs this command in the
agent's sandbox during an `assistant-request` and splices the text
between the sentinel markers into the voice assistant's system prompt.
The contract that matters:

  * the rendered tier summaries appear between the BEGIN/END markers
    on stdout (log noise on the same stream must stay outside them),
  * raw messages are never included (the webhook supplies its own
    recent-conversation context; the raw log contains internal tool
    traffic), and
  * an agent with no memory yields an empty block, not a failure.
"""

from __future__ import annotations

import importlib
import time
import uuid
from datetime import UTC, datetime

from harness.memory.export import (
    EXPORT_BEGIN_MARKER,
    EXPORT_END_MARKER,
    export_memory_context,
)


def _fresh_storage(tmp_path, monkeypatch):
    from harness.core import storage

    importlib.reload(storage)
    monkeypatch.setattr(storage, "_STORAGE_ROOT", tmp_path / "storage")
    return storage


def _insert(conn, table: str, cols: dict) -> None:
    keys = ", ".join(cols)
    ph = ", ".join("?" for _ in cols)
    conn.execute(
        f"INSERT INTO {table} (id, {keys}, created_at_ns) VALUES (?, {ph}, ?)",
        (str(uuid.uuid4()), *cols.values(), time.time_ns()),
    )


def test_export_memory_context_renders_all_tiers(tmp_path, monkeypatch):
    """Every summary tier inside its window is rendered; raw messages are not.

    Windows for now=2026-07-01T12:33Z (UTC buckets):
      5-minute [11:30, 12:30), hourly [06-30 00:00, 07-01 11:00),
      daily [06-21, 06-30), weekly [05-01, 06-21),
      monthly strictly before 2026-05.
    """
    storage = _fresh_storage(tmp_path, monkeypatch)
    conn = storage.load("agent-export-tiers")

    _insert(
        conn,
        "five_minute_summaries",
        {
            "date": "2026-07-01",
            "hour": 12,
            "minute": 10,
            "summary": "5m tier row",
            "message_count": 3,
        },
    )
    _insert(
        conn,
        "hourly_summaries",
        {"date": "2026-06-30", "hour": 5, "summary": "hourly tier row", "message_count": 10},
    )
    _insert(
        conn,
        "daily_summaries",
        {"date": "2026-06-25", "summary": "daily tier row", "message_count": 40},
    )
    _insert(
        conn,
        "weekly_summaries",
        {"week_start_date": "2026-05-31", "summary": "weekly tier row", "message_count": 200},
    )
    _insert(
        conn,
        "monthly_summaries",
        {"year": 2026, "month": 3, "summary": "monthly tier row", "message_count": 900},
    )
    conn.execute(
        "INSERT INTO messages (id, ts_ns, role, content_json) VALUES (?, ?, ?, ?)",
        ("m1", time.time_ns(), "user", '{"role": "user", "content": "raw secret message"}'),
    )
    storage.flush()

    rendered = export_memory_context(
        timezone_name="UTC",
        current_time=datetime(2026, 7, 1, 12, 33, tzinfo=UTC),
    )

    assert "5m tier row" in rendered
    assert "hourly tier row" in rendered
    assert "daily tier row" in rendered
    assert "weekly tier row" in rendered
    assert "monthly tier row" in rendered
    assert "raw secret message" not in rendered

    storage.close()


def test_export_memory_cli_wraps_payload_in_markers(tmp_path, monkeypatch, capsys):
    storage_root = tmp_path / "storage"
    monkeypatch.chdir(tmp_path)

    storage = _fresh_storage(tmp_path, monkeypatch)
    monkeypatch.setattr(storage, "_STORAGE_ROOT", storage_root)
    conn = storage.load("agent-cli-export")
    # A 2020 monthly summary is strictly before any current monthly
    # boundary, so it is included regardless of when the test runs.
    _insert(
        conn,
        "monthly_summaries",
        {"year": 2020, "month": 1, "summary": "January 2020 things happened", "message_count": 5},
    )
    conn.execute(
        "INSERT INTO messages (id, ts_ns, role, content_json) VALUES (?, ?, ?, ?)",
        ("m1", time.time_ns(), "user", '{"role": "user", "content": "raw secret message"}'),
    )
    storage.flush()
    storage.close()

    from harness.cli import main

    assert main(["export-memory", "agent-cli-export", "--log-level", "ERROR"]) == 0

    out = capsys.readouterr().out
    assert EXPORT_BEGIN_MARKER in out
    assert EXPORT_END_MARKER in out
    payload = out.split(EXPORT_BEGIN_MARKER, 1)[1].split(EXPORT_END_MARKER, 1)[0]
    assert "January 2020 things happened" in payload
    assert "raw secret message" not in payload


def test_export_memory_cli_empty_agent_yields_empty_block(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    storage = _fresh_storage(tmp_path, monkeypatch)
    assert storage is not None

    from harness.cli import main

    assert main(["export-memory", "agent-cli-export-empty", "--log-level", "ERROR"]) == 0

    out = capsys.readouterr().out
    payload = out.split(EXPORT_BEGIN_MARKER, 1)[1].split(EXPORT_END_MARKER, 1)[0]
    assert payload.strip() == ""


def _seed_all_tiers(storage, agent_id: str, *, five_minute_pad: int = 300):
    """Seed one row per coarse tier plus three 5-minute rows (oldest to
    newest) with padded summaries so budget math has something to trim.

    Window-compatible with now=2026-07-01T12:33Z, matching
    ``test_export_memory_context_renders_all_tiers``.
    """
    conn = storage.load(agent_id)
    for minute, label in ((0, "5m-old"), (5, "5m-mid"), (10, "5m-new")):
        _insert(
            conn,
            "five_minute_summaries",
            {
                "date": "2026-07-01",
                "hour": 12,
                "minute": minute,
                "summary": label + " " + "x" * five_minute_pad,
                "message_count": 3,
            },
        )
    _insert(
        conn,
        "hourly_summaries",
        {"date": "2026-06-30", "hour": 5, "summary": "hourly tier row", "message_count": 10},
    )
    _insert(
        conn,
        "daily_summaries",
        {"date": "2026-06-25", "summary": "daily tier row", "message_count": 40},
    )
    _insert(
        conn,
        "weekly_summaries",
        {"week_start_date": "2026-05-31", "summary": "weekly tier row", "message_count": 200},
    )
    _insert(
        conn,
        "monthly_summaries",
        {"year": 2026, "month": 3, "summary": "monthly tier row", "message_count": 900},
    )
    storage.flush()


def test_export_max_tokens_drops_finest_tier_first(tmp_path, monkeypatch):
    """A tight budget removes the whole 5-minute tier (header included)
    before touching any coarser tier."""
    storage = _fresh_storage(tmp_path, monkeypatch)
    _seed_all_tiers(storage, "agent-budget-tiers")

    rendered = export_memory_context(
        timezone_name="UTC",
        current_time=datetime(2026, 7, 1, 12, 33, tzinfo=UTC),
        max_tokens=100,  # ~400 chars: forces all three padded 5m rows out
    )

    assert "monthly tier row" in rendered
    assert "weekly tier row" in rendered
    assert "daily tier row" in rendered
    assert "hourly tier row" in rendered
    assert "5m-" not in rendered
    assert "=== 5-MINUTE SUMMARIES ===" not in rendered

    storage.close()


def test_export_max_tokens_trims_oldest_entries_within_tier(tmp_path, monkeypatch):
    """Partial trims drop the oldest entries of the finest tier first,
    keeping the freshest detail."""
    storage = _fresh_storage(tmp_path, monkeypatch)
    _seed_all_tiers(storage, "agent-budget-oldest")

    rendered = export_memory_context(
        timezone_name="UTC",
        current_time=datetime(2026, 7, 1, 12, 33, tzinfo=UTC),
        max_tokens=180,  # ~720 chars: room for one padded 5m row
    )

    assert "5m-new" in rendered
    assert "5m-old" not in rendered
    assert "monthly tier row" in rendered

    storage.close()


def test_export_without_max_tokens_is_unchanged(tmp_path, monkeypatch):
    """Under-budget data (the default path) renders every tier untouched."""
    storage = _fresh_storage(tmp_path, monkeypatch)
    _seed_all_tiers(storage, "agent-budget-default")

    kwargs = {
        "timezone_name": "UTC",
        "current_time": datetime(2026, 7, 1, 12, 33, tzinfo=UTC),
    }
    default_render = export_memory_context(**kwargs)
    explicit_none = export_memory_context(**kwargs, max_tokens=None)

    assert default_render == explicit_none
    for label in ("5m-old", "5m-mid", "5m-new", "hourly", "daily", "weekly", "monthly"):
        assert label in default_render

    storage.close()


def test_export_memory_cli_max_tokens_flag_reaches_renderer(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    storage = _fresh_storage(tmp_path, monkeypatch)
    conn = storage.load("agent-cli-budget")
    _insert(
        conn,
        "monthly_summaries",
        {"year": 2020, "month": 1, "summary": "January 2020 things happened", "message_count": 5},
    )
    _insert(
        conn,
        "monthly_summaries",
        {"year": 2020, "month": 2, "summary": "February 2020 things happened", "message_count": 5},
    )
    storage.flush()
    storage.close()

    from harness.cli import main

    # ~80-char budget: fits the header plus one entry, so the oldest
    # month is dropped and the newest survives.
    assert (
        main(
            [
                "export-memory",
                "agent-cli-budget",
                "--max-tokens",
                "20",
                "--log-level",
                "ERROR",
            ]
        )
        == 0
    )

    out = capsys.readouterr().out
    payload = out.split(EXPORT_BEGIN_MARKER, 1)[1].split(EXPORT_END_MARKER, 1)[0]
    assert "February 2020 things happened" in payload
    assert "January 2020 things happened" not in payload

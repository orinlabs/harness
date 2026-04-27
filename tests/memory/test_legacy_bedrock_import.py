from __future__ import annotations

import importlib
import json
from pathlib import Path


def test_import_legacy_bedrock_memory_writes_per_agent_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "HARNESS_MIGRATIONS_DIR",
        str(Path(__file__).parent.parent.parent / "src" / "harness" / "memory" / "migrations"),
    )

    from harness.core import storage as storage_module

    importlib.reload(storage_module)

    from harness.memory.legacy_bedrock_import import import_legacy_bedrock_memory

    payload_path = tmp_path / "agent-1.json"
    payload_path.write_text(
        json.dumps(
            {
                "agent_id": "agent-1",
                "messages": [
                    {
                        "id": "bedrock:1",
                        "ts_ns": 1_700_000_000_000_000_000,
                        "role": "user",
                        "content": {"role": "user", "content": "remember the blue door"},
                    }
                ],
                "one_minute_summaries": [
                    {
                        "id": "bedrock:one",
                        "date": "2026-04-24",
                        "hour": 10,
                        "minute": 30,
                        "summary": "one minute",
                        "message_count": 1,
                        "created_at_ns": 1,
                    }
                ],
                "five_minute_summaries": [],
                "hourly_summaries": [],
                "daily_summaries": [
                    {
                        "id": "bedrock:daily",
                        "date": "2026-04-24",
                        "summary": "daily",
                        "message_count": 1,
                        "created_at_ns": 2,
                    }
                ],
                "weekly_summaries": [],
                "monthly_summaries": [],
            }
        )
    )

    counts = import_legacy_bedrock_memory("agent-1", payload_path)
    assert counts.messages == 1
    assert counts.one_minute_summaries == 1
    assert counts.daily_summaries == 1

    conn = storage_module.load("agent-1")
    try:
        message = conn.execute("SELECT role, content_json FROM messages").fetchone()
        assert message["role"] == "user"
        assert json.loads(message["content_json"]) == {
            "role": "user",
            "content": "remember the blue door",
        }
        assert conn.execute("SELECT COUNT(*) AS c FROM daily_summaries").fetchone()["c"] == 1
    finally:
        storage_module.close()


def test_import_legacy_bedrock_memory_rejects_wrong_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_STORAGE_ROOT", str(tmp_path))

    from harness.memory.legacy_bedrock_import import import_legacy_bedrock_memory

    payload_path = tmp_path / "agent-1.json"
    payload_path.write_text(json.dumps({"agent_id": "agent-2"}))

    try:
        import_legacy_bedrock_memory("agent-1", payload_path)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("expected wrong-agent payload to fail")


def test_import_legacy_bedrock_memory_batches_across_chunk_boundary(
    tmp_path, monkeypatch
):
    """Imports that exceed BULK_INSERT_BATCH_ROWS must land all rows.

    Regression guard: the legacy per-row ``executemany()`` implementation was
    replaced with a chunked multi-row VALUES path. An off-by-one in the chunk
    flush — e.g. forgetting the tail batch, or emitting a malformed VALUES
    clause on odd-sized tails — would silently drop rows. Force enough rows
    to span at least 2 full batches and a partial tail, then assert every
    row round-trips back out of storage.
    """
    monkeypatch.setenv("HARNESS_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "HARNESS_MIGRATIONS_DIR",
        str(Path(__file__).parent.parent.parent / "src" / "harness" / "memory" / "migrations"),
    )

    from harness.core import storage as storage_module

    importlib.reload(storage_module)

    from harness.memory import legacy_bedrock_import
    from harness.memory.legacy_bedrock_import import import_legacy_bedrock_memory

    # Pick a row count that is not a multiple of BULK_INSERT_BATCH_ROWS so the
    # tail-flush path is exercised too. 2 full batches + a partial.
    total_messages = legacy_bedrock_import.BULK_INSERT_BATCH_ROWS * 2 + 7
    total_one_minute = legacy_bedrock_import.BULK_INSERT_BATCH_ROWS + 3

    messages = [
        {
            "id": f"bedrock:msg:{i}",
            "ts_ns": 1_700_000_000_000_000_000 + i,
            "role": "user" if i % 2 == 0 else "assistant",
            "content": {"role": "user", "content": f"msg {i}"},
        }
        for i in range(total_messages)
    ]
    one_minute = [
        {
            "id": f"bedrock:om:{i}",
            "date": "2026-04-24",
            "hour": (i // 60) % 24,
            "minute": i % 60,
            "summary": f"one minute {i}",
            "message_count": i,
            "created_at_ns": i,
        }
        for i in range(total_one_minute)
    ]

    payload_path = tmp_path / "agent-bulk.json"
    payload_path.write_text(
        json.dumps(
            {
                "agent_id": "agent-bulk",
                "messages": messages,
                "one_minute_summaries": one_minute,
                "five_minute_summaries": [],
                "hourly_summaries": [],
                "daily_summaries": [],
                "weekly_summaries": [],
                "monthly_summaries": [],
            }
        )
    )

    counts = import_legacy_bedrock_memory("agent-bulk", payload_path)
    assert counts.messages == total_messages
    assert counts.one_minute_summaries == total_one_minute

    conn = storage_module.load("agent-bulk")
    try:
        assert (
            conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"]
            == total_messages
        )
        assert (
            conn.execute("SELECT COUNT(*) AS c FROM one_minute_summaries").fetchone()[
                "c"
            ]
            == total_one_minute
        )
        # Spot-check that the first, a middle, and the last row all round-trip
        # intact — proves the batched VALUES stream correctly binds per-row
        # parameters rather than bleeding across rows.
        first = conn.execute(
            "SELECT role, content_json FROM messages WHERE id = ?",
            ["bedrock:msg:0"],
        ).fetchone()
        assert first["role"] == "user"
        assert json.loads(first["content_json"]) == {"role": "user", "content": "msg 0"}

        middle_index = legacy_bedrock_import.BULK_INSERT_BATCH_ROWS + 3
        middle = conn.execute(
            "SELECT role FROM messages WHERE id = ?",
            [f"bedrock:msg:{middle_index}"],
        ).fetchone()
        assert middle is not None
        assert middle["role"] == ("user" if middle_index % 2 == 0 else "assistant")

        last = conn.execute(
            "SELECT role FROM messages WHERE id = ?",
            [f"bedrock:msg:{total_messages - 1}"],
        ).fetchone()
        assert last is not None
    finally:
        storage_module.close()


def test_import_legacy_bedrock_memory_is_idempotent(tmp_path, monkeypatch):
    """Re-running the same import must not duplicate rows.

    Prod-deploy migrations may retry per-agent imports (timeout, pod restart,
    operator rerun). Legacy ids are stable, and the upsert path relies on
    ``INSERT OR REPLACE``. A regression that regresses back to plain INSERT
    — or that forgets to include the primary key in the VALUES clause — would
    leak into row-count doubling on replay.
    """
    monkeypatch.setenv("HARNESS_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "HARNESS_MIGRATIONS_DIR",
        str(Path(__file__).parent.parent.parent / "src" / "harness" / "memory" / "migrations"),
    )

    from harness.core import storage as storage_module

    importlib.reload(storage_module)

    from harness.memory.legacy_bedrock_import import import_legacy_bedrock_memory

    payload = {
        "agent_id": "agent-replay",
        "messages": [
            {
                "id": "bedrock:replay:1",
                "ts_ns": 1,
                "role": "user",
                "content": {"role": "user", "content": "once"},
            }
        ],
        "one_minute_summaries": [
            {
                "id": "bedrock:om:1",
                "date": "2026-04-24",
                "hour": 1,
                "minute": 2,
                "summary": "first",
                "message_count": 1,
                "created_at_ns": 10,
            }
        ],
        "five_minute_summaries": [],
        "hourly_summaries": [],
        "daily_summaries": [],
        "weekly_summaries": [],
        "monthly_summaries": [],
    }
    payload_path = tmp_path / "agent-replay.json"
    payload_path.write_text(json.dumps(payload))

    import_legacy_bedrock_memory("agent-replay", payload_path)
    # Mutate the summary text so REPLACE is observable and we can distinguish
    # the second run from the first.
    payload["one_minute_summaries"][0]["summary"] = "second"
    payload_path.write_text(json.dumps(payload))
    import_legacy_bedrock_memory("agent-replay", payload_path)

    conn = storage_module.load("agent-replay")
    try:
        assert conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"] == 1
        assert (
            conn.execute("SELECT COUNT(*) AS c FROM one_minute_summaries").fetchone()[
                "c"
            ]
            == 1
        )
        assert (
            conn.execute(
                "SELECT summary FROM one_minute_summaries WHERE id = ?",
                ["bedrock:om:1"],
            ).fetchone()["summary"]
            == "second"
        )
    finally:
        storage_module.close()

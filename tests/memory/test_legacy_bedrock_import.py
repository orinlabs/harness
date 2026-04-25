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

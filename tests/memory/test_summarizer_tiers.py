"""Tests for summarizer tier input handling."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from harness.core.llm import LLMResponse, Usage
from harness.memory.summarizer import SummaryUpdater


@pytest.fixture
def storage_env(tmp_path, monkeypatch):
    import importlib

    mig_dir = Path(__file__).parent.parent.parent / "src" / "harness" / "memory" / "migrations"
    monkeypatch.setenv("HARNESS_MIGRATIONS_DIR", str(mig_dir))

    from harness.core import storage as storage_module

    importlib.reload(storage_module)
    monkeypatch.setattr(storage_module, "_STORAGE_ROOT", tmp_path)
    storage_module.load("agent-summarizer-tiers")
    try:
        yield storage_module
    finally:
        storage_module.close()


def _insert_message(storage_env, ts: datetime, msg: dict) -> None:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    ts_ns = int(ts.timestamp() * 1_000_000_000)
    storage_env.db.execute(
        "INSERT INTO messages (id, ts_ns, role, content_json) VALUES (?, ?, ?, ?)",
        (str(uuid.uuid4()), ts_ns, str(msg.get("role") or "user"), json.dumps(msg)),
    )


def _fake_llm_response() -> LLMResponse:
    return LLMResponse(
        text="I reviewed the photo.",
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            total_cost=0.0,
            model="openai/gpt-5-nano",
        ),
        raw={"choices": [{"message": {"content": "I reviewed the photo."}}]},
    )


def _prompt_from_llm_mock(mock_llm) -> str:
    messages = mock_llm.call_args.kwargs["messages"]
    return messages[0]["content"]


def test_five_minute_tier_strips_image_bytes_before_llm(storage_env):
    now = datetime(2026, 5, 19, 12, 7, tzinfo=UTC)
    ts = now - timedelta(minutes=10)
    _insert_message(
        storage_env,
        ts,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "site photo"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + ("Z" * 50_000)},
                },
            ],
        },
    )

    updater = SummaryUpdater(timezone_name="UTC", model="openai/gpt-5-nano")

    with patch("harness.memory.summarizer.llm.complete", return_value=_fake_llm_response()) as mock_llm:
        created = updater._update_five_minute_summaries(now)

    assert created
    prompt = _prompt_from_llm_mock(mock_llm)
    assert "ZZZZ" not in prompt
    assert "[image attachment]" in prompt

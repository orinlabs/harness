"""Assistant reasoning must be persisted to memory so the summarizer can
use it -- but stripped at replay so the next agent turn doesn't re-send
provider-specific reasoning fields back to the model.

Locks in the contract introduced alongside the summarizer sanitize
fix: ``harness.harness._step`` logs the **raw** assistant message
(including ``reasoning`` plaintext and ``reasoning_details``) into
SQLite, and ``llm.complete()`` / ``_prepare_replay_messages`` continues
to strip ``reasoning`` (but keep ``reasoning_details``) on the way out
to OpenRouter on subsequent turns.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from harness.core.llm import LLMResponse, ToolCall, Usage


@pytest.fixture
def harness_storage(tmp_path, monkeypatch):
    """Fresh sqlite + applied migrations for the test, scoped to tmp_path.

    Same pattern ``tests/memory/conftest.py`` uses: reload the storage
    module so any cached connection is dropped, then monkeypatch its
    storage root to ``tmp_path`` so the test never touches the
    developer's real ``~/.harness/agents`` files.
    """
    mig_dir = Path(__file__).parent.parent / "src/harness/memory/migrations"
    monkeypatch.setenv("HARNESS_MIGRATIONS_DIR", str(mig_dir))

    from harness.core import storage

    importlib.reload(storage)
    monkeypatch.setattr(storage, "_STORAGE_ROOT", tmp_path)
    yield
    try:
        storage.close()
    except Exception:
        pass


class _RecordingLLM:
    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(
            {
                "model": kwargs.get("model"),
                "messages": [dict(m) for m in kwargs.get("messages", [])],
            }
        )
        if not self._responses:
            raise RuntimeError("RecordingLLM ran out of programmed responses")
        return self._responses.pop(0)


_REASONING_TEXT = "Step 1: I need to call sleep so the loop exits."
_REASONING_SIGNATURE = "anthropic-opaque-signature-blob"


def _assistant_with_reasoning(*, tool_calls: list[dict]) -> LLMResponse:
    raw = {
        "id": "rec-1",
        "model": "test/recording",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": "{}",
                            },
                        }
                        for tc in tool_calls
                    ],
                    "reasoning": _REASONING_TEXT,
                    "reasoning_details": [
                        {
                            "type": "reasoning.text",
                            "text": _REASONING_TEXT,
                            "signature": _REASONING_SIGNATURE,
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return LLMResponse(
        text="",
        tool_calls=[ToolCall(id=tc["id"], name=tc["name"], args={}) for tc in tool_calls],
        finish_reason="tool_calls",
        reasoning=_REASONING_TEXT,
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            total_cost=0.0,
            model="test/recording",
            llm_calls=1,
            reasoning_tokens=4,
        ),
        raw=raw,
    )


def test_assistant_reasoning_persisted_to_memory_but_stripped_on_replay(
    harness_storage, monkeypatch
):
    from harness import AgentConfig
    from harness.core import llm as llm_mod
    from harness.core import storage as storage_mod
    from harness.harness import Harness

    recording = _RecordingLLM(
        [_assistant_with_reasoning(tool_calls=[{"id": "call-sleep-1", "name": "sleep"}])]
    )
    monkeypatch.setattr(llm_mod, "complete", recording)

    config = AgentConfig(
        id="agent-reasoning-test",
        model="test/recording",
        system_prompt="test agent",
    )

    Harness(config, run_id="run-reasoning-1").run()

    # Harness.run() closes the per-agent sqlite handle in its finally
    # block (simulates process exit). Re-open it to read the persisted
    # rows -- same pattern test_memory's "messages survive reopen" uses.
    storage_mod.load("agent-reasoning-test")
    try:
        rows = storage_mod.db.execute(
            "SELECT role, content_json FROM messages "
            "WHERE role = 'assistant' ORDER BY ts_ns"
        ).fetchall()
    finally:
        storage_mod.close()
    assert rows, "expected an assistant message in memory"
    assistant = json.loads(rows[0]["content_json"])
    assert assistant["reasoning"] == _REASONING_TEXT
    details = assistant["reasoning_details"]
    assert isinstance(details, list) and len(details) == 1
    # Signature is persisted as the model returned it. Sanitize only runs
    # at summarization time.
    assert details[0]["text"] == _REASONING_TEXT
    assert details[0]["signature"] == _REASONING_SIGNATURE


def test_persisted_assistant_strips_reasoning_via_replay_prepare(
    harness_storage, monkeypatch
):
    """End-to-end check: the message harness persists into memory still
    passes cleanly through ``_prepare_replay_messages`` -- ``reasoning``
    plaintext goes away (Anthropic / OpenRouter reject replays carrying
    it), and ``reasoning_details`` survives so signed thinking blocks can
    continue extended thinking across tool calls.

    The strip itself lives in ``harness.core.llm`` and is already covered
    by ``tests/test_llm.py``; this test wires the persistence side
    (``Harness._step`` logs the raw message) through that same strip to
    lock in the round trip.
    """
    from harness import AgentConfig
    from harness.core import llm as llm_mod
    from harness.core import storage as storage_mod
    from harness.harness import Harness

    recording = _RecordingLLM(
        [_assistant_with_reasoning(tool_calls=[{"id": "call-sleep-1", "name": "sleep"}])]
    )
    monkeypatch.setattr(llm_mod, "complete", recording)

    config = AgentConfig(
        id="agent-replay-test",
        model="test/recording",
        system_prompt="test agent",
    )
    Harness(config, run_id="run-reasoning-2").run()

    storage_mod.load("agent-replay-test")
    try:
        rows = storage_mod.db.execute(
            "SELECT content_json FROM messages WHERE role = 'assistant' ORDER BY ts_ns"
        ).fetchall()
    finally:
        storage_mod.close()

    assert rows, "no assistant message in memory"
    persisted = json.loads(rows[0]["content_json"])
    # Sanity: reasoning was preserved on disk (so the summarizer can use it).
    assert persisted.get("reasoning") == _REASONING_TEXT

    [replay] = llm_mod._prepare_replay_messages([persisted])
    assert "reasoning" not in replay, replay
    assert replay.get("reasoning_details") == [
        {
            "type": "reasoning.text",
            "text": _REASONING_TEXT,
            "signature": _REASONING_SIGNATURE,
        }
    ]

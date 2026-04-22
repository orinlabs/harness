"""Chunk 8: external adapter end-to-end.

Proves the full platform-injected tool path: AgentConfig.adapters -> tool_map ->
model tool_call -> HTTP POST to fake_platform -> handler -> result back into
memory -> next turn sees the tool-result and proceeds.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

CHEAP_MODEL = "openai/gpt-4o-mini"


@pytest.fixture
def harness_env(tmp_path, monkeypatch, openrouter_key, fake_platform):
    monkeypatch.setenv("HARNESS_STORAGE_ROOT", str(tmp_path))
    mig_dir = Path(__file__).parent.parent / "src/harness/memory/migrations"
    monkeypatch.setenv("HARNESS_MIGRATIONS_DIR", str(mig_dir))

    from harness.core import llm, runtime_api, storage, tracer

    importlib.reload(storage)
    importlib.reload(tracer)
    importlib.reload(runtime_api)
    importlib.reload(llm)

    yield fake_platform


def test_external_adapter_tool_end_to_end(harness_env):
    from harness import AdapterConfig, AgentConfig, ExternalToolSpec, Harness

    echo_calls: list[dict] = []

    def echo_handler(args, envelope):
        echo_calls.append({"args": args, "envelope": envelope})
        return {"text": str(args.get("text", "")).upper()}

    harness_env.register_tool("echo", echo_handler)

    echo_spec = ExternalToolSpec(
        name="echo",
        description="Echoes the provided text uppercased.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        url=f"{harness_env.url}/fake_tools/echo",
    )

    config = AgentConfig(
        id="agent-adapter",
        model=CHEAP_MODEL,
        system_prompt=(
            'You have two tools: `echo(text)` and `sleep`. On your FIRST turn, call '
            '`echo` with text="hello". On your NEXT turn, read the echo result and '
            'then call `sleep` with until="2099-01-01T00:00:00Z" and reason equal '
            'to the exact echo result you received.'
        ),
        adapters=[
            AdapterConfig(
                name="test-adapter",
                description="Test-only echo tool.",
                tools=[echo_spec],
            )
        ],
    )

    Harness(config, run_id="run-adapter").run()

    assert len(echo_calls) == 1, f"expected exactly one echo call, got {len(echo_calls)}"
    call = echo_calls[0]
    assert call["args"] == {"text": "hello"}
    assert call["envelope"]["agent_id"] == "agent-adapter"
    assert call["envelope"]["run_id"] == "run-adapter"

    echo_posts = [
        r
        for r in harness_env.requests
        if r.method == "POST" and r.path == "/fake_tools/echo"
    ]
    assert len(echo_posts) == 1
    assert echo_posts[0].headers.get("Authorization") == "Bearer test-token"

    tool_spans = [
        s for s in harness_env.spans_open.values() if s["name"] == "tool_call"
    ]
    echo_spans = [s for s in tool_spans if s["metadata"].get("tool_name") == "echo"]
    assert len(echo_spans) == 1

    assert len(harness_env.sleep_requests) == 1
    reason = harness_env.sleep_requests[0].get("reason", "")
    assert "HELLO" in reason.upper(), (
        f"sleep reason did not carry through the echo tool result: {reason!r}"
    )

    from harness.core import storage

    storage.load("agent-adapter")
    rows = storage.db.execute(
        "SELECT role, content_json FROM messages ORDER BY ts_ns"
    ).fetchall()
    storage.close()

    tool_msgs = [r for r in rows if r["role"] == "tool"]
    assert any("HELLO" in r["content_json"] for r in tool_msgs), (
        "expected at least one tool-result message containing 'HELLO'"
    )

"""Chunk 7: end-to-end harness integration test.

Real OpenRouter + real SQLite + fake platform (HTTP). The one test that catches
regressions across the whole system.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

CHEAP_MODEL = "openai/gpt-4o-mini"


@pytest.fixture
def harness_env(tmp_path, monkeypatch, openrouter_key, fake_platform):
    """Full setup: storage root, migrations, platform URL, reloaded modules."""
    monkeypatch.setenv("HARNESS_STORAGE_ROOT", str(tmp_path))
    mig_dir = Path(__file__).parent.parent / "src/harness/memory/migrations"
    monkeypatch.setenv("HARNESS_MIGRATIONS_DIR", str(mig_dir))

    from harness.core import llm, runtime_api, storage, tracer

    importlib.reload(storage)
    importlib.reload(tracer)
    importlib.reload(runtime_api)
    importlib.reload(llm)

    yield fake_platform


def test_harness_runs_and_sleeps(harness_env):
    """Ask the model to sleep. Assert the full loop plays out against real services."""
    from harness import AgentConfig, Harness

    config = AgentConfig(
        id="agent-e2e",
        model=CHEAP_MODEL,
        system_prompt=(
            "You are a test agent. Your only task is to call the `sleep` tool with "
            'until="2099-01-01T00:00:00Z" and reason="test complete". '
            "Do not reply with text before calling the tool."
        ),
    )

    Harness(config, run_id="run-e2e-1").run()

    run_spans = [
        s for s in harness_env.spans_open.values() if s["name"] == "run_agent"
    ]
    turn_spans = [
        s for s in harness_env.spans_open.values() if s["name"].startswith("turn_")
    ]
    llm_spans = [
        s for s in harness_env.spans_open.values() if s["span_type"] == "llm"
    ]
    tool_spans = [
        s for s in harness_env.spans_open.values() if s["span_type"] == "tool"
    ]

    assert len(run_spans) == 1
    assert len(turn_spans) >= 1
    assert len(llm_spans) >= 1
    assert len(tool_spans) >= 1

    assert len(harness_env.traces_open) == 1, "one run should produce exactly one trace"
    trace_id = next(iter(harness_env.traces_open))
    for s in harness_env.spans_open.values():
        assert s["trace_id"] == trace_id

    run_span = run_spans[0]
    assert run_span["span_type"] == "text"
    assert run_span["metadata"]["agent_id"] == "agent-e2e"
    assert run_span["metadata"]["run_id"] == "run-e2e-1"

    closed = {s_id: s for s_id, s in harness_env.spans_closed.items()}
    llm_closed = [closed[s["id"]] for s in llm_spans if s["id"] in closed]
    assert any(
        c["metadata"].get("llm_cost", {}).get("total_cost_usd", 0) > 0 for c in llm_closed
    ), f"no llm span reported llm_cost: {[c['metadata'] for c in llm_closed]}"

    # The run_agent span closes with an aggregate `usage` dict that includes
    # model_breakdown. Assert cost accumulated.
    run_closed = closed[run_span["id"]]
    assert run_closed["metadata"]["usage"]["total_cost_usd"] > 0
    assert run_closed["metadata"]["usage"]["model_breakdown"], "model_breakdown empty"

    sleep_tool_spans = [
        closed[s["id"]]
        for s in tool_spans
        if s["name"] == "sleep" and s["id"] in closed
    ]
    assert sleep_tool_spans, "no tool span named 'sleep'"

    assert len(harness_env.sleep_requests) == 1
    sr = harness_env.sleep_requests[0]
    assert sr["agent_id"] == "agent-e2e"
    assert sr["until"] == "2099-01-01T00:00:00Z"

    from harness.core import storage

    storage.load("agent-e2e")
    rows = storage.db.execute(
        "SELECT role, content_json FROM messages ORDER BY ts_ns"
    ).fetchall()
    storage.close()

    roles = [r["role"] for r in rows]
    assert "assistant" in roles, f"expected at least one assistant message, got {roles}"
    assert "tool" in roles, f"expected a tool-result message, got {roles}"


def test_harness_replays_prior_messages(harness_env):
    """Pre-populate a prior turn. Verify the next run sees it in memory."""
    from harness import AgentConfig, Harness
    from harness.core import storage
    from harness.memory import MemoryService

    storage.load("agent-replay")
    MemoryService(agent_id="agent-replay", model=CHEAP_MODEL).log_messages(
        [
            {"role": "user", "content": "Remember: my favorite number is 73."},
            {"role": "assistant", "content": "Noted, your favorite number is 73."},
        ]
    )
    storage.flush()
    storage.close()

    config = AgentConfig(
        id="agent-replay",
        model=CHEAP_MODEL,
        system_prompt=(
            "You are a memory-test agent. On your FIRST turn, call the `sleep` tool "
            'with until="2099-01-01T00:00:00Z" and reason containing the user'
            "'s favorite number you previously recorded."
        ),
    )

    Harness(config, run_id="run-replay").run()

    assert len(harness_env.sleep_requests) == 1
    reason = harness_env.sleep_requests[0].get("reason", "")
    assert "73" in reason, f"sleep reason did not reference recalled number: {reason!r}"


def test_harness_import_remains_lazy():
    """Importing `harness` must not eagerly pull the loop (keeps cold start cheap)."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import harness; import sys; "
            "eager = [m for m in sys.modules if m.startswith('harness.') and "
            "m not in {'harness.config', 'harness.context', 'harness.constants'}]; "
            "print(eager)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = result.stdout.strip()
    assert loaded == "[]", f"harness modules loaded during import: {loaded}"

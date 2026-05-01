"""Chunk 7: end-to-end harness integration test.

Real OpenRouter + real SQLite + fake platform (HTTP). The one test that catches
regressions across the whole system.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

CHEAP_MODEL = "openai/gpt-4o-mini"

# Cheapest reasoning model on OpenRouter we can use for the thinking-span
# regression test. Haiku 4.5 is an Anthropic thinking model and honors
# `reasoning.enabled: true` (returns plaintext reasoning + non-zero
# reasoning_tokens), which is exactly what the assertions below depend on.
CHEAP_THINKING_MODEL = "anthropic/claude-haiku-4.5"


@pytest.fixture
def harness_env(tmp_path, monkeypatch, openrouter_key, fake_platform):
    """Full setup: storage root, migrations, platform URL, reloaded modules."""
    mig_dir = Path(__file__).parent.parent / "src/harness/memory/migrations"
    monkeypatch.setenv("HARNESS_MIGRATIONS_DIR", str(mig_dir))

    from harness.core import llm, storage, tracer

    importlib.reload(storage)
    importlib.reload(tracer)
    importlib.reload(llm)
    monkeypatch.setattr(storage, "_STORAGE_ROOT", tmp_path)

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

    run_spans = [s for s in harness_env.spans_open.values() if s["name"] == "run_agent"]
    turn_spans = [s for s in harness_env.spans_open.values() if s["name"].startswith("turn_")]
    llm_spans = [s for s in harness_env.spans_open.values() if s["span_type"] == "llm"]
    tool_spans = [s for s in harness_env.spans_open.values() if s["span_type"] == "tool"]

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
        closed[s["id"]] for s in tool_spans if s["name"] == "sleep" and s["id"] in closed
    ]
    assert sleep_tool_spans, "no tool span named 'sleep'"

    assert len(harness_env.sleep_requests) == 1
    sr = harness_env.sleep_requests[0]
    assert sr["agent_id"] == "agent-e2e"
    assert sr["until"] == "2099-01-01T00:00:00Z"

    from harness.core import storage

    storage.load("agent-e2e")
    rows = storage.db.execute("SELECT role, content_json FROM messages ORDER BY ts_ns").fetchall()
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


def test_harness_surfaces_reasoning_on_llm_span_and_sibling_thinking_span(
    harness_env,
):
    """Thinking-model run must expose reasoning in two places on the trace:

    1. On the ``openrouter_api_call`` (llm_span) metadata: a top-level
       ``reasoning`` string and a top-level ``reasoning_tokens`` int.
       Users view traces in Bedrock and expect the LLM span itself to
       show what the model was thinking, not just a token count buried
       inside ``llm_cost``.

    2. As a sibling ``thinking`` TEXT span under the same ``turn_N``
       parent, carrying the plaintext reasoning in ``output_text``.

    Regression guard for:
      - The llm_span's ``s.output`` truncation at 20kB was swallowing
        the reasoning text on any reasoning-heavy turn, because
        OpenRouter streams ``reasoning_details`` as 1-token-per-delta
        dicts that blow past the 20kB cap.
      - A buggy client (or missing ``reasoning.enabled: true``) could
        silently produce zero reasoning tokens and nobody would
        notice in the Bedrock UI, since no sibling ``thinking`` span
        and no metadata.reasoning would be visible.
    """
    from harness import AgentConfig, Harness

    config = AgentConfig(
        id="agent-thinking",
        model=CHEAP_THINKING_MODEL,
        # Force a non-trivial computation before the tool call. Anthropic
        # thinking models routinely skip reasoning on "just call this
        # tool" prompts even with `reasoning.enabled: true` -- there's
        # nothing to reason about. Embedding a small arithmetic puzzle
        # in the reason field reliably triggers reasoning_tokens > 0 on
        # Haiku 4.5.
        system_prompt=(
            "You are a test agent. Solve this step by step before you do "
            "anything else: what is 17 * 23 * 41 + 9 minus 7 squared? "
            "Show your work. Then, on the SAME turn, call the `sleep` "
            'tool with until="2099-01-01T00:00:00Z" and reason set to '
            "the numerical answer you computed. Do not call the tool "
            "until you have solved the arithmetic."
        ),
    )

    Harness(config, run_id="run-thinking-1").run()

    llm_open = [s for s in harness_env.spans_open.values() if s["span_type"] == "llm"]
    assert llm_open, "no llm spans captured on thinking-model run"

    llm_closed = [
        harness_env.spans_closed[s["id"]] for s in llm_open if s["id"] in harness_env.spans_closed
    ]
    assert llm_closed, "llm spans were opened but never closed"

    # At least one llm_span must carry top-level reasoning metadata. We
    # check the closed-span metadata (the PATCH body) because that's
    # what Bedrock persists as final. Both `reasoning` (plaintext) and
    # `reasoning_tokens` (int > 0) must be present on thinking models.
    with_reasoning = [
        c
        for c in llm_closed
        if c["metadata"].get("reasoning") and c["metadata"].get("reasoning_tokens", 0) > 0
    ]
    assert with_reasoning, (
        "no llm_span exposed top-level `reasoning` + `reasoning_tokens` "
        "metadata. Found metadata keys per closed llm span: "
        f"{[sorted(c['metadata'].keys()) for c in llm_closed]}"
    )
    md = with_reasoning[0]["metadata"]
    assert isinstance(md["reasoning"], str) and len(md["reasoning"]) > 0
    assert isinstance(md["reasoning_tokens"], int)
    assert md["reasoning_tokens"] > 0

    # Sibling `thinking` span must exist, be TEXT-typed, share the
    # trace, and carry the reasoning plaintext in output_text.
    thinking_open = [s for s in harness_env.spans_open.values() if s["name"] == "thinking"]
    assert thinking_open, (
        "no sibling `thinking` span was emitted even though reasoning "
        "tokens were used. Span names opened: "
        f"{sorted({s['name'] for s in harness_env.spans_open.values()})}"
    )
    tspan = thinking_open[0]
    assert tspan["span_type"] == "text"
    assert tspan["metadata"].get("reasoning_tokens", 0) > 0
    assert tspan["metadata"].get("has_plaintext") is True

    tclosed = harness_env.spans_closed.get(tspan["id"])
    assert tclosed is not None, "thinking span opened but never closed"
    assert tclosed["output_text"], "thinking span closed with empty output"


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

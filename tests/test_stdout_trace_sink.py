"""StdoutTraceSink must make a standalone run reproducible from stdout alone.

Production use: a supervisor spawns the harness as a subprocess, holds its
stdout pipe, and line-matches for ``@@trace {...}`` to reconstruct the run --
span ordering from the pipe, nesting from ``parent_id``, thinking text and
tool args/results from the span payloads. Same contract as the ``@@sleep``
sentinel (``test_runtime_sleep_sentinel.py``).

The reconstruction test drives the *tracer* (``text_span`` / ``llm_span`` /
``tool_span`` / ``emit_completed_span``) with the exact span structure the
harness main loop emits -- run_agent > turn_N > (openrouter_api_call,
thinking, tool) -- captured at the OS file-descriptor level (``capfd``), the
same stream a parent process's pipe would see.
"""

from __future__ import annotations

import json
import re

from harness.core.tracing import (
    TRACE_SENTINEL_PREFIX,
    NullTraceSink,
    StdoutTraceSink,
)

# The parser a supervisor would use: anchored prefix, JSON payload after.
_SENTINEL_RE = re.compile(r"^@@trace (\{.*\})$", re.MULTILINE)


def _parse_events(stream_text: str) -> list[dict]:
    return [json.loads(m.group(1)) for m in _SENTINEL_RE.finditer(stream_text)]


# ---------------------------------------------------------------------------
# Sink selection (autoconfigure + CLI plumbing)
# ---------------------------------------------------------------------------


def test_autoconfigure_standalone_defaults_to_stdout_sink(monkeypatch):
    monkeypatch.delenv("BEDROCK_URL", raising=False)
    monkeypatch.delenv("BEDROCK_TOKEN", raising=False)
    monkeypatch.delenv("HARNESS_TRACE_SINK", raising=False)

    from harness.cloud.autoconfig import autoconfigure
    from harness.core.runtime import LocalAgentRuntime

    sink, runtime = autoconfigure()
    assert isinstance(sink, StdoutTraceSink)
    assert isinstance(runtime, LocalAgentRuntime)


def test_autoconfigure_honors_null_override(monkeypatch):
    monkeypatch.delenv("BEDROCK_URL", raising=False)
    monkeypatch.delenv("BEDROCK_TOKEN", raising=False)
    monkeypatch.setenv("HARNESS_TRACE_SINK", "null")

    from harness.cloud.autoconfig import autoconfigure

    sink, _runtime = autoconfigure()
    assert isinstance(sink, NullTraceSink)


def test_autoconfigure_sink_override_does_not_change_runtime(monkeypatch):
    """--trace-sink stdout on a Bedrock-managed run must keep the Bedrock
    runtime: sleep still has to POST to the platform that owns the process."""
    monkeypatch.setenv("BEDROCK_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("BEDROCK_TOKEN", "t")
    monkeypatch.setenv("HARNESS_TRACE_SINK", "stdout")

    from harness.cloud.autoconfig import autoconfigure
    from harness.cloud.bedrock import BedrockAgentRuntime

    sink, runtime = autoconfigure()
    assert isinstance(sink, StdoutTraceSink)
    assert isinstance(runtime, BedrockAgentRuntime)


def test_autoconfigure_unknown_override_falls_back_to_auto(monkeypatch):
    monkeypatch.delenv("BEDROCK_URL", raising=False)
    monkeypatch.delenv("BEDROCK_TOKEN", raising=False)
    monkeypatch.setenv("HARNESS_TRACE_SINK", "kafka")

    from harness.cloud.autoconfig import autoconfigure

    sink, _runtime = autoconfigure()
    assert isinstance(sink, StdoutTraceSink)


def test_boot_forwards_trace_sink_flag():
    """`harness boot` execs into `harness agent`; the flag must survive."""
    import argparse

    from harness.cli import _build_agent_cmd

    args = argparse.Namespace(trace_sink="null")
    cmd = _build_agent_cmd("agent-x", "run-x", args)
    i = cmd.index("--trace-sink")
    assert cmd[i + 1] == "null"


# ---------------------------------------------------------------------------
# Run reconstruction from the stdout stream
# ---------------------------------------------------------------------------


def test_full_run_is_reconstructable_from_stdout(capfd, monkeypatch):
    """Emit the exact span structure the harness main loop produces and
    rebuild the run from the ``@@trace`` lines alone."""
    from harness.core import tracer
    from harness.core.tracer import (
        SpanType,
        emit_completed_span,
        llm_span,
        text_span,
        tool_span,
    )

    tracer.set_trace_sink(StdoutTraceSink())
    # Earlier tests in the same process intentionally leave spans dangling
    # (the close_all_open coverage in test_core_http.py) and the tracer's
    # ContextVars leak across tests. Clear them so this run starts a fresh
    # trace, as a real harness process would.
    tracer._current_trace_id.set(None)
    tracer._current_parent_span_id.set(None)

    reasoning = 'Let me think.\nStep 1: compute 17 * 23.\nThen call "sleep".'
    llm_input = json.dumps({"system": "You are a test agent.", "messages": []})
    llm_output = json.dumps({"choices": [{"message": {"role": "assistant"}}]})
    tool_args = json.dumps({"args": {"until": "2099-01-01T00:00:00Z"}, "tool_call_id": "tc_1"})

    with text_span(
        "run_agent",
        agent_id="agent-repro",
        metadata={"agent_id": "agent-repro", "run_id": "run-repro", "model": "m"},
    ) as run_span:
        with text_span("turn_0"):
            with llm_span("openrouter_api_call", metadata={"model": "m"}) as s:
                s.input(llm_input)
                s.output(llm_output)
                s.set_metadata(reasoning=reasoning, reasoning_tokens=42)
            emit_completed_span(
                "thinking",
                span_type=SpanType.TEXT,
                started_at="2026-07-03T00:00:00+00:00",
                ended_at="2026-07-03T00:00:01+00:00",
                output=reasoning,
                metadata={"model": "m", "reasoning_tokens": 42, "has_plaintext": True},
            )
            with tool_span("sleep", input=tool_args) as s:
                s.output("Sleeping until 2099-01-01T00:00:00Z.")
        run_span.set_metadata(usage={"total_cost_usd": 0.01, "llm_calls": 1})

    out, err = capfd.readouterr()
    events = _parse_events(out)

    # Sentinels go to stdout only.
    assert TRACE_SENTINEL_PREFIX not in err

    # Every event line is one of the four lifecycle kinds, and every open
    # has a matching close (nothing dangling => the run is complete).
    kinds = {e["event"] for e in events}
    assert kinds == {"open_trace", "close_trace", "open_span", "close_span"}
    opened = {e["span_id"] for e in events if e["event"] == "open_span"}
    closed = {e["span_id"] for e in events if e["event"] == "close_span"}
    assert opened == closed and len(opened) == 5

    # One trace bracketing everything, stamped with the agent.
    [trace_open] = [e for e in events if e["event"] == "open_trace"]
    assert trace_open["agent_id"] == "agent-repro"
    assert all(
        e["trace_id"] == trace_open["trace_id"] for e in events if "trace_id" in e
    )

    # Rebuild nesting from parent_id: run_agent > turn_0 > {llm, thinking, tool}.
    by_name = {e["name"]: e for e in events if e["event"] == "open_span"}
    assert by_name["run_agent"]["parent_id"] is None
    assert by_name["turn_0"]["parent_id"] == by_name["run_agent"]["span_id"]
    for child in ("openrouter_api_call", "thinking", "sleep"):
        assert by_name[child]["parent_id"] == by_name["turn_0"]["span_id"], child

    close_by_name = {e["name"]: e for e in events if e["event"] == "close_span"}

    # Thinking is reproducible: plaintext survives the round trip intact,
    # including embedded newlines and quotes (JSON-escaped, single line).
    thinking = close_by_name["thinking"]
    assert thinking["output_text"] == reasoning
    assert thinking["metadata"]["reasoning_tokens"] == 42
    llm = close_by_name["openrouter_api_call"]
    assert llm["metadata"]["reasoning"] == reasoning

    # The LLM exchange and tool call are reproducible: request, response,
    # args, and result all round-trip.
    assert llm["input_text"] == llm_input
    assert llm["output_text"] == llm_output
    assert llm["span_type"] == "llm"
    sleep = close_by_name["sleep"]
    assert sleep["span_type"] == "tool"
    assert json.loads(sleep["input_text"])["args"]["until"] == "2099-01-01T00:00:00Z"
    assert sleep["output_text"] == "Sleeping until 2099-01-01T00:00:00Z."

    # Run-level rollup lands on the close of run_agent and the trace close.
    assert close_by_name["run_agent"]["metadata"]["usage"]["total_cost_usd"] == 0.01
    [trace_close] = [e for e in events if e["event"] == "close_trace"]
    assert trace_close["metadata"]["usage"]["llm_calls"] == 1

    # Every sentinel is a single line: multi-line payloads must never split.
    sentinel_lines = [ln for ln in out.splitlines() if ln.startswith(TRACE_SENTINEL_PREFIX)]
    assert len(sentinel_lines) == len(events) == 12


def test_unserializable_metadata_does_not_break_the_line(capfd):
    """Oddball metadata values (datetimes, exceptions) must degrade to
    strings, not kill the event line -- tracing is best-effort."""
    from datetime import UTC, datetime

    sink = StdoutTraceSink()
    sink.open_span(
        span_id="s1",
        trace_id="t1",
        parent_id=None,
        name="weird",
        span_type="text",
        started_at="now",
        input_text=None,
        metadata={"when": datetime(2026, 7, 3, tzinfo=UTC), "err": ValueError("boom")},
    )

    out, _ = capfd.readouterr()
    [event] = _parse_events(out)
    assert event["metadata"]["when"].startswith("2026-07-03")
    assert "boom" in event["metadata"]["err"]

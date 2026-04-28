"""tracer + Bedrock runtime HTTP integration (real HTTP to fake_platform).

These tests exercise the BedrockTraceSink + BedrockAgentRuntime via the
public tracer API. fake_platform sets BEDROCK_URL / BEDROCK_TOKEN at its
URL, so the tracer's autoconfig selects the Bedrock-backed sink.
"""
from __future__ import annotations


def test_tracer_records_trace_and_span(fake_platform):
    from harness.core import tracer

    with tracer.text_span("root", agent_id="agent-x", metadata={"foo": "bar"}) as s:
        s.set_metadata(bar=42)

    # One trace opened + closed, one span opened + closed.
    assert len(fake_platform.traces_open) == 1
    assert len(fake_platform.traces_closed) == 1
    assert len(fake_platform.spans_open) == 1
    assert len(fake_platform.spans_closed) == 1

    trace = next(iter(fake_platform.traces_open.values()))
    assert trace["name"] == "root"
    assert trace["agent_id"] == "agent-x"

    opened = next(iter(fake_platform.spans_open.values()))
    closed = next(iter(fake_platform.spans_closed.values()))
    assert opened["name"] == "root"
    assert opened["span_type"] == "text"
    assert opened["parent_id"] is None
    assert opened["trace_id"] == trace["id"]
    assert opened["metadata"] == {"foo": "bar"}

    assert closed["error"] == ""
    assert closed["metadata"] == {"foo": "bar", "bar": 42}


def test_tracer_nests_spans_under_one_trace(fake_platform):
    from harness.core import tracer

    with tracer.text_span("parent"):
        with tracer.tool_span("child"):
            pass

    assert len(fake_platform.traces_open) == 1, "both spans should share one trace"
    assert len(fake_platform.spans_open) == 2

    trace_id = next(iter(fake_platform.traces_open))
    for s in fake_platform.spans_open.values():
        assert s["trace_id"] == trace_id

    parent = next(s for s in fake_platform.spans_open.values() if s["name"] == "parent")
    child = next(s for s in fake_platform.spans_open.values() if s["name"] == "child")
    assert parent["parent_id"] is None
    assert child["parent_id"] == parent["id"]
    assert parent["span_type"] == "text"
    assert child["span_type"] == "tool"


def test_tracer_typed_factories(fake_platform):
    from harness.core import tracer

    with tracer.text_span("a"):
        pass
    with tracer.tool_span("b"):
        pass
    with tracer.llm_span("c"):
        pass

    types = {s["name"]: s["span_type"] for s in fake_platform.spans_open.values()}
    assert types == {"a": "text", "b": "tool", "c": "llm"}


def test_tracer_captures_input_output(fake_platform):
    from harness.core import tracer

    with tracer.llm_span("c", input="hi") as s:
        s.output("there")

    closed = next(iter(fake_platform.spans_closed.values()))
    assert closed["input_text"] == "hi"
    assert closed["output_text"] == "there"


def test_tracer_records_error(fake_platform):
    from harness.core import tracer

    try:
        with tracer.text_span("boom"):
            raise ValueError("nope")
    except ValueError:
        pass

    closed = next(iter(fake_platform.spans_closed.values()))
    assert "nope" in closed["error"]
    assert closed["metadata"].get("error") and "nope" in closed["metadata"]["error"]

    trace_closed = next(iter(fake_platform.traces_closed.values()))
    assert "nope" in trace_closed["error"]


def test_tracer_sends_auth_header(fake_platform):
    from harness.core import tracer

    with tracer.text_span("x"):
        pass

    for req in fake_platform.requests:
        if req.path.startswith("/api/tracing/"):
            assert req.headers.get("Authorization") == "Bearer test-token"


def test_tracer_close_all_open_flushes_dangling_spans(fake_platform):
    """SIGTERM-style abort: a span is open, process is told to die, registry
    force-closes it.

    This mirrors what the CLI's SIGTERM handler does — we enter a `with`
    block, never exit it cleanly, and then invoke `close_all_open`. Without
    the fix, the trace on the platform side stays open forever.
    """
    from harness.core import tracer

    # Open a trace + nested span and intentionally do NOT exit either `with`.
    # We emulate the state a subprocess would be in when SIGTERM arrives.
    outer_cm = tracer.text_span("run_agent", agent_id="agent-x")
    outer = outer_cm.__enter__()
    outer.set_metadata(turns=3)
    inner_cm = tracer.tool_span("my_tool")
    inner = inner_cm.__enter__()
    inner.output("partial")

    # Platform has seen both spans opened but neither closed.
    assert len(fake_platform.traces_open) == 1
    assert len(fake_platform.spans_open) == 2
    assert len(fake_platform.traces_closed) == 0
    assert len(fake_platform.spans_closed) == 0

    tracer.close_all_open("harness killed by platform")

    # Both spans and the trace must be closed now, with the error recorded.
    assert len(fake_platform.spans_closed) == 2
    assert len(fake_platform.traces_closed) == 1

    for closed in fake_platform.spans_closed.values():
        assert closed["error"] == "harness killed by platform"
        assert closed["metadata"].get("error") == "harness killed by platform"

    trace_closed = next(iter(fake_platform.traces_closed.values()))
    assert trace_closed["error"] == "harness killed by platform"
    # Metadata accumulated on the root span is propagated to the trace.
    assert trace_closed["metadata"].get("turns") == 3

    # Output we set before the abort is preserved.
    tool_closed = next(
        c for c in fake_platform.spans_closed.values()
        if c["output_text"] == "partial"
    )
    assert tool_closed is not None

    # Idempotent: second call is a no-op (registry already empty).
    tracer.close_all_open("second call")
    assert len(fake_platform.spans_closed) == 2
    assert len(fake_platform.traces_closed) == 1


def test_tracer_close_all_open_is_noop_when_nothing_open(fake_platform):
    from harness.core import tracer

    tracer.close_all_open("nothing to do")
    assert len(fake_platform.traces_closed) == 0
    assert len(fake_platform.spans_closed) == 0


def test_tracer_sigterm_during_span_closes_trace(fake_platform):
    """End-to-end: SIGTERM arrives while a span is open; `finally` in
    `tracer.span()` runs and closes everything on the platform.

    This is the production scenario the user hit: bedrock SIGTERMs the
    harness subprocess when the agent flips to sleeping, and the open
    `run_agent` span has to be marked closed so the trace isn't stuck.
    """
    import os
    import signal
    import threading

    from harness.core import tracer

    def _raise_on_sigterm(signum, _frame):
        raise KeyboardInterrupt(signal.Signals(signum).name)

    old_handler = signal.signal(signal.SIGTERM, _raise_on_sigterm)
    try:
        # Send SIGTERM from a background thread after we're inside the span.
        def _send():
            # Small delay so we're deterministically inside the `with`.
            threading.Event().wait(0.05)
            os.kill(os.getpid(), signal.SIGTERM)

        kicker = threading.Thread(target=_send)

        try:
            with tracer.text_span("run_agent") as s:
                s.set_metadata(turn=7)
                kicker.start()
                # Busy-wait to receive the signal without Python swallowing it.
                threading.Event().wait(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            kicker.join(timeout=1.0)
    finally:
        signal.signal(signal.SIGTERM, old_handler)

    # The `finally` in `tracer.span()` must have fired, closing span + trace.
    assert len(fake_platform.spans_closed) == 1
    assert len(fake_platform.traces_closed) == 1

    closed_span = next(iter(fake_platform.spans_closed.values()))
    assert "KeyboardInterrupt" in closed_span["error"]
    assert closed_span["metadata"].get("turn") == 7

    closed_trace = next(iter(fake_platform.traces_closed.values()))
    assert "KeyboardInterrupt" in closed_trace["error"]


def test_tracer_normal_close_removes_from_registry(fake_platform):
    """Spans that close cleanly must NOT linger in the registry, or else a
    later `close_all_open()` would double-close them.
    """
    from harness.core import tracer

    with tracer.text_span("done"):
        pass

    tracer.close_all_open("should be empty")

    # Exactly one close per span/trace — no duplicates from the registry path.
    assert len(fake_platform.spans_closed) == 1
    assert len(fake_platform.traces_closed) == 1
    # And the clean close recorded no error.
    closed_span = next(iter(fake_platform.spans_closed.values()))
    assert closed_span["error"] == ""
    closed_trace = next(iter(fake_platform.traces_closed.values()))
    assert closed_trace["error"] == ""


def test_bedrock_agent_runtime_sleep(fake_platform):
    from harness.cloud.bedrock import BedrockAgentRuntime

    BedrockAgentRuntime().sleep(
        "agent-7", until="2099-01-01T00:00:00Z", reason="testing"
    )

    assert len(fake_platform.sleep_requests) == 1
    sr = fake_platform.sleep_requests[0]
    assert sr["agent_id"] == "agent-7"
    assert sr["until"] == "2099-01-01T00:00:00Z"
    assert sr["reason"] == "testing"

    req = next(r for r in fake_platform.requests if "/sleep/" in r.path)
    assert req.path == "/api/cloud/agents/agent-7/sleep/"
    assert req.headers.get("Authorization") == "Bearer test-token"

"""tracer + runtime_api HTTP integration (real HTTP to fake_platform)."""
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


def test_runtime_api_sleep(fake_platform):
    from harness.core import runtime_api

    runtime_api.sleep("agent-7", until="2099-01-01T00:00:00Z", reason="testing")

    assert len(fake_platform.sleep_requests) == 1
    sr = fake_platform.sleep_requests[0]
    assert sr["agent_id"] == "agent-7"
    assert sr["until"] == "2099-01-01T00:00:00Z"
    assert sr["reason"] == "testing"

    req = next(r for r in fake_platform.requests if "/sleep/" in r.path)
    assert req.path == "/api/cloud/agents/agent-7/sleep/"
    assert req.headers.get("Authorization") == "Bearer test-token"

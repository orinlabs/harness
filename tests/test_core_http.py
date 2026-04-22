"""Chunk 3 verification: tracer + runtime_api speak real HTTP to fake_platform."""
from __future__ import annotations


def test_tracer_records_open_and_close(fake_platform):
    from harness.core import tracer

    with tracer.span("root", foo="bar") as s:
        s.set("bar", 42)

    assert len(fake_platform.spans_open) == 1
    assert len(fake_platform.spans_closed) == 1

    opened = next(iter(fake_platform.spans_open.values()))
    closed = next(iter(fake_platform.spans_closed.values()))

    assert opened["name"] == "root"
    assert opened["parent_id"] is None
    assert opened["metadata"] == {"foo": "bar"}

    assert closed["status"] == "ok"
    assert closed["error"] is None
    assert closed["duration_ns"] > 0
    assert closed["metadata"] == {"bar": 42}


def test_tracer_nests_spans(fake_platform):
    from harness.core import tracer

    with tracer.span("parent") as _p:
        with tracer.span("child") as _c:
            pass

    assert len(fake_platform.spans_open) == 2

    parents = {s["id"]: s for s in fake_platform.spans_open.values() if s["name"] == "parent"}
    children = {s["id"]: s for s in fake_platform.spans_open.values() if s["name"] == "child"}
    assert len(parents) == 1
    assert len(children) == 1

    parent_id = next(iter(parents.keys()))
    child = next(iter(children.values()))
    assert child["parent_id"] == parent_id


def test_tracer_records_error(fake_platform):
    from harness.core import tracer

    try:
        with tracer.span("boom"):
            raise ValueError("nope")
    except ValueError:
        pass

    closed = next(iter(fake_platform.spans_closed.values()))
    assert closed["status"] == "error"
    assert "nope" in closed["error"]


def test_tracer_sends_auth_header(fake_platform):
    from harness.core import tracer

    with tracer.span("x"):
        pass

    post_req = next(r for r in fake_platform.requests if r.method == "POST")
    assert post_req.headers.get("Authorization") == "Bearer test-token"


def test_runtime_api_sleep(fake_platform):
    from harness.core import runtime_api

    runtime_api.sleep("agent-7", until="2099-01-01T00:00:00Z", reason="testing")

    assert len(fake_platform.sleep_requests) == 1
    sr = fake_platform.sleep_requests[0]
    assert sr["agent_id"] == "agent-7"
    assert sr["until"] == "2099-01-01T00:00:00Z"
    assert sr["reason"] == "testing"

    req = next(r for r in fake_platform.requests if "/sleep" in r.path)
    assert req.path == "/agents/agent-7/sleep"
    assert req.headers.get("Authorization") == "Bearer test-token"

"""Chunk 5 verification: tool contract, SleepTool, ExternalTool, build_tool_map."""
from __future__ import annotations

import time

import pytest

from harness.config import AdapterConfig, ExternalToolSpec
from harness.context import RunContext
from tests.fake_platform import FakeToolError


def _ctx(agent_id: str = "agent-1", run_id: str = "run-1") -> RunContext:
    return RunContext(agent_id=agent_id, run_id=run_id)


def test_external_tool_posts_and_returns_text(fake_platform):
    from harness.tools.external import ExternalTool

    fake_platform.register_tool(
        "echo", lambda args, env: {"text": str(args["x"]).upper()}
    )
    spec = ExternalToolSpec(
        name="echo",
        description="upper-case echo",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        url=f"{fake_platform.url}/fake_tools/echo",
    )
    tool = ExternalTool(spec)

    result = tool.call({"x": "hi"}, _ctx())
    assert result.text == "HI"

    posts = [
        r for r in fake_platform.requests if r.method == "POST" and r.path.endswith("/echo")
    ]
    assert len(posts) == 1
    req = posts[0]
    # trace_id/parent_span_id are None when not inside a tracer span context.
    assert req.body == {
        "args": {"x": "hi"},
        "agent_id": "agent-1",
        "run_id": "run-1",
        "trace_id": None,
        "parent_span_id": None,
    }
    assert req.headers.get("Authorization") == "Bearer test-token"
    assert req.headers.get("Content-Type") == "application/json"


def test_external_tool_timeout(fake_platform):
    from harness.tools.external import ExternalTool

    def slow(args, env):
        time.sleep(0.5)
        return {"text": "too late"}

    fake_platform.register_tool("slow", slow)
    spec = ExternalToolSpec(
        name="slow",
        description="sleeps",
        parameters={"type": "object", "properties": {}},
        url=f"{fake_platform.url}/fake_tools/slow",
        timeout_seconds=0.1,
    )

    result = ExternalTool(spec).call({}, _ctx())
    assert result.text.startswith("timeout after")


def test_external_tool_json_error_is_verbatim(fake_platform):
    from harness.tools.external import ExternalTool

    def boom(args, env):
        raise FakeToolError(status=500, body={"error": "slack auth expired"})

    fake_platform.register_tool("boom", boom)
    spec = ExternalToolSpec(
        name="boom",
        description="fails",
        parameters={"type": "object", "properties": {}},
        url=f"{fake_platform.url}/fake_tools/boom",
    )

    result = ExternalTool(spec).call({}, _ctx())
    import json

    assert json.loads(result.text) == {"error": "slack auth expired"}


def test_external_tool_non_json_error_fallback(fake_platform):
    from harness.tools.external import ExternalTool

    def boom(args, env):
        raise FakeToolError(status=502, body="bad gateway")

    fake_platform.register_tool("boom2", boom)
    spec = ExternalToolSpec(
        name="boom2",
        description="fails",
        parameters={"type": "object", "properties": {}},
        url=f"{fake_platform.url}/fake_tools/boom2",
    )

    result = ExternalTool(spec).call({}, _ctx())
    assert result.text.startswith("502 ")
    assert "bad gateway" in result.text


def test_sleep_tool_posts_and_flags_ctx(fake_platform):
    from harness.tools.sleep import SleepTool

    ctx = _ctx(agent_id="agent-7", run_id="run-x")
    tool = SleepTool()

    result = tool.call(
        {"until": "2099-01-01T00:00:00Z", "reason": "done"}, ctx
    )

    assert "2099-01-01T00:00:00Z" in result.text
    assert ctx.sleep_requested is True

    assert len(fake_platform.sleep_requests) == 1
    sr = fake_platform.sleep_requests[0]
    assert sr["agent_id"] == "agent-7"
    assert sr["until"] == "2099-01-01T00:00:00Z"
    assert sr["reason"] == "done"


def test_build_tool_map_merges_builtins_and_adapters(fake_platform):
    from harness.tools import build_tool_map

    spec = ExternalToolSpec(
        name="search",
        description="search",
        parameters={"type": "object", "properties": {}},
        url=f"{fake_platform.url}/fake_tools/search",
    )
    adapter = AdapterConfig(name="a1", description="", tools=[spec])

    tool_map = build_tool_map([adapter])
    assert set(tool_map.keys()) == {"sleep", "search"}


def test_build_tool_map_rejects_collision(fake_platform):
    from harness.tools import build_tool_map

    bad = ExternalToolSpec(
        name="sleep",
        description="shadow",
        parameters={"type": "object", "properties": {}},
        url=f"{fake_platform.url}/fake_tools/sleep",
    )
    adapter = AdapterConfig(name="a1", description="", tools=[bad])

    with pytest.raises(ValueError, match="collision"):
        build_tool_map([adapter])

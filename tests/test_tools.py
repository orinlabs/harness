"""Chunk 5 verification: tool contract, SleepTool, ExternalTool, build_tool_map."""
from __future__ import annotations

import time

import pytest

from harness.config import ExternalToolSpec, ToolAuth
from harness.context import RunContext
from tests.fake_platform import FakeToolError


def _bedrock_spec(**overrides) -> ExternalToolSpec:
    """Helper for tests that need a Bedrock-proxied spec (bearer auth + trace fwd).

    This is the shape ``BedrockConfigClient.fetch_harness_config`` produces; it's
    what Bedrock's adapter runtime expects on the wire. Standalone tools use
    ``ExternalToolSpec(...)`` directly with defaults.
    """
    base = dict(
        auth=ToolAuth(kind="bearer_env", token_env="BEDROCK_TOKEN"),
        forward_trace_context=True,
    )
    base.update(overrides)
    return ExternalToolSpec(**base)


def _ctx(agent_id: str = "agent-1", run_id: str = "run-1") -> RunContext:
    """RunContext wired with a Bedrock runtime (every test in this file uses
    ``fake_platform``, which points BEDROCK_URL at its own server)."""
    from harness.cloud.bedrock import BedrockAgentRuntime

    return RunContext(agent_id=agent_id, run_id=run_id, runtime=BedrockAgentRuntime())


def test_external_tool_bedrock_style_posts_and_returns_text(fake_platform):
    """Bedrock-shaped spec: bearer auth from env + trace_id/parent_span_id in body."""
    from harness.tools.external import ExternalTool

    fake_platform.register_tool(
        "echo", lambda args, env: {"text": str(args["x"]).upper()}
    )
    spec = _bedrock_spec(
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


def test_external_tool_standalone_default_omits_auth_and_trace_fields(fake_platform):
    """Standalone spec (no auth, no trace fwd): body has only the core fields, no auth header."""
    from harness.tools.external import ExternalTool

    fake_platform.register_tool(
        "bare", lambda args, env: {"text": f"hello {args.get('name', '?')}"}
    )
    spec = ExternalToolSpec(
        name="bare",
        description="no-auth tool",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}},
        url=f"{fake_platform.url}/fake_tools/bare",
    )
    tool = ExternalTool(spec)

    result = tool.call({"name": "alice"}, _ctx())
    assert result.text == "hello alice"

    posts = [
        r for r in fake_platform.requests if r.method == "POST" and r.path.endswith("/bare")
    ]
    req = posts[0]
    assert req.body == {
        "args": {"name": "alice"},
        "agent_id": "agent-1",
        "run_id": "run-1",
    }
    # No auth configured -> no Authorization header sent.
    assert "Authorization" not in {k.title() for k in req.headers.keys()}


def test_external_tool_custom_headers_auth(fake_platform):
    """ToolAuth(kind='headers') lets configs ship custom auth shapes."""
    from harness.tools.external import ExternalTool

    fake_platform.register_tool("api", lambda args, env: {"text": "ok"})
    spec = ExternalToolSpec(
        name="api",
        description="custom-auth tool",
        parameters={"type": "object", "properties": {}},
        url=f"{fake_platform.url}/fake_tools/api",
        auth=ToolAuth(kind="headers", headers={"X-API-Key": "secret-123"}),
    )
    ExternalTool(spec).call({}, _ctx())

    req = next(
        r for r in fake_platform.requests if r.method == "POST" and r.path.endswith("/api")
    )
    # Preserve the caller's header case through httpx -> BaseHTTPRequestHandler.
    assert req.headers.get("X-API-Key") == "secret-123"


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


def test_sleep_tool_allows_sleep_when_no_notifications(fake_platform):
    """list_notifications returning its empty-inbox string must NOT block sleep."""
    from harness.tools.external import ExternalTool
    from harness.tools.sleep import SleepTool

    fake_platform.register_tool(
        "list_notifications",
        lambda args, env: {"text": "You have no pending notifications."},
    )
    list_tool = ExternalTool(
        ExternalToolSpec(
            name="list_notifications",
            description="list",
            parameters={"type": "object", "properties": {}},
            url=f"{fake_platform.url}/fake_tools/list_notifications",
        )
    )

    ctx = _ctx(agent_id="agent-7", run_id="run-x")
    ctx.tool_map = {"list_notifications": list_tool}
    tool = SleepTool()

    result = tool.call({"until": "2099-01-01T00:00:00Z", "reason": "done"}, ctx)

    assert "2099-01-01T00:00:00Z" in result.text
    assert ctx.sleep_requested is True
    assert len(fake_platform.sleep_requests) == 1
    # The pre-check must have hit list_notifications exactly once.
    calls = [r for r in fake_platform.requests if r.path.endswith("/list_notifications")]
    assert len(calls) == 1


def test_sleep_tool_blocks_when_notifications_pending(fake_platform):
    """Any non-empty list_notifications output must refuse sleep and skip runtime_api.sleep."""
    from harness.tools.external import ExternalTool
    from harness.tools.sleep import SleepTool

    fake_response = (
        "You have 1 notification(s):\n\n"
        "1. Mike sent a photo [HIGH PRIORITY]\n"
        "   Notification ID: abc-123\n"
        "   Source: sms\n"
        "   Details: bring it in by 5pm\n"
    )
    fake_platform.register_tool(
        "list_notifications", lambda args, env: {"text": fake_response}
    )
    list_tool = ExternalTool(
        ExternalToolSpec(
            name="list_notifications",
            description="list",
            parameters={"type": "object", "properties": {}},
            url=f"{fake_platform.url}/fake_tools/list_notifications",
        )
    )

    ctx = _ctx(agent_id="agent-7", run_id="run-x")
    ctx.tool_map = {"list_notifications": list_tool}
    tool = SleepTool()

    result = tool.call({"until": "2099-01-01T00:00:00Z", "reason": "done"}, ctx)

    # Agent got refused...
    assert "Cannot sleep" in result.text
    assert "Mike sent a photo" in result.text
    # ...and the runtime_api.sleep POST was NOT issued.
    assert ctx.sleep_requested is False
    assert fake_platform.sleep_requests == []


def test_sleep_tool_allows_sleep_when_list_notifications_errors(fake_platform):
    """Transient list_notifications errors must not wedge the agent awake."""
    from harness.tools.external import ExternalTool
    from harness.tools.sleep import SleepTool

    fake_platform.register_tool(
        "list_notifications",
        lambda args, env: {"text": "Error listing notifications: db timeout"},
    )
    list_tool = ExternalTool(
        ExternalToolSpec(
            name="list_notifications",
            description="list",
            parameters={"type": "object", "properties": {}},
            url=f"{fake_platform.url}/fake_tools/list_notifications",
        )
    )

    ctx = _ctx(agent_id="agent-7", run_id="run-x")
    ctx.tool_map = {"list_notifications": list_tool}
    tool = SleepTool()

    result = tool.call({"until": "2099-01-01T00:00:00Z", "reason": "done"}, ctx)

    assert "2099-01-01T00:00:00Z" in result.text
    assert ctx.sleep_requested is True
    assert len(fake_platform.sleep_requests) == 1


def test_sleep_tool_no_list_notifications_tool_available(fake_platform):
    """When the adapter set doesn't provide list_notifications, sleep proceeds normally."""
    from harness.tools.sleep import SleepTool

    ctx = _ctx(agent_id="agent-7", run_id="run-x")
    ctx.tool_map = {}  # no list_notifications registered
    tool = SleepTool()

    result = tool.call({"until": "2099-01-01T00:00:00Z", "reason": "done"}, ctx)

    assert "2099-01-01T00:00:00Z" in result.text
    assert ctx.sleep_requested is True
    assert len(fake_platform.sleep_requests) == 1


def test_build_tool_map_merges_builtins_and_tools(fake_platform):
    from harness.tools import build_tool_map

    spec = ExternalToolSpec(
        name="search",
        description="search",
        parameters={"type": "object", "properties": {}},
        url=f"{fake_platform.url}/fake_tools/search",
    )

    tool_map = build_tool_map([spec])
    assert set(tool_map.keys()) == {"sleep", "search"}


def test_build_tool_map_rejects_collision(fake_platform):
    from harness.tools import build_tool_map

    bad = ExternalToolSpec(
        name="sleep",
        description="shadow",
        parameters={"type": "object", "properties": {}},
        url=f"{fake_platform.url}/fake_tools/sleep",
    )

    with pytest.raises(ValueError, match="collision"):
        build_tool_map([bad])

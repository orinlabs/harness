"""External tool wrapper.

`ExternalTool(spec)` satisfies the `Tool` protocol by POSTing args to a URL the
platform registered in `AgentConfig`. The harness never sees the tool's actual
implementation — auth, proxying, and dispatching all live on the platform side.

Wire contract:
  POST {spec.url}
  Headers: Authorization: Bearer $HARNESS_PLATFORM_TOKEN, Content-Type: application/json
  Body:    {"args": {...}, "agent_id": "...", "run_id": "..."}
  200:     {"text": str, "images": [base64, ...] | null}
  non-2xx: surfaced verbatim to the model as ToolResult.text (JSON body -> JSON
           string; non-JSON -> "<status> <reason>: <text>")
  timeout: ToolResult.text = "timeout after Ns"
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from harness.config import ExternalToolSpec
from harness.context import RunContext
from harness.tools.base import ToolResult, ToolSchema

logger = logging.getLogger(__name__)

_client: httpx.Client | None = None


def _http() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client()
    return _client


def _auth_header() -> dict[str, str]:
    token = os.environ.get("HARNESS_PLATFORM_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


class ExternalTool:
    def __init__(self, spec: ExternalToolSpec):
        self._spec = spec

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def description(self) -> str:
        return self._spec.description

    @property
    def parameters(self) -> dict:
        return self._spec.parameters

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(self._spec.name, self._spec.description, self._spec.parameters)

    def call(self, args: dict, ctx: RunContext) -> ToolResult:
        # Include tracing context so the adapter runtime can parent its own
        # tool span under our active harness tree rather than creating a
        # sibling trace at the root.
        from harness.core.tracer import get_current_span_id, get_current_trace_id

        body = {
            "args": args,
            "agent_id": ctx.agent_id,
            "run_id": ctx.run_id,
            "trace_id": get_current_trace_id(),
            "parent_span_id": get_current_span_id(),
        }
        headers = {"Content-Type": "application/json", **_auth_header()}

        try:
            resp = _http().post(
                self._spec.url,
                json=body,
                headers=headers,
                timeout=self._spec.timeout_seconds,
            )
        except httpx.TimeoutException:
            logger.warning("external tool %s timed out", self._spec.name)
            return ToolResult(text=f"timeout after {self._spec.timeout_seconds}s")
        except httpx.HTTPError as e:
            logger.warning("external tool %s transport error: %s", self._spec.name, e)
            return ToolResult(text=f"transport error: {e}")

        if 200 <= resp.status_code < 300:
            return self._parse_success(resp.json())

        return self._format_error(resp)

    @staticmethod
    def _parse_success(data: Any) -> ToolResult:
        if not isinstance(data, dict):
            return ToolResult(text=json.dumps(data))
        text = str(data.get("text") or "")
        images = data.get("images")
        if images is not None and not isinstance(images, list):
            images = None
        return ToolResult(text=text, images=images)

    @staticmethod
    def _format_error(resp: httpx.Response) -> ToolResult:
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            try:
                return ToolResult(text=json.dumps(resp.json()))
            except ValueError:
                pass
        body = resp.text
        truncated = body if len(body) <= 500 else body[:500] + "..."
        return ToolResult(text=f"{resp.status_code} {resp.reason_phrase}: {truncated}")

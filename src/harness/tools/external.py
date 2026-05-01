"""External tool wrapper.

``ExternalTool(spec)`` satisfies the ``Tool`` protocol by POSTing args to
``spec.url``. The harness never sees the tool's implementation.

Wire contract (standalone):
  POST {spec.url}
  Body:    {"args": {...}, "agent_id": "...", "run_id": "..."}
  200:     {"text": str, "images": [base64, ...] | null}

When ``spec.auth`` is configured, an ``Authorization`` header is added.
When ``spec.forward_trace_context=True`` the body also carries
``trace_id`` / ``parent_span_id`` so a backend that re-emits its own span
(e.g. Bedrock's adapter runtime) can nest under our active harness trace.

Failure modes:
  non-2xx: surfaced verbatim to the model as ToolResult.text (JSON body ->
           JSON string; non-JSON -> "<status> <reason>: <text>")
  timeout: ToolResult.text = "timeout after Ns"
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from harness.config import ExternalToolSpec, ToolAuth
from harness.context import RunContext
from harness.tools.base import ToolResult, ToolSchema

logger = logging.getLogger(__name__)

_client: httpx.Client | None = None


def _http() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client()
    return _client


def _headers_for(auth: ToolAuth) -> dict[str, str]:
    """Translate a ``ToolAuth`` into HTTP headers."""
    if auth.kind == "bearer_env":
        token = os.environ.get(auth.token_env or "", "")
        return {"Authorization": f"Bearer {token}"} if token else {}
    if auth.kind == "bearer_literal":
        return {"Authorization": f"Bearer {auth.token}"} if auth.token else {}
    if auth.kind == "headers":
        return dict(auth.headers)
    return {}


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
        body: dict[str, Any] = {
            "args": args,
            "agent_id": ctx.agent_id,
            "run_id": ctx.run_id,
        }
        if self._spec.forward_trace_context:
            # Only pulled when the backend actually consumes these (Bedrock's
            # adapter runtime does, standalone HTTP tools don't). Deferred
            # import avoids dragging tracer into tools/ imports.
            from harness.core.tracer import (
                get_current_span_id,
                get_current_trace_id,
            )

            body["trace_id"] = get_current_trace_id()
            body["parent_span_id"] = get_current_span_id()

        headers = {"Content-Type": "application/json", **_headers_for(self._spec.auth)}

        try:
            resp = _http().post(
                self._spec.url,
                json=body,
                headers=headers,
                timeout=self._spec.timeout_seconds,
            )
        except httpx.TimeoutException:
            logger.warning(
                "external tool %s timed out calling %s",
                self._spec.name,
                self._spec.url,
            )
            return ToolResult(text=f"timeout after {self._spec.timeout_seconds}s")
        except httpx.HTTPError as e:
            logger.warning(
                "external tool %s transport error calling %s: %s",
                self._spec.name,
                self._spec.url,
                e,
            )
            return ToolResult(text=f"transport error calling {self._spec.url}: {e}")

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

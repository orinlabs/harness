"""Bedrock-backed ``TraceSink``.

Wire protocol (authed with BEDROCK_TOKEN):

    POST   {PLATFORM}/api/tracing/traces/              -> create trace
      body:  {id, name, started_at, agent_id, metadata}

    PATCH  {PLATFORM}/api/tracing/traces/{id}/         -> close trace
      body:  {ended_at, error, metadata}
      (metadata merges; the /end/ POST endpoint exists but ignores metadata
       updates, so we PATCH instead.)

    POST   {PLATFORM}/api/tracing/spans/               -> create span
      body:  {id, trace_id, parent_id, name, span_type, started_at,
              input_text, metadata, agent_id}

    PATCH  {PLATFORM}/api/tracing/spans/{id}/          -> close span
      body:  {ended_at, input_text, output_text, error, metadata}

All methods swallow transport errors and log at WARNING -- tracing is
best-effort so a flaky platform can never take down an agent run.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from harness.cloud.bedrock.client import auth_header, http, platform_url_or_none

logger = logging.getLogger(__name__)


class BedrockTraceSink:
    """Ship trace/span lifecycle events to Bedrock's /api/tracing/* endpoints.

    Reads ``BEDROCK_URL`` / ``BEDROCK_TOKEN`` lazily on each call so tests that
    set the env inside a fixture don't have to reinstantiate the sink. When
    ``BEDROCK_URL`` is unset every call short-circuits -- callers can install
    this sink defensively and it'll behave like ``NullTraceSink`` until the
    env shows up.
    """

    def open_trace(
        self,
        *,
        trace_id: str,
        name: str,
        started_at: str,
        agent_id: str | None,
    ) -> None:
        base = platform_url_or_none()
        if base is None:
            return
        body = {
            "id": trace_id,
            "name": name,
            "started_at": started_at,
            "agent_id": agent_id,
            "metadata": {},
        }
        try:
            http().post(
                f"{base}/api/tracing/traces/", json=body, headers=auth_header()
            )
        except httpx.HTTPError as e:
            logger.warning("BedrockTraceSink: failed to open trace %s: %s", name, e)

    def close_trace(
        self,
        *,
        trace_id: str,
        name: str,
        agent_id: str | None,
        ended_at: str,
        error: str | None,
        metadata: dict[str, Any],
    ) -> None:
        base = platform_url_or_none()
        if base is None:
            return
        body = {
            "ended_at": ended_at,
            "error": error or "",
            "metadata": _safe_json(metadata),
        }
        try:
            http().patch(
                f"{base}/api/tracing/traces/{trace_id}/",
                json=body,
                headers=auth_header(),
            )
        except httpx.HTTPError as e:
            logger.warning(
                "BedrockTraceSink: failed to close trace %s: %s", trace_id, e
            )

    def open_span(
        self,
        *,
        span_id: str,
        trace_id: str,
        parent_id: str | None,
        name: str,
        span_type: str,
        started_at: str,
        input_text: str | None,
        metadata: dict[str, Any],
    ) -> None:
        base = platform_url_or_none()
        if base is None:
            return
        body = {
            "id": span_id,
            "trace_id": trace_id,
            "parent_id": parent_id,
            "name": name,
            "span_type": span_type,
            "started_at": started_at,
            "input_text": input_text,
            "metadata": _safe_json(metadata),
        }
        try:
            http().post(
                f"{base}/api/tracing/spans/", json=body, headers=auth_header()
            )
        except httpx.HTTPError as e:
            logger.warning("BedrockTraceSink: failed to open span %s: %s", name, e)

    def close_span(
        self,
        *,
        span_id: str,
        trace_id: str,
        parent_id: str | None,
        name: str,
        span_type: str,
        ended_at: str,
        input_text: str | None,
        output_text: str | None,
        error: str | None,
        metadata: dict[str, Any],
    ) -> None:
        base = platform_url_or_none()
        if base is None:
            return
        body = {
            "ended_at": ended_at,
            "input_text": input_text or "",
            "output_text": output_text or "",
            "error": error or "",
            "metadata": _safe_json(metadata),
        }
        try:
            http().patch(
                f"{base}/api/tracing/spans/{span_id}/",
                json=body,
                headers=auth_header(),
            )
        except httpx.HTTPError as e:
            logger.warning(
                "BedrockTraceSink: failed to close span %s: %s", span_id, e
            )


def _safe_json(obj: Any) -> Any:
    """Recursively coerce to JSON-serializable primitives."""
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)

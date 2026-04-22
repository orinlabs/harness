"""Trace client.

Emits nested spans to the infra platform's trace API. Each span does:
  - POST /traces/spans   on enter (returns a span id)
  - PATCH /traces/spans/{id}   on exit (status, duration, metadata)

Nesting happens via a ContextVar so callers don't thread parent-ids through
their code. The context manager returns a handle whose `set(k, v)` mutates the
local metadata dict; all metadata is shipped in the final PATCH.

Failures to reach the platform are logged but never raised — tracing is
best-effort and must not break a run.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_current_parent: ContextVar[str | None] = ContextVar("harness_trace_parent", default=None)

_client: httpx.Client | None = None


def _platform_url() -> str:
    url = os.environ.get("HARNESS_PLATFORM_URL")
    if not url:
        raise RuntimeError(
            "HARNESS_PLATFORM_URL is not set. Core clients need it to reach the infra platform."
        )
    return url.rstrip("/")


def _auth_header() -> dict[str, str]:
    token = os.environ.get("HARNESS_PLATFORM_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _http() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=10.0)
    return _client


class Span:
    def __init__(self, span_id: str):
        self.id = span_id
        self._meta: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._meta[key] = value


@contextmanager
def span(name: str, **metadata: Any) -> Iterator[Span]:
    """Open a span. Nests automatically under the current parent if any."""
    span_id = str(uuid.uuid4())
    parent_id = _current_parent.get()
    started_at_ns = time.time_ns()

    body = {
        "id": span_id,
        "name": name,
        "parent_id": parent_id,
        "started_at_ns": started_at_ns,
        "metadata": dict(metadata),
    }
    try:
        _http().post(
            f"{_platform_url()}/traces/spans",
            json=body,
            headers=_auth_header(),
        )
    except httpx.HTTPError as e:
        logger.warning("tracer: failed to open span %s: %s", name, e)

    handle = Span(span_id)
    token = _current_parent.set(span_id)
    status = "ok"
    error_msg: str | None = None
    try:
        yield handle
    except BaseException as e:
        status = "error"
        error_msg = f"{type(e).__name__}: {e}"
        raise
    finally:
        _current_parent.reset(token)
        ended_at_ns = time.time_ns()
        patch_body = {
            "ended_at_ns": ended_at_ns,
            "duration_ns": ended_at_ns - started_at_ns,
            "status": status,
            "error": error_msg,
            "metadata": handle._meta,
        }
        try:
            _http().patch(
                f"{_platform_url()}/traces/spans/{span_id}",
                json=patch_body,
                headers=_auth_header(),
            )
        except httpx.HTTPError as e:
            logger.warning("tracer: failed to close span %s: %s", name, e)

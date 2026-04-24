"""Trace + span client.

Matches the platform's existing trace shape (two-tier model of `Trace` + `Span`,
typed spans, separate input/output text fields, and standard metadata keys
like `usage`, `llm_cost`, `model_breakdown`, `summaries_created`).

Wire protocol (authed with BEDROCK_TOKEN):

    POST   {PLATFORM}/api/tracing/traces/              -> create trace
      body:  {id, name, started_at, agent_id, metadata}

    PATCH  {PLATFORM}/api/tracing/traces/{id}/         -> close trace
      body:  {ended_at, error, metadata}
      (metadata merges; bedrock's /end/ POST ignores metadata updates, so we
       use PATCH which accepts ended_at + metadata in one call.)

    POST   {PLATFORM}/api/tracing/spans/               -> create span
      body:  {id, trace_id, parent_id, name, span_type, started_at,
              input_text, metadata, agent_id}

    PATCH  {PLATFORM}/api/tracing/spans/{id}/          -> close span
      body:  {ended_at, input_text, output_text, error, metadata}

The first span opened in a context creates the trace implicitly. Nested spans
reuse the same trace_id via ContextVar-based parent tracking. Failures to reach
the platform are logged but never raised — tracing is best-effort.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterator

import httpx

logger = logging.getLogger(__name__)


class SpanType(str, Enum):
    TEXT = "text"
    TOOL = "tool"
    LLM = "llm"
    FUNCTION = "function"
    AUDIO = "audio"
    CHECKPOINT = "checkpoint"


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_current_trace_id: ContextVar[str | None] = ContextVar("harness_trace_id", default=None)
_current_trace_name: ContextVar[str | None] = ContextVar(
    "harness_trace_name", default=None
)
_current_trace_agent_id: ContextVar[str | None] = ContextVar(
    "harness_trace_agent_id", default=None
)
_current_parent_span_id: ContextVar[str | None] = ContextVar(
    "harness_parent_span_id", default=None
)
_client: httpx.Client | None = None

# Emergency registry of still-open traces/spans.
#
# ContextVars are scoped per-task, so a signal handler or atexit hook can't see
# them. We mirror open-trace and open-span info into these module-level dicts
# so `close_all_open(error)` can force-close anything left dangling when the
# process is killed (SIGTERM from the platform, crash, etc.). Entries are
# removed on normal close.
_open_traces: dict[str, dict[str, Any]] = {}
_open_spans: dict[str, dict[str, Any]] = {}


def _platform_url() -> str | None:
    """Return the platform base URL or None if tracing is disabled.

    Tracing is best-effort. When `BEDROCK_URL` is unset, all tracer
    HTTP calls short-circuit. This is what lets unit tests that don't need a
    platform (e.g. the memory port) run without spinning up fake_platform.
    """
    url = os.environ.get("BEDROCK_URL")
    return url.rstrip("/") if url else None


def _auth_header() -> dict[str, str]:
    token = os.environ.get("BEDROCK_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _http() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=10.0)
    return _client


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def get_current_trace_id() -> str | None:
    return _current_trace_id.get()


def get_current_span_id() -> str | None:
    return _current_parent_span_id.get()


# ---------------------------------------------------------------------------
# Span handle
# ---------------------------------------------------------------------------


class Span:
    """In-process handle for an open span.

    Callers mutate `input_text`, `output_text`, and `metadata` via the helper
    methods; they're shipped to the platform when the span closes.
    """

    def __init__(self, span_id: str, trace_id: str, name: str, span_type: SpanType):
        self.id = span_id
        self.trace_id = trace_id
        self.name = name
        self.span_type = span_type
        self._input: str | None = None
        self._output: str | None = None
        self._metadata: dict[str, Any] = {}

    def input(self, text: str) -> None:
        self._input = text

    def output(self, text: str) -> None:
        self._output = text

    def set_metadata(self, **kwargs: Any) -> None:
        self._metadata.update(kwargs)


# ---------------------------------------------------------------------------
# Core context manager (and the typed factories around it)
# ---------------------------------------------------------------------------


@contextmanager
def span(
    name: str,
    *,
    span_type: SpanType = SpanType.TEXT,
    input: str | None = None,
    metadata: dict[str, Any] | None = None,
    agent_id: str | None = None,
) -> Iterator[Span]:
    """Open a span. If no trace is active, one is created implicitly.

    - `span_type` controls the enum tag (text/tool/llm/function/audio).
    - `input` / `metadata` are initial values; callers may add more before close
      via `.input(...)`, `.output(...)`, `.set_metadata(...)`.
    - `agent_id` is attached to the trace on trace creation only; ignored when
      a trace is already active.
    """
    span_id = str(uuid.uuid4())
    started_at = _now_iso()

    trace_id = _current_trace_id.get()
    trace_name_token = None
    trace_id_token = None
    trace_agent_token = None
    created_trace = False
    if trace_id is None:
        trace_id = str(uuid.uuid4())
        created_trace = True
        _open_trace(
            trace_id=trace_id,
            name=name,
            started_at=started_at,
            agent_id=agent_id,
        )
        trace_id_token = _current_trace_id.set(trace_id)
        trace_name_token = _current_trace_name.set(name)
        trace_agent_token = _current_trace_agent_id.set(agent_id)

    parent_id = _current_parent_span_id.get()

    _open_span(
        span_id=span_id,
        trace_id=trace_id,
        parent_id=parent_id,
        name=name,
        span_type=span_type,
        started_at=started_at,
        input_text=input,
        metadata=metadata or {},
    )

    handle = Span(span_id, trace_id, name, span_type)
    if input is not None:
        handle._input = input
    if metadata:
        handle._metadata.update(metadata)

    # Register in the emergency map so `close_all_open()` can flush this span
    # (and any trace we created) if we get killed before the finally runs.
    _open_spans[span_id] = {
        "span_id": span_id,
        "trace_id": trace_id,
        "parent_id": parent_id,
        "name": name,
        "span_type": span_type,
        "handle": handle,
    }
    if created_trace:
        _open_traces[trace_id] = {
            "trace_id": trace_id,
            "name": name,
            "agent_id": agent_id,
            "handle": handle,
        }

    parent_token = _current_parent_span_id.set(span_id)
    status = "ok"
    error_msg: str | None = None
    try:
        yield handle
    except BaseException as e:
        status = "error"
        error_msg = f"{type(e).__name__}: {e}"
        handle._metadata["error"] = error_msg
        raise
    finally:
        _current_parent_span_id.reset(parent_token)
        ended_at = _now_iso()
        _open_spans.pop(span_id, None)
        _close_span(
            span_id=span_id,
            trace_id=trace_id,
            parent_id=parent_id,
            name=name,
            span_type=span_type,
            ended_at=ended_at,
            input_text=handle._input,
            output_text=handle._output,
            error=error_msg if status == "error" else None,
            metadata=handle._metadata,
        )
        if created_trace:
            _open_traces.pop(trace_id, None)
            _close_trace(
                trace_id=trace_id,
                name=name,
                agent_id=agent_id,
                ended_at=ended_at,
                error=error_msg if status == "error" else None,
                # Propagate the root span's metadata up to the trace so the
                # platform can show a usage summary on the trace row itself.
                metadata=dict(handle._metadata),
            )
            if trace_id_token is not None:
                _current_trace_id.reset(trace_id_token)
            if trace_name_token is not None:
                _current_trace_name.reset(trace_name_token)
            if trace_agent_token is not None:
                _current_trace_agent_id.reset(trace_agent_token)


def text_span(
    name: str,
    *,
    input: str | None = None,
    metadata: dict[str, Any] | None = None,
    agent_id: str | None = None,
):
    return span(
        name,
        span_type=SpanType.TEXT,
        input=input,
        metadata=metadata,
        agent_id=agent_id,
    )


def tool_span(
    name: str,
    *,
    input: str | None = None,
    metadata: dict[str, Any] | None = None,
    agent_id: str | None = None,
):
    return span(
        name,
        span_type=SpanType.TOOL,
        input=input,
        metadata=metadata,
        agent_id=agent_id,
    )


def llm_span(
    name: str,
    *,
    input: str | None = None,
    metadata: dict[str, Any] | None = None,
    agent_id: str | None = None,
):
    return span(
        name,
        span_type=SpanType.LLM,
        input=input,
        metadata=metadata,
        agent_id=agent_id,
    )


def emit_completed_span(
    name: str,
    *,
    span_type: SpanType,
    started_at: str,
    ended_at: str,
    input: str | None = None,
    output: str | None = None,
    metadata: dict[str, Any] | None = None,
    error: str | None = None,
    agent_id: str | None = None,
) -> None:
    """Emit a span that represents work already completed.

    Unlike `span()`, this doesn't open a context — useful when you want to log
    a span *only if* something happened, and the work was already bracketed
    with manual timestamps. Nests under whichever span is currently active
    via the same ContextVars.

    If no trace is active when this is called, a trace is created and closed
    around this single span (to preserve the invariant that every span
    belongs to a trace). When `agent_id` is supplied, it's attached to that
    standalone trace so the span is discoverable under the agent in the
    platform UI rather than appearing as an orphan.
    """
    span_id = str(uuid.uuid4())
    metadata = metadata or {}

    trace_id = _current_trace_id.get()
    standalone_trace = False
    if trace_id is None:
        trace_id = str(uuid.uuid4())
        standalone_trace = True
        _open_trace(
            trace_id=trace_id,
            name=name,
            started_at=started_at,
            agent_id=agent_id,
        )

    parent_id = _current_parent_span_id.get()
    _open_span(
        span_id=span_id,
        trace_id=trace_id,
        parent_id=parent_id,
        name=name,
        span_type=span_type,
        started_at=started_at,
        input_text=input,
        metadata=metadata,
    )
    _close_span(
        span_id=span_id,
        trace_id=trace_id,
        parent_id=parent_id,
        name=name,
        span_type=span_type,
        ended_at=ended_at,
        input_text=input,
        output_text=output,
        error=error,
        metadata=metadata,
    )
    if standalone_trace:
        _close_trace(
            trace_id=trace_id,
            name=name,
            agent_id=agent_id,
            ended_at=ended_at,
            error=error,
            metadata=dict(metadata),
        )


# ---------------------------------------------------------------------------
# HTTP emission (fire-and-log-on-failure)
# ---------------------------------------------------------------------------


def _open_trace(
    *,
    trace_id: str,
    name: str,
    started_at: str,
    agent_id: str | None,
) -> None:
    base = _platform_url()
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
        _http().post(
            f"{base}/api/tracing/traces/", json=body, headers=_auth_header()
        )
    except httpx.HTTPError as e:
        logger.warning("tracer: failed to open trace %s: %s", name, e)


def _close_trace(
    *,
    trace_id: str,
    name: str,
    agent_id: str | None,
    ended_at: str,
    error: str | None,
    metadata: dict[str, Any],
) -> None:
    base = _platform_url()
    if base is None:
        return
    # PATCH merges `metadata` with what was set at create time, and applies
    # ended_at/error in the same call. The /end/ POST endpoint exists but
    # ignores metadata updates, so we avoid it.
    body = {
        "ended_at": ended_at,
        "error": error or "",
        "metadata": _safe_json(metadata),
    }
    try:
        _http().patch(
            f"{base}/api/tracing/traces/{trace_id}/",
            json=body,
            headers=_auth_header(),
        )
    except httpx.HTTPError as e:
        logger.warning("tracer: failed to close trace %s: %s", trace_id, e)


def _open_span(
    *,
    span_id: str,
    trace_id: str,
    parent_id: str | None,
    name: str,
    span_type: SpanType,
    started_at: str,
    input_text: str | None,
    metadata: dict[str, Any],
) -> None:
    base = _platform_url()
    if base is None:
        return
    body = {
        "id": span_id,
        "trace_id": trace_id,
        "parent_id": parent_id,
        "name": name,
        "span_type": span_type.value,
        "started_at": started_at,
        "input_text": input_text,
        "metadata": _safe_json(metadata),
    }
    try:
        _http().post(
            f"{base}/api/tracing/spans/", json=body, headers=_auth_header()
        )
    except httpx.HTTPError as e:
        logger.warning("tracer: failed to open span %s: %s", name, e)


def _close_span(
    *,
    span_id: str,
    trace_id: str,
    parent_id: str | None,
    name: str,
    span_type: SpanType,
    ended_at: str,
    input_text: str | None,
    output_text: str | None,
    error: str | None,
    metadata: dict[str, Any],
) -> None:
    base = _platform_url()
    if base is None:
        return
    # PATCH merges metadata with what was set at create time; /end/ POST does
    # not, so we PATCH instead.
    body = {
        "ended_at": ended_at,
        "input_text": input_text or "",
        "output_text": output_text or "",
        "error": error or "",
        "metadata": _safe_json(metadata),
    }
    try:
        _http().patch(
            f"{base}/api/tracing/spans/{span_id}/",
            json=body,
            headers=_auth_header(),
        )
    except httpx.HTTPError as e:
        logger.warning("tracer: failed to close span %s: %s", span_id, e)


def close_all_open(error: str) -> None:
    """Force-close every still-open span and trace with `error`.

    Called from the CLI's signal handlers / atexit hook when the harness is
    killed mid-run (e.g. platform sends SIGTERM when the agent flips to
    sleeping). Iterates over a snapshot of the registries so closing is
    idempotent even if the normal `finally` path also fires.
    """
    if not _open_spans and not _open_traces:
        return
    ended_at = _now_iso()
    logger.info(
        "tracer: force-closing %d span(s) and %d trace(s) (%s)",
        len(_open_spans),
        len(_open_traces),
        error,
    )

    # Close spans first (leaves-to-root would be ideal, but PATCH is
    # idempotent on the platform so order doesn't matter for correctness).
    for span_id, info in list(_open_spans.items()):
        handle: Span = info["handle"]
        handle._metadata.setdefault("error", error)
        _close_span(
            span_id=span_id,
            trace_id=info["trace_id"],
            parent_id=info["parent_id"],
            name=info["name"],
            span_type=info["span_type"],
            ended_at=ended_at,
            input_text=handle._input,
            output_text=handle._output,
            error=error,
            metadata=handle._metadata,
        )
        _open_spans.pop(span_id, None)

    for trace_id, info in list(_open_traces.items()):
        handle = info["handle"]
        _close_trace(
            trace_id=trace_id,
            name=info["name"],
            agent_id=info["agent_id"],
            ended_at=ended_at,
            error=error,
            metadata=dict(handle._metadata),
        )
        _open_traces.pop(trace_id, None)


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

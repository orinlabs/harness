"""Trace + span API.

Transport-agnostic. Every HTTP / storage concern lives in a ``TraceSink``
(``harness.core.tracing``); this module just manages the trace/span
lifecycle, ContextVar-based parenting, and the emergency "close everything
on SIGTERM" registry.

Module-level ``_sink`` is the currently-installed sink. It auto-populates on
first use via ``harness.cloud.autoconfigure()`` (returns a Bedrock sink when
the env is set, else ``NullTraceSink``). Callers that want explicit control
(most notably ``Harness.__init__`` when the user passed ``trace_sink=...``)
call ``set_trace_sink(sink)`` before the first span opens.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from harness.core.tracing import TraceSink

logger = logging.getLogger(__name__)


class SpanType(StrEnum):
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
_current_trace_name: ContextVar[str | None] = ContextVar("harness_trace_name", default=None)
_current_trace_agent_id: ContextVar[str | None] = ContextVar("harness_trace_agent_id", default=None)
_current_parent_span_id: ContextVar[str | None] = ContextVar("harness_parent_span_id", default=None)

# Emergency registry of still-open traces/spans.
#
# ContextVars are scoped per-task, so a signal handler or atexit hook can't see
# them. We mirror open-trace and open-span info into these module-level dicts
# so `close_all_open(error)` can force-close anything left dangling when the
# process is killed. Entries are removed on normal close.
_open_traces: dict[str, dict[str, Any]] = {}
_open_spans: dict[str, dict[str, Any]] = {}


_sink: TraceSink | None = None


def set_trace_sink(sink: TraceSink) -> None:
    """Install a ``TraceSink`` for subsequent span events.

    Called from ``Harness.__init__`` when the user wants an explicit sink
    (or from tests). Replaces any previously-installed sink.
    """
    global _sink
    _sink = sink


def get_trace_sink() -> TraceSink:
    """Return the active ``TraceSink``, auto-configuring on first access.

    Deferred import of ``harness.cloud.autoconfig`` keeps ``harness.core`` free
    of any cloud-provider dependency at import time.
    """
    global _sink
    if _sink is None:
        from harness.cloud.autoconfig import autoconfigure

        sink, _runtime = autoconfigure()
        _sink = sink
    return _sink


def _reset_sink_for_tests() -> None:
    """Clear the cached sink so the next ``get_trace_sink()`` re-runs autoconfig.

    Test-only helper; ``tests/conftest.py`` calls this between tests so env
    changes from one test don't leak into the next.
    """
    global _sink
    _sink = None


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def get_current_trace_id() -> str | None:
    return _current_trace_id.get()


def get_current_span_id() -> str | None:
    return _current_parent_span_id.get()


# ---------------------------------------------------------------------------
# Span handle
# ---------------------------------------------------------------------------


class Span:
    """In-process handle for an open span.

    Callers mutate ``input_text``, ``output_text``, and ``metadata`` via the
    helper methods; they're shipped to the sink when the span closes.
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
    """Open a span. If no trace is active, one is created implicitly."""
    sink = get_trace_sink()

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
        try:
            sink.open_trace(
                trace_id=trace_id,
                name=name,
                started_at=started_at,
                agent_id=agent_id,
            )
        except Exception:  # noqa: BLE001
            logger.exception("tracer: sink.open_trace failed")
        trace_id_token = _current_trace_id.set(trace_id)
        trace_name_token = _current_trace_name.set(name)
        trace_agent_token = _current_trace_agent_id.set(agent_id)

    parent_id = _current_parent_span_id.get()

    try:
        sink.open_span(
            span_id=span_id,
            trace_id=trace_id,
            parent_id=parent_id,
            name=name,
            span_type=span_type.value,
            started_at=started_at,
            input_text=input,
            metadata=metadata or {},
        )
    except Exception:  # noqa: BLE001
        logger.exception("tracer: sink.open_span failed")

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
        try:
            sink.close_span(
                span_id=span_id,
                trace_id=trace_id,
                parent_id=parent_id,
                name=name,
                span_type=span_type.value,
                ended_at=ended_at,
                input_text=handle._input,
                output_text=handle._output,
                error=error_msg if status == "error" else None,
                metadata=handle._metadata,
            )
        except Exception:  # noqa: BLE001
            logger.exception("tracer: sink.close_span failed")
        if created_trace:
            _open_traces.pop(trace_id, None)
            try:
                sink.close_trace(
                    trace_id=trace_id,
                    name=name,
                    agent_id=agent_id,
                    ended_at=ended_at,
                    error=error_msg if status == "error" else None,
                    # Propagate the root span's metadata up to the trace so
                    # backends can show a usage summary on the trace row.
                    metadata=dict(handle._metadata),
                )
            except Exception:  # noqa: BLE001
                logger.exception("tracer: sink.close_trace failed")
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

    Unlike ``span()``, this doesn't open a context -- useful when you want
    to log a span *only if* something happened, and the work was already
    bracketed with manual timestamps. Nests under whichever span is
    currently active via the same ContextVars.

    If no trace is active when this is called, a trace is created and
    closed around this single span (to preserve the invariant that every
    span belongs to a trace).
    """
    sink = get_trace_sink()
    span_id = str(uuid.uuid4())
    metadata = metadata or {}

    trace_id = _current_trace_id.get()
    standalone_trace = False
    if trace_id is None:
        trace_id = str(uuid.uuid4())
        standalone_trace = True
        try:
            sink.open_trace(
                trace_id=trace_id,
                name=name,
                started_at=started_at,
                agent_id=agent_id,
            )
        except Exception:  # noqa: BLE001
            logger.exception("tracer: sink.open_trace failed")

    parent_id = _current_parent_span_id.get()
    try:
        sink.open_span(
            span_id=span_id,
            trace_id=trace_id,
            parent_id=parent_id,
            name=name,
            span_type=span_type.value,
            started_at=started_at,
            input_text=input,
            metadata=metadata,
        )
        sink.close_span(
            span_id=span_id,
            trace_id=trace_id,
            parent_id=parent_id,
            name=name,
            span_type=span_type.value,
            ended_at=ended_at,
            input_text=input,
            output_text=output,
            error=error,
            metadata=metadata,
        )
    except Exception:  # noqa: BLE001
        logger.exception("tracer: sink.open_span/close_span failed")

    if standalone_trace:
        try:
            sink.close_trace(
                trace_id=trace_id,
                name=name,
                agent_id=agent_id,
                ended_at=ended_at,
                error=error,
                metadata=dict(metadata),
            )
        except Exception:  # noqa: BLE001
            logger.exception("tracer: sink.close_trace failed")


def close_all_open(error: str) -> None:
    """Force-close every still-open span and trace with ``error``.

    Called from the CLI's signal handlers / atexit hook when the harness is
    killed mid-run. Iterates over a snapshot of the registries so closing
    is idempotent even if the normal ``finally`` path also fires.
    """
    if not _open_spans and not _open_traces:
        return
    sink = get_trace_sink()
    ended_at = _now_iso()
    logger.info(
        "tracer: force-closing %d span(s) and %d trace(s) (%s)",
        len(_open_spans),
        len(_open_traces),
        error,
    )

    for span_id, info in list(_open_spans.items()):
        handle: Span = info["handle"]
        handle._metadata.setdefault("error", error)
        try:
            sink.close_span(
                span_id=span_id,
                trace_id=info["trace_id"],
                parent_id=info["parent_id"],
                name=info["name"],
                span_type=info["span_type"].value,
                ended_at=ended_at,
                input_text=handle._input,
                output_text=handle._output,
                error=error,
                metadata=handle._metadata,
            )
        except Exception:  # noqa: BLE001
            logger.exception("tracer: sink.close_span failed during drain")
        _open_spans.pop(span_id, None)

    for trace_id, info in list(_open_traces.items()):
        handle = info["handle"]
        try:
            sink.close_trace(
                trace_id=trace_id,
                name=info["name"],
                agent_id=info["agent_id"],
                ended_at=ended_at,
                error=error,
                metadata=dict(handle._metadata),
            )
        except Exception:  # noqa: BLE001
            logger.exception("tracer: sink.close_trace failed during drain")
        _open_traces.pop(trace_id, None)

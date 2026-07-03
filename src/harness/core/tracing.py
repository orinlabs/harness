"""Trace-sink protocol + built-in sinks.

`harness.core.tracer` holds a module-level ``TraceSink`` and delegates every
HTTP emission through it. The tracer module itself is transport-agnostic;
concrete backends (e.g. ``BedrockTraceSink`` in ``harness.cloud.bedrock``)
implement this protocol.

Three built-in sinks live here because they have no external deps:

* ``NullTraceSink`` -- drops everything. Kept for callers that explicitly
  want tracing off.
* ``StdoutTraceSink`` -- prints every trace/span event as one
  machine-readable ``@@trace {...}`` line on stdout. The default when no
  cloud backend is configured, so a standalone run is fully reproducible
  from its stdout stream (same supervisor contract as the ``@@sleep``
  sentinel in ``harness.core.runtime``).
* ``InMemoryTraceSink`` -- collects traces/spans into plain dicts for tests.
  Useful when a test wants to assert on tracer behavior without spinning up
  a fake HTTP server.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Sentinel prefix for the machine-readable trace lines StdoutTraceSink
# prints. Same contract as SLEEP_SENTINEL_PREFIX in harness.core.runtime:
# supervisors holding the stdout pipe match lines starting with this prefix
# and parse the JSON that follows. Must never appear at the start of any
# other harness output.
TRACE_SENTINEL_PREFIX = "@@trace "


@runtime_checkable
class TraceSink(Protocol):
    """Transport for trace + span lifecycle events.

    Every method is expected to be best-effort and non-raising. Implementations
    that talk to a remote service should log-and-swallow transport errors so a
    flaky backend can never take down the agent run.
    """

    def open_trace(
        self,
        *,
        trace_id: str,
        name: str,
        started_at: str,
        agent_id: str | None,
    ) -> None: ...

    def close_trace(
        self,
        *,
        trace_id: str,
        name: str,
        agent_id: str | None,
        ended_at: str,
        error: str | None,
        metadata: dict[str, Any],
    ) -> None: ...

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
    ) -> None: ...

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
    ) -> None: ...


class NullTraceSink:
    """Trace sink that drops every event. Default when no backend is wired."""

    def open_trace(self, **_: Any) -> None:
        return None

    def close_trace(self, **_: Any) -> None:
        return None

    def open_span(self, **_: Any) -> None:
        return None

    def close_span(self, **_: Any) -> None:
        return None


class StdoutTraceSink:
    """Print every trace/span lifecycle event as one stdout line.

    Line format (mirrors the ``TraceSink`` protocol kwargs, plus ``event``)::

        @@trace {"event": "open_span", "span_id": "...", "trace_id": "...", ...}

    Because the harness routes *everything* through spans -- the ``run_agent``
    root, per-turn spans, ``openrouter_api_call`` (with request/response
    JSON), ``thinking`` (with reasoning plaintext), and one tool span per
    tool call (args in ``input_text``, result in ``output_text``) -- a
    supervisor that captures stdout can reconstruct the entire run from
    these lines alone: ordering from the pipe, nesting from
    ``parent_id``/``trace_id``, timing from ``started_at``/``ended_at``.

    Same delivery contract as the ``@@sleep`` sentinel: one line per event,
    JSON keeps embedded newlines escaped, flushed explicitly so lines can't
    sit in a buffered stdout if the process dies. Without a supervisor the
    lines are inert log noise.

    Best-effort and non-raising, per the ``TraceSink`` protocol: emission
    failures are logged at WARNING and swallowed.
    """

    def open_trace(
        self,
        *,
        trace_id: str,
        name: str,
        started_at: str,
        agent_id: str | None,
    ) -> None:
        self._emit(
            "open_trace",
            {
                "trace_id": trace_id,
                "name": name,
                "started_at": started_at,
                "agent_id": agent_id,
            },
        )

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
        self._emit(
            "close_trace",
            {
                "trace_id": trace_id,
                "name": name,
                "agent_id": agent_id,
                "ended_at": ended_at,
                "error": error or "",
                "metadata": metadata,
            },
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
        self._emit(
            "open_span",
            {
                "span_id": span_id,
                "trace_id": trace_id,
                "parent_id": parent_id,
                "name": name,
                "span_type": span_type,
                "started_at": started_at,
                "input_text": input_text,
                "metadata": metadata,
            },
        )

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
        self._emit(
            "close_span",
            {
                "span_id": span_id,
                "trace_id": trace_id,
                "parent_id": parent_id,
                "name": name,
                "span_type": span_type,
                "ended_at": ended_at,
                "input_text": input_text or "",
                "output_text": output_text or "",
                "error": error or "",
                "metadata": metadata,
            },
        )

    @staticmethod
    def _emit(event: str, payload: dict[str, Any]) -> None:
        try:
            # default=str keeps oddball metadata values (datetimes, enums)
            # from killing the line; json escaping keeps it single-line.
            line = json.dumps({"event": event, **payload}, default=str, ensure_ascii=False)
            print(f"{TRACE_SENTINEL_PREFIX}{line}", file=sys.stdout, flush=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("StdoutTraceSink: failed to emit %s event: %s", event, e)


class InMemoryTraceSink:
    """Sink that stores trace/span events as plain dicts.

    Intended for tests that want to assert on what the tracer emitted without
    standing up a fake HTTP server. Mirrors the shape of the wire payloads:

        traces_open[trace_id] = {"id", "name", "started_at", "agent_id", "metadata"}
        traces_closed[trace_id] = {"ended_at", "error", "metadata"}
        spans_open[span_id] = {"id", "trace_id", "parent_id", "name", "span_type",
                               "started_at", "input_text", "metadata"}
        spans_closed[span_id] = {"ended_at", "input_text", "output_text",
                                 "error", "metadata"}
    """

    def __init__(self) -> None:
        self.traces_open: dict[str, dict[str, Any]] = {}
        self.traces_closed: dict[str, dict[str, Any]] = {}
        self.spans_open: dict[str, dict[str, Any]] = {}
        self.spans_closed: dict[str, dict[str, Any]] = {}

    def open_trace(
        self,
        *,
        trace_id: str,
        name: str,
        started_at: str,
        agent_id: str | None,
    ) -> None:
        self.traces_open[trace_id] = {
            "id": trace_id,
            "name": name,
            "started_at": started_at,
            "agent_id": agent_id,
            "metadata": {},
        }

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
        self.traces_closed[trace_id] = {
            "ended_at": ended_at,
            "error": error or "",
            "metadata": metadata,
        }

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
        self.spans_open[span_id] = {
            "id": span_id,
            "trace_id": trace_id,
            "parent_id": parent_id,
            "name": name,
            "span_type": span_type,
            "started_at": started_at,
            "input_text": input_text,
            "metadata": metadata,
        }

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
        self.spans_closed[span_id] = {
            "ended_at": ended_at,
            "input_text": input_text or "",
            "output_text": output_text or "",
            "error": error or "",
            "metadata": metadata,
        }

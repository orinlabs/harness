"""Pick a trace sink and agent runtime based on the environment.

Trace sink selection:

1. ``HARNESS_TRACE_SINK`` env var, when set to a known value:
   ``bedrock`` | ``stdout`` | ``null``. The CLI's ``--trace-sink`` flag
   normalizes onto this var (same pattern as ``--bedrock-url`` ->
   ``$BEDROCK_URL``).
2. Otherwise automatic: ``bedrock`` when both ``BEDROCK_URL`` and
   ``BEDROCK_TOKEN`` are present, else ``stdout`` (standalone runs stay
   fully reproducible from their stdout stream).

Runtime selection is independent of the sink choice and keyed on Bedrock
env alone: the runtime is a process-lifecycle concern (who SIGTERMs and
re-spawns us at wake time), so it follows whether a platform actually
manages this process -- not where traces are shipped. Forcing
``--trace-sink stdout`` on a Bedrock-managed run still POSTs sleep to the
platform.

Callers that want explicit control construct their sink/runtime directly and
pass them into ``Harness(..., trace_sink=..., runtime=...)``.
"""

from __future__ import annotations

import logging
import os

from harness.core.runtime import AgentRuntime, LocalAgentRuntime
from harness.core.tracing import NullTraceSink, StdoutTraceSink, TraceSink

logger = logging.getLogger(__name__)

_SINK_CHOICES = ("bedrock", "stdout", "null")


def _have_bedrock_env() -> bool:
    return bool(os.environ.get("BEDROCK_URL") and os.environ.get("BEDROCK_TOKEN"))


def _sink_choice() -> str:
    """Resolve the trace-sink choice: explicit env override, else automatic."""
    choice = os.environ.get("HARNESS_TRACE_SINK", "").strip().lower()
    if choice in _SINK_CHOICES:
        return choice
    if choice:
        logger.warning(
            "autoconfigure: unknown HARNESS_TRACE_SINK=%r (expected one of %s); "
            "falling back to automatic selection",
            choice,
            "|".join(_SINK_CHOICES),
        )
    return "bedrock" if _have_bedrock_env() else "stdout"


def _build_sink(choice: str) -> TraceSink:
    if choice == "bedrock":
        from harness.cloud.bedrock import BedrockTraceSink

        if not _have_bedrock_env():
            # BedrockTraceSink short-circuits per call while BEDROCK_URL is
            # unset, so this behaves like the null sink until the env shows
            # up. Surface that so an explicit `--trace-sink bedrock` without
            # creds isn't silently traceless.
            logger.warning(
                "autoconfigure: trace sink 'bedrock' selected but BEDROCK_URL/"
                "BEDROCK_TOKEN are not both set; spans will be dropped until "
                "the env appears"
            )
        return BedrockTraceSink()
    if choice == "null":
        return NullTraceSink()
    return StdoutTraceSink()


def autoconfigure() -> tuple[TraceSink, AgentRuntime]:
    """Return ``(trace_sink, runtime)`` chosen from the current environment.

    Deferred imports keep ``harness.core`` independent of
    ``harness.cloud.bedrock`` at import time.
    """
    sink_choice = _sink_choice()
    sink = _build_sink(sink_choice)

    if _have_bedrock_env():
        from harness.cloud.bedrock import BedrockAgentRuntime

        runtime: AgentRuntime = BedrockAgentRuntime()
    else:
        runtime = LocalAgentRuntime()

    logger.info(
        "autoconfigure: trace_sink=%s runtime=%s",
        type(sink).__name__,
        type(runtime).__name__,
    )
    return sink, runtime

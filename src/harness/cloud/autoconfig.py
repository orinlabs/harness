"""Pick a trace sink and agent runtime based on the environment.

The contract is simple: if Bedrock credentials are present, return the
Bedrock-backed implementations; otherwise return the no-op / local defaults.

Callers that want explicit control construct their sink/runtime directly and
pass them into ``Harness(..., trace_sink=..., runtime=...)``.
"""

from __future__ import annotations

import logging
import os

from harness.core.runtime import AgentRuntime, LocalAgentRuntime
from harness.core.tracing import NullTraceSink, TraceSink

logger = logging.getLogger(__name__)


def _have_bedrock_env() -> bool:
    return bool(os.environ.get("BEDROCK_URL") and os.environ.get("BEDROCK_TOKEN"))


def autoconfigure() -> tuple[TraceSink, AgentRuntime]:
    """Return ``(trace_sink, runtime)`` chosen from the current environment.

    Order of precedence today (add providers here as they land):
      1. Bedrock -- when both ``BEDROCK_URL`` and ``BEDROCK_TOKEN`` are set.
      2. Fallback -- ``NullTraceSink`` + ``LocalAgentRuntime`` (standalone).

    Deferred imports keep ``harness.core`` independent of
    ``harness.cloud.bedrock`` at import time.
    """
    if _have_bedrock_env():
        from harness.cloud.bedrock import BedrockAgentRuntime, BedrockTraceSink

        logger.info("autoconfigure: using Bedrock trace sink + runtime")
        return BedrockTraceSink(), BedrockAgentRuntime()

    logger.info("autoconfigure: Bedrock env not set, using NullTraceSink + LocalAgentRuntime")
    return NullTraceSink(), LocalAgentRuntime()

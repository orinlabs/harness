"""Agent-runtime protocol + local (standalone) implementation.

An ``AgentRuntime`` is the process-lifecycle counterpart to a ``TraceSink``:
it's what the built-in ``SleepTool`` calls into when the model asks the agent
to go idle. Backends vary by environment:

* ``LocalAgentRuntime`` (this file) -- standalone runs. ``sleep`` does no
  network I/O. The caller is expected to set ``ctx.sleep_requested = True``
  immediately after, which causes the harness loop to exit cleanly at the
  end of the current turn. There is no process supervisor to wake us back
  up, so "sleep" effectively means "stop this run". This matches the
  behavior the user sees on Bedrock (the process exits), minus the resume.

* ``BedrockAgentRuntime`` (``harness.cloud.bedrock``) -- production. POSTs
  the sleep request to the platform, which then takes responsibility for
  SIGTERM-ing this process and spawning a fresh one at wake time.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Sentinel prefix for the machine-readable sleep line LocalAgentRuntime
# prints on stdout. Supervisors that spawn the harness as a subprocess
# match lines starting with this prefix and parse the JSON that follows.
# Must never appear at the start of any other harness output.
SLEEP_SENTINEL_PREFIX = "@@sleep "


@runtime_checkable
class AgentRuntime(Protocol):
    """Platform operations that are scoped to a single agent.

    Currently just ``sleep``. As more agent-scoped lifecycle hooks land on
    the platform (e.g. ``wake_now``, ``report_state``), they go here.
    """

    def sleep(self, agent_id: str, *, until: str, reason: str) -> dict[str, Any]: ...


class LocalAgentRuntime:
    """Standalone runtime. ``sleep`` prints a sentinel line and returns.

    The ``SleepTool`` sets ``ctx.sleep_requested = True`` after calling us, so
    the harness loop exits cleanly at the end of the current turn. There is no
    platform to notify; instead we print one machine-readable line to stdout::

        @@sleep {"until": "2026-06-30T09:00:00Z", "reason": "..."}

    A supervisor running the harness as a subprocess (e.g. a simulated
    environment that wants to advance its world clock to the wake time)
    parses that line from the pipe it already holds. The pipe is ordered,
    dies with the process, and needs no cleanup. Flushed explicitly so the
    line can't be lost in a buffered stdout when the process exits right
    after sleeping. Without a supervisor the line is inert log noise.
    """

    def sleep(self, agent_id: str, *, until: str, reason: str) -> dict[str, Any]:
        logger.info(
            "LocalAgentRuntime.sleep: agent=%s until=%s reason=%s "
            "(standalone mode -- run will exit; re-launch to resume)",
            agent_id,
            until,
            reason,
        )
        payload = json.dumps(
            {"agent_id": agent_id, "until": until, "reason": reason},
            ensure_ascii=False,
        )
        print(f"{SLEEP_SENTINEL_PREFIX}{payload}", file=sys.stdout, flush=True)
        return {}

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

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class AgentRuntime(Protocol):
    """Platform operations that are scoped to a single agent.

    Currently just ``sleep``. As more agent-scoped lifecycle hooks land on
    the platform (e.g. ``wake_now``, ``report_state``), they go here.
    """

    def sleep(self, agent_id: str, *, until: str, reason: str) -> dict[str, Any]: ...


class LocalAgentRuntime:
    """Standalone runtime. ``sleep`` is a log-only no-op.

    The ``SleepTool`` sets ``ctx.sleep_requested = True`` after calling us, so
    the harness loop exits cleanly at the end of the current turn. For
    standalone runs there's no supervisor to re-launch the process at ``until``
    -- if the user wants that they need a real platform (e.g. Bedrock).
    """

    def sleep(self, agent_id: str, *, until: str, reason: str) -> dict[str, Any]:
        logger.info(
            "LocalAgentRuntime.sleep: agent=%s until=%s reason=%s "
            "(standalone mode -- run will exit; re-launch to resume)",
            agent_id,
            until,
            reason,
        )
        return {}

"""Built-in sleep tool.

Calling this tool tells the infra platform to put the agent to sleep. The
harness process exits cleanly after the current turn. On wake, infra spins
up a new process with the same agent_id and a new run_id.

Before actually sleeping we ask the agent's own ``list_notifications`` tool
(if one is registered) whether there are any pending notifications. If so,
we refuse to sleep and hand the notification list back to the model -- the
agent should address / clear each item before going idle. This mirrors the
pre-harness bedrock SleepTool's urgent-notification gate, but implemented
as a cross-tool call so the harness doesn't need its own notifications API
client.
"""
from __future__ import annotations

import logging

from harness.context import RunContext
from harness.core import runtime_api
from harness.tools.base import ToolResult, ToolSchema

logger = logging.getLogger(__name__)

_LIST_NOTIFICATIONS_TOOL = "list_notifications"
# The ``list_notifications`` handler (bedrock-side) returns exactly this
# sentence when there are no active, uncleared notifications. We match on
# a substring so minor punctuation drift ("You have no pending
# notifications" vs "... notifications.") still reads as "empty inbox".
_EMPTY_MARKER = "no pending notifications"


class SleepTool:
    name = "sleep"
    description = (
        "Put yourself to sleep until a specific time (or indefinitely). "
        "Use this when you have finished your current work and nothing else is pending. "
        "The infra platform will wake you up again at the specified time."
    )
    parameters = {
        "type": "object",
        "properties": {
            "until": {
                "type": "string",
                "description": (
                    'ISO-8601 timestamp (e.g. "2099-01-01T00:00:00Z") when you should wake up, '
                    'or the string "indefinite" if you should only be woken by an external event.'
                ),
            },
            "reason": {
                "type": "string",
                "description": "Short explanation of why you are going to sleep.",
            },
        },
        "required": ["until"],
    }

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(self.name, self.description, self.parameters)

    def call(self, args: dict, ctx: RunContext) -> ToolResult:
        until = str(args.get("until") or "indefinite")
        reason = str(args.get("reason") or "")

        blocked = _notifications_block_sleep(ctx)
        if blocked is not None:
            # Don't flip ctx.sleep_requested -- the agent must keep running
            # so it can actually address the notifications.
            logger.info(
                "sleep refused for agent=%s: pending notifications present",
                ctx.agent_id,
            )
            return ToolResult(
                text=(
                    "Cannot sleep: there are pending notifications you need to address "
                    "and clear first.\n\n" + blocked
                )
            )

        runtime_api.sleep(ctx.agent_id, until=until, reason=reason)
        ctx.sleep_requested = True
        return ToolResult(text=f"Sleeping until {until}.")


def _notifications_block_sleep(ctx: RunContext) -> str | None:
    """Return the notifications-listing text if sleep should be blocked.

    Invokes the agent's own ``list_notifications`` tool (if registered) and
    returns its text when it reports any active notifications. Returns
    ``None`` when sleep is allowed -- no tool registered, empty inbox,
    the call raised, or the tool reported an error. Any failure mode
    is logged and falls through to "allow sleep"; we don't want a broken
    notifications adapter to permanently pin the agent awake.
    """
    tool = ctx.tool_map.get(_LIST_NOTIFICATIONS_TOOL)
    if tool is None:
        return None

    try:
        result = tool.call({}, ctx)
    except Exception:  # noqa: BLE001
        logger.warning(
            "list_notifications raised during sleep pre-check for agent=%s; allowing sleep",
            ctx.agent_id,
            exc_info=True,
        )
        return None

    text = (getattr(result, "text", "") or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if _EMPTY_MARKER in lowered:
        return None
    # The handler returns an "Error listing notifications: ..." string on
    # its own exception path -- treat that like "couldn't determine" and
    # let the agent sleep rather than wedging it over a transient failure.
    if lowered.startswith("error"):
        logger.warning(
            "list_notifications returned an error during sleep pre-check for agent=%s: %s",
            ctx.agent_id,
            text[:200],
        )
        return None
    return text

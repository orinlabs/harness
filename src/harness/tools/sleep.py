"""Built-in sleep tool.

Delegates to ``ctx.runtime.sleep(...)`` (an ``AgentRuntime``). In standalone
mode the runtime is a no-op and the harness process exits cleanly after the
current turn. In Bedrock mode the runtime POSTs to the platform, which then
SIGTERMs this process and re-spawns a fresh one at wake time.

Before actually sleeping we ask the agent's own ``list_notifications`` tool
(if one is registered) whether there are any pending notifications. If so,
we refuse to sleep and hand the notification list back to the model -- the
agent should address / clear each item before going idle. This is
implemented as a cross-tool call so the harness doesn't need its own
notifications API client.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from harness.context import RunContext
from harness.tools.base import ToolResult, ToolSchema

logger = logging.getLogger(__name__)

_LIST_NOTIFICATIONS_TOOL = "list_notifications"
# The ``list_notifications`` handler returns exactly this sentence when there
# are no active, uncleared notifications. We match on a substring so minor
# punctuation drift ("You have no pending notifications" vs "...
# notifications.") still reads as "empty inbox".
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
                    'ISO-8601 timestamp (e.g. "2099-01-01T09:00:00") when you should wake up. '
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

        schedule = _resolve_sleep_schedule(
            until,
            max_until_utc=ctx.max_sleep_until,
            timezone_name=ctx.timezone_name,
        )

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

        if ctx.runtime is None:
            # Defensive: Harness always wires a runtime onto ctx. A missing
            # runtime means the caller built a RunContext by hand for a test
            # and forgot to attach one. Fall back to "request sleep, no
            # platform side effect" so the model's state machine still
            # behaves -- tests that care about the network call inject an
            # AgentRuntime explicitly.
            logger.warning(
                "SleepTool called without ctx.runtime; treating as local no-op "
                "(agent=%s, until=%s)",
                ctx.agent_id,
                schedule.runtime_until,
            )
        else:
            ctx.runtime.sleep(ctx.agent_id, until=schedule.runtime_until, reason=reason)
        ctx.sleep_requested = True
        if schedule.clamped:
            logger.info(
                "sleep clamped for agent=%s: requested until=%s, capped at %s",
                ctx.agent_id,
                until,
                schedule.runtime_until,
            )
            return ToolResult(
                text=(
                    f"Sleeping until {schedule.display_until}. (Your requested wake time of "
                    f"{until!r} was past the platform-enforced maximum, "
                    f"so it was capped at {schedule.display_until}.)"
                )
            )
        return ToolResult(text=f"Sleeping until {schedule.display_until}.")


@dataclass(frozen=True)
class _SleepSchedule:
    runtime_until: str
    display_until: str
    clamped: bool


def _resolve_sleep_schedule(
    until: str,
    *,
    max_until_utc: datetime | None,
    timezone_name: str | None,
) -> _SleepSchedule:
    """Convert the agent-local wake time to UTC and apply an optional cap.

    ``runtime_until`` is what goes to Bedrock/local runtime (always UTC for
    timestamp sleeps). ``display_until`` is what the agent sees back. When
    the agent has a configured timezone, naive timestamps are interpreted
    in that zone and display text stays in that zone; otherwise UTC is used.
    """
    if until == "indefinite":
        if max_until_utc is None:
            return _SleepSchedule("indefinite", "indefinite", clamped=False)
        return _SleepSchedule(
            _format_utc(max_until_utc),
            _format_for_agent(max_until_utc, timezone_name),
            clamped=True,
        )

    requested_utc: datetime | None = None
    try:
        requested_utc = _parse_agent_timestamp(until, timezone_name)
    except ValueError:
        if max_until_utc is None:
            logger.warning(
                "sleep until=%r is not parseable ISO-8601; passing through without cap",
                until,
            )
            return _SleepSchedule(until, until, clamped=False)
        logger.warning(
            "sleep until=%r is not parseable ISO-8601; clamping to max %s",
            until,
            max_until_utc,
        )

    if requested_utc is None or (max_until_utc is not None and requested_utc > max_until_utc):
        assert max_until_utc is not None
        return _SleepSchedule(
            _format_utc(max_until_utc),
            _format_for_agent(max_until_utc, timezone_name),
            clamped=True,
        )

    return _SleepSchedule(
        _format_utc(requested_utc),
        _format_for_agent(requested_utc, timezone_name),
        clamped=False,
    )


def _parse_agent_timestamp(value: str, timezone_name: str | None) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_agent_tz(timezone_name))
    return dt.astimezone(UTC)


def _agent_tz(timezone_name: str | None):
    return ZoneInfo(timezone_name) if timezone_name else UTC


def _format_utc(dt: datetime) -> str:
    """Render an aware datetime as UTC ISO-8601 with a ``Z`` suffix."""
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _format_for_agent(dt: datetime, timezone_name: str | None) -> str:
    if timezone_name:
        return dt.astimezone(_agent_tz(timezone_name)).isoformat()
    return _format_utc(dt)


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

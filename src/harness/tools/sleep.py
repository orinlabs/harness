"""Built-in sleep tool.

Calling this tool tells the infra platform to put the agent to sleep. The
harness process exits cleanly after the current turn. On wake, infra spins
up a new process with the same agent_id and a new run_id.
"""
from __future__ import annotations

from harness.context import RunContext
from harness.core import runtime_api
from harness.tools.base import ToolResult, ToolSchema


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
        runtime_api.sleep(ctx.agent_id, until=until, reason=reason)
        ctx.sleep_requested = True
        return ToolResult(text=f"Sleeping until {until}.")

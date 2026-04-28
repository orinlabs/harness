"""In-process fake SMS adapter for evals.

Mirrors ``defaults.sms`` (Twilio-backed in production) but stores
messages in the agent's sqlite DB. Conversations are keyed by phone
number since we deliberately do not depend on the contacts fake at the
schema level (scenarios that enable both adapters still work; scenarios
that enable only SMS still work).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from harness.tools.base import Tool, ToolResult, ToolSchema

from .base import new_id, now_iso, require_db

if TYPE_CHECKING:
    from harness.context import RunContext


AGENT_PHONE_NUMBER = "+15550000000"


# ---------------------------------------------------------------------------
# Inbound injection
# ---------------------------------------------------------------------------


def inject_inbound(contact_phone: str, body: str, sent_at: str | None = None) -> str:
    """Simulate an inbound SMS from ``contact_phone``. Returns message id."""
    db = require_db()
    msg_id = new_id("sim_sms")
    db.execute(
        "INSERT INTO fake_sms_message (id, contact_phone, direction, body, sent_at) "
        "VALUES (?, ?, 'inbound', ?, ?)",
        (msg_id, contact_phone, body, sent_at or now_iso()),
    )
    return msg_id


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class _ToolBase:
    name: str = ""
    description: str = ""
    parameters: dict = {}

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(self.name, self.description, self.parameters)


class ListConversationsTool(_ToolBase):
    name = "list_conversations"
    description = (
        "List SMS conversations ordered by most recent activity. Shows "
        "latest message preview for each contact."
    )
    parameters = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max conversations to return (default 20, max 100).",
            },
        },
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        db = require_db()
        limit = min(int(args.get("limit") or 20), 100)
        rows = db.execute(
            "SELECT contact_phone, MAX(sent_at) AS latest_at, COUNT(*) AS count "
            "FROM fake_sms_message GROUP BY contact_phone "
            "ORDER BY latest_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

        if not rows:
            return ToolResult(text="No SMS conversations yet.")

        lines = ["SMS Conversations:", ""]
        for row in rows:
            latest = db.execute(
                "SELECT direction, body, sent_at FROM fake_sms_message "
                "WHERE contact_phone = ? ORDER BY sent_at DESC LIMIT 1",
                (row["contact_phone"],),
            ).fetchone()
            lines.append(f"• {row['contact_phone']}")
            lines.append(f"  Messages: {row['count']}")
            if latest:
                who = "You" if latest["direction"] == "outbound" else row["contact_phone"]
                lines.append(f"  Latest ({latest['sent_at']}):")
                lines.append(f"    {who}: {latest['body']}")
            lines.append("")

        return ToolResult(text="\n".join(lines))


class GetConversationTool(_ToolBase):
    name = "get_conversation"
    description = (
        "Get messages in a conversation with a specific phone number. "
        "Returns the most recent messages first."
    )
    parameters = {
        "type": "object",
        "properties": {
            "phone": {
                "type": "string",
                "description": "Phone number in E.164 format (e.g. +15551234567).",
            },
            "limit": {
                "type": "integer",
                "description": "Max messages to return (default 25, max 100).",
            },
        },
        "required": ["phone"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        phone = args.get("phone")
        if not phone:
            return ToolResult(text="Error: phone is required.")

        db = require_db()
        limit = min(int(args.get("limit") or 25), 100)
        rows = db.execute(
            "SELECT id, direction, body, sent_at FROM fake_sms_message "
            "WHERE contact_phone = ? ORDER BY sent_at DESC LIMIT ?",
            (phone, limit),
        ).fetchall()

        if not rows:
            return ToolResult(text=f"No messages with {phone}.")

        lines = [f"Conversation with {phone}", f"Showing {len(rows)} messages", ""]
        for row in reversed(list(rows)):
            who = "You" if row["direction"] == "outbound" else phone
            lines.append(f"[{row['sent_at']}] {who}:")
            lines.append(f"  {row['body']}")
            lines.append("")

        return ToolResult(text="\n".join(lines))


class SendSmsTool(_ToolBase):
    name = "send_sms"
    description = "Send an SMS message to a phone number."
    parameters = {
        "type": "object",
        "properties": {
            "phone": {
                "type": "string",
                "description": "Recipient phone number in E.164 format.",
            },
            "to": {
                "type": "string",
                "description": "Alias for 'phone' (matches production schema).",
            },
            "body": {"type": "string", "description": "Message body (max 1600 chars)."},
        },
        "required": ["body"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        phone = args.get("phone") or args.get("to")
        body = args.get("body") or ""
        if not phone or not body:
            return ToolResult(text="Error: phone and body are required.")

        db = require_db()
        msg_id = new_id("sim_sms_out")
        db.execute(
            "INSERT INTO fake_sms_message (id, contact_phone, direction, body, sent_at) "
            "VALUES (?, ?, 'outbound', ?, ?)",
            (msg_id, phone, body, now_iso()),
        )
        return ToolResult(text=f'SMS sent to {phone}:\n"{body}"')


class OpenAttachmentTool(_ToolBase):
    """Stub that always reports 'no attachments'.

    The fake SMS adapter does not carry media -- scenarios that
    exercised ``open_attachment`` in bedrock were MMS-focused. We keep
    the tool so schemas line up; the model gets a clean 'no attachments'
    response if it tries.
    """

    name = "open_attachment"
    description = (
        "Open an attachment from an SMS message. The fake adapter does "
        "not carry attachments; this tool always reports none available."
    )
    parameters = {
        "type": "object",
        "properties": {
            "message_id": {"type": "string", "description": "Message ID."},
            "attachment_index": {
                "type": "integer",
                "description": "Attachment index (default 0).",
            },
        },
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        return ToolResult(text="No message with attachments found.")


# ---------------------------------------------------------------------------
# Adapter assembly
# ---------------------------------------------------------------------------


class FakeSMSAdapter:
    name = "FakeSMS"
    description = "In-process SMS adapter for evals (sqlite-backed)."
    TOOLS = [
        SendSmsTool,
        ListConversationsTool,
        GetConversationTool,
        OpenAttachmentTool,
    ]

    @classmethod
    def make_tools(cls) -> list[Tool]:
        return [T() for T in cls.TOOLS]

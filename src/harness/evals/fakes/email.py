"""In-process fake email adapter for evals.

Mimics the production ``defaults.email`` tools (AgentMail-backed) but
stores threads / messages in the agent's sqlite DB. All six tools return
plain text responses shaped like the production handlers so scenarios
that grep agent output (or check ``_checkpoint_trace_events``) work
without change.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from harness.tools.base import Tool, ToolResult, ToolSchema

from .base import new_id, now_iso, require_db

if TYPE_CHECKING:
    from harness.context import RunContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_list(value: Any) -> list[str]:
    """Normalize ``to`` / ``cc`` / ``bcc`` fields.

    Production ``send_email`` accepts either a list or a comma-separated
    string; we mirror that so scenarios can pass either shape.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


def _thread_participants(thread_id: str) -> list[str]:
    db = require_db()
    row = db.execute(
        "SELECT participants FROM fake_email_thread WHERE id = ?",
        (thread_id,),
    ).fetchone()
    if not row:
        return []
    try:
        return list(json.loads(row["participants"]))
    except (json.JSONDecodeError, TypeError):
        return []


def _merge_participants(existing: list[str], *more: str | list[str]) -> list[str]:
    out = list(existing)
    for group in more:
        items = [group] if isinstance(group, str) else list(group)
        for item in items:
            if item and item not in out:
                out.append(item)
    return out


def _record_outbound_message(
    thread_id: str | None,
    subject: str,
    from_addr: str,
    to_addrs: list[str],
    cc_addrs: list[str],
    body: str,
    sent_at: str | None = None,
) -> tuple[str, str]:
    """Persist an outbound message (and create a thread if needed).

    Returns (message_id, thread_id).
    """
    db = require_db()
    ts = sent_at or now_iso()
    msg_id = new_id("sim_msg")

    if thread_id is None:
        thread_id = new_id("sim_thread")
        db.execute(
            "INSERT INTO fake_email_thread (id, subject, participants, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                thread_id,
                subject,
                json.dumps(_merge_participants([from_addr], to_addrs, cc_addrs)),
                ts,
                ts,
            ),
        )
    else:
        participants = _merge_participants(
            _thread_participants(thread_id), [from_addr], to_addrs, cc_addrs
        )
        db.execute(
            "UPDATE fake_email_thread SET participants = ?, updated_at = ? WHERE id = ?",
            (json.dumps(participants), ts, thread_id),
        )

    db.execute(
        "INSERT INTO fake_email_message "
        "(id, thread_id, direction, from_addr, to_addrs, cc_addrs, subject, body, sent_at) "
        "VALUES (?, ?, 'outbound', ?, ?, ?, ?, ?, ?)",
        (
            msg_id,
            thread_id,
            from_addr,
            json.dumps(to_addrs),
            json.dumps(cc_addrs),
            subject,
            body,
            ts,
        ),
    )
    return msg_id, thread_id


# ---------------------------------------------------------------------------
# Inbound injection (called by scenarios, not the LLM)
# ---------------------------------------------------------------------------


def inject_inbound(
    thread_id: str | None,
    from_email: str,
    to_email: str | list[str],
    subject: str,
    body: str,
    sent_at: str | None = None,
) -> str:
    """Simulate an inbound email landing in the agent's inbox.

    If ``thread_id`` is None, a new thread is created using ``subject``.
    Returns the new message id.
    """
    db = require_db()
    ts = sent_at or now_iso()
    to_addrs = _split_list(to_email)
    msg_id = new_id("sim_inmsg")

    if thread_id is None:
        thread_id = new_id("sim_thread")
        db.execute(
            "INSERT INTO fake_email_thread (id, subject, participants, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                thread_id,
                subject,
                json.dumps(_merge_participants([from_email], to_addrs)),
                ts,
                ts,
            ),
        )
    else:
        participants = _merge_participants(
            _thread_participants(thread_id), [from_email], to_addrs
        )
        db.execute(
            "UPDATE fake_email_thread SET participants = ?, updated_at = ? WHERE id = ?",
            (json.dumps(participants), ts, thread_id),
        )

    db.execute(
        "INSERT INTO fake_email_message "
        "(id, thread_id, direction, from_addr, to_addrs, cc_addrs, subject, body, sent_at) "
        "VALUES (?, ?, 'inbound', ?, ?, '[]', ?, ?, ?)",
        (
            msg_id,
            thread_id,
            from_email,
            json.dumps(to_addrs),
            subject,
            body,
            ts,
        ),
    )
    return msg_id


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


AGENT_EMAIL_ADDRESS = "agent@eval.test"


class _ToolBase:
    """Tiny helper so all fake tools share the boilerplate schema property."""

    name: str = ""
    description: str = ""
    parameters: dict = {}

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(self.name, self.description, self.parameters)


class ListThreadsTool(_ToolBase):
    name = "list_threads"
    description = (
        "List email threads from your inbox. Shows subject, participants, "
        "and message counts. Use get_thread to read the full messages."
    )
    parameters = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max threads to return (default 10, max 100).",
            },
        },
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        db = require_db()
        limit = min(int(args.get("limit") or 10), 100)
        rows = db.execute(
            "SELECT id, subject, participants, updated_at "
            "FROM fake_email_thread ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

        if not rows:
            return ToolResult(text="No email threads in your inbox.")

        lines = [f"Email Threads ({len(rows)}):", ""]
        for row in rows:
            subject = row["subject"] or "(No subject)"
            try:
                participants = json.loads(row["participants"]) or []
            except (json.JSONDecodeError, TypeError):
                participants = []
            # Count messages and grab most recent preview.
            count_row = db.execute(
                "SELECT COUNT(*) AS c FROM fake_email_message WHERE thread_id = ?",
                (row["id"],),
            ).fetchone()
            preview_row = db.execute(
                "SELECT body FROM fake_email_message WHERE thread_id = ? "
                "ORDER BY sent_at DESC LIMIT 1",
                (row["id"],),
            ).fetchone()
            msg_count = count_row["c"] if count_row else 0
            preview = (preview_row["body"] if preview_row else "") or ""
            if len(preview) > 120:
                preview = preview[:120] + "..."

            lines.append(f"• {subject}")
            lines.append(f"  Thread ID: {row['id']}")
            if participants:
                lines.append(f"  Participants: {', '.join(participants)}")
            lines.append(f"  Messages: {msg_count}")
            if preview:
                lines.append(f"  Preview: {preview}")
            lines.append("")

        return ToolResult(text="\n".join(lines))


class GetThreadTool(_ToolBase):
    name = "get_thread"
    description = (
        "Get all messages in an email thread. Use this to read the full "
        "content of emails in a thread."
    )
    parameters = {
        "type": "object",
        "properties": {
            "thread_id": {
                "type": "string",
                "description": "The ID of the thread to retrieve",
            },
        },
        "required": ["thread_id"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        thread_id = args.get("thread_id")
        if not thread_id:
            return ToolResult(text="Error: thread_id is required.")

        db = require_db()
        thread = db.execute(
            "SELECT id, subject FROM fake_email_thread WHERE id = ?",
            (thread_id,),
        ).fetchone()
        if not thread:
            return ToolResult(text=f"Error: thread {thread_id} not found.")

        messages = db.execute(
            "SELECT id, direction, from_addr, to_addrs, cc_addrs, subject, body, sent_at "
            "FROM fake_email_message WHERE thread_id = ? ORDER BY sent_at ASC",
            (thread_id,),
        ).fetchall()

        lines = [
            f"Thread: {thread['subject'] or '(No subject)'}",
            f"Thread ID: {thread_id}",
            f"Messages: {len(messages)}",
            "",
            "-" * 50,
        ]
        for msg in messages:
            try:
                to_addrs = json.loads(msg["to_addrs"]) or []
            except (json.JSONDecodeError, TypeError):
                to_addrs = []
            try:
                cc_addrs = json.loads(msg["cc_addrs"]) or []
            except (json.JSONDecodeError, TypeError):
                cc_addrs = []
            lines.append("")
            lines.append(f"From: {msg['from_addr']}")
            lines.append(f"To: {', '.join(to_addrs)}")
            if cc_addrs:
                lines.append(f"CC: {', '.join(cc_addrs)}")
            lines.append(f"Date: {msg['sent_at']}")
            lines.append(f"Message ID: {msg['id']}")
            lines.append(f"Direction: {msg['direction']}")
            lines.append("")
            lines.append(msg["body"])
            lines.append("")
            lines.append("-" * 50)

        return ToolResult(text="\n".join(lines))


class SendEmailTool(_ToolBase):
    name = "send_email"
    description = (
        "Send a new email message (creates a new thread). Provide the "
        "recipient(s), subject, and body text."
    )
    parameters = {
        "type": "object",
        "properties": {
            "to": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "description": "Recipient(s). String or list of email addresses.",
            },
            "subject": {"type": "string", "description": "Email subject line."},
            "body": {"type": "string", "description": "Email body (plain text)."},
            "text": {
                "type": "string",
                "description": "Alias for 'body' (matches production schema).",
            },
            "cc": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "description": "Optional CC recipient(s).",
            },
        },
        "required": ["to", "subject"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        to = _split_list(args.get("to"))
        cc = _split_list(args.get("cc"))
        subject = args.get("subject") or ""
        body = args.get("body") or args.get("text") or ""
        if not to or not subject or not body:
            return ToolResult(text="Error: to, subject, and body are required.")

        _msg_id, thread_id = _record_outbound_message(
            thread_id=None,
            subject=subject,
            from_addr=AGENT_EMAIL_ADDRESS,
            to_addrs=to,
            cc_addrs=cc,
            body=body,
        )

        lines = [
            "Email sent successfully!",
            f"To: {', '.join(to)}",
            f"Subject: {subject}",
        ]
        if cc:
            lines.append(f"CC: {', '.join(cc)}")
        lines.append(f"Thread ID: {thread_id}")
        lines.append(f"Body: {body}")
        return ToolResult(text="\n".join(lines))


class ReplyToEmailTool(_ToolBase):
    name = "reply_to_email"
    description = (
        "Reply to an existing email message in a thread. Use to continue "
        "an existing conversation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "message_id": {"type": "string", "description": "Message ID to reply to."},
            "to": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "description": "Recipient(s).",
            },
            "body": {"type": "string", "description": "Reply body."},
            "text": {"type": "string", "description": "Alias for 'body'."},
        },
        "required": ["message_id", "to"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        message_id = args.get("message_id")
        to = _split_list(args.get("to"))
        body = args.get("body") or args.get("text") or ""
        if not message_id or not to or not body:
            return ToolResult(text="Error: message_id, to, and body are required.")

        db = require_db()
        parent = db.execute(
            "SELECT thread_id, subject FROM fake_email_message WHERE id = ?",
            (message_id,),
        ).fetchone()
        if not parent:
            return ToolResult(text=f"Error: message {message_id} not found.")

        thread_id = parent["thread_id"]
        parent_subject = parent["subject"] or ""
        reply_subject = (
            parent_subject
            if parent_subject.lower().startswith("re:")
            else f"Re: {parent_subject}"
        )
        _record_outbound_message(
            thread_id=thread_id,
            subject=reply_subject,
            from_addr=AGENT_EMAIL_ADDRESS,
            to_addrs=to,
            cc_addrs=[],
            body=body,
        )
        return ToolResult(text=f"Reply sent to {', '.join(to)}.\nMessage: {body}")


class SearchEmailsTool(_ToolBase):
    name = "search_emails"
    description = (
        "Search through emails by keyword, sender, or direction. Returns "
        "matching messages with subject, sender, and preview."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Free-text search across subject and body.",
            },
            "from_email": {
                "type": "string",
                "description": "Filter by sender (partial match).",
            },
            "direction": {
                "type": "string",
                "enum": ["inbound", "outbound"],
                "description": "Filter by direction.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 20, max 50).",
            },
        },
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        db = require_db()
        clauses: list[str] = []
        params: list[Any] = []

        query = args.get("query")
        if query:
            clauses.append("(subject LIKE ? OR body LIKE ?)")
            like = f"%{query}%"
            params.extend([like, like])

        from_email = args.get("from_email")
        if from_email:
            clauses.append("from_addr LIKE ?")
            params.append(f"%{from_email}%")

        direction = args.get("direction")
        if direction in ("inbound", "outbound"):
            clauses.append("direction = ?")
            params.append(direction)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit = min(int(args.get("limit") or 20), 50)
        params.append(limit)

        rows = db.execute(
            f"SELECT id, thread_id, direction, from_addr, to_addrs, subject, body, sent_at "
            f"FROM fake_email_message {where} ORDER BY sent_at DESC LIMIT ?",
            params,
        ).fetchall()

        if not rows:
            return ToolResult(text="No emails found matching your criteria.")

        lines = [f"Found {len(rows)} email(s):", ""]
        for row in rows:
            try:
                to_addrs = json.loads(row["to_addrs"]) or []
            except (json.JSONDecodeError, TypeError):
                to_addrs = []
            preview = row["body"][:100].replace("\n", " ")
            if len(row["body"]) > 100:
                preview += "..."
            icon = "📥" if row["direction"] == "inbound" else "📤"
            lines.append(f"{icon} {row['subject']}")
            lines.append(f"  From: {row['from_addr']}")
            lines.append(f"  To: {', '.join(to_addrs) or 'N/A'}")
            lines.append(f"  Date: {row['sent_at']}")
            lines.append(f"  Thread ID: {row['thread_id']}")
            lines.append(f"  Message ID: {row['id']}")
            lines.append(f"  Preview: {preview}")
            lines.append("")

        return ToolResult(text="\n".join(lines))


class GetInboxInfoTool(_ToolBase):
    name = "get_inbox_info"
    description = "Get information about your email inbox, including your email address."
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        return ToolResult(
            text=f"Your email address: {AGENT_EMAIL_ADDRESS}\nDisplay name: Eval Agent"
        )


# ---------------------------------------------------------------------------
# Adapter assembly
# ---------------------------------------------------------------------------


class FakeEmailAdapter:
    name = "FakeEmail"
    description = "In-process email adapter for evals (sqlite-backed)."
    TOOLS = [
        SendEmailTool,
        ListThreadsTool,
        GetThreadTool,
        ReplyToEmailTool,
        SearchEmailsTool,
        GetInboxInfoTool,
    ]

    @classmethod
    def make_tools(cls) -> list[Tool]:
        return [T() for T in cls.TOOLS]

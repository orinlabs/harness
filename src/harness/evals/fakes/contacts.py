"""In-process fake contacts adapter for evals.

Mirrors the production ``defaults.contacts`` tools. CRUD against the
``fake_contact`` table in the per-agent sqlite DB.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from harness.config import AdapterConfig
from harness.tools.base import ToolResult, ToolSchema

from .base import new_id, now_iso, require_db

if TYPE_CHECKING:
    from harness.context import RunContext


class _ToolBase:
    name: str = ""
    description: str = ""
    parameters: dict = {}

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(self.name, self.description, self.parameters)


class CreateContactTool(_ToolBase):
    name = "create_contact"
    description = (
        "Create a new contact in your contact book. You must provide a "
        "name, and optionally a phone number, email, and notes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The contact's full name."},
            "phone": {
                "type": "string",
                "description": "Phone number in E.164 format.",
            },
            "email": {"type": "string", "description": "Email address."},
            "notes": {"type": "string", "description": "Any notes about this contact."},
        },
        "required": ["name"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: "RunContext | None") -> ToolResult:
        name = args.get("name")
        if not name:
            return ToolResult(text="Error: name is required.")

        db = require_db()
        contact_id = new_id("sim_contact")
        ts = now_iso()
        db.execute(
            "INSERT INTO fake_contact (id, name, phone, email, notes, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                contact_id,
                name,
                args.get("phone") or "",
                args.get("email") or "",
                args.get("notes") or "",
                ts,
                ts,
            ),
        )

        lines = [f"Contact created: {name}", f"ID: {contact_id}"]
        if args.get("phone"):
            lines.append(f"Phone: {args['phone']}")
        if args.get("email"):
            lines.append(f"Email: {args['email']}")
        if args.get("notes"):
            lines.append(f"Notes: {args['notes']}")
        return ToolResult(text="\n".join(lines))


class GetContactTool(_ToolBase):
    name = "get_contact"
    description = "Get details of a specific contact by their ID."
    parameters = {
        "type": "object",
        "properties": {
            "contact_id": {"type": "string", "description": "The contact ID."},
        },
        "required": ["contact_id"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: "RunContext | None") -> ToolResult:
        cid = args.get("contact_id")
        if not cid:
            return ToolResult(text="Error: contact_id is required.")

        db = require_db()
        row = db.execute(
            "SELECT * FROM fake_contact WHERE id = ?",
            (cid,),
        ).fetchone()
        if not row:
            return ToolResult(text=f"Error: Contact with ID {cid} not found.")

        lines = [f"Contact: {row['name']}", f"ID: {row['id']}"]
        if row["phone"]:
            lines.append(f"Phone: {row['phone']}")
        if row["email"]:
            lines.append(f"Email: {row['email']}")
        if row["notes"]:
            lines.append(f"Notes: {row['notes']}")
        lines.append(f"Created: {row['created_at']}")
        lines.append(f"Last updated: {row['updated_at']}")
        return ToolResult(text="\n".join(lines))


class ListContactsTool(_ToolBase):
    name = "list_contacts"
    description = (
        "List contacts in your contact book. Optionally filter by a search "
        "query that matches name, phone, or email."
    )
    parameters = {
        "type": "object",
        "properties": {
            "search": {
                "type": "string",
                "description": "Optional search query.",
            },
            "limit": {
                "type": "integer",
                "description": "Max contacts to return (default 50, max 100).",
            },
        },
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: "RunContext | None") -> ToolResult:
        db = require_db()
        limit = min(int(args.get("limit") or 50), 100)
        search = args.get("search") or ""
        if search:
            like = f"%{search}%"
            rows = db.execute(
                "SELECT * FROM fake_contact "
                "WHERE name LIKE ? OR phone LIKE ? OR email LIKE ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (like, like, like, limit),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT * FROM fake_contact ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        if not rows:
            if search:
                return ToolResult(text=f'No contacts found matching "{search}".')
            return ToolResult(text="You have no contacts.")

        header = f"Found {len(rows)} contact(s):" + (
            f' (matching "{search}")' if search else ""
        )
        lines = [header, ""]
        for row in rows:
            lines.append(f"• {row['name']}")
            lines.append(f"  ID: {row['id']}")
            if row["phone"]:
                lines.append(f"  Phone: {row['phone']}")
            if row["email"]:
                lines.append(f"  Email: {row['email']}")
            if row["notes"]:
                lines.append(f"  Notes: {row['notes']}")
            lines.append("")

        return ToolResult(text="\n".join(lines))


class UpdateContactTool(_ToolBase):
    name = "update_contact"
    description = (
        "Update an existing contact's information. Only provided fields "
        "will be updated."
    )
    parameters = {
        "type": "object",
        "properties": {
            "contact_id": {"type": "string", "description": "Contact ID."},
            "name": {"type": "string"},
            "phone": {"type": "string"},
            "email": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": ["contact_id"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: "RunContext | None") -> ToolResult:
        cid = args.get("contact_id")
        if not cid:
            return ToolResult(text="Error: contact_id is required.")

        db = require_db()
        existing = db.execute(
            "SELECT * FROM fake_contact WHERE id = ?",
            (cid,),
        ).fetchone()
        if not existing:
            return ToolResult(text=f"Error: Contact with ID {cid} not found.")

        updates: list[str] = []
        params: list[object] = []
        for field in ("name", "phone", "email", "notes"):
            if field in args:
                updates.append(f"{field} = ?")
                params.append(args[field])

        if not updates:
            return ToolResult(text=f"No updates provided for contact {existing['name']}.")

        params.append(now_iso())
        params.append(cid)
        db.execute(
            f"UPDATE fake_contact SET {', '.join(updates)}, updated_at = ? WHERE id = ?",
            params,
        )
        changed = [u.split(" = ")[0] for u in updates]
        return ToolResult(
            text=f'Contact "{existing["name"]}" updated: {", ".join(changed)}.'
        )


class DeleteContactTool(_ToolBase):
    name = "delete_contact"
    description = "Delete a contact from your contact book."
    parameters = {
        "type": "object",
        "properties": {
            "contact_id": {"type": "string", "description": "Contact ID."},
        },
        "required": ["contact_id"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: "RunContext | None") -> ToolResult:
        cid = args.get("contact_id")
        if not cid:
            return ToolResult(text="Error: contact_id is required.")

        db = require_db()
        existing = db.execute(
            "SELECT name FROM fake_contact WHERE id = ?",
            (cid,),
        ).fetchone()
        if not existing:
            return ToolResult(text=f"Error: Contact with ID {cid} not found.")

        db.execute("DELETE FROM fake_contact WHERE id = ?", (cid,))
        return ToolResult(text=f'Contact "{existing["name"]}" deleted successfully.')


# ---------------------------------------------------------------------------
# Adapter assembly
# ---------------------------------------------------------------------------


class FakeContactsAdapter:
    name = "FakeContacts"
    description = "In-process contacts adapter for evals (sqlite-backed)."
    TOOLS = [
        CreateContactTool,
        GetContactTool,
        ListContactsTool,
        UpdateContactTool,
        DeleteContactTool,
    ]

    @classmethod
    def make_adapter_config(cls) -> AdapterConfig:
        return AdapterConfig(
            name=cls.name,
            description=cls.description,
            tools=[T() for T in cls.TOOLS],
        )

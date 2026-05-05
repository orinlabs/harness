"""In-process fake contacts adapter for evals.

Mirrors the production ``defaults.contacts`` tools after the
multi-address migration:

* A contact owns one or more phone numbers and email addresses.
* Each address tracks ``is_primary`` and ``status``.
* ``update_contact`` is strictly additive (``add_phones`` /
  ``add_emails`` / ``primary_phone_id`` / ``primary_email_id``);
  destructive ops live in ``remove_addresses_from_contact``.

The fake DB is per-eval and the schema is hard-cut over to the new
shape (see ``migrations/0002_contact_addresses.sql``).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from harness.tools.base import Tool, ToolResult, ToolSchema

from .base import new_id, now_iso, require_db

if TYPE_CHECKING:
    from harness.context import RunContext


# ---------------------------------------------------------------------------
# Normalization helpers — mirror defaults.contacts.models.normalize_*
# ---------------------------------------------------------------------------

_PHONE_STRIP_RE = re.compile(r"[\s\-().]")


def _normalize_phone(value: str) -> str:
    if not value:
        return ""
    return _PHONE_STRIP_RE.sub("", value).lstrip("+")


def _normalize_email(value: str) -> str:
    if not value:
        return ""
    return value.strip().lower()


# ---------------------------------------------------------------------------
# Address row helpers
# ---------------------------------------------------------------------------


def _list_phones(db, contact_id: str) -> list[dict]:
    rows = db.execute(
        "SELECT * FROM fake_contact_phone WHERE contact_id = ? "
        "ORDER BY is_primary DESC, created_at ASC",
        (contact_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _list_emails(db, contact_id: str) -> list[dict]:
    rows = db.execute(
        "SELECT * FROM fake_contact_email WHERE contact_id = ? "
        "ORDER BY is_primary DESC, created_at ASC",
        (contact_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _ensure_only_one_primary_phone(db, contact_id: str, primary_id: str) -> None:
    db.execute(
        "UPDATE fake_contact_phone SET is_primary = 0, updated_at = ? "
        "WHERE contact_id = ? AND id != ?",
        (now_iso(), contact_id, primary_id),
    )
    db.execute(
        "UPDATE fake_contact_phone SET is_primary = 1, updated_at = ? WHERE id = ?",
        (now_iso(), primary_id),
    )


def _ensure_only_one_primary_email(db, contact_id: str, primary_id: str) -> None:
    db.execute(
        "UPDATE fake_contact_email SET is_primary = 0, updated_at = ? "
        "WHERE contact_id = ? AND id != ?",
        (now_iso(), contact_id, primary_id),
    )
    db.execute(
        "UPDATE fake_contact_email SET is_primary = 1, updated_at = ? WHERE id = ?",
        (now_iso(), primary_id),
    )


def _auto_promote_primary_phone(db, contact_id: str) -> None:
    """If no phone is primary but phones exist, promote the longest-lived one."""
    has_primary = db.execute(
        "SELECT 1 FROM fake_contact_phone WHERE contact_id = ? AND is_primary = 1",
        (contact_id,),
    ).fetchone()
    if has_primary:
        return
    row = db.execute(
        "SELECT id FROM fake_contact_phone WHERE contact_id = ? "
        "ORDER BY created_at ASC LIMIT 1",
        (contact_id,),
    ).fetchone()
    if row:
        db.execute(
            "UPDATE fake_contact_phone SET is_primary = 1, updated_at = ? WHERE id = ?",
            (now_iso(), row["id"]),
        )


def _auto_promote_primary_email(db, contact_id: str) -> None:
    has_primary = db.execute(
        "SELECT 1 FROM fake_contact_email WHERE contact_id = ? AND is_primary = 1",
        (contact_id,),
    ).fetchone()
    if has_primary:
        return
    row = db.execute(
        "SELECT id FROM fake_contact_email WHERE contact_id = ? "
        "ORDER BY created_at ASC LIMIT 1",
        (contact_id,),
    ).fetchone()
    if row:
        db.execute(
            "UPDATE fake_contact_email SET is_primary = 1, updated_at = ? WHERE id = ?",
            (now_iso(), row["id"]),
        )


def _add_phone_for_contact(
    db,
    contact_id: str,
    phone: str,
    *,
    label: str = "",
    is_primary: bool | None = None,
) -> str:
    if not phone:
        return ""
    normalized = _normalize_phone(phone)
    # Reuse a row already on this contact if normalized form matches.
    existing = db.execute(
        "SELECT id FROM fake_contact_phone WHERE contact_id = ? AND phone = ?",
        (contact_id, phone),
    ).fetchone() or db.execute(
        "SELECT id, phone FROM fake_contact_phone WHERE contact_id = ?",
        (contact_id,),
    ).fetchall()

    # The first ``execute`` returns either a single row, or a list when
    # we fell through to the second query (.fetchall returns a list).
    if isinstance(existing, list):
        match_id = next(
            (r["id"] for r in existing if _normalize_phone(r["phone"]) == normalized),
            None,
        )
    else:
        match_id = existing["id"] if existing else None

    if match_id:
        if is_primary:
            _ensure_only_one_primary_phone(db, contact_id, match_id)
        return match_id

    pid = new_id("sim_phone")
    ts = now_iso()
    db.execute(
        "INSERT INTO fake_contact_phone "
        "(id, contact_id, phone, label, is_primary, status, verification_source, "
        " verified_at, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (pid, contact_id, phone, label, 0, "verified", "portal_create", ts, ts, ts),
    )
    if is_primary:
        _ensure_only_one_primary_phone(db, contact_id, pid)
    else:
        _auto_promote_primary_phone(db, contact_id)
    return pid


def _add_email_for_contact(
    db,
    contact_id: str,
    email: str,
    *,
    label: str = "",
    is_primary: bool | None = None,
) -> str:
    if not email:
        return ""
    normalized = _normalize_email(email)
    rows = db.execute(
        "SELECT id, email FROM fake_contact_email WHERE contact_id = ?",
        (contact_id,),
    ).fetchall()
    match_id = next(
        (r["id"] for r in rows if _normalize_email(r["email"]) == normalized),
        None,
    )
    if match_id:
        if is_primary:
            _ensure_only_one_primary_email(db, contact_id, match_id)
        return match_id

    eid = new_id("sim_email")
    ts = now_iso()
    db.execute(
        "INSERT INTO fake_contact_email "
        "(id, contact_id, email, label, is_primary, status, verification_source, "
        " verified_at, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (eid, contact_id, email, label, 0, "verified", "portal_create", ts, ts, ts),
    )
    if is_primary:
        _ensure_only_one_primary_email(db, contact_id, eid)
    else:
        _auto_promote_primary_email(db, contact_id)
    return eid


def _format_contact_lines(db, row: dict, indent: str = "") -> list[str]:
    """Render a contact's addresses for tool output."""
    lines: list[str] = []
    phones = _list_phones(db, row["id"])
    for ph in phones:
        tag = " (primary)" if ph["is_primary"] else ""
        status = "" if ph["status"] == "verified" else f" [{ph['status']}]"
        label = f" — {ph['label']}" if ph["label"] else ""
        lines.append(f"{indent}Phone: {ph['phone']}{tag}{label}{status} (id={ph['id']})")
    emails = _list_emails(db, row["id"])
    for em in emails:
        tag = " (primary)" if em["is_primary"] else ""
        status = "" if em["status"] == "verified" else f" [{em['status']}]"
        label = f" — {em['label']}" if em["label"] else ""
        lines.append(f"{indent}Email: {em['email']}{tag}{label}{status} (id={em['id']})")
    if row.get("notes"):
        lines.append(f"{indent}Notes: {row['notes']}")
    return lines


# ---------------------------------------------------------------------------
# Tool descriptors
# ---------------------------------------------------------------------------


class _ToolBase:
    name: str = ""
    description: str = ""
    parameters: dict = {}

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(self.name, self.description, self.parameters)


_PHONE_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "phone": {"type": "string", "description": "Phone in E.164 format."},
        "label": {"type": "string"},
        "is_primary": {"type": "boolean"},
    },
    "required": ["phone"],
    "additionalProperties": False,
}

_EMAIL_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "email": {"type": "string"},
        "label": {"type": "string"},
        "is_primary": {"type": "boolean"},
    },
    "required": ["email"],
    "additionalProperties": False,
}


class CreateContactTool(_ToolBase):
    name = "create_contact"
    description = (
        "Create a new contact. Provide a name and optional notes plus one or more "
        "phone numbers / email addresses. Mark one phone (and one email) as "
        "is_primary; if you don't, the first item is treated as primary."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The contact's full name."},
            "phones": {"type": "array", "items": _PHONE_ITEM_SCHEMA},
            "emails": {"type": "array", "items": _EMAIL_ITEM_SCHEMA},
            "notes": {"type": "string", "description": "Any notes about this contact."},
        },
        "required": ["name"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        name = args.get("name")
        if not name:
            return ToolResult(text="Error: name is required.")

        db = require_db()
        contact_id = new_id("sim_contact")
        ts = now_iso()
        db.execute(
            "INSERT INTO fake_contact (id, name, notes, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (contact_id, name, args.get("notes") or "", ts, ts),
        )

        for ph in args.get("phones") or []:
            _add_phone_for_contact(
                db,
                contact_id,
                ph.get("phone") or "",
                label=ph.get("label") or "",
                is_primary=ph.get("is_primary"),
            )
        for em in args.get("emails") or []:
            _add_email_for_contact(
                db,
                contact_id,
                em.get("email") or "",
                label=em.get("label") or "",
                is_primary=em.get("is_primary"),
            )
        _auto_promote_primary_phone(db, contact_id)
        _auto_promote_primary_email(db, contact_id)

        row = dict(
            db.execute(
                "SELECT * FROM fake_contact WHERE id = ?", (contact_id,)
            ).fetchone()
        )
        lines = [f"Contact created: {name}", f"ID: {contact_id}"]
        lines.extend(_format_contact_lines(db, row, indent="  "))
        return ToolResult(text="\n".join(lines))


class GetContactTool(_ToolBase):
    name = "get_contact"
    description = (
        "Get details of a contact. Output includes every phone and email with "
        "their per-address ID; use those IDs with send tools (phone_id / "
        "email_id) when you need to target a non-primary address."
    )
    parameters = {
        "type": "object",
        "properties": {
            "contact_id": {"type": "string", "description": "The contact ID."},
        },
        "required": ["contact_id"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        cid = args.get("contact_id")
        if not cid:
            return ToolResult(text="Error: contact_id is required.")

        db = require_db()
        row = db.execute("SELECT * FROM fake_contact WHERE id = ?", (cid,)).fetchone()
        if not row:
            return ToolResult(text=f"Error: Contact with ID {cid} not found.")
        row = dict(row)
        lines = [f"Contact: {row['name']}", f"ID: {row['id']}"]
        lines.extend(_format_contact_lines(db, row, indent="  "))
        lines.append(f"Created: {row['created_at']}")
        lines.append(f"Last updated: {row['updated_at']}")
        return ToolResult(text="\n".join(lines))


class ListContactsTool(_ToolBase):
    name = "list_contacts"
    description = (
        "List contacts in your contact book. Optionally filter by a search "
        "query that matches name, any phone, or any email."
    )
    parameters = {
        "type": "object",
        "properties": {
            "search": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        db = require_db()
        limit = min(int(args.get("limit") or 50), 100)
        search = args.get("search") or ""
        if search:
            like = f"%{search}%"
            rows = db.execute(
                "SELECT DISTINCT c.* FROM fake_contact c "
                "LEFT JOIN fake_contact_phone p ON p.contact_id = c.id "
                "LEFT JOIN fake_contact_email e ON e.contact_id = c.id "
                "WHERE c.name LIKE ? OR p.phone LIKE ? OR e.email LIKE ? "
                "ORDER BY c.updated_at DESC LIMIT ?",
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

        header = f"Found {len(rows)} contact(s):" + (f' (matching "{search}")' if search else "")
        lines = [header, ""]
        for row in rows:
            row = dict(row)
            lines.append(f"• {row['name']}")
            lines.append(f"  ID: {row['id']}")
            lines.extend(_format_contact_lines(db, row, indent="  "))
            lines.append("")

        return ToolResult(text="\n".join(lines))


class UpdateContactTool(_ToolBase):
    name = "update_contact"
    description = (
        "Update a contact. Strictly additive for addresses: you can add new "
        "phones / emails and switch which existing one is primary, but you "
        "cannot remove an address with this tool — use "
        "remove_addresses_from_contact for that."
    )
    parameters = {
        "type": "object",
        "properties": {
            "contact_id": {"type": "string"},
            "name": {"type": "string"},
            "notes": {"type": "string"},
            "add_phones": {"type": "array", "items": _PHONE_ITEM_SCHEMA},
            "add_emails": {"type": "array", "items": _EMAIL_ITEM_SCHEMA},
            "primary_phone_id": {"type": "string"},
            "primary_email_id": {"type": "string"},
        },
        "required": ["contact_id"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        cid = args.get("contact_id")
        if not cid:
            return ToolResult(text="Error: contact_id is required.")

        db = require_db()
        existing = db.execute(
            "SELECT * FROM fake_contact WHERE id = ?", (cid,)
        ).fetchone()
        if not existing:
            return ToolResult(text=f"Error: Contact with ID {cid} not found.")
        existing = dict(existing)

        scalar_changes: list[str] = []
        for field in ("name", "notes"):
            if field in args:
                db.execute(
                    f"UPDATE fake_contact SET {field} = ?, updated_at = ? WHERE id = ?",
                    (args[field], now_iso(), cid),
                )
                scalar_changes.append(field)

        for ph in args.get("add_phones") or []:
            _add_phone_for_contact(
                db,
                cid,
                ph.get("phone") or "",
                label=ph.get("label") or "",
                is_primary=ph.get("is_primary"),
            )
        for em in args.get("add_emails") or []:
            _add_email_for_contact(
                db,
                cid,
                em.get("email") or "",
                label=em.get("label") or "",
                is_primary=em.get("is_primary"),
            )

        ppid = args.get("primary_phone_id")
        if ppid:
            row = db.execute(
                "SELECT id FROM fake_contact_phone WHERE id = ? AND contact_id = ?",
                (ppid, cid),
            ).fetchone()
            if not row:
                return ToolResult(
                    text=f"Error: phone_id {ppid} does not belong to contact {cid}."
                )
            _ensure_only_one_primary_phone(db, cid, ppid)
        peid = args.get("primary_email_id")
        if peid:
            row = db.execute(
                "SELECT id FROM fake_contact_email WHERE id = ? AND contact_id = ?",
                (peid, cid),
            ).fetchone()
            if not row:
                return ToolResult(
                    text=f"Error: email_id {peid} does not belong to contact {cid}."
                )
            _ensure_only_one_primary_email(db, cid, peid)

        _auto_promote_primary_phone(db, cid)
        _auto_promote_primary_email(db, cid)

        any_change = (
            scalar_changes
            or args.get("add_phones")
            or args.get("add_emails")
            or ppid
            or peid
        )
        if not any_change:
            return ToolResult(text=f"No updates provided for contact {existing['name']}.")
        return ToolResult(text=f'Contact "{existing["name"]}" updated.')


class RemoveAddressesFromContactTool(_ToolBase):
    name = "remove_addresses_from_contact"
    description = (
        "Use sparingly. Only remove an address when the contact has explicitly "
        "told you it changed or is incorrect, or when an owner/operator told "
        "you to remove it. The backend auto-promotes the next remaining "
        "address to primary if you remove the current primary."
    )
    parameters = {
        "type": "object",
        "properties": {
            "contact_id": {"type": "string"},
            "phones": {"type": "array", "items": {"type": "string"}},
            "emails": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["contact_id"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        cid = args.get("contact_id")
        if not cid:
            return ToolResult(text="Error: contact_id is required.")

        db = require_db()
        existing = db.execute(
            "SELECT name FROM fake_contact WHERE id = ?", (cid,)
        ).fetchone()
        if not existing:
            return ToolResult(text=f"Error: Contact with ID {cid} not found.")

        phones = [p for p in (args.get("phones") or []) if isinstance(p, str) and p.strip()]
        emails = [e for e in (args.get("emails") or []) if isinstance(e, str) and e.strip()]
        if not phones and not emails:
            return ToolResult(text="Error: provide at least one phone or email to remove.")

        removed: list[str] = []
        unknown: list[str] = []

        for raw in phones:
            normalized = _normalize_phone(raw)
            rows = db.execute(
                "SELECT id, phone FROM fake_contact_phone WHERE contact_id = ?",
                (cid,),
            ).fetchall()
            match = next(
                (r for r in rows if _normalize_phone(r["phone"]) == normalized), None
            )
            if not match:
                unknown.append(raw)
                continue
            db.execute(
                "DELETE FROM fake_contact_phone WHERE id = ?", (match["id"],)
            )
            removed.append(f"phone={match['phone']}")
        for raw in emails:
            normalized = _normalize_email(raw)
            rows = db.execute(
                "SELECT id, email FROM fake_contact_email WHERE contact_id = ?",
                (cid,),
            ).fetchall()
            match = next(
                (r for r in rows if _normalize_email(r["email"]) == normalized), None
            )
            if not match:
                unknown.append(raw)
                continue
            db.execute(
                "DELETE FROM fake_contact_email WHERE id = ?", (match["id"],)
            )
            removed.append(f"email={match['email']}")

        _auto_promote_primary_phone(db, cid)
        _auto_promote_primary_email(db, cid)

        if unknown:
            row = dict(db.execute("SELECT * FROM fake_contact WHERE id = ?", (cid,)).fetchone())
            current_lines = _format_contact_lines(db, row)
            current = ", ".join(line.strip() for line in current_lines) or "(none)"
            return ToolResult(
                text=(
                    "Error: the following addresses are not on this contact: "
                    + ", ".join(unknown)
                    + f". Current addresses: {current}."
                )
            )

        return ToolResult(
            text=f'Contact "{existing["name"]}" addresses removed: {", ".join(removed)}.'
        )


class DeleteContactTool(_ToolBase):
    name = "delete_contact"
    description = "Delete a contact (and all their addresses) from your contact book."
    parameters = {
        "type": "object",
        "properties": {
            "contact_id": {"type": "string"},
        },
        "required": ["contact_id"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        cid = args.get("contact_id")
        if not cid:
            return ToolResult(text="Error: contact_id is required.")

        db = require_db()
        existing = db.execute(
            "SELECT name FROM fake_contact WHERE id = ?", (cid,)
        ).fetchone()
        if not existing:
            return ToolResult(text=f"Error: Contact with ID {cid} not found.")

        db.execute("DELETE FROM fake_contact_phone WHERE contact_id = ?", (cid,))
        db.execute("DELETE FROM fake_contact_email WHERE contact_id = ?", (cid,))
        db.execute("DELETE FROM fake_contact WHERE id = ?", (cid,))
        return ToolResult(text=f'Contact "{existing["name"]}" deleted successfully.')


# ---------------------------------------------------------------------------
# Adapter assembly
# ---------------------------------------------------------------------------


class TestContactsAdapter:
    name = "TestContacts"
    description = "In-process test contacts adapter for evals (sqlite-backed)."
    TOOLS = [
        CreateContactTool,
        GetContactTool,
        ListContactsTool,
        UpdateContactTool,
        RemoveAddressesFromContactTool,
        DeleteContactTool,
    ]

    @classmethod
    def make_tools(cls) -> list[Tool]:
        return [T() for T in cls.TOOLS]


class FakeContactsAdapter(TestContactsAdapter):
    """Backward-compatible fake naming for existing eval imports."""

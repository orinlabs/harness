"""In-process test Documents adapter for evals.

Mirrors the production ``defaults.documents`` tools from bedrock-api-adapters,
but stores per-agent documents in the local sqlite DB. The adapter identity is
``TestDocuments`` so local test configs do not get mistaken for Bedrock's
production ``Documents`` adapter.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from harness.tools.base import Tool, ToolResult, ToolSchema

from .base import new_id, now_iso, require_db

if TYPE_CHECKING:
    from harness.context import RunContext

_ALLOWED_KINDS = {"note", "skill"}


def _normalize_kind(raw: object, default: str = "note") -> str:
    if raw is None:
        return default
    value = str(raw).strip().lower()
    return value if value in _ALLOWED_KINDS else default


def seed_document(
    title: str,
    content: str,
    *,
    kind: str = "note",
    created_at: str | None = None,
) -> str:
    """Seed a document for a scenario. Returns the created document id."""
    db = require_db()
    doc_id = new_id("sim_doc")
    ts = created_at or now_iso()
    db.execute(
        "INSERT INTO fake_document (id, title, content, kind, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (doc_id, title, content, _normalize_kind(kind), ts, ts),
    )
    return doc_id


class _ToolBase:
    name: str = ""
    description: str = ""
    parameters: dict = {}

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(self.name, self.description, self.parameters)


class CreateDocumentTool(_ToolBase):
    name = "create_document"
    description = (
        "Create a new document. Use kind='note' (default) for ephemeral scratchpad "
        "notes. Use kind='skill' for curated knowledge you expect to reuse."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "The title of the document."},
            "content": {"type": "string", "description": "The content of the document."},
            "kind": {
                "type": "string",
                "enum": ["note", "skill"],
                "description": "Document kind. Defaults to 'note'.",
            },
        },
        "required": ["title", "content"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        title = args.get("title")
        content = args.get("content")
        if not title or content is None:
            return ToolResult(text="Error: title and content are required.")

        doc_id = seed_document(
            title=str(title),
            content=str(content),
            kind=_normalize_kind(args.get("kind")),
        )
        kind = _normalize_kind(args.get("kind"))
        return ToolResult(
            text=(
                f'Document created: "{title}"\n'
                f"Document ID: {doc_id}\n"
                f"Kind: {kind}\n"
                f"Content:\n{content}"
            )
        )


class ListDocumentsTool(_ToolBase):
    name = "list_documents"
    description = "List all documents for the agent."
    parameters = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max documents to return (default 50, max 100).",
            },
            "offset": {
                "type": "integer",
                "description": "Number of documents to skip (default 0).",
            },
        },
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        db = require_db()
        limit = min(int(args.get("limit") or 50), 100)
        offset = max(int(args.get("offset") or 0), 0)
        rows = db.execute(
            "SELECT id, title, content, kind FROM fake_document "
            "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

        if not rows:
            return ToolResult(text="You have no documents.")

        lines = [f"Documents ({len(rows)}):", ""]
        for row in rows:
            content = row["content"] or ""
            truncated = content[:50] + "..." if len(content) > 50 else content
            lines.append(f"• [{row['kind']}] {row['title']}")
            lines.append(f"  Document ID: {row['id']}")
            lines.append(f"  Content: {truncated}")
            lines.append("")

        return ToolResult(text="\n".join(lines))


class ListSkillsTool(_ToolBase):
    name = "list_skills"
    description = (
        "List the agent's curated skill documents. Supports an optional query "
        "matching title or content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Optional search query."},
            "limit": {
                "type": "integer",
                "description": "Max skills to return (default 100, max 100).",
            },
        },
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        db = require_db()
        query = (args.get("query") or "").strip()
        limit = min(int(args.get("limit") or 100), 100)
        if query:
            like = f"%{query}%"
            rows = db.execute(
                "SELECT id, title, content FROM fake_document "
                "WHERE kind = 'skill' AND (title LIKE ? OR content LIKE ?) "
                "ORDER BY updated_at DESC LIMIT ?",
                (like, like, limit),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT id, title, content FROM fake_document "
                "WHERE kind = 'skill' ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        if not rows:
            if query:
                return ToolResult(text=f'No skills matched "{query}".')
            return ToolResult(
                text=(
                    "You have no skill documents yet. Use create_document with kind='skill' "
                    "to capture stable facts or SOPs."
                )
            )

        lines = [f"Skills ({len(rows)}):", ""]
        for row in rows:
            content = row["content"] or ""
            preview = content[:80] + "..." if len(content) > 80 else content
            lines.append(f"• {row['title']}")
            lines.append(f"  Document ID: {row['id']}")
            lines.append(f"  Preview: {preview}")
            lines.append("")

        lines.append(
            "Use get_document to read a skill in full before acting on it. Use "
            "update_document to revise a skill after a mistake or when you learn "
            "something new."
        )
        return ToolResult(text="\n".join(lines))


class GetDocumentTool(_ToolBase):
    name = "get_document"
    description = "Get a specific document by ID."
    parameters = {
        "type": "object",
        "properties": {
            "document_id": {"type": "string", "description": "The document ID."},
        },
        "required": ["document_id"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        document_id = args.get("document_id")
        if not document_id:
            return ToolResult(text="Error: document_id is required.")

        db = require_db()
        row = db.execute(
            "SELECT * FROM fake_document WHERE id = ?",
            (document_id,),
        ).fetchone()
        if not row:
            return ToolResult(text=f"Error: Document with ID {document_id} not found.")

        return ToolResult(
            text=(
                f"Document: {row['title']}\n"
                f"Document ID: {row['id']}\n"
                f"Kind: {row['kind']}\n"
                f"Created: {row['created_at']}\n"
                f"Last updated: {row['updated_at']}\n\n"
                f"--- Content ---\n{row['content']}"
            )
        )


class UpdateDocumentTool(_ToolBase):
    name = "update_document"
    description = "Update an existing document. Can also change kind (note or skill)."
    parameters = {
        "type": "object",
        "properties": {
            "document_id": {"type": "string", "description": "The document ID."},
            "title": {"type": "string"},
            "content": {"type": "string"},
            "kind": {"type": "string", "enum": ["note", "skill"]},
        },
        "required": ["document_id"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        document_id = args.get("document_id")
        if not document_id:
            return ToolResult(text="Error: document_id is required.")

        db = require_db()
        existing = db.execute(
            "SELECT title, kind FROM fake_document WHERE id = ?",
            (document_id,),
        ).fetchone()
        if not existing:
            return ToolResult(text=f"Error: Document with ID {document_id} not found.")

        updates: list[str] = []
        params: list[object] = []
        for field in ("title", "content"):
            if field in args:
                updates.append(f"{field} = ?")
                params.append(args[field])
        if "kind" in args and args["kind"]:
            new_kind = _normalize_kind(args["kind"], default=existing["kind"])
            if new_kind != existing["kind"]:
                updates.append("kind = ?")
                params.append(new_kind)

        if not updates:
            return ToolResult(text=f'No updates provided for document "{existing["title"]}".')

        params.append(now_iso())
        params.append(document_id)
        db.execute(
            f"UPDATE fake_document SET {', '.join(updates)}, updated_at = ? WHERE id = ?",
            params,
        )
        changed = [u.split(" = ")[0] for u in updates]
        next_title = args.get("title") or existing["title"]
        return ToolResult(text=f'Document "{next_title}" updated: {", ".join(changed)}.')


class DeleteDocumentTool(_ToolBase):
    name = "delete_document"
    description = "Delete a document."
    parameters = {
        "type": "object",
        "properties": {
            "document_id": {"type": "string", "description": "The document ID."},
        },
        "required": ["document_id"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        document_id = args.get("document_id")
        if not document_id:
            return ToolResult(text="Error: document_id is required.")

        db = require_db()
        row = db.execute(
            "SELECT title FROM fake_document WHERE id = ?",
            (document_id,),
        ).fetchone()
        if not row:
            return ToolResult(text=f"Error: Document with ID {document_id} not found.")

        db.execute("DELETE FROM fake_document WHERE id = ?", (document_id,))
        return ToolResult(text=f'Document "{row["title"]}" deleted successfully.')


class TestDocumentsAdapter:
    name = "TestDocuments"
    description = "In-process test documents adapter for evals (sqlite-backed)."
    TOOLS = [
        CreateDocumentTool,
        ListDocumentsTool,
        ListSkillsTool,
        GetDocumentTool,
        UpdateDocumentTool,
        DeleteDocumentTool,
    ]

    @classmethod
    def make_tools(cls) -> list[Tool]:
        return [T() for T in cls.TOOLS]


class FakeDocumentsAdapter(TestDocumentsAdapter):
    """Backward-compatible fake naming for callers that prefer the old style."""


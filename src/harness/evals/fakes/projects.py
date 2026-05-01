"""In-process test Projects adapter for evals.

Mirrors the production ``defaults.calendars`` project tools from
bedrock-api-adapters, but stores project state in the local sqlite DB. The
adapter identity is ``TestProjects`` so it remains clearly separate from
Bedrock's production ``Projects`` adapter.
"""
from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from harness.tools.base import Tool, ToolResult, ToolSchema

from .base import new_id, now_iso, require_db

if TYPE_CHECKING:
    from harness.context import RunContext

_VISIBILITIES = {"internal", "external"}
_STATUSES = {"incomplete", "complete"}


def _format_date(value: str | None) -> str:
    if not value:
        return "No date set"
    try:
        return date.fromisoformat(value).strftime("%B %d, %Y")
    except ValueError:
        return value


def _normalize_choice(value: object, allowed: set[str], default: str) -> str:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    return normalized if normalized in allowed else default


def seed_project(
    title: str,
    objective: str,
    *,
    start_date: str,
    end_date: str,
    visibility: str = "internal",
    status: str = "incomplete",
    parent_project_id: str | None = None,
    created_at: str | None = None,
) -> str:
    """Seed a project for a scenario. Returns the created project id."""
    db = require_db()
    project_id = new_id("sim_project")
    ts = created_at or now_iso()
    db.execute(
        "INSERT INTO fake_project "
        "(id, title, objective, start_date, end_date, visibility, status, "
        "parent_project_id, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            project_id,
            title,
            objective,
            start_date,
            end_date,
            _normalize_choice(visibility, _VISIBILITIES, "internal"),
            _normalize_choice(status, _STATUSES, "incomplete"),
            parent_project_id,
            ts,
            ts,
        ),
    )
    return project_id


def _children(project_id: str):
    db = require_db()
    return db.execute(
        "SELECT * FROM fake_project WHERE parent_project_id = ? ORDER BY start_date",
        (project_id,),
    ).fetchall()


def _complete_project_tree(project_id: str, ts: str) -> int:
    db = require_db()
    count = 0
    for child in _children(project_id):
        count += _complete_project_tree(child["id"], ts)
    db.execute(
        "UPDATE fake_project SET status = 'complete', updated_at = ? WHERE id = ?",
        (ts, project_id),
    )
    return count + 1


def _delete_project_tree(project_id: str) -> int:
    db = require_db()
    count = 0
    for child in _children(project_id):
        count += _delete_project_tree(child["id"])
    db.execute("DELETE FROM fake_project WHERE id = ?", (project_id,))
    return count + 1


class _ToolBase:
    name: str = ""
    description: str = ""
    parameters: dict = {}

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(self.name, self.description, self.parameters)


class ListProjectsTool(_ToolBase):
    name = "list_projects"
    description = (
        "List top-level projects for this agent. Returns projects ordered by "
        "start date. Use get_project to inspect sub-projects."
    )
    parameters = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max projects to return (default 50, max 100).",
            },
            "status": {
                "type": "string",
                "enum": ["incomplete", "complete"],
                "description": "Filter by status. Omit to see all.",
            },
            "visibility": {
                "type": "string",
                "enum": ["internal", "external"],
                "description": "Filter by visibility. Omit to see all.",
            },
        },
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        db = require_db()
        limit = min(int(args.get("limit") or 50), 100)
        clauses = ["parent_project_id IS NULL"]
        params: list[object] = []

        status_filter = args.get("status")
        if status_filter:
            clauses.append("status = ?")
            params.append(status_filter)
        visibility_filter = args.get("visibility")
        if visibility_filter:
            clauses.append("visibility = ?")
            params.append(visibility_filter)

        params.append(limit)
        rows = db.execute(
            "SELECT * FROM fake_project "
            f"WHERE {' AND '.join(clauses)} ORDER BY start_date LIMIT ?",
            params,
        ).fetchall()

        if not rows:
            suffix = f" (filtered by status: {status_filter})" if status_filter else ""
            return ToolResult(text="You have no projects." + suffix)

        lines = [f"You have {len(rows)} top-level project(s):", ""]
        for row in rows:
            sub_count = db.execute(
                "SELECT COUNT(*) AS c FROM fake_project WHERE parent_project_id = ?",
                (row["id"],),
            ).fetchone()["c"]
            lines.append(f"• {row['title']} [{row['status']}] ({row['visibility']})")
            lines.append(f"  Project ID: {row['id']}")
            lines.append(f"  Objective: {row['objective']}")
            lines.append(
                f"  Period: {_format_date(row['start_date'])} to {_format_date(row['end_date'])}"
            )
            if sub_count > 0:
                lines.append(f"  Sub-projects: {sub_count} (use get_project to inspect)")
            lines.append("")

        return ToolResult(text="\n".join(lines))


class GetProjectTool(_ToolBase):
    name = "get_project"
    description = "Get details of a single project, including its sub-projects."
    parameters = {
        "type": "object",
        "properties": {
            "project_id": {"type": "string", "description": "The project ID."},
        },
        "required": ["project_id"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        project_id = args.get("project_id")
        if not project_id:
            return ToolResult(text="Error: project_id is required.")

        db = require_db()
        project = db.execute(
            "SELECT * FROM fake_project WHERE id = ?",
            (project_id,),
        ).fetchone()
        if not project:
            return ToolResult(text=f"Error: Project with ID {project_id} not found.")

        lines = [
            f"• {project['title']} [{project['status']}] ({project['visibility']})",
            f"  Project ID: {project['id']}",
            f"  Objective: {project['objective']}",
            (
                f"  Period: {_format_date(project['start_date'])} "
                f"to {_format_date(project['end_date'])}"
            ),
        ]
        if project["parent_project_id"]:
            parent = db.execute(
                "SELECT id, title FROM fake_project WHERE id = ?",
                (project["parent_project_id"],),
            ).fetchone()
            if parent:
                lines.append(
                    f"  Parent project: {parent['title']} (Project ID: {parent['id']})"
                )

        sub_projects = _children(project["id"])
        if sub_projects:
            lines.append(f"  Sub-projects ({len(sub_projects)}):")
            for sub in sub_projects:
                nested_count = db.execute(
                    "SELECT COUNT(*) AS c FROM fake_project WHERE parent_project_id = ?",
                    (sub["id"],),
                ).fetchone()["c"]
                lines.append(f"    • {sub['title']} [{sub['status']}] ({sub['visibility']})")
                lines.append(f"      Project ID: {sub['id']}")
                lines.append(f"      Objective: {sub['objective']}")
                lines.append(
                    f"      Period: {_format_date(sub['start_date'])} "
                    f"to {_format_date(sub['end_date'])}"
                )
                if nested_count > 0:
                    lines.append(f"      Sub-projects: {nested_count} (use get_project to inspect)")

        return ToolResult(text="\n".join(lines))


class CreateProjectTool(_ToolBase):
    name = "create_project"
    description = "Create a new project."
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Project title."},
            "objective": {"type": "string", "description": "Project objective."},
            "start_date": {"type": "string", "description": "Start date as YYYY-MM-DD."},
            "end_date": {"type": "string", "description": "End date as YYYY-MM-DD."},
            "parent_project_id": {
                "type": "string",
                "description": "Optional parent project ID for sub-projects.",
            },
            "visibility": {
                "type": "string",
                "enum": ["internal", "external"],
                "description": "Project visibility. Defaults to internal.",
            },
        },
        "required": ["title", "objective", "start_date", "end_date"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        required = ("title", "objective", "start_date", "end_date")
        if any(not args.get(field) for field in required):
            return ToolResult(
                text="Error: title, objective, start_date, and end_date are required."
            )

        db = require_db()
        parent_project_id = args.get("parent_project_id")
        parent_title = ""
        if parent_project_id:
            parent = db.execute(
                "SELECT title FROM fake_project WHERE id = ?",
                (parent_project_id,),
            ).fetchone()
            if not parent:
                return ToolResult(
                    text=f"Error: Parent project with ID {parent_project_id} not found."
                )
            parent_title = parent["title"]

        try:
            start = date.fromisoformat(str(args["start_date"]))
            end = date.fromisoformat(str(args["end_date"]))
        except ValueError as exc:
            return ToolResult(text=f"Error creating project: {exc}")
        if start >= end:
            return ToolResult(
                text="Error creating project: Project start date must be before end date"
            )

        visibility = _normalize_choice(args.get("visibility"), _VISIBILITIES, "internal")
        project_id = seed_project(
            title=str(args["title"]),
            objective=str(args["objective"]),
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            visibility=visibility,
            parent_project_id=parent_project_id,
        )

        result = (
            f'Project created: "{args["title"]}"\n'
            f"ID: {project_id}\n"
            f"Objective: {args['objective']}\n"
            f"Period: {_format_date(start.isoformat())} to {_format_date(end.isoformat())}"
        )
        if parent_title:
            result += f'\nParent project: "{parent_title}"'
        return ToolResult(text=result)


class UpdateProjectTool(_ToolBase):
    name = "update_project"
    description = "Update an existing project."
    parameters = {
        "type": "object",
        "properties": {
            "project_id": {"type": "string", "description": "The project ID."},
            "title": {"type": "string"},
            "objective": {"type": "string"},
            "start_date": {"type": "string", "description": "Start date as YYYY-MM-DD."},
            "end_date": {"type": "string", "description": "End date as YYYY-MM-DD."},
            "status": {"type": "string", "enum": ["incomplete", "complete"]},
            "visibility": {"type": "string", "enum": ["internal", "external"]},
        },
        "required": ["project_id"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        project_id = args.get("project_id")
        if not project_id:
            return ToolResult(text="Error: project_id is required.")

        db = require_db()
        project = db.execute(
            "SELECT * FROM fake_project WHERE id = ?",
            (project_id,),
        ).fetchone()
        if not project:
            return ToolResult(text=f"Error: Project with ID {project_id} not found.")

        updates: list[str] = []
        params: list[object] = []
        labels: list[str] = []
        for field in ("title", "objective"):
            if field in args:
                updates.append(f"{field} = ?")
                params.append(args[field])
                labels.append(f'title to "{args[field]}"' if field == "title" else "objective")

        for field in ("start_date", "end_date"):
            if field in args:
                try:
                    parsed = date.fromisoformat(str(args[field])).isoformat()
                except ValueError as exc:
                    return ToolResult(text=f"Error updating project: {exc}")
                updates.append(f"{field} = ?")
                params.append(parsed)
                labels.append(f"{field.replace('_', ' ')} to {_format_date(parsed)}")

        if "status" in args:
            status = _normalize_choice(args["status"], _STATUSES, project["status"])
            updates.append("status = ?")
            params.append(status)
            labels.append(f"status to {status}")
        if "visibility" in args:
            visibility = _normalize_choice(args["visibility"], _VISIBILITIES, project["visibility"])
            updates.append("visibility = ?")
            params.append(visibility)
            labels.append(f"visibility to {visibility}")

        if not updates:
            return ToolResult(text=f'No updates provided for project "{project["title"]}".')

        next_start = args.get("start_date", project["start_date"])
        next_end = args.get("end_date", project["end_date"])
        if date.fromisoformat(str(next_start)) >= date.fromisoformat(str(next_end)):
            return ToolResult(
                text="Error updating project: Project start date must be before end date"
            )

        params.append(now_iso())
        params.append(project_id)
        db.execute(
            f"UPDATE fake_project SET {', '.join(updates)}, updated_at = ? WHERE id = ?",
            params,
        )
        next_title = args.get("title") or project["title"]
        return ToolResult(text=f'Project "{next_title}" updated: {", ".join(labels)}.')


class CompleteProjectTool(_ToolBase):
    name = "complete_project"
    description = "Mark a project as complete, including its sub-projects."
    parameters = {
        "type": "object",
        "properties": {
            "project_id": {"type": "string", "description": "The project ID."},
        },
        "required": ["project_id"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        project_id = args.get("project_id")
        if not project_id:
            return ToolResult(text="Error: project_id is required.")

        db = require_db()
        project = db.execute(
            "SELECT title FROM fake_project WHERE id = ?",
            (project_id,),
        ).fetchone()
        if not project:
            return ToolResult(text=f"Error: Project with ID {project_id} not found.")

        affected = _complete_project_tree(project_id, now_iso())
        sub_projects = max(affected - 1, 0)
        result = f'Project "{project["title"]}" marked as complete!'
        if sub_projects > 0:
            result += f"\n{sub_projects} sub-project(s) also marked as complete."
        return ToolResult(text=result)


class DeleteProjectTool(_ToolBase):
    name = "delete_project"
    description = "Delete a project, including its sub-projects."
    parameters = {
        "type": "object",
        "properties": {
            "project_id": {"type": "string", "description": "The project ID."},
        },
        "required": ["project_id"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: RunContext | None) -> ToolResult:
        project_id = args.get("project_id")
        if not project_id:
            return ToolResult(text="Error: project_id is required.")

        db = require_db()
        project = db.execute(
            "SELECT title FROM fake_project WHERE id = ?",
            (project_id,),
        ).fetchone()
        if not project:
            return ToolResult(text=f"Error: Project with ID {project_id} not found.")

        affected = _delete_project_tree(project_id)
        sub_projects = max(affected - 1, 0)
        result = f'Project "{project["title"]}" deleted successfully.'
        if sub_projects > 0:
            result += f"\n{sub_projects} sub-project(s) also deleted."
        return ToolResult(text=result)


class TestProjectsAdapter:
    name = "TestProjects"
    description = "In-process test projects adapter for evals (sqlite-backed)."
    TOOLS = [
        ListProjectsTool,
        GetProjectTool,
        CreateProjectTool,
        UpdateProjectTool,
        CompleteProjectTool,
        DeleteProjectTool,
    ]

    @classmethod
    def make_tools(cls) -> list[Tool]:
        return [T() for T in cls.TOOLS]


class FakeProjectsAdapter(TestProjectsAdapter):
    """Backward-compatible fake naming for callers that prefer the old style."""


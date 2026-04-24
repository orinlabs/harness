"""In-process fake computer adapter for evals.

Mirrors the production ``defaults.computer`` tools (Blaxel sandbox) but
backs them with a per-agent temp directory plus :func:`subprocess.run`
for ``computer_exec``. This is deliberately narrower than the production
sandbox: no network, no long-running processes, no attachment download
helpers -- scenarios that exercise those should extend the fake or pin
to the production adapter over HTTP.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from harness.config import AdapterConfig
from harness.tools.base import ToolResult, ToolSchema

from .base import now_iso, require_db

if TYPE_CHECKING:
    from harness.context import RunContext


# ---------------------------------------------------------------------------
# tmpdir provisioning
# ---------------------------------------------------------------------------


def _get_tmpdir(agent_id: str) -> Path:
    """Return (creating if needed) the fake computer tmpdir for this agent."""
    db = require_db()
    row = db.execute(
        "SELECT tmpdir FROM fake_computer_state WHERE agent_id = ?",
        (agent_id,),
    ).fetchone()
    if row and row["tmpdir"]:
        path = Path(row["tmpdir"])
        path.mkdir(parents=True, exist_ok=True)
        return path

    path = Path(tempfile.mkdtemp(prefix=f"fake-computer-{agent_id}-"))
    db.execute(
        "INSERT OR REPLACE INTO fake_computer_state (agent_id, tmpdir, created_at) "
        "VALUES (?, ?, ?)",
        (agent_id, str(path), now_iso()),
    )
    return path


def _resolve(agent_id: str, path: str) -> Path:
    """Resolve a tool-supplied path inside the agent's fake workspace.

    Absolute paths are rebased onto the tmpdir so a model using
    ``/workspace/foo`` still stays sandboxed. We also block ``..``
    traversal that escapes the tmpdir.
    """
    tmpdir = _get_tmpdir(agent_id)
    p = (path or "").strip() or "/"
    # Strip leading slash so joinpath keeps us inside tmpdir.
    rel = p.lstrip("/")
    candidate = (tmpdir / rel).resolve()
    tmpdir_resolved = tmpdir.resolve()
    try:
        candidate.relative_to(tmpdir_resolved)
    except ValueError as e:
        raise ValueError(f"path escapes sandbox: {path!r}") from e
    return candidate


def teardown_tmpdir(agent_id: str) -> None:
    """Remove the tmpdir and its DB row. Test teardown helper."""
    db = require_db()
    row = db.execute(
        "SELECT tmpdir FROM fake_computer_state WHERE agent_id = ?",
        (agent_id,),
    ).fetchone()
    if row and row["tmpdir"]:
        shutil.rmtree(row["tmpdir"], ignore_errors=True)
    db.execute("DELETE FROM fake_computer_state WHERE agent_id = ?", (agent_id,))


# ---------------------------------------------------------------------------
# Inbound injection (no-op: computer has no "inbound" channel)
# ---------------------------------------------------------------------------


def inject_inbound(agent_id: str, path: str, content: str) -> str:
    """Drop a file into the agent's fake workspace as "environment state".

    Returns the absolute path inside the tmpdir. Scenarios use this to
    seed files the agent will read (e.g. a CSV the scenario expects the
    agent to process).
    """
    target = _resolve(agent_id, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return str(target)


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


def _agent_id(ctx: "RunContext | None") -> str:
    """Resolve the current agent id.

    ``ctx`` is optional (tests pass ``None``). Falls back to
    :func:`harness.context.get_agent_id` which raises if nothing is set.
    """
    if ctx is not None:
        return ctx.agent_id
    from harness.context import get_agent_id

    return get_agent_id()


class ComputerExecTool(_ToolBase):
    name = "computer_exec"
    description = (
        "Execute a shell command on your computer. Your workspace is a "
        "sandboxed directory; files are persisted between turns."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run."},
            "working_dir": {
                "type": "string",
                "description": "Working directory (default: workspace root).",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (1-60, default 60).",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: "RunContext | None") -> ToolResult:
        command = (args.get("command") or "").strip()
        if not command:
            return ToolResult(text="Error: No command provided.")

        agent_id = _agent_id(ctx)
        tmpdir = _get_tmpdir(agent_id)
        wd_arg = (args.get("working_dir") or "").strip()
        try:
            cwd = _resolve(agent_id, wd_arg) if wd_arg else tmpdir
        except ValueError as e:
            return ToolResult(text=f"Error: {e}")
        timeout = min(max(int(args.get("timeout") or 60), 1), 60)

        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "HOME": str(tmpdir)},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(text=f"Error: command timed out after {timeout}s")
        except Exception as e:
            return ToolResult(text=f"Error executing command: {e}")

        lines = [f"Exit code: {proc.returncode}"]
        if proc.stdout.strip():
            lines.append("--- stdout ---")
            lines.append(proc.stdout.rstrip())
        if proc.stderr.strip():
            lines.append("--- stderr ---")
            lines.append(proc.stderr.rstrip())
        return ToolResult(text="\n".join(lines))


class ComputerReadFileTool(_ToolBase):
    name = "computer_read_file"
    description = "Read the contents of a text file from your computer."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file."},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: "RunContext | None") -> ToolResult:
        path = (args.get("path") or "").strip()
        if not path:
            return ToolResult(text="Error: No path provided.")

        try:
            target = _resolve(_agent_id(ctx), path)
        except ValueError as e:
            return ToolResult(text=f"Error: {e}")

        if not target.exists():
            return ToolResult(text=f"Error reading file: file not found: {path}")
        if target.is_dir():
            return ToolResult(text=f"Error reading file: {path} is a directory")

        try:
            return ToolResult(text=target.read_text())
        except Exception as e:
            return ToolResult(text=f"Error reading file: {e}")


class ComputerWriteFileTool(_ToolBase):
    name = "computer_write_file"
    description = "Write content to a file on your computer."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file."},
            "content": {"type": "string", "description": "Content to write."},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: "RunContext | None") -> ToolResult:
        path = (args.get("path") or "").strip()
        if not path:
            return ToolResult(text="Error: No path provided.")
        content = args.get("content") or ""

        try:
            target = _resolve(_agent_id(ctx), path)
        except ValueError as e:
            return ToolResult(text=f"Error: {e}")

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        except Exception as e:
            return ToolResult(text=f"Error writing file: {e}")

        return ToolResult(text=f"Wrote {len(content)} bytes to {path}")


class ComputerListFilesTool(_ToolBase):
    name = "computer_list_files"
    description = "List files and directories at a path on your computer."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list (default: workspace root).",
            },
        },
        "additionalProperties": False,
    }

    def call(self, args: dict, ctx: "RunContext | None") -> ToolResult:
        path = args.get("path") or "/"
        try:
            target = _resolve(_agent_id(ctx), path)
        except ValueError as e:
            return ToolResult(text=f"Error: {e}")

        if not target.exists():
            return ToolResult(text=f"Error listing files: not found: {path}")
        if not target.is_dir():
            return ToolResult(text=f"Error listing files: not a directory: {path}")

        lines = [f"Directory: {path}", ""]
        # Sort subdirs first, then files -- matches the production shape.
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        for entry in entries:
            if entry.is_dir():
                lines.append(f"  {entry.name}/")
            else:
                lines.append(f"  {entry.name}")
        return ToolResult(text="\n".join(lines))


# ---------------------------------------------------------------------------
# Adapter assembly
# ---------------------------------------------------------------------------


class FakeComputerAdapter:
    name = "FakeComputer"
    description = "In-process computer adapter for evals (tmpdir-backed)."
    TOOLS = [
        ComputerExecTool,
        ComputerReadFileTool,
        ComputerWriteFileTool,
        ComputerListFilesTool,
    ]

    @classmethod
    def make_adapter_config(cls) -> AdapterConfig:
        return AdapterConfig(
            name=cls.name,
            description=cls.description,
            tools=[T() for T in cls.TOOLS],
        )

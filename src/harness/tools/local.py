"""Built-in local tools for workflow mode: bash / read_file / write_file.

These run **in-process, inside the sandbox** — no HTTP hop — and are only
registered for `harness workflow` agent steps (see
``harness.workflow.runner``). The regular `harness agent` path never sees
them: an agent-loop run gets its tools from the platform config, while a
workflow agent step is explicitly a "work on files in this run's working
directory" affair.

All three tools are path-restricted to the run working directory they were
constructed with: relative paths resolve against it, and anything that
escapes it (``../``, absolute paths outside, symlinks pointing out) is
refused with an error ToolResult rather than an exception — the model can
read the message and correct itself.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from harness.context import RunContext
from harness.tools.base import ToolResult, ToolSchema

logger = logging.getLogger(__name__)

BASH_TIMEOUT_SECONDS = 120.0
# Keep tool results bounded so one chatty command can't blow up the LLM
# context. Tail, not head: the end of the output is where errors live.
_MAX_OUTPUT_CHARS = 20_000


def _resolve_inside(root: Path, relative: str) -> Path:
    """Resolve ``relative`` against ``root`` and require the result to stay
    inside ``root``. Raises ``ValueError`` on escape attempts (``..``,
    absolute paths outside the root, symlinks pointing elsewhere)."""
    candidate = (root / relative).resolve()
    root = root.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path {relative!r} escapes the run working directory")
    return candidate


def _clip(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    clipped = len(text) - _MAX_OUTPUT_CHARS
    return f"[... {clipped} chars truncated ...]\n" + text[-_MAX_OUTPUT_CHARS:]


class BashTool:
    name = "bash"
    description = (
        "Run a shell command in the run working directory and return its "
        "stdout and stderr. Commands time out after 120 seconds."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute (bash -c).",
            },
        },
        "required": ["command"],
    }

    def __init__(self, working_dir: Path):
        self._working_dir = Path(working_dir)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(self.name, self.description, self.parameters)

    def call(self, args: dict, ctx: RunContext) -> ToolResult:
        command = str(args.get("command") or "").strip()
        if not command:
            return ToolResult(text="Error: 'command' is required.")
        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                cwd=str(self._working_dir),
                capture_output=True,
                text=True,
                timeout=BASH_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                text=f"Error: command timed out after {BASH_TIMEOUT_SECONDS:.0f}s: {command}"
            )
        parts = [f"exit code: {proc.returncode}"]
        if proc.stdout:
            parts.append(f"stdout:\n{proc.stdout}")
        if proc.stderr:
            parts.append(f"stderr:\n{proc.stderr}")
        return ToolResult(text=_clip("\n".join(parts)))


class ReadFileTool:
    name = "read_file"
    description = "Read a text file. The path is relative to the run working directory."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the run working directory.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, working_dir: Path):
        self._working_dir = Path(working_dir)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(self.name, self.description, self.parameters)

    def call(self, args: dict, ctx: RunContext) -> ToolResult:
        raw = str(args.get("path") or "")
        if not raw:
            return ToolResult(text="Error: 'path' is required.")
        try:
            path = _resolve_inside(self._working_dir, raw)
        except ValueError as e:
            return ToolResult(text=f"Error: {e}")
        if not path.is_file():
            return ToolResult(text=f"Error: no such file: {raw}")
        try:
            return ToolResult(text=_clip(path.read_text()))
        except UnicodeDecodeError:
            return ToolResult(text=f"Error: {raw} is not a UTF-8 text file.")


class WriteFileTool:
    name = "write_file"
    description = (
        "Write a text file (parent directories are created). The path is "
        "relative to the run working directory."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the run working directory.",
            },
            "content": {
                "type": "string",
                "description": "Full file contents to write.",
            },
        },
        "required": ["path", "content"],
    }

    def __init__(self, working_dir: Path):
        self._working_dir = Path(working_dir)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(self.name, self.description, self.parameters)

    def call(self, args: dict, ctx: RunContext) -> ToolResult:
        raw = str(args.get("path") or "")
        if not raw:
            return ToolResult(text="Error: 'path' is required.")
        content = args.get("content")
        if not isinstance(content, str):
            return ToolResult(text="Error: 'content' must be a string.")
        try:
            path = _resolve_inside(self._working_dir, raw)
        except ValueError as e:
            return ToolResult(text=f"Error: {e}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return ToolResult(text=f"Wrote {len(content)} chars to {raw}.")


def workflow_local_tools(working_dir: Path) -> list:
    """The built-in tool set for workflow-mode agent steps."""
    return [BashTool(working_dir), ReadFileTool(working_dir), WriteFileTool(working_dir)]

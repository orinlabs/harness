"""Workflow-mode local built-ins (bash / read_file / write_file).

The critical property is path restriction: all three tools are constructed
with the run working directory and must refuse anything that escapes it —
with an error ToolResult the model can read, never an exception.
"""

from __future__ import annotations

import pytest

from harness.context import RunContext
from harness.tools.local import BashTool, ReadFileTool, WriteFileTool, workflow_local_tools


@pytest.fixture
def ctx():
    return RunContext(agent_id="wf-agent", run_id="wf-run")


@pytest.fixture
def workdir(tmp_path):
    wd = tmp_path / "wf"
    wd.mkdir()
    return wd


def test_write_then_read_roundtrip(workdir, ctx):
    write = WriteFileTool(workdir)
    read = ReadFileTool(workdir)

    result = write.call({"path": "out/nested/report.txt", "content": "hello"}, ctx)
    assert "Wrote" in result.text  # parents created on demand
    assert (workdir / "out" / "nested" / "report.txt").read_text() == "hello"

    assert read.call({"path": "out/nested/report.txt"}, ctx).text == "hello"


def test_file_tools_refuse_paths_that_escape_the_working_dir(workdir, tmp_path, ctx):
    secret = tmp_path / "secret.txt"
    secret.write_text("do not read")

    # Relative traversal out of the working dir.
    res = WriteFileTool(workdir).call({"path": "../evil.txt", "content": "x"}, ctx)
    assert "escapes" in res.text
    assert not (tmp_path / "evil.txt").exists()

    # Absolute path outside the working dir.
    res = ReadFileTool(workdir).call({"path": str(secret)}, ctx)
    assert "escapes" in res.text

    res = ReadFileTool(workdir).call({"path": "../secret.txt"}, ctx)
    assert "escapes" in res.text


def test_symlink_escape_is_refused(workdir, tmp_path, ctx):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (workdir / "link").symlink_to(outside)

    res = ReadFileTool(workdir).call({"path": "link"}, ctx)
    assert "escapes" in res.text


def test_bash_runs_with_cwd_pinned_to_the_working_dir(workdir, ctx):
    result = BashTool(workdir).call({"command": "pwd && echo marker-ok"}, ctx)
    assert str(workdir.resolve()) in result.text
    assert "marker-ok" in result.text
    assert "exit code: 0" in result.text


def test_bash_reports_exit_code_and_stderr(workdir, ctx):
    result = BashTool(workdir).call({"command": "echo oops >&2; exit 7"}, ctx)
    assert "exit code: 7" in result.text
    assert "oops" in result.text


def test_workflow_local_tools_exports_exactly_the_three(workdir):
    names = [t.name for t in workflow_local_tools(workdir)]
    assert names == ["bash", "read_file", "write_file"]

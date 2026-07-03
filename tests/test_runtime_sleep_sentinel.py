"""LocalAgentRuntime.sleep must emit a machine-readable sentinel line.

Production use: a supervisor (e.g. a simulated environment) spawns the
harness as a subprocess, holds its stdout pipe, and line-matches for
``@@sleep {...}`` to learn that the agent went to sleep and until when --
so it can advance its world clock to the wake time before re-launching.

These tests exercise the same path production takes: the built-in
``SleepTool`` calling ``ctx.runtime.sleep(...)`` on a ``LocalAgentRuntime``,
captured at the OS file-descriptor level (``capfd``) -- the same stream a
parent process's pipe would see -- and parsed with the kind of regex the
supervisor uses.
"""

from __future__ import annotations

import json
import re

from harness.context import RunContext
from harness.core.runtime import SLEEP_SENTINEL_PREFIX, LocalAgentRuntime
from harness.tools.sleep import SleepTool

# The parser a supervisor would use: anchored prefix, JSON payload after.
_SENTINEL_RE = re.compile(r"^@@sleep (\{.*\})$", re.MULTILINE)


def _parse_sentinels(stream_text: str) -> list[dict]:
    return [json.loads(m.group(1)) for m in _SENTINEL_RE.finditer(stream_text)]


def test_sleep_tool_emits_parseable_sentinel_on_stdout(capfd):
    ctx = RunContext(agent_id="agent-1", run_id="run-1", runtime=LocalAgentRuntime())
    ctx.tool_map = {}

    result = SleepTool().call(
        {"until": "2026-06-30T09:00:00Z", "reason": "batch done"}, ctx
    )

    out, err = capfd.readouterr()
    sentinels = _parse_sentinels(out)
    assert sentinels == [
        {
            "agent_id": "agent-1",
            "until": "2026-06-30T09:00:00Z",
            "reason": "batch done",
        }
    ]
    # Sentinel goes to stdout only; it must not be duplicated on stderr
    # where log output lives.
    assert SLEEP_SENTINEL_PREFIX not in err
    # The sleep itself still proceeds as before.
    assert ctx.sleep_requested is True
    assert "2026-06-30T09:00:00Z" in result.text


def test_indefinite_sleep_and_missing_reason(capfd):
    ctx = RunContext(agent_id="agent-2", run_id="run-2", runtime=LocalAgentRuntime())
    ctx.tool_map = {}

    SleepTool().call({"until": "indefinite"}, ctx)

    out, _ = capfd.readouterr()
    [payload] = _parse_sentinels(out)
    assert payload == {"agent_id": "agent-2", "until": "indefinite", "reason": ""}


def test_sentinel_line_is_unambiguous_amid_other_stdout(capfd):
    """A supervisor parses one combined stream; interleaved prints must not
    confuse the line parser, and the sentinel must be a single line."""
    print("unrelated harness output")
    LocalAgentRuntime().sleep("agent-3", until="2099-01-01T00:00:00Z", reason='say "hi"\nthen stop')
    print("more output after")

    out, _ = capfd.readouterr()
    [payload] = _parse_sentinels(out)
    # json.dumps escapes the newline, keeping the sentinel on one line.
    assert payload["reason"] == 'say "hi"\nthen stop'
    sentinel_lines = [ln for ln in out.splitlines() if ln.startswith(SLEEP_SENTINEL_PREFIX)]
    assert len(sentinel_lines) == 1

"""Simulated clock: `--sim-start-time` must offset every harness time read.

Production use: a supervisor launches `harness agent --sim-start-time
<epoch>` (or sets $HARNESS_SIM_START_TIME on `harness boot`) so the agent
believes it is running at an arbitrary moment. Everything downstream that
stamps or windows by "now" — memory message timestamps, the summarizer's
bucket selection, tier windows in export — must see simulated time =
start + real elapsed run time, or memory buckets land in the wrong day
and summaries silently never fire.

The tests below exercise the exact production seams:

  * `MemoryService.log_messages` with no explicit ts (how `Harness.run`
    logs every message),
  * `SummaryUpdater.update_all()` with no `current_time` (how
    `MemoryService.update_summaries` invokes it each turn),
  * the `export-memory` CLI end-to-end via `main([...])`,
  * `boot`'s exec argv forwarding, and
  * the eval runner's `SimulatedClock` driving the same core offset.
"""

from __future__ import annotations

import argparse
import importlib
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from harness.core import clock

NS = 1_000_000_000

# A moment far from any plausible wall clock, so a test that accidentally
# reads wall time can never pass by coincidence.
SIM_NOW = datetime(2030, 1, 1, 12, 33, 0, tzinfo=UTC)
SIM_EPOCH = SIM_NOW.timestamp()


@pytest.fixture
def storage_env(tmp_path, monkeypatch):
    """Fresh sqlite + applied migrations for a scratch agent."""
    mig_dir = Path(__file__).parent.parent / "src/harness/memory/migrations"
    monkeypatch.setenv("HARNESS_MIGRATIONS_DIR", str(mig_dir))

    from harness.core import storage as storage_module

    importlib.reload(storage_module)
    monkeypatch.setattr(storage_module, "_STORAGE_ROOT", tmp_path / "storage")
    storage_module.load("agent-simclock")
    try:
        yield storage_module
    finally:
        storage_module.close()


# ---------------------------------------------------------------------------
# Core offset math
# ---------------------------------------------------------------------------


def test_now_is_start_plus_elapsed_run_time():
    clock.set_start_epoch(SIM_EPOCH)
    first = clock.now()
    time.sleep(0.05)
    second = clock.now()

    # Anchored at the epoch, not wall-clock...
    assert abs((first - SIM_NOW).total_seconds()) < 5.0
    # ...and still ticking with real elapsed time underneath.
    assert timedelta(seconds=0.05) <= second - first < timedelta(seconds=5)


def test_time_ns_and_now_agree():
    clock.set_start_epoch(SIM_EPOCH)
    delta_s = clock.time_ns() / NS - clock.now().timestamp()
    assert abs(delta_s) < 1.0


def test_without_offset_clock_is_wall_clock():
    delta = clock.now() - datetime.now(tz=UTC)
    assert abs(delta.total_seconds()) < 5.0


# ---------------------------------------------------------------------------
# Memory writes + summarization (the seams Harness.run actually calls)
# ---------------------------------------------------------------------------


def test_log_messages_stamps_simulated_time(storage_env):
    from harness.memory import MemoryService

    clock.set_start_epoch(SIM_EPOCH)
    m = MemoryService(agent_id="agent-simclock")
    m.log_messages([{"role": "user", "content": "hello from the future"}])

    row = storage_env.db.execute("SELECT ts_ns FROM messages").fetchone()
    assert abs(row["ts_ns"] / NS - SIM_EPOCH) < 5.0


def test_summarizer_default_now_buckets_by_simulated_time(storage_env, monkeypatch):
    """`update_all()` with no current_time must bucket against the simulated
    clock: messages logged 6 simulated minutes ago land in a *completed*
    5-minute bucket dated 2030, which can only happen if the summarizer's
    default "now" reads the offset clock (wall clock is 202x)."""
    from harness.core import llm
    from harness.memory import MemoryService

    def fake_complete(**kwargs):
        return llm.LLMResponse(
            text="summary of future things",
            tool_calls=[],
            finish_reason="stop",
            usage=llm.Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2, total_cost=0.0),
        )

    monkeypatch.setattr(llm, "complete", fake_complete)

    clock.set_start_epoch(SIM_EPOCH)
    m = MemoryService(agent_id="agent-simclock")
    m.log_messages(
        [{"role": "user", "content": "six sim-minutes ago"}],
        ts_ns=clock.time_ns() - 6 * 60 * NS,
    )

    m.update_summaries()  # production call: no explicit current_time

    # The summarizer buckets by naive local-time fields (see
    # `memory.marks.force_timezone`); mirror that conversion so the
    # expectation holds in any machine timezone.
    msg_local = datetime.fromtimestamp(SIM_EPOCH - 6 * 60)
    rows = storage_env.db.execute("SELECT date, hour, minute FROM five_minute_summaries").fetchall()
    assert len(rows) == 1
    assert rows[0]["date"] == msg_local.date().isoformat()
    assert rows[0]["hour"] == msg_local.hour
    assert rows[0]["minute"] == msg_local.minute - (msg_local.minute % 5)


# ---------------------------------------------------------------------------
# CLI end-to-end: flag and env var reach the export window
# ---------------------------------------------------------------------------


def _seed_five_minute_row(storage_module, agent_id: str) -> None:
    conn = storage_module.load(agent_id)
    conn.execute(
        "INSERT INTO five_minute_summaries "
        "(id, date, hour, minute, summary, message_count, created_at_ns) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), "2030-01-01", 12, 10, "future 5m row", 3, time.time_ns()),
    )
    storage_module.flush()
    storage_module.close()


def _export_payload(capsys) -> str:
    from harness.memory.export import EXPORT_BEGIN_MARKER, EXPORT_END_MARKER

    out = capsys.readouterr().out
    return out.split(EXPORT_BEGIN_MARKER, 1)[1].split(EXPORT_END_MARKER, 1)[0]


def test_export_memory_cli_sim_start_time_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HARNESS_SIM_START_TIME", raising=False)

    from harness.core import storage as storage_module

    importlib.reload(storage_module)
    monkeypatch.setattr(storage_module, "_STORAGE_ROOT", tmp_path / "storage")
    _seed_five_minute_row(storage_module, "agent-cli-sim")

    from harness.cli import main

    # Without the flag "now" is wall-clock, so the 2030 window excludes the row.
    assert main(["export-memory", "agent-cli-sim", "--log-level", "ERROR"]) == 0
    assert "future 5m row" not in _export_payload(capsys)

    clock.reset()  # main() left no offset, but keep runs independent
    assert (
        main(
            [
                "export-memory",
                "agent-cli-sim",
                "--sim-start-time",
                str(SIM_EPOCH),
                "--log-level",
                "ERROR",
            ]
        )
        == 0
    )
    assert "future 5m row" in _export_payload(capsys)


def test_export_memory_cli_sim_start_time_env_fallback(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    from harness.core import storage as storage_module

    importlib.reload(storage_module)
    monkeypatch.setattr(storage_module, "_STORAGE_ROOT", tmp_path / "storage")
    _seed_five_minute_row(storage_module, "agent-env-sim")

    monkeypatch.setenv("HARNESS_SIM_START_TIME", str(SIM_EPOCH))

    from harness.cli import main

    assert main(["export-memory", "agent-env-sim", "--log-level", "ERROR"]) == 0
    assert "future 5m row" in _export_payload(capsys)


def test_export_memory_cli_rejects_bad_env_value(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HARNESS_SIM_START_TIME", "not-a-number")

    from harness.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["export-memory", "agent-bad-env", "--log-level", "ERROR"])
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# boot forwards the flag into the exec'd `harness agent` argv
# ---------------------------------------------------------------------------


def test_build_agent_cmd_forwards_sim_start_time():
    from harness.cli import _build_agent_cmd

    args = argparse.Namespace(
        bedrock_token=None,
        bedrock_url=None,
        local=False,
        model=None,
        reasoning_effort=None,
        max_tokens=None,
        log_level=None,
        sim_start_time=SIM_EPOCH,
    )
    cmd = _build_agent_cmd("agent-xyz", run_id=None, args=args)

    assert "--sim-start-time" in cmd
    assert cmd[cmd.index("--sim-start-time") + 1] == str(SIM_EPOCH)


# ---------------------------------------------------------------------------
# Eval SimulatedClock drives the same core offset
# ---------------------------------------------------------------------------


def test_eval_clock_offsets_core_clock_and_restores_prior():
    from harness.evals.clock import simulated_clock_context

    # A pre-existing CLI offset must survive an eval clock's lifecycle.
    clock.set_start_epoch(SIM_EPOCH)
    prior_offset = clock.get_offset()

    eval_start = datetime(2031, 6, 15, 8, 0, 0, tzinfo=UTC)
    with simulated_clock_context(eval_start) as sim:
        assert abs((clock.now() - eval_start).total_seconds()) < 5.0

        target = eval_start + timedelta(hours=3)
        sim.advance_to(target)
        # Production call sites (memory, tracer, fakes) read core clock and
        # must see the advanced time without being handed the clock object.
        assert abs((clock.now() - target).total_seconds()) < 5.0

    assert clock.get_offset() == prior_offset

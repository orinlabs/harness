"""MemoryService end-to-end: real SQLite + real OpenRouter."""
from __future__ import annotations

import importlib
import time
from pathlib import Path

import pytest

NS_PER_MINUTE = 60 * 1_000_000_000


@pytest.fixture
def memory_env(tmp_path, monkeypatch, openrouter_key):
    """Fresh sqlite + loaded migrations + clean storage module."""
    monkeypatch.setenv("HARNESS_STORAGE_ROOT", str(tmp_path))
    mig_dir = Path(__file__).parent.parent / "src/harness/memory/migrations"
    monkeypatch.setenv("HARNESS_MIGRATIONS_DIR", str(mig_dir))

    from harness.core import storage as storage_module

    importlib.reload(storage_module)
    storage_module.load("agent-memtest")
    try:
        yield storage_module
    finally:
        storage_module.close()


def _make(recent_limit: int | None = None):
    from harness.memory import MemoryService

    kwargs = {"agent_id": "agent-memtest", "model": "openai/gpt-4o-mini"}
    if recent_limit is not None:
        kwargs["recent_limit"] = recent_limit
    return MemoryService(**kwargs)


def test_fresh_db_returns_system_and_empty_messages(memory_env):
    m = _make()
    system, messages = m.build_llm_inputs("you are helpful")

    assert system == "you are helpful"
    assert messages == []


def test_log_messages_persists_in_order(memory_env):
    m = _make()
    m.log_messages(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you"},
        ]
    )

    system, messages = m.build_llm_inputs("you are helpful")
    assert [msg["role"] for msg in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == "hello"
    assert messages[-1]["content"] == "how are you"


def test_messages_survive_reopen(memory_env):
    m = _make()
    m.log_messages([{"role": "user", "content": "first"}])
    memory_env.flush()
    memory_env.close()

    memory_env.load("agent-memtest")
    m2 = _make()
    _, messages = m2.build_llm_inputs("you are helpful")
    assert len(messages) == 1
    assert messages[0]["content"] == "first"


def test_nudge_appends_user_message(memory_env):
    from harness.memory.service import NUDGE_TEXT

    m = _make()
    m.log_messages([{"role": "assistant", "content": "no tool call here"}])
    m.nudge()

    _, messages = m.build_llm_inputs("sys")
    assert messages[-1] == {"role": "user", "content": NUDGE_TEXT}


def test_five_minute_summaries_generated_for_completed_buckets(memory_env):
    """Log messages with timestamps in a past 5-minute bucket; assert a 5m summary appears."""
    m = _make()

    # 6 minutes ago guarantees we're in a completed (non-current) 5m
    # bucket regardless of when the test runs within the current window.
    now = time.time_ns()
    six_min_ago = now - 6 * NS_PER_MINUTE

    m.log_messages(
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "Tell me a fact about Paris."},
            {
                "role": "assistant",
                "content": "Paris has a famous tower called the Eiffel Tower.",
            },
        ],
        ts_ns=six_min_ago,
    )
    m.update_summaries()

    rows = memory_env.db.execute(
        "SELECT date, hour, minute, summary FROM five_minute_summaries"
    ).fetchall()
    assert len(rows) >= 1, "expected at least one 5-minute summary"

    combined = "\n".join(r["summary"] for r in rows).lower()
    assert "paris" in combined, f"summary did not reference Paris: {combined!r}"


def test_summaries_not_regenerated_for_current_five_minute_bucket(memory_env):
    """Messages in the current (incomplete) 5m bucket must not produce a 5m summary yet."""
    m = _make()
    m.log_messages([{"role": "user", "content": "just happened"}])

    count = memory_env.db.execute(
        "SELECT COUNT(*) AS c FROM five_minute_summaries"
    ).fetchone()["c"]
    assert count == 0


def test_build_llm_inputs_renders_summary_block(memory_env):
    """Messages old enough to be summarized should push the summary header into system."""
    m = _make(recent_limit=4)

    # 6 minutes ago: guaranteed to be in a completed 5m bucket.
    now = time.time_ns()
    long_ago = now - 6 * NS_PER_MINUTE

    m.log_messages(
        [
            {"role": "user", "content": "The secret word is 'kumquat'."},
            {"role": "assistant", "content": "Understood, the secret word is kumquat."},
        ],
        ts_ns=long_ago,
    )
    m.update_summaries()

    system, _ = m.build_llm_inputs("base sys")

    assert "5-MINUTE SUMMARIES" in system or "HOURLY SUMMARIES" in system, (
        f"expected a tier header in rendered system, got: {system!r}"
    )
    assert "kumquat" in system.lower(), (
        f"summary did not preserve the kumquat fact: {system!r}"
    )

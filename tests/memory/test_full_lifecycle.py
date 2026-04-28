"""End-to-end: seed messages across history -> summarize through all 6 tiers ->
agent recalls a fact from memory via SMS.

This test is slower than the rest (dozens of real OpenRouter calls to build
summaries) but exercises the full memory + harness stack the way production
will. Budget ~$0.01-0.03 per run on openai/gpt-4o-mini.
"""
from __future__ import annotations

import importlib
import json
import sys
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# Reuse the in-process fake platform from tests/fake_platform.py.
sys.path.insert(0, str(Path(__file__).parent.parent))
from fake_platform import FakePlatform  # noqa: E402

CHEAP_MODEL = "openai/gpt-4o-mini"


@pytest.fixture
def env(tmp_path, monkeypatch, openrouter_key):
    """Fresh storage + fake platform + core modules reloaded to pick up env."""
    monkeypatch.setenv("HARNESS_STORAGE_ROOT", str(tmp_path))
    mig_dir = Path(__file__).parent.parent.parent / "src/harness/memory/migrations"
    monkeypatch.setenv("HARNESS_MIGRATIONS_DIR", str(mig_dir))

    platform = FakePlatform()
    platform.start()
    monkeypatch.setenv("BEDROCK_URL", platform.url)
    monkeypatch.setenv("BEDROCK_TOKEN", "lifecycle-test")

    from harness.core import llm, storage, tracer

    importlib.reload(storage)
    importlib.reload(tracer)
    importlib.reload(llm)

    try:
        yield platform
    finally:
        platform.stop()
        try:
            storage.close()
        except Exception:
            pass


def _insert_message(ts: datetime, role: str, content: str) -> None:
    """Write one message at a precise timestamp without triggering summarization."""
    from harness.core import storage

    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    ts_ns = int(ts.timestamp() * 1_000_000_000)
    msg = {"role": role, "content": content}
    storage.db.execute(
        "INSERT INTO messages (id, ts_ns, role, content_json) VALUES (?, ?, ?, ?)",
        (str(uuid.uuid4()), ts_ns, role, json.dumps(msg)),
    )


def _seed_conversation_at(ts: datetime, fact: str) -> None:
    """Write a short user+assistant exchange at consecutive microseconds.

    The messages share a bucket key (same 5-minute slot) so they
    produce exactly one 5-minute summary.
    """
    _insert_message(ts, "user", f"Just so you remember: {fact}")
    _insert_message(ts + timedelta(milliseconds=1), "assistant", f"Got it — I'll remember: {fact}")


SECRET_CODE = "7314"
FAVORITE_DESSERT = "tiramisu"
CITY = "Reykjavik"


def _seed_history(now: datetime) -> None:
    """Drop facts at a spread of historical points so every tier gets populated
    by `update_summaries(current_time=now)`.

    Bucket layout after rollup (for `now` = right now):
      - ~2 months ago: gets rolled into monthly
      - ~3 weeks ago:  weekly (and monthly if spans month boundary)
      - ~5 days ago:   daily + weekly
      - ~6 hours ago:  hourly + daily
      - ~30 min ago:   5-minute + hourly
      - ~7 min ago:    5-minute (bucket already complete)
    """
    _seed_conversation_at(
        now - timedelta(days=62, hours=3),
        f"the secret code is {SECRET_CODE}",
    )
    _seed_conversation_at(
        now - timedelta(days=21, hours=2),
        f"my favorite dessert is {FAVORITE_DESSERT}",
    )
    _seed_conversation_at(
        now - timedelta(days=5, hours=1),
        f"I'm planning a trip to {CITY}",
    )
    _seed_conversation_at(
        now - timedelta(hours=6),
        "I had coffee with Alex this morning",
    )
    _seed_conversation_at(
        now - timedelta(minutes=30),
        "the meeting is at 3 pm tomorrow",
    )
    _seed_conversation_at(
        now - timedelta(minutes=7),
        "reminder: grocery run after work",
    )


def test_tiered_summaries_build_across_all_six_layers(env):
    """Seed history, run summarisation once, assert every tier got rows and the
    facts we planted propagate through at least one of the summaries."""
    from harness.core import storage
    from harness.memory import MemoryService

    storage.load("agent-lifecycle")
    now = datetime.now(tz=UTC).replace(second=0, microsecond=0)

    _seed_history(now)

    usage = MemoryService(agent_id="agent-lifecycle", model=CHEAP_MODEL).update_summaries(
        current_time=now
    )

    def _count(table: str) -> int:
        return storage.db.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]

    counts = {
        "5m": _count("five_minute_summaries"),
        "hourly": _count("hourly_summaries"),
        "daily": _count("daily_summaries"),
        "weekly": _count("weekly_summaries"),
        "monthly": _count("monthly_summaries"),
    }

    print(f"\n[lifecycle] tier counts: {counts}")
    print(
        f"[lifecycle] llm usage: {usage.llm_calls} calls, "
        f"{usage.input_tokens} in / {usage.output_tokens} out, "
        f"${usage.total_cost:.4f}"
    )

    assert counts["5m"] >= 1, (
        f"expected >=1 5m summary (at least from the ~7min-ago seed), got {counts}"
    )
    assert counts["hourly"] >= 1
    assert counts["daily"] >= 1
    assert counts["weekly"] >= 1
    assert counts["monthly"] >= 1

    all_summary_text = "\n".join(
        row["summary"]
        for table in (
            "five_minute_summaries",
            "hourly_summaries",
            "daily_summaries",
            "weekly_summaries",
            "monthly_summaries",
        )
        for row in storage.db.execute(f"SELECT summary FROM {table}")
    ).lower()

    for fact_fragment in (SECRET_CODE, FAVORITE_DESSERT.lower(), CITY.lower()):
        assert fact_fragment in all_summary_text, (
            f"fact {fact_fragment!r} did not propagate through any summary tier. "
            f"Combined summaries: {all_summary_text!r}"
        )

    storage.close()


def test_agent_recalls_fact_from_summarized_memory(env):
    """Seed old facts, build the full tier stack, then run a Harness with an
    SMS inbox. The inbox asks for a fact that only exists in a monthly-tier
    summary — so the agent has to rely on build_llm_inputs pulling that
    summary into the system prompt."""
    from harness import AgentConfig, ExternalToolSpec, Harness
    from harness.core import storage
    from harness.memory import MemoryService

    storage.load("agent-lifecycle-2")
    now = datetime.now(tz=UTC).replace(second=0, microsecond=0)

    _seed_history(now)
    MemoryService(agent_id="agent-lifecycle-2", model=CHEAP_MODEL).update_summaries(
        current_time=now
    )
    storage.flush()
    storage.close()

    # Mock SMS inbox; the incoming question targets a fact from ~2 months ago
    # (only present in the monthly/weekly summary tier by now).
    incoming_body = "hey, what was that secret code i told you ages ago?"
    outgoing: list[dict] = []

    def sms_check_handler(args, envelope):
        msgs = [{"id": "m1", "from": "+15551234567", "body": incoming_body}]
        return {"text": "New SMS:\n" + json.dumps(msgs)}

    def sms_send_handler(args, envelope):
        outgoing.append({"to": args.get("to"), "body": args.get("body")})
        return {"text": f"Sent to {args.get('to')}"}

    env.register_tool("sms_check_inbox", sms_check_handler)
    env.register_tool("sms_send", sms_send_handler)

    sms_check = ExternalToolSpec(
        name="sms_check_inbox",
        description="Read unread SMS messages. Marks them read.",
        parameters={"type": "object", "properties": {}},
        url=f"{env.url}/fake_tools/sms_check_inbox",
    )
    sms_send = ExternalToolSpec(
        name="sms_send",
        description="Send an SMS reply.",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "body"],
        },
        url=f"{env.url}/fake_tools/sms_send",
    )

    config = AgentConfig(
        id="agent-lifecycle-2",
        model=CHEAP_MODEL,
        system_prompt=(
            "You are an SMS concierge. On wake: 1) call sms_check_inbox to see "
            "new messages, 2) use your memory (summaries from past days/weeks/"
            "months) to answer, 3) reply via sms_send to the sender, 4) call "
            "sleep with a future time once done."
        ),
        tools=[sms_check, sms_send],
    )

    start = time.perf_counter()
    Harness(config, run_id="run-lifecycle").run()
    elapsed = time.perf_counter() - start
    print(f"\n[lifecycle] agent run: {elapsed:.2f}s")

    assert outgoing, "agent never sent an SMS reply"
    reply_body = outgoing[-1]["body"]
    print(f"[lifecycle] agent replied: {reply_body!r}")

    assert SECRET_CODE in reply_body, (
        f"agent failed to recall the secret code from summarized memory. "
        f"Reply: {reply_body!r}"
    )

    assert env.sleep_requests, "agent never called sleep"

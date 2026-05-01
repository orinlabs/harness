"""Real-sqlite tests for the in-process fake adapters.

No mocks: every test opens a real per-agent sqlite DB under a tmp_path,
runs the fake adapter migrations, and exercises the Tool.call contract
end-to-end. This mirrors how an eval scenario invokes them at runtime.
"""

from __future__ import annotations

import pytest

from harness.config import AgentConfig, ExternalToolSpec
from harness.core import storage
from harness.evals.fakes import (
    FakeComputerAdapter,
    FakeContactsAdapter,
    FakeEmailAdapter,
    FakeSMSAdapter,
)
from harness.evals.fakes import (
    email as email_fake,
)
from harness.evals.fakes import (
    sms as sms_fake,
)
from harness.evals.fakes.base import apply_migrations


def _by_name(tools: list) -> dict:
    return {t.name: t for t in tools}


@pytest.fixture
def agent_db(tmp_path, monkeypatch):
    """Open a real sqlite DB under tmp_path, run fake-adapter migrations.

    Test-local: mirrors the eval runner's pre-scenario setup.
    """
    monkeypatch.setattr(storage, "_STORAGE_ROOT", tmp_path)
    storage.load("test-agent-fakes")
    apply_migrations()
    try:
        yield "test-agent-fakes"
    finally:
        storage.flush()
        storage.close()


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


def test_email_list_and_get_thread_after_inject(agent_db):
    msg_id = email_fake.inject_inbound(
        thread_id=None,
        from_email="alice@example.com",
        to_email="agent@eval.test",
        subject="Hi",
        body="Hello there!",
        sent_at="2026-04-23T10:00:00Z",
    )
    assert msg_id.startswith("sim_inmsg_")

    by_name = _by_name(FakeEmailAdapter.make_tools())

    list_result = by_name["list_threads"].call({}, ctx=None)
    assert "Hi" in list_result.text
    assert "alice@example.com" in list_result.text

    # Pull the thread_id out of the listing output.
    import re

    thread_id_match = re.search(r"Thread ID: (sim_thread_\w+)", list_result.text)
    assert thread_id_match, list_result.text
    thread_id = thread_id_match.group(1)

    thread_result = by_name["get_thread"].call({"thread_id": thread_id}, ctx=None)
    assert "Hello there!" in thread_result.text
    assert "alice@example.com" in thread_result.text
    assert "Direction: inbound" in thread_result.text


def test_email_send_creates_thread(agent_db):
    by_name = _by_name(FakeEmailAdapter.make_tools())

    send_result = by_name["send_email"].call(
        {"to": ["bob@example.com"], "subject": "Question", "body": "Are you free?"},
        ctx=None,
    )
    assert "Email sent successfully" in send_result.text
    assert "bob@example.com" in send_result.text

    # Verify persistence.
    rows = storage.db.execute(
        "SELECT direction, from_addr, body FROM fake_email_message"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["direction"] == "outbound"
    assert rows[0]["body"] == "Are you free?"

    # And it shows up in list_threads.
    list_result = by_name["list_threads"].call({}, ctx=None)
    assert "Question" in list_result.text


def test_email_reply_and_search(agent_db):
    # Seed an inbound; capture its message id for reply.
    email_fake.inject_inbound(
        thread_id=None,
        from_email="carol@example.com",
        to_email="agent@eval.test",
        subject="Project update",
        body="Shipping tomorrow.",
        sent_at="2026-04-23T10:00:00Z",
    )
    by_name = _by_name(FakeEmailAdapter.make_tools())

    search_result = by_name["search_emails"].call({"query": "Shipping"}, ctx=None)
    assert "Project update" in search_result.text
    import re

    mid_match = re.search(r"Message ID: (sim_inmsg_\w+)", search_result.text)
    assert mid_match, search_result.text
    message_id = mid_match.group(1)

    reply_result = by_name["reply_to_email"].call(
        {"message_id": message_id, "to": "carol@example.com", "body": "Great, thanks!"},
        ctx=None,
    )
    assert "Reply sent to carol@example.com" in reply_result.text

    # Outbound message is persisted with Re: subject.
    rows = storage.db.execute(
        "SELECT subject FROM fake_email_message WHERE direction = 'outbound'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["subject"] == "Re: Project update"


def test_email_inbox_info_and_empty_list(agent_db):
    by_name = _by_name(FakeEmailAdapter.make_tools())

    info_result = by_name["get_inbox_info"].call({}, ctx=None)
    assert "agent@eval.test" in info_result.text

    empty_result = by_name["list_threads"].call({}, ctx=None)
    assert "No email threads" in empty_result.text


# ---------------------------------------------------------------------------
# SMS
# ---------------------------------------------------------------------------


def test_sms_roundtrip(agent_db):
    sms_fake.inject_inbound("+15551234567", "yo", sent_at="2026-04-23T10:00:00Z")
    by_name = _by_name(FakeSMSAdapter.make_tools())

    list_result = by_name["list_conversations"].call({}, ctx=None)
    assert "+15551234567" in list_result.text
    assert "yo" in list_result.text

    get_result = by_name["get_conversation"].call({"phone": "+15551234567"}, ctx=None)
    assert "yo" in get_result.text


def test_sms_send_persists_outbound(agent_db):
    by_name = _by_name(FakeSMSAdapter.make_tools())

    result = by_name["send_sms"].call({"phone": "+15559999999", "body": "on my way"}, ctx=None)
    assert "SMS sent to +15559999999" in result.text

    rows = storage.db.execute("SELECT direction, body FROM fake_sms_message").fetchall()
    assert len(rows) == 1
    assert rows[0]["direction"] == "outbound"
    assert rows[0]["body"] == "on my way"


def test_sms_empty_state(agent_db):
    by_name = _by_name(FakeSMSAdapter.make_tools())
    assert "No SMS conversations" in by_name["list_conversations"].call({}, ctx=None).text


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------


def test_contacts_full_crud(agent_db):
    by_name = _by_name(FakeContactsAdapter.make_tools())

    create_result = by_name["create_contact"].call(
        {"name": "Jamie", "email": "jamie@example.com", "phone": "+15550001111"},
        ctx=None,
    )
    assert "Contact created: Jamie" in create_result.text

    import re

    cid_match = re.search(r"ID: (sim_contact_\w+)", create_result.text)
    assert cid_match, create_result.text
    contact_id = cid_match.group(1)

    get_result = by_name["get_contact"].call({"contact_id": contact_id}, ctx=None)
    assert "Jamie" in get_result.text
    assert "jamie@example.com" in get_result.text

    list_result = by_name["list_contacts"].call({}, ctx=None)
    assert "Jamie" in list_result.text

    search_result = by_name["list_contacts"].call({"search": "Jamie"}, ctx=None)
    assert "Jamie" in search_result.text

    update_result = by_name["update_contact"].call(
        {"contact_id": contact_id, "notes": "Allergic to tree nuts"}, ctx=None
    )
    assert "updated" in update_result.text

    get_after_update = by_name["get_contact"].call({"contact_id": contact_id}, ctx=None)
    assert "Allergic to tree nuts" in get_after_update.text

    delete_result = by_name["delete_contact"].call({"contact_id": contact_id}, ctx=None)
    assert "deleted successfully" in delete_result.text

    empty_result = by_name["list_contacts"].call({}, ctx=None)
    assert "no contacts" in empty_result.text.lower()


def test_contacts_missing_id(agent_db):
    by_name = _by_name(FakeContactsAdapter.make_tools())
    result = by_name["get_contact"].call({"contact_id": "missing"}, ctx=None)
    assert "not found" in result.text.lower()


# ---------------------------------------------------------------------------
# Computer
# ---------------------------------------------------------------------------


class _StubCtx:
    """Stand-in for harness.context.RunContext -- tests only need agent_id."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.run_id = "test-run"
        self.turn = 0
        self.sleep_requested = False


def test_computer_write_read_roundtrip(agent_db):
    by_name = _by_name(FakeComputerAdapter.make_tools())
    ctx = _StubCtx(agent_db)

    write_result = by_name["computer_write_file"].call(
        {"path": "/notes.txt", "content": "hello world"}, ctx=ctx
    )
    assert "Wrote 11 bytes" in write_result.text

    read_result = by_name["computer_read_file"].call({"path": "/notes.txt"}, ctx=ctx)
    assert read_result.text == "hello world"


def test_computer_list_and_exec(agent_db):
    by_name = _by_name(FakeComputerAdapter.make_tools())
    ctx = _StubCtx(agent_db)

    by_name["computer_write_file"].call({"path": "/workspace/a.txt", "content": "a"}, ctx=ctx)
    by_name["computer_write_file"].call({"path": "/workspace/b.txt", "content": "b"}, ctx=ctx)

    ls_result = by_name["computer_list_files"].call({"path": "/workspace"}, ctx=ctx)
    assert "a.txt" in ls_result.text
    assert "b.txt" in ls_result.text

    exec_result = by_name["computer_exec"].call({"command": "echo hello"}, ctx=ctx)
    assert "Exit code: 0" in exec_result.text
    assert "hello" in exec_result.text


def test_computer_sandbox_escape_blocked(agent_db):
    by_name = _by_name(FakeComputerAdapter.make_tools())
    ctx = _StubCtx(agent_db)

    # Relative traversal must not escape the tmpdir.
    result = by_name["computer_write_file"].call(
        {"path": "../../escape.txt", "content": "x"}, ctx=ctx
    )
    assert "escapes sandbox" in result.text


# ---------------------------------------------------------------------------
# Integration with build_tool_map -- this is the path production hits.
# ---------------------------------------------------------------------------


def test_build_tool_map_with_fakes_and_external_mix(agent_db):
    from harness.tools.registry import build_tool_map

    remote_spec = ExternalToolSpec(
        name="remote_op",
        description="remote op",
        parameters={"type": "object", "properties": {}},
        url="https://example.invalid/remote",
    )

    tm = build_tool_map(
        [
            *FakeEmailAdapter.make_tools(),
            *FakeContactsAdapter.make_tools(),
            remote_spec,
        ]
    )
    # Email, contacts, remote, plus the builtin sleep.
    assert "send_email" in tm
    assert "create_contact" in tm
    assert "remote_op" in tm
    assert "sleep" in tm


def test_agent_config_accepts_fake_tools():
    """Scenarios should be able to splat fake tools straight into AgentConfig."""
    ac = AgentConfig(
        id="whoever",
        model="openai/gpt-4o-mini",
        system_prompt="",
        tools=[*FakeEmailAdapter.make_tools()],
    )
    assert any(t.name == "send_email" for t in ac.tools)

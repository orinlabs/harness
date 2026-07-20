"""Workflow runner tests against a fake `current` API (real HTTP, fake impl).

Covers the four invariants:
  * journal first (transition/record ordering asserted on the wire),
  * exit at gates (proposal record + waiting_on_gate + exit 0),
  * resume from journal (succeeded steps skipped, gate decisions honored,
    `when:` approved/rejected branches),
  * terminal run_report before exit.

Plus retry/on_failure semantics and agent steps with a stubbed LLM (no
OpenRouter traffic — same ``monkeypatch llm.complete`` pattern as
``tests/test_assistant_reasoning_logged.py``).
"""

from __future__ import annotations

import base64
import importlib
import json
from pathlib import Path

import pytest

from harness.cloud.current.client import CurrentClient
from harness.core.llm import LLMResponse, ToolCall, Usage
from tests.fake_current import FakeCurrent

RUN_ID = "run-1"
PROJECT = "11111111-1111-1111-1111-111111111111"


# ---------------------------------------------------------------------------
# Envelope / runner builders
# ---------------------------------------------------------------------------


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def script_step(step_id: str, code: str, **extra) -> dict:
    return {
        "id": step_id,
        "kind": "script",
        "run": f"work/{step_id}.py",
        "content_b64": _b64(code),
        **extra,
    }


def gate_step(step_id: str, proposal: str, **extra) -> dict:
    return {"id": step_id, "kind": "gate", "proposal": proposal, **extra}


def envelope_response(
    steps: list[dict],
    *,
    control: dict | None = None,
    steps_state: list[dict] | None = None,
    decisions: list[dict] | None = None,
    project: str | None = PROJECT,
) -> dict:
    return {
        "run_id": RUN_ID,
        "workspace": "22222222-2222-2222-2222-222222222222",
        "project": project,
        "envelope": {
            "envelope_version": 1,
            "name": "test-workflow",
            "kind": "scripted",
            "control": {
                "display_name": "Test Workflow",
                "required_adapters": [],
                "inputs": [],
                "outputs": [],
                "files": [],
                **(control or {}),
            },
            "emits": [],
            "steps": steps,
            "rendered_hash": "sha256:feedbeef",
        },
        "steps_state": (
            steps_state
            if steps_state is not None
            else [{"step_id": s["id"], "status": "pending", "attempts": 0} for s in steps]
        ),
        "decisions": decisions or [],
    }


@pytest.fixture
def fake_current():
    fake = FakeCurrent(run_id=RUN_ID)
    fake.start()
    try:
        yield fake
    finally:
        fake.stop()


def make_runner(fake: FakeCurrent, tmp_path: Path, **kwargs):
    from harness.workflow import WorkflowRunner

    kwargs.setdefault("working_dir", tmp_path / "wf")
    kwargs.setdefault("data_root", tmp_path / "data")
    client = CurrentClient(fake.url, "test-run-token", fake.run_id)
    return WorkflowRunner(client, **kwargs)


def run_report(fake: FakeCurrent) -> dict:
    reports = fake.records_of("run_report")
    assert len(reports) == 1, f"expected exactly one run_report, got {len(reports)}"
    return reports[0]


# ---------------------------------------------------------------------------
# (a) Scripted happy path
# ---------------------------------------------------------------------------


def test_scripted_happy_path_journals_in_order(fake_current, tmp_path):
    """Two script steps run in order; every transition is journaled before the
    next action; control files are staged; outputs are promoted on success;
    the run_report record is the last thing on the wire."""
    (tmp_path / "data").mkdir()  # workspace volume present -> promotion active
    fake_current.envelope_response = envelope_response(
        [
            script_step(
                "s1",
                # cwd must be the working dir; control file must be staged.
                "import pathlib\n"
                "assert pathlib.Path('work/seed.txt').read_text() == 'seed'\n"
                "pathlib.Path('work/s1-ran').write_text('yes')\n",
            ),
            script_step(
                "s2",
                "import pathlib\n"
                "assert pathlib.Path('work/s1-ran').exists()\n"
                "pathlib.Path('out/result.txt').write_text('final answer')\n",
            ),
        ],
        control={
            "files": [{"dest": "work/seed.txt", "content_b64": _b64("seed")}],
            "outputs": ["project"],
        },
    )

    exit_code = make_runner(fake_current, tmp_path).run()

    assert exit_code == 0
    assert fake_current.transition_tuples() == [
        ("s1", "running", 1),
        ("s1", "succeeded", 1),
        ("s2", "running", 1),
        ("s2", "succeeded", 1),
    ]
    # Journal-first ordering across both endpoints: all four transitions,
    # then the terminal run_report record, nothing after it.
    kinds = [(kind, body.get("step_id"), body.get("status") or body.get("record_type"))
             for kind, body in fake_current.events]
    assert kinds[-1] == ("record", None, "run_report")
    assert all(kind == "transition" for kind, _, _ in kinds[:-1])

    # Every request carried the run-scoped bearer token.
    assert set(fake_current.auth_headers) == {"Bearer test-run-token"}

    report = run_report(fake_current)
    assert report["project"] == PROJECT
    data = report["data"]
    assert data["workflow"] == "test-workflow"
    assert data["definition_hash"] == "sha256:feedbeef"
    assert data["status"] == "succeeded"
    assert data["started_at"] and data["finished_at"]
    assert data["steps"] == [
        {"id": "s1", "status": "succeeded", "attempts": 1},
        {"id": "s2", "status": "succeeded", "attempts": 1},
    ]

    # out/ was promoted to /data/projects/{project}/workflow-outputs/{run_id}/.
    promoted = data["promoted_outputs"]
    assert [p["path"] for p in promoted] == ["out/result.txt"]
    dest = tmp_path / "data" / "projects" / PROJECT / "workflow-outputs" / RUN_ID / "result.txt"
    assert dest.read_text() == "final answer"
    import hashlib

    expected = "sha256:" + hashlib.sha256(b"final answer").hexdigest()
    assert promoted[0]["content_hash"] == expected


def test_project_input_hydration(fake_current, tmp_path):
    """`control.inputs: [project]` copies /data/projects/{project}/ into
    inputs/project/; absent sources are skipped silently."""
    src = tmp_path / "data" / "projects" / PROJECT
    src.mkdir(parents=True)
    (src / "brief.md").write_text("the brief")
    fake_current.envelope_response = envelope_response(
        [
            script_step(
                "s1",
                "import pathlib\n"
                "assert pathlib.Path('inputs/project/brief.md').read_text() == 'the brief'\n"
                # 'company' was requested too but /data/company doesn't exist:
                # hydration must skip it silently, not fail the run.
                "assert not pathlib.Path('inputs/company').exists()\n",
            )
        ],
        control={"inputs": ["project", "company"]},
    )

    assert make_runner(fake_current, tmp_path).run() == 0
    assert fake_current.transition_tuples()[-1] == ("s1", "succeeded", 1)


# ---------------------------------------------------------------------------
# (b) Gate exit + resume with decision (approved and rejected branches)
# ---------------------------------------------------------------------------


def _gated_steps() -> list[dict]:
    return [
        script_step(
            "prep",
            "import json, pathlib\n"
            "pathlib.Path('out').mkdir(exist_ok=True)\n"
            "pathlib.Path('out/proposal.json').write_text(json.dumps("
            "{'summary': 'Ship the Q3 report.', 'total': 42}))\n",
        ),
        gate_step("approve", "ship_report"),
        script_step("on-approved", "print('approved path')", when="approve.approved"),
        script_step("on-rejected", "print('rejected path')", when="approve.rejected"),
    ]


def test_gate_posts_proposal_and_exits_zero(fake_current, tmp_path):
    fake_current.envelope_response = envelope_response(_gated_steps())

    exit_code = make_runner(fake_current, tmp_path).run()

    assert exit_code == 0
    assert fake_current.transition_tuples() == [
        ("prep", "running", 1),
        ("prep", "succeeded", 1),
        ("approve", "waiting_on_gate", 1),
    ]
    # Journal-first at the gate: the proposal record hits the wire *before*
    # the waiting_on_gate transition.
    event_kinds = [
        (kind, body.get("record_type") or body.get("status"))
        for kind, body in fake_current.events
    ]
    assert event_kinds.index(("record", "agent_proposal")) < event_kinds.index(
        ("transition", "waiting_on_gate")
    )

    [proposal] = fake_current.records_of("agent_proposal")
    assert proposal["step_id"] == "approve"
    assert proposal["project"] == PROJECT
    data = proposal["data"]
    assert data["proposal"] == "ship_report"
    # v0 payload = the JSON content of out/proposal.json staged by `prep`;
    # its agent-authored summary is used verbatim.
    assert data["payload"] == {"summary": "Ship the Q3 report.", "total": 42}
    assert data["summary"] == "Ship the Q3 report."
    assert data["idempotency_key"] == f"wf-{RUN_ID}-approve"
    # Gate exits are NOT terminal: no run_report.
    assert fake_current.records_of("run_report") == []


def _resume_after_gate(fake_current, decision_state: str | None) -> None:
    """Mutate the fake's journal to look like a resume after the gate."""
    fake_current.envelope_response["steps_state"] = [
        {"step_id": "prep", "status": "succeeded", "attempts": 1},
        {"step_id": "approve", "status": "waiting_on_gate", "attempts": 1},
        {"step_id": "on-approved", "status": "pending", "attempts": 0},
        {"step_id": "on-rejected", "status": "pending", "attempts": 0},
    ]
    if decision_state is not None:
        fake_current.envelope_response["decisions"] = [
            {
                "gate_step_id": "approve",
                "proposal_record_id": "33333333-3333-3333-3333-333333333333",
                "state": decision_state,
            }
        ]
    fake_current.reset_journal()


def test_gate_resume_runs_approved_branch_and_skips_rejected(fake_current, tmp_path):
    fake_current.envelope_response = envelope_response(_gated_steps())
    assert make_runner(fake_current, tmp_path).run() == 0  # parks at the gate

    _resume_after_gate(fake_current, "approved")
    exit_code = make_runner(fake_current, tmp_path).run()  # fresh process, same journal

    assert exit_code == 0
    assert fake_current.transition_tuples() == [
        ("approve", "succeeded", 1),
        ("on-approved", "running", 1),
        ("on-approved", "succeeded", 1),
        ("on-rejected", "skipped", 1),
    ]
    data = run_report(fake_current)["data"]
    assert data["status"] == "succeeded"
    assert data["steps"] == [
        {"id": "prep", "status": "succeeded", "attempts": 1},
        {"id": "approve", "status": "succeeded", "attempts": 1},
        {"id": "on-approved", "status": "succeeded", "attempts": 1},
        {"id": "on-rejected", "status": "skipped", "attempts": 0},
    ]


def test_gate_resume_runs_rejected_branch_and_skips_approved(fake_current, tmp_path):
    fake_current.envelope_response = envelope_response(_gated_steps())
    assert make_runner(fake_current, tmp_path).run() == 0

    _resume_after_gate(fake_current, "rejected")
    exit_code = make_runner(fake_current, tmp_path).run()

    assert exit_code == 0
    assert fake_current.transition_tuples() == [
        ("approve", "succeeded", 1),
        ("on-approved", "skipped", 1),
        ("on-rejected", "running", 1),
        ("on-rejected", "succeeded", 1),
    ]


def test_gate_resume_without_decision_exits_again(fake_current, tmp_path):
    """Re-dispatched before anyone decided: re-post waiting_on_gate
    (idempotent) and get out without re-running anything or re-posting the
    proposal record."""
    fake_current.envelope_response = envelope_response(_gated_steps())
    assert make_runner(fake_current, tmp_path).run() == 0

    _resume_after_gate(fake_current, None)
    exit_code = make_runner(fake_current, tmp_path).run()

    assert exit_code == 0
    assert fake_current.transition_tuples() == [("approve", "waiting_on_gate", 1)]
    assert fake_current.records == []


# ---------------------------------------------------------------------------
# (c) Retry + on_failure
# ---------------------------------------------------------------------------

_FAILING_SCRIPT = "import sys\nsys.stderr.write('boom: cannot reticulate')\nsys.exit(3)\n"


def test_retry_exhaustion_then_on_failure_continue(fake_current, tmp_path):
    fake_current.envelope_response = envelope_response(
        [
            script_step(
                "flaky", _FAILING_SCRIPT, retry={"attempts": 2}, on_failure="continue"
            ),
            script_step("after", "print('still ran')"),
        ]
    )

    exit_code = make_runner(fake_current, tmp_path).run()

    assert exit_code == 0
    assert fake_current.transition_tuples() == [
        ("flaky", "running", 1),
        ("flaky", "running", 2),
        ("flaky", "failed", 2),
        ("after", "running", 1),
        ("after", "succeeded", 1),
    ]
    failed = [t for t in fake_current.transitions if t["status"] == "failed"]
    assert "boom: cannot reticulate" in failed[0]["error"]  # stderr tail captured

    data = run_report(fake_current)["data"]
    assert data["status"] == "succeeded"  # the *run* completed
    [flaky_entry] = [s for s in data["steps"] if s["id"] == "flaky"]
    assert flaky_entry["status"] == "failed"
    assert flaky_entry["attempts"] == 2
    assert "boom" in flaky_entry["error"]


def test_failure_aborts_run_by_default(fake_current, tmp_path):
    """`on_failure: fail` (the default): terminal failed run, downstream steps
    never run, run_report (status=failed) still posted after the failed
    transition."""
    fake_current.envelope_response = envelope_response(
        [
            script_step("broken", _FAILING_SCRIPT),
            script_step("never", "print('unreachable')"),
        ]
    )

    exit_code = make_runner(fake_current, tmp_path).run()

    assert exit_code == 1
    assert fake_current.transition_tuples() == [
        ("broken", "running", 1),
        ("broken", "failed", 1),
    ]
    data = run_report(fake_current)["data"]
    assert data["status"] == "failed"
    assert data["steps"] == [
        {
            "id": "broken",
            "status": "failed",
            "attempts": 1,
            "error": data["steps"][0]["error"],
        },
        {"id": "never", "status": "pending", "attempts": 0},
    ]
    assert data["promoted_outputs"] == []
    # run_report is the last event; the failed transition preceded it.
    assert fake_current.events[-1][0] == "record"
    assert fake_current.events[-1][1]["record_type"] == "run_report"


# ---------------------------------------------------------------------------
# (d) Resume skips already-finished steps
# ---------------------------------------------------------------------------


def test_resume_skips_already_succeeded_steps(fake_current, tmp_path):
    fake_current.envelope_response = envelope_response(
        [
            # Would fail loudly if re-run: the journal, not the script, must
            # decide whether it runs again.
            script_step("done-before", "raise SystemExit('must not re-run')"),
            script_step("still-todo", "print('running')"),
        ],
        steps_state=[
            {"step_id": "done-before", "status": "succeeded", "attempts": 1},
            {"step_id": "still-todo", "status": "pending", "attempts": 0},
        ],
    )

    exit_code = make_runner(fake_current, tmp_path).run()

    assert exit_code == 0
    assert fake_current.transition_tuples() == [
        ("still-todo", "running", 1),
        ("still-todo", "succeeded", 1),
    ]
    data = run_report(fake_current)["data"]
    assert data["steps"] == [
        {"id": "done-before", "status": "succeeded", "attempts": 1},
        {"id": "still-todo", "status": "succeeded", "attempts": 1},
    ]


# ---------------------------------------------------------------------------
# Agent steps (stubbed LLM — no OpenRouter traffic)
# ---------------------------------------------------------------------------


@pytest.fixture
def harness_storage(tmp_path, monkeypatch):
    """Fresh sqlite + migrations scoped to tmp_path (the pattern
    ``tests/test_assistant_reasoning_logged.py`` uses)."""
    mig_dir = Path(__file__).parent.parent / "src/harness/memory/migrations"
    monkeypatch.setenv("HARNESS_MIGRATIONS_DIR", str(mig_dir))

    from harness.core import storage

    importlib.reload(storage)
    monkeypatch.setattr(storage, "_STORAGE_ROOT", tmp_path / "storage")
    yield
    try:
        storage.close()
    except Exception:
        pass


class _ScriptedLLM:
    """Deterministic ``llm.complete`` stand-in. Also records every call's
    kwargs so tests can assert what tools the model was offered."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of programmed responses")
        return self._responses.pop(0)

    def offered_tool_names(self, call_index: int = 0) -> set[str]:
        tools = self.calls[call_index].get("tools") or []
        return {t["function"]["name"] for t in tools}


def _llm_response(*, content: str | None = None, tool_calls: list[tuple[str, str, dict]] = ()):
    """Build an LLMResponse. ``tool_calls`` is [(id, name, args), ...]."""
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": tc_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for tc_id, name, args in tool_calls
        ]
    raw = {
        "id": "stub-1",
        "model": "test/stub",
        "choices": [
            {
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return LLMResponse(
        text=content or "",
        tool_calls=[ToolCall(id=tc_id, name=name, args=args) for tc_id, name, args in tool_calls],
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            total_cost=0.0,
            model="test/stub",
        ),
        raw=raw,
    )


def agent_step(step_id: str, **extra) -> dict:
    return {
        "id": step_id,
        "kind": "agent",
        "config": {
            "model": "test/stub",
            "system_prompt": "You are a workflow step. Do the task, then stop.",
        },
        "max_turns": 5,
        **extra,
    }


def test_agent_step_uses_local_tools_and_ends_when_idle(
    fake_current, tmp_path, harness_storage, monkeypatch
):
    """The agent step gets bash/read_file/write_file (and NOT sleep), its
    tool calls act on the run working dir, and the step succeeds when the
    model stops calling tools."""
    from harness.core import llm as llm_mod

    scripted = _ScriptedLLM(
        [
            _llm_response(
                tool_calls=[
                    ("call-1", "write_file", {"path": "out/answer.txt", "content": "42"})
                ]
            ),
            _llm_response(content="Wrote the answer. Done."),
        ]
    )
    monkeypatch.setattr(llm_mod, "complete", scripted)

    fake_current.envelope_response = envelope_response([agent_step("think")])
    exit_code = make_runner(fake_current, tmp_path).run()

    assert exit_code == 0
    assert fake_current.transition_tuples() == [
        ("think", "running", 1),
        ("think", "succeeded", 1),
    ]
    assert (tmp_path / "wf" / "out" / "answer.txt").read_text() == "42"

    offered = scripted.offered_tool_names()
    assert {"bash", "read_file", "write_file"} <= offered
    assert "sleep" not in offered  # never offered in workflow mode

    assert run_report(fake_current)["data"]["status"] == "succeeded"


def test_agent_step_max_turns_exhaustion_is_a_step_failure(
    fake_current, tmp_path, harness_storage, monkeypatch
):
    from harness.core import llm as llm_mod

    # The model never goes idle: every turn calls bash. With max_turns=2 the
    # loop exhausts and the step fails (honoring default on_failure=fail).
    scripted = _ScriptedLLM(
        [
            _llm_response(tool_calls=[("c1", "bash", {"command": "true"})]),
            _llm_response(tool_calls=[("c2", "bash", {"command": "true"})]),
        ]
    )
    monkeypatch.setattr(llm_mod, "complete", scripted)

    fake_current.envelope_response = envelope_response(
        [agent_step("loops-forever", max_turns=2)]
    )
    exit_code = make_runner(fake_current, tmp_path).run()

    assert exit_code == 1
    assert fake_current.transition_tuples() == [
        ("loops-forever", "running", 1),
        ("loops-forever", "failed", 1),
    ]
    [failed] = [t for t in fake_current.transitions if t["status"] == "failed"]
    assert "max_turns=2" in failed["error"]
    assert run_report(fake_current)["data"]["status"] == "failed"


# ---------------------------------------------------------------------------
# Records endpoint 422 -> step failure (gate proposal rejected by schema)
# ---------------------------------------------------------------------------


def test_proposal_record_422_fails_the_gate_step(fake_current, tmp_path):
    fake_current.envelope_response = envelope_response([gate_step("g1", "bad_proposal")])
    fake_current.record_failures["agent_proposal"] = (
        422,
        {"detail": "payload does not match declared schema"},
    )

    exit_code = make_runner(fake_current, tmp_path).run()

    assert exit_code == 1
    assert fake_current.transition_tuples() == [("g1", "failed", 1)]
    [failed] = fake_current.transitions
    assert "422" in failed["error"]
    assert "schema" in failed["error"]
    data = run_report(fake_current)["data"]
    assert data["status"] == "failed"

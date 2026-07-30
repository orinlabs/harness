"""Workflow runner tests against a fake `current` API (real HTTP, fake impl).

Covers the four invariants:
  * journal first (transition/record ordering asserted on the wire),
  * exit at gates (proposal record + waiting_on_gate + exit 0),
  * resume from journal (succeeded steps skipped, gate approvals honored,
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
# Definition / runner builders
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


def definition_response(
    steps: list[dict],
    *,
    control: dict | None = None,
    emits: list[dict] | None = None,
    steps_state: list[dict] | None = None,
    approvals: list[dict] | None = None,
    project: str | None = PROJECT,
) -> dict:
    return {
        "run_id": RUN_ID,
        "workspace": "22222222-2222-2222-2222-222222222222",
        "project": project,
        "definition": {
            "definition_version": 1,
            "name": "test-workflow",
            "kind": "scripted",
            "control": {
                "display_name": "Test Workflow",
                "required_integrations": [],
                "inputs": [],
                "outputs": [],
                "files": [],
                **(control or {}),
            },
            "emits": emits or [],
            "steps": steps,
            "rendered_hash": "sha256:feedbeef",
        },
        "steps_state": (
            steps_state
            if steps_state is not None
            else [{"step_id": s["id"], "status": "pending", "attempts": 0} for s in steps]
        ),
        "approvals": approvals or [],
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


def test_default_working_dir_is_under_workflow_home():
    """When `working_dir` isn't passed, the runner stages under a stable
    `~/workflow/<run_id>/` root -- not the run-id-bearing `~/wf-run-<id>`
    name every other test here overrides via `make_runner`'s
    `kwargs.setdefault("working_dir", ...)`."""
    from harness.workflow import WorkflowRunner

    client = CurrentClient("http://example.invalid", "test-run-token", RUN_ID)
    runner = WorkflowRunner(client)
    assert runner.working_dir == Path.home() / "workflow" / RUN_ID


# ---------------------------------------------------------------------------
# (a) Scripted happy path
# ---------------------------------------------------------------------------


def test_scripted_happy_path_journals_in_order(fake_current, tmp_path):
    """Two script steps run in order; every transition is journaled before the
    next action; control files are staged; outputs are promoted on success;
    the run_report record is the last thing on the wire."""
    (tmp_path / "data").mkdir()  # workspace volume present -> promotion active
    fake_current.definition_response = definition_response(
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
    fake_current.definition_response = definition_response(
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


def test_company_outputs_promote_to_a_stable_overwritable_path(fake_current, tmp_path):
    """`control.outputs: [company]` promotes out/ to /data/company/ with no
    run-id segment, overwriting what a prior run left and leaving files this
    run didn't rewrite alone -- the round trip that makes a workspace-wide
    store readable back through inputs/company on the next run."""
    company = tmp_path / "data" / "company"
    company.mkdir(parents=True)
    (company / "state.json").write_text('{"watermark": "old"}')
    (company / "untouched.txt").write_text("still here")
    fake_current.definition_response = definition_response(
        [
            script_step(
                "s1",
                "import pathlib\n"
                # Prior run's state hydrated in; this run supersedes it.
                "assert pathlib.Path('inputs/company/state.json').read_text()"
                " == '{\"watermark\": \"old\"}'\n"
                "pathlib.Path('out/sitetracker').mkdir(parents=True)\n"
                "pathlib.Path('out/state.json').write_text('{\"watermark\": \"new\"}')\n"
                "pathlib.Path('out/sitetracker/inventory.json').write_text('{}')\n",
            )
        ],
        control={"inputs": ["company"], "outputs": ["company"]},
    )

    assert make_runner(fake_current, tmp_path).run() == 0

    assert (company / "state.json").read_text() == '{"watermark": "new"}'
    assert (company / "sitetracker" / "inventory.json").read_text() == "{}"
    # Promotion touches only what out/ holds; the rest of the store survives.
    assert (company / "untouched.txt").read_text() == "still here"
    promoted = run_report(fake_current)["data"]["promoted_outputs"]
    assert sorted(p["path"] for p in promoted) == [
        "out/sitetracker/inventory.json",
        "out/state.json",
    ]
    # No run-id segment anywhere: a reader needs no knowledge of run ids.
    assert all(RUN_ID not in p["dest"] for p in promoted)


def test_company_outputs_promote_without_a_project(fake_current, tmp_path):
    """Company scope is workspace-wide, so it must promote for a run that
    carries no project -- which every discovery-style run does."""
    (tmp_path / "data").mkdir()
    fake_current.definition_response = definition_response(
        [script_step("s1", "import pathlib; pathlib.Path('out/x.txt').write_text('x')")],
        control={"outputs": ["company"]},
        project=None,
    )

    assert make_runner(fake_current, tmp_path).run() == 0
    assert (tmp_path / "data" / "company" / "x.txt").read_text() == "x"


def test_project_scope_alone_never_writes_the_company_store(fake_current, tmp_path):
    (tmp_path / "data").mkdir()
    fake_current.definition_response = definition_response(
        [script_step("s1", "import pathlib; pathlib.Path('out/x.txt').write_text('x')")],
        control={"outputs": ["project"]},
    )

    assert make_runner(fake_current, tmp_path).run() == 0
    assert not (tmp_path / "data" / "company").exists()


# ---------------------------------------------------------------------------
# (b) Gate exit + resume with approval (approved and rejected branches)
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
    fake_current.definition_response = definition_response(_gated_steps())

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


def _resume_after_gate(fake_current, approval_state: str | None) -> None:
    """Mutate the fake's journal to look like a resume after the gate."""
    fake_current.definition_response["steps_state"] = [
        {"step_id": "prep", "status": "succeeded", "attempts": 1},
        {"step_id": "approve", "status": "waiting_on_gate", "attempts": 1},
        {"step_id": "on-approved", "status": "pending", "attempts": 0},
        {"step_id": "on-rejected", "status": "pending", "attempts": 0},
    ]
    if approval_state is not None:
        fake_current.definition_response["approvals"] = [
            {
                "gate_step_id": "approve",
                "proposal_record_id": "33333333-3333-3333-3333-333333333333",
                "state": approval_state,
            }
        ]
    fake_current.reset_journal()


def test_gate_resume_runs_approved_branch_and_skips_rejected(fake_current, tmp_path):
    fake_current.definition_response = definition_response(_gated_steps())
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
        {"id": "on-rejected", "status": "skipped"},
    ]


def test_gate_resume_runs_rejected_branch_and_skips_approved(fake_current, tmp_path):
    fake_current.definition_response = definition_response(_gated_steps())
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


def test_gate_on_missing_proposal_skip_lets_run_complete(fake_current, tmp_path):
    """A gate with `on_missing_proposal: skip` and no staged out/proposal.json
    has nothing to review: it records `skipped`, emits no agent_proposal, and
    the run completes without parking. Downstream `when: <gate>.approved`
    steps skip too (no approval ever lands)."""
    fake_current.definition_response = definition_response(
        [
            script_step("prep", "print('nothing actionable; no proposal staged')"),
            gate_step("approve", "ship_report", on_missing_proposal="skip"),
            script_step("on-approved", "print('approved path')", when="approve.approved"),
        ]
    )

    exit_code = make_runner(fake_current, tmp_path).run()

    assert exit_code == 0
    assert fake_current.transition_tuples() == [
        ("prep", "running", 1),
        ("prep", "succeeded", 1),
        ("approve", "skipped", 1),
        ("on-approved", "skipped", 1),
    ]
    assert fake_current.records_of("agent_proposal") == []
    data = run_report(fake_current)["data"]
    assert data["status"] == "succeeded"
    assert data["steps"] == [
        {"id": "prep", "status": "succeeded", "attempts": 1},
        {"id": "approve", "status": "skipped"},
        {"id": "on-approved", "status": "skipped"},
    ]


def test_gate_on_missing_proposal_skip_still_parks_when_proposal_staged(
    fake_current, tmp_path
):
    """`on_missing_proposal: skip` only fires when nothing was staged: with an
    out/proposal.json present the gate parks exactly like a default gate."""
    fake_current.definition_response = definition_response(
        [
            script_step(
                "prep",
                "import json, pathlib\n"
                "pathlib.Path('out').mkdir(exist_ok=True)\n"
                "pathlib.Path('out/proposal.json').write_text(json.dumps("
                "{'summary': 'Ship the Q3 report.'}))\n",
            ),
            gate_step("approve", "ship_report", on_missing_proposal="skip"),
        ]
    )

    exit_code = make_runner(fake_current, tmp_path).run()

    assert exit_code == 0
    assert fake_current.transition_tuples() == [
        ("prep", "running", 1),
        ("prep", "succeeded", 1),
        ("approve", "waiting_on_gate", 1),
    ]
    [proposal] = fake_current.records_of("agent_proposal")
    assert proposal["data"]["summary"] == "Ship the Q3 report."


def test_gate_resume_without_approval_exits_again(fake_current, tmp_path):
    """Re-dispatched before anyone decided: re-post waiting_on_gate
    (idempotent) and get out without re-running anything or re-posting the
    proposal record."""
    fake_current.definition_response = definition_response(_gated_steps())
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
    fake_current.definition_response = definition_response(
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
    fake_current.definition_response = definition_response(
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
        {"id": "never", "status": "pending"},
    ]
    assert data["promoted_outputs"] == []
    # run_report is the last event; the failed transition preceded it.
    assert fake_current.events[-1][0] == "record"
    assert fake_current.events[-1][1]["record_type"] == "run_report"


# ---------------------------------------------------------------------------
# (d) Resume skips already-finished steps
# ---------------------------------------------------------------------------


def test_resume_skips_already_succeeded_steps(fake_current, tmp_path):
    fake_current.definition_response = definition_response(
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

    fake_current.definition_response = definition_response([agent_step("think")])
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

    fake_current.definition_response = definition_response(
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
    fake_current.definition_response = definition_response([gate_step("g1", "bad_proposal")])
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


# ---------------------------------------------------------------------------
# (e) Staged-record auto-commit: out/records/<type>.json + emits[].auto_commit
# ---------------------------------------------------------------------------

PO_FACT_EMIT = {
    "record_type": "po_fact",
    "schema": {"type": "object"},
    "auto_commit": True,
}


def _stage_records_script(step_id: str, filename: str, payload: object) -> dict:
    """A script step that stages ``out/records/<filename>`` with the given
    JSON-serializable payload (a dict or a list of dicts)."""
    code = (
        "import json, pathlib\n"
        "pathlib.Path('out/records').mkdir(parents=True, exist_ok=True)\n"
        f"pathlib.Path('out/records/{filename}').write_text(json.dumps({payload!r}))\n"
    )
    return script_step(step_id, code)


def _record_event_types(fake: FakeCurrent) -> list[str]:
    return [body["record_type"] for kind, body in fake.events if kind == "record"]


def test_staged_record_auto_commit_posts_proposal_then_records(fake_current, tmp_path):
    """A definition declaring `po_fact` with `auto_commit: true` gets its
    staged out/records/po_fact.json committed at run end: one auto-approved
    agent_proposal covering every entry, then each record -- no gate, no
    human decision, and nothing hits the wire until the run has already
    succeeded."""
    entries = [
        {
            "scope": "vendor",
            "vendor_name": "Loop Global, Inc.",
            "kind": "vendor_address",
            "text": "Loop Global HQ: 1700 E Walnut Ave, Fl 6, El Segundo, CA 90245",
            "source": "Slack, @mohit, 2026-07-22",
        }
    ]
    fake_current.definition_response = definition_response(
        [_stage_records_script("teach", "po_fact.json", entries)],
        emits=[PO_FACT_EMIT],
    )

    exit_code = make_runner(fake_current, tmp_path).run()

    assert exit_code == 0
    [proposal] = fake_current.records_of("agent_proposal")
    assert proposal["data"]["proposal"] == "auto:po_fact"
    assert proposal["data"]["auto_approved"] is True
    assert proposal["data"]["idempotency_key"] == f"wf-{RUN_ID}-records-po_fact"
    assert proposal["data"]["payload"] == {"record_type": "po_fact", "entries": entries}

    [record] = fake_current.records_of("po_fact")
    assert record["project"] == PROJECT
    assert record["data"] == entries[0]
    assert "supersedes" not in record  # nothing to supersede in this entry

    # Proposal, then the record, then (later) the terminal run_report --
    # never a record with no matching proposal ahead of it.
    assert _record_event_types(fake_current) == ["agent_proposal", "po_fact", "run_report"]


def test_staged_record_without_auto_commit_is_left_alone(fake_current, tmp_path):
    """emits[] without `auto_commit: true` (the default, e.g.
    purchase_order) is never auto-committed by the runner -- a script step
    owns posting that type itself, and the runner must not double-post
    from the same out/records/ staging convention."""
    fake_current.definition_response = definition_response(
        [_stage_records_script("draft", "purchase_order.json", {"status": "draft"})],
        emits=[{"record_type": "purchase_order", "schema": {"type": "object"}}],
    )

    exit_code = make_runner(fake_current, tmp_path).run()

    assert exit_code == 0
    assert fake_current.records_of("purchase_order") == []
    assert fake_current.records_of("agent_proposal") == []


def test_staged_record_undeclared_type_is_left_alone(fake_current, tmp_path):
    """A staged file whose stem isn't any declared emits[] record_type is
    ignored -- the harness never invents commits for a type the
    definition never declared at all."""
    fake_current.definition_response = definition_response(
        [_stage_records_script("teach", "mystery_fact.json", {"x": 1})],
        emits=[PO_FACT_EMIT],
    )

    exit_code = make_runner(fake_current, tmp_path).run()

    assert exit_code == 0
    assert fake_current.records_of("mystery_fact") == []
    assert fake_current.records_of("agent_proposal") == []


def test_staged_record_schema_rejection_is_skipped_not_fatal(fake_current, tmp_path):
    """A 422 from the records endpoint (a schema-invalid staged entry) is
    logged and skipped, not fatal -- this commit is best-effort at the very
    end of a run that has already succeeded."""
    fake_current.definition_response = definition_response(
        [_stage_records_script("teach", "po_fact.json", [{"bad": "shape"}])],
        emits=[PO_FACT_EMIT],
    )
    fake_current.record_failures["po_fact"] = (422, {"detail": "does not match schema"})

    exit_code = make_runner(fake_current, tmp_path).run()

    assert exit_code == 0
    [proposal] = fake_current.records_of("agent_proposal")
    assert proposal["data"]["proposal"] == "auto:po_fact"
    assert fake_current.records_of("po_fact") == []  # rejected, not retried
    assert run_report(fake_current)["data"]["status"] == "succeeded"


def test_staged_record_proposal_rejection_skips_the_whole_type(fake_current, tmp_path):
    """When even the auto-approved proposal itself is rejected, no record of
    that type is posted either -- a record with no matching proposal ahead
    of it would break the "every state change has a proposal" audit
    guarantee."""
    fake_current.definition_response = definition_response(
        [_stage_records_script("teach", "po_fact.json", [{"scope": "global"}])],
        emits=[PO_FACT_EMIT],
    )
    fake_current.record_failures["agent_proposal"] = (422, {"detail": "bad payload"})

    exit_code = make_runner(fake_current, tmp_path).run()

    assert exit_code == 0
    assert fake_current.records_of("agent_proposal") == []
    assert fake_current.records_of("po_fact") == []
    assert run_report(fake_current)["data"]["status"] == "succeeded"


def test_staged_record_supersedes_and_project_id_sidecars(fake_current, tmp_path):
    """`supersedes` and `project_id` are reserved sidecar keys on a staged
    entry: popped off before the entry becomes the record's `data`, and
    routed to the record POST's own `supersedes` / `project` fields."""
    other_project = "44444444-4444-4444-4444-444444444444"
    entries = [
        {
            "scope": "project",
            "kind": "tax_rate",
            "text": "Confirmed 6% for this site",
            "source": "Teams, @mohit, 2026-07-22",
            "project_id": other_project,
            "supersedes": "55555555-5555-5555-5555-555555555555",
        }
    ]
    fake_current.definition_response = definition_response(
        [_stage_records_script("teach", "po_fact.json", entries)],
        emits=[PO_FACT_EMIT],
    )

    exit_code = make_runner(fake_current, tmp_path).run()

    assert exit_code == 0
    [record] = fake_current.records_of("po_fact")
    assert record["project"] == other_project
    assert record["supersedes"] == "55555555-5555-5555-5555-555555555555"
    assert record["data"] == {
        "scope": "project",
        "kind": "tax_rate",
        "text": "Confirmed 6% for this site",
        "source": "Teams, @mohit, 2026-07-22",
    }
    [proposal] = fake_current.records_of("agent_proposal")
    proposal_entry = proposal["data"]["payload"]["entries"][0]
    assert "supersedes" not in proposal_entry
    assert "project_id" not in proposal_entry


def test_staged_record_single_object_not_wrapped_in_list_is_accepted(fake_current, tmp_path):
    """A staged file may be one bare record object, not just a list of
    them -- the common case of teaching exactly one fact this run."""
    fake_current.definition_response = definition_response(
        [
            _stage_records_script(
                "teach",
                "po_fact.json",
                {"scope": "global", "kind": "other", "text": "hi", "source": "test"},
            )
        ],
        emits=[PO_FACT_EMIT],
    )

    exit_code = make_runner(fake_current, tmp_path).run()

    assert exit_code == 0
    [record] = fake_current.records_of("po_fact")
    assert record["data"]["text"] == "hi"


# ---------------------------------------------------------------------------
# (f) Terminal-sequence resilience: promotion failure / runner crash
# ---------------------------------------------------------------------------

_PO_FACT_ENTRIES = [{"scope": "global", "kind": "other", "text": "hi", "source": "test"}]


def _promoting_definition() -> dict:
    """One step that stages both an auto-commit record and a plain output
    file, with company promotion declared -- the shape whose terminal
    sequence (commit records, promote out/, post run_report) these tests
    stress."""
    return definition_response(
        [
            script_step(
                "produce",
                "import json, pathlib\n"
                "pathlib.Path('out/records').mkdir(parents=True, exist_ok=True)\n"
                "pathlib.Path('out/records/po_fact.json').write_text(json.dumps("
                f"{_PO_FACT_ENTRIES!r}))\n"
                "pathlib.Path('out/big.txt').write_text('archive payload')\n",
            )
        ],
        emits=[PO_FACT_EMIT],
        control={"outputs": ["company"]},
    )


def test_promotion_failure_still_commits_records_and_posts_run_report(
    fake_current, tmp_path, monkeypatch
):
    """Promotion is the riskiest I/O of the run (a bulk copy onto the
    network-backed workspace volume). When it blows up, the staged records
    must already be committed and the run_report must still post, carrying
    the promotion error in `summary` -- otherwise the run lingers `running`
    until the stale sweep and the ledger silently loses the run's records."""
    (tmp_path / "data").mkdir()
    fake_current.definition_response = _promoting_definition()

    def exploding_copy2(src, dst, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("harness.workflow.runner.shutil.copy2", exploding_copy2)

    exit_code = make_runner(fake_current, tmp_path).run()

    assert exit_code == 0  # every step succeeded; only promotion failed
    [record] = fake_current.records_of("po_fact")
    assert record["data"]["text"] == "hi"
    data = run_report(fake_current)["data"]
    assert data["status"] == "succeeded"
    assert data["promoted_outputs"] == []
    assert data["summary"].startswith("Output promotion failed: OSError")


def test_staged_records_commit_before_outputs_promote(fake_current, tmp_path, monkeypatch):
    """Wire order: the durable ledger (staged records) commits before the
    first byte of promotion I/O, so a promotion hang or crash can never
    cost the records."""
    import shutil as shutil_mod

    (tmp_path / "data").mkdir()
    fake_current.definition_response = _promoting_definition()

    real_copy2 = shutil_mod.copy2
    wire_at_first_copy: list[list[str]] = []

    def spying_copy2(src, dst, **kwargs):
        if not wire_at_first_copy:
            wire_at_first_copy.append(_record_event_types(fake_current))
        return real_copy2(src, dst, **kwargs)

    monkeypatch.setattr("harness.workflow.runner.shutil.copy2", spying_copy2)

    assert make_runner(fake_current, tmp_path).run() == 0

    assert wire_at_first_copy == [["agent_proposal", "po_fact"]]
    data = run_report(fake_current)["data"]
    assert data["status"] == "succeeded"
    assert sorted(p["path"] for p in data["promoted_outputs"]) == [
        "out/big.txt",
        "out/records/po_fact.json",
    ]


def test_runner_crash_posts_failed_run_report_and_reraises(fake_current, tmp_path, monkeypatch):
    """An uncaught exception anywhere past the definition fetch must still
    finalize the run: a best-effort failed run_report posts (with the steps
    journaled so far and the crash in `summary`), then the exception
    propagates so the process exits nonzero."""
    from harness.workflow import WorkflowRunner

    fake_current.definition_response = definition_response([script_step("s1", "print('ok')")])

    def boom(self, definition):
        raise RuntimeError("terminal bookkeeping blew up")

    monkeypatch.setattr(WorkflowRunner, "_commit_staged_records", boom)

    with pytest.raises(RuntimeError, match="terminal bookkeeping blew up"):
        make_runner(fake_current, tmp_path).run()

    data = run_report(fake_current)["data"]
    assert data["status"] == "failed"
    assert data["summary"] == "Runner crashed: RuntimeError: terminal bookkeeping blew up"
    assert data["steps"] == [{"id": "s1", "status": "succeeded", "attempts": 1}]
    assert data["promoted_outputs"] == []

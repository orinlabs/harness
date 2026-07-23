"""Workflow runner: interpret a `current` definition inside a disposable sandbox.

``harness workflow`` is the third harness mode (after ``agent`` and
``boot``): current dispatches a sandbox with ``CURRENT_URL`` /
``CURRENT_RUN_TOKEN`` / ``HARNESS_RUN_ID`` set, the runner fetches the
run's definition (an ordered list of script / agent / gate steps plus the
journal so far), executes it, and reports every state change back over
HTTP. Postgres on the current side is the source of truth.

The four invariants this module exists to uphold:

1. **Journal first.** Every transition/record is POSTed (and confirmed)
   *before* the runner advances to its next action. Runner memory is
   disposable; the journal is not.
2. **Exit at gates.** A gate step posts an ``agent_proposal`` record, posts
   ``waiting_on_gate``, and exits 0. A process is never parked waiting for
   a human — current re-dispatches a fresh sandbox once an approval lands.
   Exception: a gate with ``on_missing_proposal: skip`` and no staged
   ``out/proposal.json`` has nothing to review; it records ``skipped`` and
   the run continues (downstream ``when:`` guards on it stay unmet).
3. **Resume from the journal.** On start we replay ``steps_state``: steps
   already ``succeeded``/``skipped`` are not re-run; a gate sitting in
   ``waiting_on_gate`` with an approval present resolves and downstream
   ``when:`` guards are evaluated. Re-posting an already-recorded
   transition is safe (the endpoint is idempotent).
4. **Terminal report.** Before a terminal exit (run succeeded or failed —
   not a gate exit) the runner posts a ``run_report`` record summarizing
   step outcomes, records emitted, and promoted outputs.

Step working directory layout (``~/workflow/{run_id}/`` by default)::

    inputs/   hydrated from the workspace volume (data_root) when present
    work/     scratch
    out/      step outputs; promoted to data_root on run success when the
              definition declares "project" in control.outputs

``out/records/<record_type>.json`` is a second, narrower convention: a
staged record (or list of them) for a type the definition declares in
``emits[]`` with ``auto_commit: true`` — e.g. a model-learned fact like
``po_fact``. At run end (alongside promotion) the runner commits every
such file itself: an auto-approved ``agent_proposal`` first (so "which
runs changed workspace state" stays a proposal query away, even though no
human decided anything), then the record(s). Types without
``auto_commit: true`` are untouched here — e.g. ``purchase_order`` is
posted explicitly by its own script step, which reads (but never
deletes) the same staging file.

``data_root`` defaults to ``/data`` but is sourced from the
``CURRENT_DATA_ROOT`` env var when current sets it (single-sourced from
current's own volume-mount-path constant, so the two sides can't
silently drift) — see ``--data-root`` in ``cli.py``.

Failure semantics: ``retry: {attempts: N}`` is a *total* attempt budget per
step; ``on_failure: fail`` (the default) aborts the run as terminal
failed, ``on_failure: continue`` records the failure and proceeds.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from harness.cloud.current.client import CurrentAPIError, CurrentClient
from harness.config_loader import build_agent_config
from harness.core import clock

logger = logging.getLogger(__name__)

# Cap for agent steps that don't declare `max_turns`. Deliberately lower
# than the interactive loop's MAX_TURNS: a workflow step is a bounded task,
# and max-turns exhaustion is a *step failure* here, not a soft landing.
DEFAULT_AGENT_MAX_TURNS = 30

_STDERR_TAIL_CHARS = 2000
_STAGING_DIRS = ("inputs", "work", "out")


def _now_iso() -> str:
    return clock.now_iso()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _tail(text: str, limit: int = _STDERR_TAIL_CHARS) -> str:
    text = text or ""
    return text if len(text) <= limit else text[-limit:]


class WorkflowRunner:
    """Interpret one workflow run. ``run()`` returns the process exit code."""

    def __init__(
        self,
        client: CurrentClient,
        *,
        working_dir: Path | None = None,
        data_root: Path = Path("/data"),
    ):
        self.client = client
        self.run_id = client.run_id
        self.working_dir = (
            Path(working_dir) if working_dir else Path.home() / "workflow" / self.run_id
        )
        self.data_root = Path(data_root)
        self.project: str | None = None
        # record_type -> count of records this process POSTed (for run_report).
        self._records_emitted: Counter[str] = Counter()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> int:
        resp = self.client.get_definition()
        definition: dict[str, Any] = resp["definition"]
        self.project = resp.get("project")
        steps_state = {s["step_id"]: dict(s) for s in resp.get("steps_state") or []}
        approvals = {a["gate_step_id"]: dict(a) for a in resp.get("approvals") or []}

        logger.info(
            "workflow run start: run=%s workflow=%r kind=%s steps=%d workdir=%s",
            self.run_id,
            definition.get("name"),
            definition.get("kind"),
            len(definition.get("steps") or []),
            self.working_dir,
        )
        self._prepare_working_dir(definition)

        started_at = _now_iso()
        report_steps: list[dict[str, Any]] = []
        run_failed = False

        steps: list[dict[str, Any]] = definition.get("steps") or []
        for index, step in enumerate(steps):
            step_id = step["id"]
            state = steps_state.get(step_id) or {"status": "pending", "attempts": 0}
            prior_attempts = int(state.get("attempts") or 0)

            # Resume: never re-run finished work.
            if state["status"] in ("succeeded", "skipped"):
                logger.info("step %s already %s; skipping", step_id, state["status"])
                report_steps.append(
                    {"id": step_id, "status": state["status"], "attempts": prior_attempts}
                )
                continue

            # A gate we previously parked on: resolved iff an approval landed.
            if step.get("kind") == "gate" and state["status"] == "waiting_on_gate":
                if step_id not in approvals:
                    logger.info("gate %s still awaiting approval; exiting", step_id)
                    self.client.post_transition(
                        step_id=step_id,
                        status="waiting_on_gate",
                        attempt=max(prior_attempts, 1),
                    )
                    return 0
                logger.info(
                    "gate %s resolved: %s", step_id, approvals[step_id].get("state")
                )
                self.client.post_transition(
                    step_id=step_id, status="succeeded", attempt=max(prior_attempts, 1)
                )
                report_steps.append(
                    {"id": step_id, "status": "succeeded", "attempts": max(prior_attempts, 1)}
                )
                continue

            # `when:` guard — "<gate-step-id>.approved|rejected".
            if not self._when_satisfied(step, approvals):
                logger.info("step %s skipped (when=%r unmet)", step_id, step.get("when"))
                self.client.post_transition(
                    step_id=step_id, status="skipped", attempt=max(prior_attempts, 1)
                )
                # No `attempts`: the run_report schema requires >= 1 when
                # present, and a skipped step was never attempted.
                report_steps.append({"id": step_id, "status": "skipped"})
                continue

            if step.get("kind") == "gate":
                # `on_missing_proposal: skip`: no prior step staged
                # out/proposal.json, so there is nothing to review — record
                # the gate as skipped and keep going. Downstream
                # `when: <gate>.approved` steps then skip too (no approval
                # ever lands), so the run completes without parking.
                if (
                    step.get("on_missing_proposal") == "skip"
                    and not (self.working_dir / "out" / "proposal.json").is_file()
                ):
                    logger.info(
                        "gate %s: no out/proposal.json and on_missing_proposal=skip; skipping",
                        step_id,
                    )
                    self.client.post_transition(
                        step_id=step_id, status="skipped", attempt=max(prior_attempts, 1)
                    )
                    report_steps.append({"id": step_id, "status": "skipped"})
                    continue
                outcome = self._run_gate_step(step)
                if outcome is None:
                    # Proposal + waiting_on_gate journaled; leave the sandbox.
                    return 0
                # Posting the proposal failed schema validation (422) —
                # that's a step failure and gates have no retry: terminal.
                self.client.post_transition(
                    step_id=step_id, status="failed", attempt=1, error=outcome
                )
                report_steps.append(
                    {"id": step_id, "status": "failed", "attempts": 1, "error": outcome}
                )
                run_failed = True
                report_steps.extend(self._pending_report(steps[index + 1 :], steps_state))
                break

            # script / agent step with a total-attempts retry budget.
            total_attempts = max(int((step.get("retry") or {}).get("attempts") or 1), 1)
            attempts_done = prior_attempts
            error_text: str | None = None
            succeeded = False
            while attempts_done < total_attempts:
                attempt = attempts_done + 1
                self.client.post_transition(step_id=step_id, status="running", attempt=attempt)
                logger.info(
                    "step %s (%s) attempt %d/%d",
                    step_id,
                    step.get("kind"),
                    attempt,
                    total_attempts,
                )
                ok, error_text = self._execute_step(step)
                attempts_done = attempt
                if ok:
                    self.client.post_transition(
                        step_id=step_id, status="succeeded", attempt=attempt
                    )
                    succeeded = True
                    break
                logger.warning(
                    "step %s attempt %d/%d failed: %s",
                    step_id,
                    attempt,
                    total_attempts,
                    error_text,
                )

            if succeeded:
                report_steps.append(
                    {"id": step_id, "status": "succeeded", "attempts": attempts_done}
                )
                continue

            if error_text is None:
                # Resume found the attempt budget already spent (a previous
                # sandbox died between the last attempt and its transition).
                error_text = f"attempt budget exhausted ({attempts_done}/{total_attempts})"
            self.client.post_transition(
                step_id=step_id,
                status="failed",
                attempt=max(attempts_done, 1),
                error=error_text,
            )
            report_steps.append(
                {
                    "id": step_id,
                    "status": "failed",
                    "attempts": attempts_done,
                    "error": error_text,
                }
            )
            if (step.get("on_failure") or "fail") == "continue":
                logger.info("step %s failed but on_failure=continue; proceeding", step_id)
                continue
            run_failed = True
            report_steps.extend(self._pending_report(steps[index + 1 :], steps_state))
            break

        # Terminal: promote outputs and commit staged records (success
        # only), then journal the report.
        status = "failed" if run_failed else "succeeded"
        promoted = [] if run_failed else self._promote_outputs(definition)
        if not run_failed:
            self._commit_staged_records(definition)
        self._post_run_report(
            definition,
            status=status,
            started_at=started_at,
            finished_at=_now_iso(),
            report_steps=report_steps,
            promoted=promoted,
        )
        logger.info("workflow run %s: %s", self.run_id, status)
        return 1 if run_failed else 0

    # ------------------------------------------------------------------
    # Working directory: staging, control files, hydration, promotion
    # ------------------------------------------------------------------

    def _prepare_working_dir(self, definition: dict[str, Any]) -> None:
        for name in _STAGING_DIRS:
            (self.working_dir / name).mkdir(parents=True, exist_ok=True)

        control = definition.get("control") or {}
        for entry in control.get("files") or []:
            dest = self.working_dir / entry["dest"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(base64.b64decode(entry["content_b64"]))
            logger.info("staged control file %s", entry["dest"])

        # v0 hydration: best-effort copies from the workspace volume.
        # Missing volume or missing sources are skipped silently by design —
        # a workflow must be runnable before any data exists.
        inputs = control.get("inputs") or []
        if "project" in inputs and self.project:
            self._hydrate(
                self.data_root / "projects" / str(self.project),
                self.working_dir / "inputs" / "project",
            )
        if "company" in inputs:
            self._hydrate(self.data_root / "company", self.working_dir / "inputs" / "company")

    @staticmethod
    def _hydrate(src: Path, dest: Path) -> None:
        if not src.is_dir():
            return
        shutil.copytree(src, dest, dirs_exist_ok=True)
        logger.info("hydrated %s -> %s", src, dest)

    def _promote_outputs(self, definition: dict[str, Any]) -> list[dict[str, str]]:
        """Copy ``out/`` to the project's workflow-outputs area on the volume.

        Only when the definition declares ``"project"`` in ``control.outputs``
        and both a project and the ``/data`` volume exist. Returns the
        ``promoted_outputs`` entries for the run_report.
        """
        control = definition.get("control") or {}
        if "project" not in (control.get("outputs") or []) or not self.project:
            return []
        if not self.data_root.is_dir():
            logger.info("no %s volume; skipping output promotion", self.data_root)
            return []

        out_dir = self.working_dir / "out"
        dest_root = (
            self.data_root / "projects" / str(self.project) / "workflow-outputs" / self.run_id
        )
        promoted: list[dict[str, str]] = []
        for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
            rel = path.relative_to(out_dir)
            dest = dest_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            promoted.append(
                {
                    "path": str(Path("out") / rel),
                    "dest": str(dest),
                    "content_hash": _sha256_file(dest),
                }
            )
        logger.info("promoted %d output file(s) to %s", len(promoted), dest_root)
        return promoted

    def _commit_staged_records(self, definition: dict[str, Any]) -> None:
        """Commit ``out/records/<type>.json`` for every ``emits[]`` type this
        definition marks ``auto_commit: true`` — the filesystem write path
        for facts a definition just appends/supersedes (e.g. ``po_fact``),
        with no human approval needed. Every commit still posts an
        auto-approved ``agent_proposal`` first, so "which runs changed
        workspace state" stays one proposal query away even though nothing
        was decided by a person — see the module docstring.

        Types without ``auto_commit: true`` are left completely alone: a
        script step may stage (and consume) the same directory convention
        itself, e.g. ``purchase_order`` (posted explicitly by
        ``po-generation``'s ``save`` step, which reads but never deletes
        ``out/records/purchase_order.json``) — committing it here too would
        double-post it.
        """
        auto_commit_types = {
            e["record_type"]
            for e in definition.get("emits") or []
            if isinstance(e, dict) and e.get("auto_commit") and e.get("record_type")
        }
        if not auto_commit_types:
            return
        records_dir = self.working_dir / "out" / "records"
        if not records_dir.is_dir():
            return
        for path in sorted(p for p in records_dir.glob("*.json") if p.is_file()):
            record_type = path.stem
            if record_type not in auto_commit_types:
                continue
            try:
                loaded = json.loads(path.read_text())
            except (ValueError, OSError) as e:
                logger.warning("out/records/%s is unreadable; skipping: %s", path.name, e)
                continue
            entries = loaded if isinstance(loaded, list) else [loaded]
            entries = [e for e in entries if isinstance(e, dict)]
            if not entries:
                continue
            self._commit_record_entries(record_type, entries)

    def _commit_record_entries(self, record_type: str, entries: list[dict[str, Any]]) -> None:
        """Post one auto-approved ``agent_proposal`` covering every staged
        entry of ``record_type``, then the record(s) themselves.

        Each entry is the record's ``data`` block, with two reserved
        sidecar keys popped off before validation: ``supersedes`` (id of
        the record this corrects) and ``project_id`` (defaults to the
        run's own project). A proposal-post rejection (e.g. schema-invalid
        payload text) skips the whole file loudly; an individual record
        rejection skips just that entry — either way nothing crashes the
        run, since these are best-effort commits at the very end of it.
        """
        clean_entries = []
        for entry in entries:
            entry = dict(entry)
            supersedes = entry.pop("supersedes", None)
            project_id = entry.pop("project_id", None)
            clean_entries.append(
                {"data": entry, "supersedes": supersedes, "project_id": project_id}
            )

        try:
            self.client.post_record(
                record_type="agent_proposal",
                step_id=None,
                project=self.project,
                produced_at=_now_iso(),
                data={
                    "proposal": f"auto:{record_type}",
                    "summary": (
                        f"Auto-approved: this run staged {len(clean_entries)} "
                        f"{record_type} record(s) — an existing-type append/"
                        "correction, not a new schema, so no human approval "
                        "was needed."
                    ),
                    "payload": {
                        "record_type": record_type,
                        "entries": [c["data"] for c in clean_entries],
                    },
                    "idempotency_key": f"wf-{self.run_id}-records-{record_type}",
                    "auto_approved": True,
                },
            )
        except CurrentAPIError as e:
            logger.warning(
                "auto-approved proposal for staged %s records rejected; not committing "
                "any of them: %s",
                record_type,
                e,
            )
            return
        self._records_emitted["agent_proposal"] += 1

        for clean in clean_entries:
            try:
                self.client.post_record(
                    record_type=record_type,
                    step_id=None,
                    project=clean["project_id"] or self.project,
                    produced_at=_now_iso(),
                    data=clean["data"],
                    supersedes=clean["supersedes"],
                )
            except CurrentAPIError as e:
                logger.warning("staged %s record rejected; skipping it: %s", record_type, e)
                continue
            self._records_emitted[record_type] += 1

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def _execute_step(self, step: dict[str, Any]) -> tuple[bool, str | None]:
        kind = step.get("kind")
        if kind == "script":
            return self._run_script_step(step)
        if kind == "agent":
            return self._run_agent_step(step)
        return False, f"unknown step kind: {kind!r}"

    def _run_script_step(self, step: dict[str, Any]) -> tuple[bool, str | None]:
        """Write the script body to a temp file and run ``python3 <file>``
        with cwd = the run working directory and full env passthrough."""
        content = base64.b64decode(step["content_b64"])
        with tempfile.NamedTemporaryFile(
            "wb", prefix=f"wf-step-{step['id']}-", suffix=".py", delete=False
        ) as f:
            f.write(content)
            script_path = f.name
        try:
            proc = subprocess.run(
                ["python3", script_path],
                cwd=str(self.working_dir),
                capture_output=True,
                text=True,
            )
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass
        if proc.returncode == 0:
            return True, None
        return False, (
            f"script exited {proc.returncode}; stderr tail:\n{_tail(proc.stderr)}"
        )

    def _run_agent_step(self, step: dict[str, Any]) -> tuple[bool, str | None]:
        """Run the in-process agent loop for one step.

        The step ``config`` is an AgentConfig subset (no ``id`` — we
        synthesize ``wf-{run_id}-{step_id}``); ``tools`` inside it are
        platform-injected ExternalToolSpec dicts. On top of those we
        register the workflow-only local built-ins (bash / read_file /
        write_file), path-restricted to the working dir. The sleep tool is
        NOT offered: the step ends when the model stops calling tools, and
        hitting ``max_turns`` first is a step failure.
        """
        # Deferred imports: keep `harness workflow` startup (and definition
        # fetch failures) from paying the agent-loop import cost.
        from harness.core.runtime import LocalAgentRuntime
        from harness.core.tracing import StdoutTraceSink
        from harness.harness import Harness
        from harness.tools.local import workflow_local_tools

        config_data = dict(step.get("config") or {})
        config_data.setdefault("id", f"wf-{self.run_id}-{step['id']}")
        try:
            config = build_agent_config(config_data)
        except (ValueError, KeyError) as e:
            return False, f"invalid agent step config: {e}"
        config = dataclasses.replace(
            config, tools=[*config.tools, *workflow_local_tools(self.working_dir)]
        )
        max_turns = int(step.get("max_turns") or DEFAULT_AGENT_MAX_TURNS)

        prev_cwd = os.getcwd()
        os.chdir(self.working_dir)
        try:
            harness = Harness(
                config,
                run_id=config.id,
                trace_sink=StdoutTraceSink(),
                runtime=LocalAgentRuntime(),
                max_turns=max_turns,
                stop_when_idle=True,
                include_builtin_tools=False,
            )
            harness.run()
        except Exception as e:  # noqa: BLE001 — attempt failure, retry may fix it
            logger.exception("agent step %s raised", step["id"])
            return False, f"agent step raised {type(e).__name__}: {_tail(str(e))}"
        finally:
            os.chdir(prev_cwd)

        if harness.exhausted_max_turns:
            return False, f"agent step hit max_turns={max_turns} without finishing"
        return True, None

    def _run_gate_step(self, step: dict[str, Any]) -> str | None:
        """Post the ``agent_proposal`` record then ``waiting_on_gate``.

        Returns ``None`` on success (caller exits 0) or an error string when
        the platform rejected the proposal record (422 schema failure).
        """
        step_id = step["id"]
        payload: dict[str, Any] = {}
        proposal_file = self.working_dir / "out" / "proposal.json"
        if proposal_file.is_file():
            try:
                loaded = json.loads(proposal_file.read_text())
                if isinstance(loaded, dict):
                    payload = loaded
                else:
                    logger.warning("out/proposal.json is not a JSON object; using {}")
            except (ValueError, OSError):
                logger.warning("out/proposal.json is unreadable; using {}", exc_info=True)

        summary = payload.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = (
                f"Workflow step {step_id!r} proposes {step.get('proposal')!r} "
                f"for run {self.run_id}. Review the attached payload and "
                "approve or reject to continue the run."
            )

        try:
            self.client.post_record(
                record_type="agent_proposal",
                step_id=step_id,
                project=self.project,
                produced_at=_now_iso(),
                data={
                    "proposal": step.get("proposal"),
                    "summary": summary,
                    "payload": payload,
                    "idempotency_key": f"wf-{self.run_id}-{step_id}",
                },
            )
        except CurrentAPIError as e:
            if e.status_code == 422:
                return f"agent_proposal record rejected (422): {_tail(e.body)}"
            raise
        self._records_emitted["agent_proposal"] += 1
        self.client.post_transition(step_id=step_id, status="waiting_on_gate", attempt=1)
        logger.info("gate %s parked (waiting_on_gate); exiting 0", step_id)
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _when_satisfied(step: dict[str, Any], approvals: dict[str, dict]) -> bool:
        when = step.get("when")
        if not when:
            return True
        gate_id, _, expected = str(when).partition(".")
        approval = approvals.get(gate_id)
        return approval is not None and approval.get("state") == expected

    @staticmethod
    def _pending_report(
        remaining: list[dict[str, Any]], steps_state: dict[str, dict]
    ) -> list[dict[str, Any]]:
        """Report entries for steps we never reached because the run aborted."""
        entries = []
        for step in remaining:
            state = steps_state.get(step["id"]) or {}
            entry: dict[str, Any] = {
                "id": step["id"],
                "status": state.get("status") or "pending",
            }
            # No `attempts` key for never-attempted steps: the run_report
            # schema requires >= 1 when present.
            attempts = int(state.get("attempts") or 0)
            if attempts > 0:
                entry["attempts"] = attempts
            entries.append(entry)
        return entries

    def _post_run_report(
        self,
        definition: dict[str, Any],
        *,
        status: str,
        started_at: str,
        finished_at: str,
        report_steps: list[dict[str, Any]],
        promoted: list[dict[str, str]],
    ) -> None:
        self.client.post_record(
            record_type="run_report",
            step_id=None,
            project=self.project,
            produced_at=finished_at,
            data={
                "workflow": definition.get("name"),
                "definition_hash": definition.get("rendered_hash"),
                "status": status,
                "started_at": started_at,
                "finished_at": finished_at,
                "steps": report_steps,
                "records_emitted": [
                    {"record_type": rt, "count": count}
                    for rt, count in sorted(self._records_emitted.items())
                ],
                "promoted_outputs": promoted,
            },
        )

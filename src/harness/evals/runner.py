"""Unified runner for simulation-based evals (harness edition).

This is the Django-stripped fork of bedrock's
``api/evals/simulation/runner.py``. Where the original wrote to
``EvalRun`` / ``EvalScenario`` / ``EvalEvent`` / ``EvalCheckpoint`` Django
models and installed Celery/Django patches, this version:

* Emits structured stdout/stderr lines in lieu of the DB writes. A
  follow-up task (T7) will POST them to Bedrock's
  ``/api/tracing/spans/`` (``span_type=checkpoint`` and friends).
* Leaves the fake-adapter I/O injection to T6 (``patch_external_clients``
  does not exist in harness).
* Uses ``override_template_id`` to scope eval agents to an
  ``AgentTemplate``; T7 will do the HTTP lookup to resolve it into a
  template snapshot.

The bits that *are* ported:

* Building the merged timeline from ``Simulation.build_timeline``.
* Stepping through it, advancing the simulated clock between events.
* Running user-actor replies that pile up during agent turns.

The bits that are deliberately TODO:

* Fake adapters (T6)
* HTTP posting of traces/checkpoints back to Bedrock (T7)
* Feature flag rewiring (deferred)
* Running the agent loop itself — that needs the harness runtime plus
  the fake adapters, both landed in T6/T7.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time as wall_time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal

from harness.core.tracer import SpanType, emit_completed_span

from .base import ScheduledEvent, ScheduledEventType, Simulation
from .clock import SimulatedClock, simulated_clock_context
from .context import set_simulation
from .types import MemorySeedEntry

logger = logging.getLogger(__name__)


def _simulation_hash(sim: type[Simulation]) -> str:
    """Content hash for reproducibility metadata."""
    data = {
        "name": sim.name,
        "description": sim.description,
        "duration_days": sim.duration_days,
        "eval_mode": sim.eval_mode,
        "agent_overrides": asdict(sim.agent_overrides),
        "feature_flags": sim.feature_flags,
        "users": [asdict(u) for u in sim.users],
    }
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()[:40]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Lightweight stand-ins for the Django models the bedrock runner wrote to.
# Each one will be POSTed to Bedrock in T7; for now the runner just prints
# them so eval runs remain observable on stdout.
# ---------------------------------------------------------------------------


@dataclass
class EvalRunRecord:
    scenario_name: str
    agent_id: str
    eval_mode: str
    status: str
    simulated_start: datetime | None = None
    simulated_end: datetime | None = None
    started_at_wall: datetime | None = None
    completed_at_wall: datetime | None = None
    config_overrides: dict = field(default_factory=dict)
    scenario_hash: str = ""
    git_sha: str = ""
    total_wall_time_seconds: float = 0.0
    total_llm_cost_usd: Decimal = Decimal("0")


def _emit(kind: str, payload: dict) -> None:
    """Structured stdout line for local eval runs.

    The durable artifact for cloud eval runs is the HarnessRun trace stream
    posted from ``harness.core``; these stdout lines exist so a developer
    running the eval locally still gets a readable timeline.
    """
    print(f"[eval-trace] {kind} {json.dumps(payload, default=str)}")


class SimulationRunner:
    """High-level API for running a simulation-based eval (harness edition)."""

    def __init__(
        self,
        simulation_cls: type[Simulation],
        *,
        agent_id: str | None = None,
        agent_config=None,
        override_model: str | None = None,
        # T7 will do the HTTP lookup to resolve this into a template snapshot.
        override_template_id: str | None = None,
        override_feature_flags: dict[str, str] | None = None,
        override_reasoning_effort: str = "",
    ):
        self.simulation_cls = simulation_cls
        self._agent_id = agent_id
        # Pre-built harness.config.AgentConfig. When provided the runner
        # actually wakes the agent between events via `Harness(...).run()`;
        # when None the runner just walks the timeline without driving the
        # agent loop (useful for scenario-shape unit tests).
        self._agent_config = agent_config
        self._override_model = override_model
        self._override_template_id = override_template_id
        self._override_feature_flags = override_feature_flags or {}
        self._override_reasoning_effort = override_reasoning_effort
        self._run: EvalRunRecord | None = None
        self.checkpoint_results: list[dict] = []

    def execute(self) -> EvalRunRecord:
        import uuid

        wall_start = wall_time.monotonic()
        sim_cls = self.simulation_cls

        if self._override_template_id:
            print(f"[sim] Using template override: {self._override_template_id}")

        # In bedrock this was `Agent.objects.create(...)`. In the harness the
        # agent is just an opaque UUID that indexes into the sqlite file
        # opened by `harness.core.storage.load(agent_id)`.
        agent_id = self._agent_id or f"sim-{sim_cls.name}-{uuid.uuid4().hex[:8]}"
        logger.info("Using simulation agent id %s", agent_id)

        from dataclasses import replace

        overrides = replace(sim_cls.agent_overrides)
        if self._override_model:
            overrides.model = self._override_model
        effective_reasoning = self._override_reasoning_effort or overrides.reasoning_effort

        sim_cls.ensure_tools()

        # Merge scenario-declared feature flags with any runtime overrides
        # the caller supplied; runtime wins. The merged dict gets stamped
        # onto ``self._agent_config.feature_flags`` below so ``Harness``
        # picks them up via ``AgentConfig.is_enabled(...)`` exactly as it
        # would on a Bedrock-backed run.
        merged_feature_flags = {**sim_cls.feature_flags, **self._override_feature_flags}
        if merged_feature_flags:
            logger.info("Applying feature flags: %s", merged_feature_flags)
            if self._agent_config is not None:
                from dataclasses import replace as _dc_replace

                merged_with_existing = {
                    **(self._agent_config.feature_flags or {}),
                    **merged_feature_flags,
                }
                self._agent_config = _dc_replace(
                    self._agent_config, feature_flags=merged_with_existing
                )

        content_hash = _simulation_hash(sim_cls)
        # stdout for local eval runs; HarnessRun trace stream is the durable artifact.
        _emit(
            "scenario",
            {
                "name": sim_cls.name,
                "description": sim_cls.description,
                "content_hash": content_hash,
                "template_id": self._override_template_id,
            },
        )

        # Bedrock used `timezone.now()` (Django-aware UTC). Harness uses
        # plain UTC-aware datetimes.
        from datetime import UTC

        sim_start = datetime.now(tz=UTC).replace(hour=8, minute=0, second=0, microsecond=0)

        config_overrides = asdict(overrides)
        config_overrides["duration_days"] = sim_cls.duration_days
        if merged_feature_flags:
            config_overrides["feature_flags"] = merged_feature_flags
        if effective_reasoning:
            config_overrides["reasoning_effort"] = effective_reasoning

        run = EvalRunRecord(
            scenario_name=sim_cls.name,
            agent_id=agent_id,
            eval_mode=sim_cls.eval_mode,
            status="running",
            simulated_start=sim_start,
            started_at_wall=datetime.now(tz=UTC),
            config_overrides=config_overrides,
            scenario_hash=content_hash,
            git_sha=_get_git_sha(),
        )
        self._run = run
        # stdout for local eval runs; HarnessRun trace stream is the durable artifact.
        _emit(
            "run_started",
            {
                "scenario": run.scenario_name,
                "agent_id": run.agent_id,
                "scenario_hash": run.scenario_hash,
                "git_sha": run.git_sha,
            },
        )

        from .actors import UserAgent

        user_agents: dict[str, UserAgent] = {}
        for user_def in sim_cls.users:
            ua = UserAgent(user_def, eval_mode=sim_cls.eval_mode)
            user_agents[user_def.id] = ua

        ms = sim_cls.memory_seed
        if ms and ms.generate:
            from .memory_gen import generate_memory_seeds

            generated = generate_memory_seeds(
                instruction=ms.generate,
                simulation=sim_cls,
                explicit_entries=ms.entries,
            )
            ms.entries.extend(generated)

        if ms and ms.entries:
            self._seed_memory(agent_id, ms.entries, sim_start)

        # Ensure the fake-adapter tables exist in this agent's sqlite DB
        # before anyone (scenario, fake tool) tries to touch them.
        if self._agent_config is not None:
            from harness.core import storage
            from harness.evals.fakes.base import apply_migrations

            storage.load(agent_id)
            apply_migrations()
            # Commit DDL/INSERTs into applied_migrations before Harness.run()
            # opens a fresh connection to the same remote db.
            storage.flush()

        try:
            with simulated_clock_context(sim_start) as clock:
                sim = sim_cls(
                    agent_id=agent_id,
                    clock=clock,
                    data_store=None,
                    user_agents=user_agents,
                )
                set_simulation(sim)

                try:
                    self._execute_timeline(sim, run, clock, sim_start)
                finally:
                    set_simulation(None)

            run.status = "awaiting_review"
            logger.info(
                "Run %s completed in %.1fs. Status: %s",
                run.scenario_name,
                wall_time.monotonic() - wall_start,
                run.status,
            )

        except Exception as e:
            logger.exception("Simulation run %s failed: %s", run.scenario_name, e)
            run.status = "failed"
            raise

        finally:
            self._cleanup(run, wall_time.monotonic() - wall_start)

        return run

    # ------------------------------------------------------------------
    # Core execution loop
    # ------------------------------------------------------------------

    def _execute_timeline(
        self,
        sim: Simulation,
        run: EvalRunRecord,
        clock: SimulatedClock,
        sim_start: datetime,
    ):
        timeline = sim.build_timeline(sim_start)
        total_events = len(timeline)
        sequence = 0
        total_wall_seconds = 0.0
        total_cost = Decimal("0")

        for i, sched in enumerate(timeline, 1):
            if sim.is_terminal():
                print(f"[sim] Simulation terminated early at day {sched.day}")
                break

            print(f"\n{'=' * 60}")
            print(
                f"[sim] Event {i}/{total_events} — "
                f"{sched.event_type.value} at {sched.sim_time.strftime('%Y-%m-%d %H:%M')}"
            )

            if sched.is_checkpoint:
                self._handle_checkpoint(
                    sim, run, sched, clock, sequence, total_cost, total_wall_seconds
                )
            else:
                if sim._pending_checkpoint_reset:
                    sim._checkpoint_trace_events = []
                    sim._pending_checkpoint_reset = False

                from_time = clock.now()
                clock.advance_to(sched.sim_time)
                sequence += 1
                self._log_event(
                    run,
                    sequence,
                    "clock_advance",
                    "system",
                    sched.sim_time,
                    {
                        "from_sim_time": from_time.isoformat(),
                        "to_sim_time": sched.sim_time.isoformat(),
                    },
                )

                if sched.event_type == ScheduledEventType.EVENT:
                    self._handle_event(sim, run, sched, clock, sequence)
                elif sched.event_type == ScheduledEventType.PERIODIC:
                    self._handle_periodic(sim, run, sched, clock, sequence)

                if sched.wake_agent:
                    print("[sim] Running agent to completion...")
                    wall_start = wall_time.monotonic()
                    self._run_agent_to_completion(sim)
                    wall_elapsed = wall_time.monotonic() - wall_start
                    total_wall_seconds += wall_elapsed
                    print(f"[sim] Agent finished in {wall_elapsed:.1f}s")

                    if sim._pending_user_replies:
                        print(
                            f"[sim] Processing {len(sim._pending_user_replies)} user reply(ies)..."
                        )
                    self._process_pending_replies(sim, run, clock, sequence)

        run.simulated_end = clock.now()
        run.total_wall_time_seconds = total_wall_seconds
        run.total_llm_cost_usd = total_cost

        score = sim.score()
        if score:
            print(f"\n[sim] Final score: {score}")

    def _handle_event(
        self,
        sim: Simulation,
        run: EvalRunRecord,
        sched: ScheduledEvent,
        clock: SimulatedClock,
        sequence: int,
    ):
        method = getattr(sim, sched.method_name)
        content = method()

        if content and sched.actor:
            user_agent = sim.user_agents.get(sched.actor)
            if user_agent:
                channel = sched.channel or (
                    user_agent.channels[0] if user_agent.channels else "sms"
                )
                content_str = content.strip() if isinstance(content, str) else str(content)
                preview = content_str[:100]
                print(f"[sim] MESSAGE from {sched.actor} via {channel}: {preview}")

                if channel == "sms":
                    sim.inject_inbound_sms(user_agent, content_str)
                elif channel == "email":
                    sim.inject_inbound_email(user_agent, content_str)
                else:
                    sim.inject_inbound_sms(user_agent, content_str)

                self._log_event(
                    run,
                    sequence + 1,
                    "message",
                    sched.actor,
                    clock.now(),
                    {"message": content_str, "direction": "inbound"},
                    channel=channel,
                )
            else:
                print(f"[sim] ENVIRONMENT EVENT: {sched.method_name}")
                self._log_event(
                    run,
                    sequence + 1,
                    "environment",
                    "environment",
                    clock.now(),
                    {"method": sched.method_name},
                )
        elif not sched.actor:
            print(f"[sim] ENVIRONMENT EVENT: {sched.method_name}")
            self._log_event(
                run,
                sequence + 1,
                "environment",
                "environment",
                clock.now(),
                {"method": sched.method_name},
            )

    def _handle_periodic(
        self,
        sim: Simulation,
        run: EvalRunRecord,
        sched: ScheduledEvent,
        clock: SimulatedClock,
        sequence: int,
    ):
        label = sched.periodic_description or sched.method_name
        print(f"[sim] PERIODIC: {label}")
        method = getattr(sim, sched.method_name)
        method()

        self._log_event(
            run,
            sequence + 1,
            "environment",
            "periodic",
            clock.now(),
            {"hook": sched.method_name, "description": sched.periodic_description},
        )

    def _handle_checkpoint(
        self,
        sim: Simulation,
        run: EvalRunRecord,
        sched: ScheduledEvent,
        clock: SimulatedClock,
        sequence: int,
        total_cost: Decimal,
        total_wall_seconds: float,
    ):
        method = getattr(sim, sched.method_name)
        passed = bool(method())

        description = sched.checkpoint_description or sched.checkpoint_name
        assertions_passed = 1 if passed else 0
        assertions_total = 1
        assertion_results = [{"passed": passed, "description": description}]

        # Emit a span of type=checkpoint so the bedrock side (T5) can
        # aggregate checkpoint pass rates via the existing Span rollup.
        cp_iso = clock.now().isoformat()
        emit_completed_span(
            name=sched.checkpoint_name,
            span_type=SpanType.CHECKPOINT,
            started_at=cp_iso,
            ended_at=cp_iso,
            agent_id=run.agent_id,
            metadata={
                "passed": passed,
                "description": description,
                "simulated_time": cp_iso,
                "scenario": run.scenario_name,
                "agent_id": run.agent_id,
                "assertions": assertion_results,
                "assertions_passed": assertions_passed,
                "assertions_total": assertions_total,
                "cumulative_cost_usd": str(total_cost),
                "cumulative_wall_seconds": total_wall_seconds,
            },
        )
        self.checkpoint_results.append(
            {
                "name": sched.checkpoint_name,
                "passed": passed,
                "description": description,
            }
        )
        _emit(
            "checkpoint",
            {
                "run_scenario": run.scenario_name,
                "run_agent_id": run.agent_id,
                "name": sched.checkpoint_name,
                "simulated_time": cp_iso,
                "description": description,
                "assertion_results": assertion_results,
                "assertions_passed": assertions_passed,
                "assertions_total": assertions_total,
                "cumulative_cost_usd": str(total_cost),
                "cumulative_wall_seconds": total_wall_seconds,
            },
        )

        self._log_event(
            run,
            sequence + 1,
            "checkpoint",
            "system",
            clock.now(),
            {
                "checkpoint": sched.checkpoint_name,
                "passed": passed,
                "assertions_passed": assertions_passed,
                "assertions_total": assertions_total,
                "results": assertion_results,
            },
        )

        sim._pending_checkpoint_reset = True

        status = "PASS" if passed else "FAIL"
        print(f"[sim] Checkpoint '{sched.checkpoint_name}': {status}")

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    def _run_agent_to_completion(self, sim: Simulation):
        """Drive one Harness run against the eval agent.

        T7: when `agent_config` is set on the runner, actually invoke
        `Harness(config, run_id).run()` between events. The fake adapters'
        tool handlers write into the per-agent sqlite DB synchronously,
        so when we re-open storage afterwards we can scan for new outbound
        messages and call `sim.process_new_outbound()` to queue user
        replies. No agent_config -> no-op (scenario-shape tests).
        """
        if self._agent_config is None:
            logger.debug(
                "Skipping agent run for sim %s (no agent_config on runner)",
                sim.name,
            )
            return

        import uuid as _uuid

        from harness.core import storage
        from harness.harness import Harness

        run_id = str(_uuid.uuid4())
        try:
            Harness(self._agent_config, run_id=run_id).run()
        except Exception:
            logger.exception(
                "Harness run failed during sim %s; continuing timeline", sim.name
            )

        # Harness.run() closed storage in its own `finally`. Re-open so
        # the fakes can be queried for new outbound messages and so any
        # subsequent inject_inbound_* calls on this sim have a live db.
        # Harness already committed its own writes via storage.flush()
        # before closing, so the outbound rows the fakes persisted during
        # the run are visible on the reopened connection.
        storage.load(sim.agent_id)

        tally = sim.process_new_outbound()
        if tally["emails"] or tally["sms"]:
            logger.info(
                "processed outbound: emails=%d sms=%d", tally["emails"], tally["sms"]
            )

    def _process_pending_replies(
        self,
        sim: Simulation,
        run: EvalRunRecord,
        clock: SimulatedClock,
        sequence: int,
    ):
        while sim._pending_user_replies:
            replies = list(sim._pending_user_replies)
            sim._pending_user_replies.clear()

            for user_agent, content, channel in replies:
                if channel == "sms":
                    sim.inject_inbound_sms(user_agent, content)
                elif channel == "email":
                    sim.inject_inbound_email(user_agent, content)

                self._log_event(
                    run,
                    sequence + 1,
                    "message",
                    user_agent.id,
                    clock.now(),
                    {"message": content, "direction": "inbound", "is_reply": True},
                    channel=channel,
                )

            self._run_agent_to_completion(sim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_event(
        self,
        run: EvalRunRecord,
        sequence: int,
        event_type: str,
        actor: str,
        sim_time: datetime,
        content: dict,
        channel: str = "",
    ):
        from .clock import _original_now

        # stdout for local eval runs; HarnessRun trace stream is the durable artifact.
        _emit(
            "event",
            {
                "run_scenario": run.scenario_name,
                "run_agent_id": run.agent_id,
                "sequence": sequence,
                "event_type": event_type,
                "actor": actor,
                "channel": channel,
                "simulated_time": sim_time.isoformat(),
                "wall_time": _original_now().isoformat(),
                "content": content,
            },
        )

    @staticmethod
    def _seed_memory(agent_id: str, entries: list, sim_start: datetime):
        """Seed pre-simulation five-minute summaries into agent memory.

        Bedrock wrote directly to `memory.models.FiveMinuteSummary`. In the
        harness the equivalent rows live inside the per-agent sqlite file
        opened by `harness.core.storage.load(agent_id)`. Writing them
        requires importing harness internals; do it lazily so the top-level
        evals module stays fast to import.
        """
        if not entries:
            return

        from harness.core import storage
        from harness.memory import rows as memory_rows

        seen: set[tuple] = set()
        records = []
        for entry in entries:
            if not isinstance(entry, MemorySeedEntry):
                continue
            date, hour, minute = entry.resolve(sim_start)
            key = (date, hour, minute)
            if key in seen:
                continue
            seen.add(key)
            records.append((date, hour, minute, entry.summary, entry.message_count))

        if not records:
            return

        # Open the per-agent sqlite file. If the harness runtime already
        # opened it this is a no-op.
        storage.load(agent_id)

        # TODO(T7): confirm `memory_rows` exposes a public bulk-insert API for
        # five-minute summaries. For now we just count what we *would* have
        # inserted so the rest of the pipeline is exercisable without a real
        # DB write. The exact insert will be finalised alongside the T7
        # rewire of `seed_memory` — it's trivial once we know the final row
        # shape, but this file should not hard-depend on an internal shape
        # that the memory package might still adjust.
        _ = memory_rows  # keep the import resolvable
        logger.info(
            "Would seed %d five-minute summaries for agent %s (T7: wire real insert)",
            len(records),
            agent_id,
        )

    def _cleanup(self, run: EvalRunRecord, wall_seconds: float | None = None):
        from datetime import UTC

        run.completed_at_wall = datetime.now(tz=UTC)
        if wall_seconds is not None:
            run.total_wall_time_seconds = wall_seconds
        # stdout for local eval runs; HarnessRun trace stream is the durable artifact.
        _emit(
            "run_completed",
            {
                "scenario": run.scenario_name,
                "agent_id": run.agent_id,
                "status": run.status,
                "wall_time_seconds": run.total_wall_time_seconds,
                "simulated_end": run.simulated_end.isoformat() if run.simulated_end else None,
            },
        )


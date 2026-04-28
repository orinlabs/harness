"""Base class for eval simulations.

Subclasses define behavior via three decorators:

- ``@periodic(at="HH:MM")`` -- fires every simulated day at that time
- ``@event(day=N, time="HH:MM")`` -- fires once at a specific simulated time
- ``@checkpoint(day=N, name="...")`` -- evaluates assertions after that day's events

The runner builds a merged timeline from all decorated methods and steps
through it, advancing the simulated clock and running the agent between
events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Callable

from .types import AgentOverrides, MemorySeed, UserDefinition

if TYPE_CHECKING:
    from .actors import UserAgent
    from .clock import SimulatedClock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def periodic(at: str, description: str = "", wake_agent: bool = True):
    """Mark a method as a periodic hook that fires every simulated day.

    Set ``wake_agent=False`` for internal bookkeeping hooks that should
    not trigger an agent run (e.g. sales calculations, delivery processing).
    """
    parts = at.split(":")
    hook_time = time(int(parts[0]), int(parts[1]))

    def decorator(func: Callable) -> Callable:
        func._sim_periodic = True
        func._sim_at = hook_time
        func._sim_periodic_description = description
        func._sim_periodic_wake_agent = wake_agent
        return func

    return decorator


def event(day: int, time_str: str = "00:00", actor: str = "", channel: str = ""):
    """Mark a method as a one-shot event at a specific simulated time."""
    parts = time_str.split(":")
    event_time = time(int(parts[0]), int(parts[1]))

    def decorator(func: Callable) -> Callable:
        func._sim_event = True
        func._sim_day = day
        func._sim_time = event_time
        func._sim_actor = actor
        func._sim_channel = channel
        return func

    return decorator


def checkpoint(day: int, name: str, description: str = ""):
    """Mark a method as a checkpoint that returns True/False."""

    def decorator(func: Callable) -> Callable:
        func._sim_checkpoint = True
        func._sim_day = day
        func._sim_checkpoint_name = name
        func._sim_checkpoint_description = description
        return func

    return decorator


# ---------------------------------------------------------------------------
# Scheduled event types
# ---------------------------------------------------------------------------


class ScheduledEventType(Enum):
    PERIODIC = "periodic"
    EVENT = "event"
    CHECKPOINT = "checkpoint"


@dataclass
class ScheduledEvent:
    sim_time: datetime
    event_type: ScheduledEventType
    method_name: str
    day: int = 0
    actor: str = ""
    channel: str = ""
    checkpoint_name: str = ""
    checkpoint_description: str = ""
    periodic_description: str = ""
    wake_agent: bool = True

    @property
    def is_checkpoint(self) -> bool:
        return self.event_type == ScheduledEventType.CHECKPOINT

    @property
    def sort_key(self) -> tuple:
        type_order = 0 if not self.is_checkpoint else 1
        return (self.sim_time, type_order)


# ---------------------------------------------------------------------------
# Metaclass
# ---------------------------------------------------------------------------


class SimulationMeta(type):
    """Collects @periodic, @event, and @checkpoint methods at class creation."""

    def __new__(mcs, name: str, bases: tuple, namespace: dict):
        cls = super().__new__(mcs, name, bases, namespace)

        periodic_hooks: list[dict] = []
        event_hooks: list[dict] = []
        checkpoint_hooks: list[dict] = []

        for base in reversed(cls.__mro__):
            for attr_name, attr in vars(base).items():
                if not callable(attr):
                    continue
                if getattr(attr, "_sim_periodic", False):
                    entry = {
                        "time": attr._sim_at,
                        "method": attr_name,
                        "description": getattr(attr, "_sim_periodic_description", ""),
                        "wake_agent": getattr(attr, "_sim_periodic_wake_agent", True),
                    }
                    if entry not in periodic_hooks:
                        periodic_hooks.append(entry)
                if getattr(attr, "_sim_event", False):
                    entry = {
                        "day": attr._sim_day,
                        "time": attr._sim_time,
                        "method": attr_name,
                        "actor": attr._sim_actor,
                        "channel": attr._sim_channel,
                    }
                    if entry not in event_hooks:
                        event_hooks.append(entry)
                if getattr(attr, "_sim_checkpoint", False):
                    entry = {
                        "day": attr._sim_day,
                        "name": attr._sim_checkpoint_name,
                        "description": attr._sim_checkpoint_description,
                        "method": attr_name,
                    }
                    if entry not in checkpoint_hooks:
                        checkpoint_hooks.append(entry)

        periodic_hooks.sort(key=lambda x: (x["time"].hour, x["time"].minute))
        cls._periodic_hooks = periodic_hooks
        cls._event_hooks = event_hooks
        cls._checkpoint_hooks = checkpoint_hooks
        return cls


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Simulation(metaclass=SimulationMeta):
    """Base class for all eval simulations."""

    # Subclasses set these as class attributes
    name: str = ""
    description: str = ""
    duration_days: int = 1
    eval_mode: str = "deterministic"
    agent_overrides: AgentOverrides = AgentOverrides()
    # TODO(T9): rewire feature flags post-product-deletion. Scenarios can still
    # declare `feature_flags = {...}` but the runner ignores them for now.
    feature_flags: dict[str, str] = {}
    users: list[UserDefinition] = []
    memory_seed: MemorySeed | None = None

    @classmethod
    def ensure_tools(cls):
        """Create any adapter/tool records this simulation needs.

        Called by the runner before adapter assignment. Override in
        subclasses that register custom adapters (e.g. VendingBenchSimulation).
        The default implementation is a no-op.
        """

    # Collected by metaclass
    _periodic_hooks: list[dict]
    _event_hooks: list[dict]
    _checkpoint_hooks: list[dict]

    def __init__(
        self,
        agent_id: str,
        clock: "SimulatedClock",
        data_store=None,
        user_agents: "dict[str, UserAgent] | None" = None,
    ):
        # In bedrock the first arg was a Django Agent model. In the harness
        # the agent is just a UUID string that indexes into the per-agent
        # sqlite storage used by `harness.memory.MemoryService`.
        self.agent_id = agent_id
        self.clock = clock
        # TODO(T6): `data_store` is the fake environment (SimulatedDataStore in
        # bedrock) that backs `inject_environment_event(...)` etc. Scenarios
        # call into it; the runner will wire a real fake in T6. For import-
        # cleanliness this can be None.
        self.data_store = data_store
        self.user_agents = user_agents or {}
        self._memory_service = None
        self._pending_user_replies: list[tuple["UserAgent", str, str]] = []
        self._trace_events: list[dict] = []
        self._checkpoint_trace_events: list[dict] = []
        self._pending_checkpoint_reset = False
        self._processed_outbound_email_ids: set[str] = set()
        self._processed_outbound_sms_ids: set[str] = set()

    # Backwards-compat alias: bedrock scenarios touched `self.agent` (a Django
    # model) for `self.agent.id`-style reads. They now get a string agent id.
    @property
    def agent(self) -> str:
        return self.agent_id

    @property
    def trace(self) -> list[dict]:
        """Tool call events since the last checkpoint. Use in @checkpoint methods."""
        return self._checkpoint_trace_events

    @property
    def all_trace(self) -> list[dict]:
        """All tool call events across the entire simulation run."""
        return self._trace_events

    @property
    def memory_service(self):
        """Lazy-initialised harness memory service for this simulation's agent.

        Bedrock's version constructed ``MemoryService(self.agent, sync=True)``
        with a Django Agent instance. The harness's
        ``harness.memory.MemoryService`` takes a string ``agent_id`` plus a
        model slug, and writes into the per-agent sqlite file opened via
        ``harness.core.storage.load(agent_id)``. Callers that actually invoke
        ``log_messages`` / ``update_summaries`` are responsible for opening
        storage beforehand.
        """
        if self._memory_service is None:
            from harness.memory import MemoryService

            model = self.agent_overrides.summarizer_model or "openai/gpt-4o-mini"
            self._memory_service = MemoryService(agent_id=str(self.agent_id), model=model)
        return self._memory_service

    # ------------------------------------------------------------------
    # Timeline construction
    # ------------------------------------------------------------------

    def build_timeline(self, sim_start: datetime) -> list[ScheduledEvent]:
        """Build a sorted list of all scheduled events for the simulation."""
        events: list[ScheduledEvent] = []

        for hook_info in self._event_hooks:
            day_offset = timedelta(days=hook_info["day"] - 1)
            t = hook_info["time"]
            sim_time = datetime.combine(sim_start.date() + day_offset, t, tzinfo=sim_start.tzinfo)
            events.append(
                ScheduledEvent(
                    sim_time=sim_time,
                    event_type=ScheduledEventType.EVENT,
                    method_name=hook_info["method"],
                    day=hook_info["day"],
                    actor=hook_info["actor"],
                    channel=hook_info["channel"],
                )
            )

        for cp_info in self._checkpoint_hooks:
            day_offset = timedelta(days=cp_info["day"] - 1)
            sim_time = datetime.combine(
                sim_start.date() + day_offset,
                time(23, 59, 59),
                tzinfo=sim_start.tzinfo,
            )
            events.append(
                ScheduledEvent(
                    sim_time=sim_time,
                    event_type=ScheduledEventType.CHECKPOINT,
                    method_name=cp_info["method"],
                    day=cp_info["day"],
                    checkpoint_name=cp_info["name"],
                    checkpoint_description=cp_info["description"],
                )
            )

        for day_num in range(1, self.duration_days + 1):
            for hook_info in self._periodic_hooks:
                day_offset = timedelta(days=day_num - 1)
                sim_time = datetime.combine(
                    sim_start.date() + day_offset,
                    hook_info["time"],
                    tzinfo=sim_start.tzinfo,
                )
                events.append(
                    ScheduledEvent(
                        sim_time=sim_time,
                        event_type=ScheduledEventType.PERIODIC,
                        method_name=hook_info["method"],
                        periodic_description=hook_info.get("description", ""),
                        wake_agent=hook_info.get("wake_agent", True),
                        day=day_num,
                    )
                )

        events.sort(key=lambda e: e.sort_key)
        return events

    # ------------------------------------------------------------------
    # Time advancement (for simulations with tool-call time costs)
    # ------------------------------------------------------------------

    def advance_to(self, target_time: datetime):
        """Advance clock to target, firing periodic hooks crossed along the way."""
        current = self.clock.now()
        if target_time <= current:
            return

        self._fire_periodic_hooks_between(current, target_time)
        self.clock.advance_to(target_time)

    def advance(self, minutes: int):
        """Advance clock by N minutes, firing periodic hooks crossed."""
        target = self.clock.now() + timedelta(minutes=minutes)
        self.advance_to(target)

    def _fire_periodic_hooks_between(self, start: datetime, end: datetime):
        """Fire any @periodic hooks whose time falls in (start, end]."""
        if not self._periodic_hooks:
            return

        current_date = start.date()
        end_date = end.date()

        while current_date <= end_date:
            for hook_info in self._periodic_hooks:
                hook_dt = datetime.combine(current_date, hook_info["time"], tzinfo=start.tzinfo)
                if start < hook_dt <= end:
                    logger.debug("Firing periodic hook %s at %s", hook_info["method"], hook_dt)
                    getattr(self, hook_info["method"])()
            current_date += timedelta(days=1)

    # ------------------------------------------------------------------
    # Message injection (stubs -- real wiring lands in T6/T7)
    # ------------------------------------------------------------------
    #
    # Bedrock's versions created Django Contact/Message/Notification rows and
    # fed the agent's memory via `MemoryService.log_messages`. In the harness
    # the equivalent is: (1) drop the message onto the fake SMS/email
    # channel (T6), and (2) push a corresponding user-role entry into the
    # agent's memory (`harness.memory.MemoryService.log_messages`). Scenarios
    # call these at runtime during an eval run; this task only guarantees the
    # signatures parse so scenario files import cleanly.

    # Side-effect impls filled in at T7: drop inbound into the fake channel
    # and log a user-role entry into harness memory so the next turn's
    # build_llm_inputs() sees it.

    def inject_inbound_sms(self, user_agent: "UserAgent", content: str):
        from harness.core import storage
        from harness.evals.fakes import sms as sms_fake
        from harness.evals.fakes.base import apply_migrations

        # Ensure the fake tables exist -- storage may have just been
        # reopened after a Harness.run() cycle which called storage.close().
        storage.load(self.agent_id)
        apply_migrations()
        sent_at = self.clock.now().isoformat()
        sms_fake.inject_inbound(
            contact_phone=user_agent.phone or "+10000000000",
            body=content,
            sent_at=sent_at,
        )
        # Log as a user-role entry keyed by channel so the agent sees it
        # on the next turn even without calling list_conversations.
        self.memory_service.log_messages(
            [
                {
                    "role": "user",
                    "content": (
                        f"[inbound SMS from {user_agent.name} "
                        f"({user_agent.phone})]: {content}"
                    ),
                }
            ]
        )
        self.create_notification(
            title=f"SMS from {user_agent.name}",
            body=content,
            priority="high",
        )
        # Commit so the subsequent Harness.run() -- which opens a fresh
        # storage connection -- sees the persisted rows.
        storage.flush()

    def inject_inbound_email(self, user_agent: "UserAgent", content: str):
        from harness.core import storage
        from harness.evals.fakes import email as email_fake
        from harness.evals.fakes.base import apply_migrations

        storage.load(self.agent_id)
        apply_migrations()
        sent_at = self.clock.now().isoformat()
        subject_line = content.splitlines()[0][:60] if content else "(no subject)"
        email_fake.inject_inbound(
            thread_id=None,
            from_email=user_agent.email or f"{user_agent.id}@example.com",
            to_email="agent@eval.test",
            subject=subject_line,
            body=content,
            sent_at=sent_at,
        )
        self.memory_service.log_messages(
            [
                {
                    "role": "user",
                    "content": (
                        f"[inbound email from {user_agent.name} "
                        f"<{user_agent.email}>] {subject_line}\n\n{content}"
                    ),
                }
            ]
        )
        self.create_notification(
            title=f"Email from {user_agent.name}: {subject_line}",
            body=content[:400],
            priority="high",
        )
        storage.flush()

    # ------------------------------------------------------------------
    # Outbound processing (called by the runner after each agent wake)
    # ------------------------------------------------------------------
    #
    # In the harness-edition eval flow, outbound tool calls (send_email /
    # send_sms) are persisted synchronously by the fake adapters during
    # the Harness run. The Simulation needs to inspect those new rows
    # *after* the agent finishes so it can synthesize UserAgent replies
    # and queue them via `_pending_user_replies`. These IDs live on the
    # sim instance so we don't re-process the same message twice.

    def process_new_outbound(self):
        """Scan for new outbound fake-adapter messages, invoke on_outbound_*.

        Returns a tally dict -- the runner logs it.
        """
        from harness.core import storage
        from harness.evals.fakes.base import apply_migrations

        storage.load(self.agent_id)
        apply_migrations()

        import json as _json

        db = storage.db
        if db is None:
            return {"emails": 0, "sms": 0}

        emails_seen = 0
        sms_seen = 0

        # Emails
        try:
            rows = db.execute(
                "SELECT id, to_addrs, subject, body FROM fake_email_message "
                "WHERE direction = 'outbound' ORDER BY sent_at ASC"
            ).fetchall()
        except Exception:
            rows = []
        for row in rows:
            msg_id = row["id"]
            if msg_id in self._processed_outbound_email_ids:
                continue
            self._processed_outbound_email_ids.add(msg_id)
            try:
                to_addrs = _json.loads(row["to_addrs"]) or []
            except Exception:
                to_addrs = []
            self.on_outbound_email(
                to=to_addrs,
                subject=row["subject"] or "",
                body=row["body"] or "",
            )
            emails_seen += 1

        # SMS
        try:
            rows = db.execute(
                "SELECT id, contact_phone, body FROM fake_sms_message "
                "WHERE direction = 'outbound' ORDER BY sent_at ASC"
            ).fetchall()
        except Exception:
            rows = []
        for row in rows:
            msg_id = row["id"]
            if msg_id in self._processed_outbound_sms_ids:
                continue
            self._processed_outbound_sms_ids.add(msg_id)
            self.on_outbound_sms(
                to=row["contact_phone"] or "",
                body=row["body"] or "",
                sid=msg_id,
            )
            sms_seen += 1

        return {"emails": emails_seen, "sms": sms_seen}

    def on_outbound_sms(self, to: str, body: str, sid: str):
        print(f"[sim] >>> AGENT SMS to {to}: {body[:200]}")
        for ua in self.user_agents.values():
            if ua.phone == to:
                response = ua.generate_response(body, "sms")
                if response:
                    print(f"[sim] <<< USER REPLY from {ua.name}: {response[:200]}")
                    self._pending_user_replies.append((ua, response, "sms"))
                break

    def on_outbound_email(self, to: list[str], subject: str, body: str):
        print(f"[sim] >>> AGENT EMAIL to {to}: {subject} — {body[:200]}")
        for ua in self.user_agents.values():
            if ua.email in to:
                response = ua.generate_response(body, "email")
                if response:
                    self._pending_user_replies.append((ua, response, "email"))

    def on_computer_exec(self, command: str, result: str):
        print(f"[sim] >>> AGENT computer_exec: {command[:200]}")

    # ------------------------------------------------------------------
    # Notification helper
    # ------------------------------------------------------------------

    def create_notification(self, title: str, body: str = "", priority: str = "high"):
        """Log a notification to stdout.

        The fake adapter set does not currently ship a `fake_notification`
        table -- scenarios that want a persisted notification feed should
        extend the fakes in a follow-up task. For now we just echo to
        stdout so scenarios calling this don't crash. We deliberately do
        NOT attempt a speculative INSERT against a non-existent table
        because that would poison the active sqlite transaction and
        nuke any prior-in-this-connection writes on commit.
        """
        print(f"[sim] NOTIFICATION ({priority}) {title}: {body[:200]}")

    # ------------------------------------------------------------------
    # Scoring / termination (override in subclasses)
    # ------------------------------------------------------------------

    def score(self) -> dict | None:
        """Return aggregate score dict, or None if not applicable."""
        return None

    def is_terminal(self) -> bool:
        """Return True if the simulation should end early."""
        return False

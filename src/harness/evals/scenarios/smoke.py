"""Minimal smoke scenario for the `harness eval` CLI path.

One day. One inbound email. One checkpoint that always passes. The goal
here is to validate the end-to-end wiring -- fake-adapter tables
migrated, Harness driven to completion, checkpoint span POSTed to
bedrock -- not to test any particular agent behavior. Heavier scenarios
are the subject of T8.
"""

from __future__ import annotations

from harness.evals import (
    AgentOverrides,
    ResponsePolicy,
    Simulation,
    UserDefinition,
    checkpoint,
    event,
)


class SmokeScenario(Simulation):
    name = "smoke"
    description = "Minimal 1-day smoke: one inbound email, one always-pass checkpoint."
    duration_days = 1
    eval_mode = "deterministic"
    agent_overrides = AgentOverrides(model="claude-haiku-4-5", max_turns=3)
    users = [
        UserDefinition(
            id="user",
            name="Smoke Tester",
            email="smoke-tester@example.com",
            phone="+15550001111",
            channels=["email"],
            response_policy=[
                ResponsePolicy(trigger="default", response="Thanks!"),
            ],
        ),
    ]

    @event(day=1, time_str="09:00", actor="user", channel="email")
    def user_says_hi(self):
        return (
            "Hi! This is a smoke test. Please reply by email once to confirm "
            "you received this, then go back to sleep."
        )

    @checkpoint(
        day=1,
        name="smoke_reached_end_of_day",
        description="Smoke scenario executed end-to-end (always passes).",
    )
    def check_smoke(self):
        # Always true: the point of the smoke checkpoint is to prove the
        # emit_completed_span(span_type=CHECKPOINT) pipeline reaches
        # bedrock. T8 exercises real-behavior checkpoints.
        return True

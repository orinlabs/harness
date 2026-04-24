"""Harness eval framework.

This subpackage is **not** imported by the top-level ``harness`` package —
production cold-start should not pay the price of importing eval-only
dependencies. Load it only from the ``harness eval ...`` CLI path (T7).
"""

from __future__ import annotations

from .base import (
    ScheduledEvent,
    ScheduledEventType,
    Simulation,
    checkpoint,
    event,
    periodic,
)
from .context import get_simulation, set_simulation
from .runner import EvalRunRecord, SimulationRunner
from .types import (
    AgentOverrides,
    CalendarEventData,
    EmailEventData,
    EnvironmentData,
    GmailMessageData,
    MemorySeed,
    MemorySeedEntry,
    MemorySeedInstruction,
    ResponsePolicy,
    UserDefinition,
)

__all__ = [
    "AgentOverrides",
    "CalendarEventData",
    "EmailEventData",
    "EnvironmentData",
    "EvalRunRecord",
    "GmailMessageData",
    "MemorySeed",
    "MemorySeedEntry",
    "MemorySeedInstruction",
    "ResponsePolicy",
    "ScheduledEvent",
    "ScheduledEventType",
    "Simulation",
    "SimulationRunner",
    "UserDefinition",
    "checkpoint",
    "event",
    "get_simulation",
    "periodic",
    "set_simulation",
]

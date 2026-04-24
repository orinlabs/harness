"""Typed dataclass definitions used by eval simulations.

These define the shared vocabulary for user definitions, environment
data, memory seeding, and agent configuration overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class ResponsePolicy:
    trigger: str
    response: str
    channel: str = ""


@dataclass
class UserDefinition:
    id: str
    name: str
    phone: str = ""
    email: str = ""
    channels: list[str] = field(default_factory=list)
    response_policy: list[ResponsePolicy] = field(default_factory=list)
    instructions: str = ""
    model: str = ""


@dataclass
class CalendarEventData:
    summary: str
    start: str
    end: str
    id: str = ""
    description: str = ""
    location: str = ""
    attendees: list[str] = field(default_factory=list)
    calendar_owner: str = ""


@dataclass
class EmailEventData:
    from_: str
    to: list[str]
    subject: str = ""
    body: str = ""
    message_id: str = ""
    thread_id: str = ""
    timestamp: str = ""


@dataclass
class GmailMessageData:
    from_: str
    to: str
    subject: str = ""
    body: str = ""
    id: str = ""
    thread_id: str = ""
    date: str = ""


EnvironmentData = CalendarEventData | EmailEventData | GmailMessageData


@dataclass
class MemorySeedEntry:
    """A single pre-populated five-minute summary to seed before the eval runs."""

    day: int
    time_str: str
    summary: str
    message_count: int = 3

    def resolve(self, sim_start: datetime) -> tuple:
        """Return (date, hour, minute) for the DB record."""
        parts = self.time_str.split(":")
        hour = int(parts[0]) if len(parts) >= 1 else 9
        minute = int(parts[1]) if len(parts) >= 2 else 0
        day_offset = timedelta(days=self.day - 1)
        target_date = sim_start.date() + day_offset
        return target_date, hour, minute


@dataclass
class MemorySeedInstruction:
    """LLM-based generation of background five-minute summaries."""

    instruction: str
    model: str = "claude-haiku-4-5"
    count: int = 30
    time_range_days: int = 365


@dataclass
class MemorySeed:
    entries: list[MemorySeedEntry] = field(default_factory=list)
    generate: MemorySeedInstruction | None = None


@dataclass
class AgentOverrides:
    model: str = ""
    max_turns: int | None = None
    system_prompt: str = ""
    summarizer_model: str = ""
    adapters: list[str] = field(default_factory=list)
    reasoning_effort: str = ""

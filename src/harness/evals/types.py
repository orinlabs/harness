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
    """A simulated user with one or more addresses.

    The new shape is multi-address: ``phones`` is the full ordered
    list of phone numbers (primary first), ``emails`` is the full
    ordered list of email addresses (primary first). The legacy
    ``phone`` / ``email`` scalar fields remain as inputs for
    backward-compatible scenario fixtures — they're folded into
    ``phones`` / ``emails`` in :meth:`__post_init__` and you should
    read them via :attr:`primary_phone` / :attr:`primary_email`.
    """

    id: str
    name: str
    # Legacy scalar inputs — accepted for back-compat with existing
    # scenario fixtures. New scenarios should set ``phones`` / ``emails``.
    phone: str = ""
    email: str = ""
    phones: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    response_policy: list[ResponsePolicy] = field(default_factory=list)
    instructions: str = ""
    model: str = ""

    def __post_init__(self) -> None:
        # Fold scalar phone/email into the multi-value lists when the
        # caller used the legacy shape, deduping while preserving order.
        if self.phone and self.phone not in self.phones:
            self.phones = [self.phone, *self.phones]
        if self.emails is None:
            self.emails = []
        if self.email and self.email not in self.emails:
            self.emails = [self.email, *self.emails]
        # Keep the scalar mirrors in sync with the primary value so legacy
        # call sites still see the right thing.
        self.phone = self.phones[0] if self.phones else ""
        self.email = self.emails[0] if self.emails else ""

    @property
    def primary_phone(self) -> str:
        return self.phones[0] if self.phones else ""

    @property
    def primary_email(self) -> str:
        return self.emails[0] if self.emails else ""

    def matches_phone(self, value: str) -> bool:
        """True when ``value`` is one of this user's phone numbers."""
        return bool(value) and value in self.phones

    def matches_email(self, value: str) -> bool:
        return bool(value) and value in self.emails


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
    max_tokens: int | None = None

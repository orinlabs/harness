"""Typed row dataclasses for summary and message queries.

These are what `MemoryContextBuilder.fetch_data` returns and what tests
assert against.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from typing import Any


@dataclass
class MessageRow:
    id: str
    ts_ns: int
    role: str
    content: dict[str, Any]  # OpenAI chat-format message dict


@dataclass
class OneMinuteSummary:
    id: str
    date: _date
    hour: int
    minute: int
    summary: str
    message_count: int


@dataclass
class FiveMinuteSummary:
    id: str
    date: _date
    hour: int
    minute: int
    summary: str
    message_count: int


@dataclass
class HourlySummary:
    id: str
    date: _date
    hour: int
    summary: str
    message_count: int


@dataclass
class DailySummary:
    id: str
    date: _date
    summary: str
    message_count: int


@dataclass
class WeeklySummary:
    id: str
    week_start_date: _date
    summary: str
    message_count: int


@dataclass
class MonthlySummary:
    id: str
    year: int
    month: int
    summary: str
    message_count: int

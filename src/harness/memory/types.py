from enum import Enum
from textwrap import dedent

from pydantic import BaseModel


class PeriodType(Enum):
    FIVE_MINUTE = "five_minute"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class PeriodMeta(BaseModel):
    max_length: str
    focus: str
    time_period: str


PERIOD_META: dict[PeriodType, PeriodMeta] = {
    PeriodType.FIVE_MINUTE: PeriodMeta(
        max_length="2 sentences",
        focus=dedent(
            """
            Summarize these messages while preserving the most amount of semantic information possible, like a historian would. Try
            and lose as little information as possible.

            IMPORTANT: Always preserve the user's exact instructions, preferences, corrections, and emotional reactions verbatim
            or near-verbatim. If the user says "don't use markdown" or "stop doing X", that must appear in the summary. User
            directives and feelings are the highest-priority information to retain.
            """
        ),
        time_period="these five minutes",
    ),
    PeriodType.HOURLY: PeriodMeta(
        max_length="2-4 sentences",
        focus=dedent(
            """
            Summarize this time period while preserving the most amount of semantic information possible, like a historian would. Try
            and lose as little information as possible. Give higher priority to trends that should continue, surprising information,
            learnings (even if you have to create the abstracted learnings from the context), and any other information that is important
            to the agent's current situation.
            """
        ),
        time_period="this hour",
    ),
    PeriodType.DAILY: PeriodMeta(
        max_length="4-6 sentences",
        focus=dedent(
            """
            Summarize this time period while preserving the most amount of semantic information possible, like a historian would. Try
            and lose as little information as possible. Give higher priority to trends that should continue, surprising information,
            learnings (even if you have to create the abstracted learnings from the context), and any other information that is important
            to the agent's current situation.
            """
        ),
        time_period="today",
    ),
    PeriodType.WEEKLY: PeriodMeta(
        max_length="6-8 sentences",
        focus=dedent(
            """
            Summarize this time period while preserving the most amount of semantic information possible, like a historian would. Try
            and lose as little information as possible. Give higher priority to trends that should continue, surprising information,
            learnings (even if you have to create the abstracted learnings from the context), and any other information that is important
            to the agent's current situation.
            """
        ),
        time_period="this week",
    ),
    PeriodType.MONTHLY: PeriodMeta(
        max_length="8-10 sentences",
        focus=dedent(
            """
            Summarize this time period while preserving the most amount of semantic information possible, like a historian would. Try
            and lose as little information as possible. Give higher priority to trends that should continue, surprising information,
            learnings (even if you have to create the abstracted learnings from the context), and any other information that is important
            to the agent's current situation.
            """
        ),
        time_period="this month",
    ),
}

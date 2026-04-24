"""Generate five-minute memory-seed entries from a high-level instruction.

Uses a cheap LLM call to produce realistic background memories that
complement any explicit (hand-written) entries in the simulation.

The bedrock version split Claude and OpenAI code paths through the in-tree
``core.llm`` abstractions. In the harness we route every call through
``harness.core.llm.complete`` (OpenRouter), which speaks the OpenAI
chat-completions format for all provider families, and extract the JSON
tool-call payload.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from .types import MemorySeedEntry, MemorySeedInstruction, UserDefinition

logger = logging.getLogger(__name__)


@runtime_checkable
class HasSimulationContext(Protocol):
    """Anything with users and a description (Simulation class or similar)."""

    users: list[UserDefinition]
    description: str


_SYSTEM_PROMPT = """\
You are a memory-seed generator for an AI agent eval harness.

Your job is to produce realistic five-minute summary entries that will be
inserted into the agent's memory system. Each entry represents a single
five-minute window of activity the agent observed in the past.

Write every summary in FIRST PERSON ("I helped with...", "I coordinated...").
Keep each summary to 1-3 sentences. Vary the time-of-day and day-of-week
realistically (business hours, occasional early/late, weekday-heavy).

Do NOT duplicate or rephrase any of the pinned entries provided below.
Weave your generated entries around them to form a coherent timeline.
"""


class GeneratedMemoryEntry(BaseModel):
    day: int = Field(description="Negative integer — days before simulation start")
    time: str = Field(description="HH:MM in 24-hour format")
    summary: str = Field(description="First-person summary, 1-3 sentences")
    message_count: int = Field(
        ge=3, le=15, description="Number of messages in this five-minute window"
    )


class GeneratedMemories(BaseModel):
    entries: list[GeneratedMemoryEntry]


def _to_seed_entries(items: list[GeneratedMemoryEntry]) -> list[MemorySeedEntry]:
    return [
        MemorySeedEntry(
            day=e.day,
            time_str=e.time,
            summary=e.summary,
            message_count=e.message_count,
        )
        for e in items
    ]


def _build_prompt(
    instruction: MemorySeedInstruction,
    simulation: HasSimulationContext,
    explicit_entries: list[MemorySeedEntry],
) -> str:
    parts: list[str] = []

    parts.append(f"INSTRUCTION: {instruction.instruction.strip()}")
    parts.append(f"TARGET COUNT: ~{instruction.count} entries")
    parts.append(
        f"TIME RANGE: day -{instruction.time_range_days} through day -1 "
        f"(relative to simulation start)"
    )

    if simulation.users:
        user_names = [u.name for u in simulation.users]
        parts.append(f"USERS: {', '.join(user_names)}")

    if simulation.description:
        parts.append(f"CONTEXT: {simulation.description.strip()}")

    if explicit_entries:
        parts.append("PINNED ENTRIES (do NOT duplicate these):")
        for entry in explicit_entries:
            parts.append(
                f"  day={entry.day} time={entry.time_str}: "
                f"{entry.summary[:200]}{'...' if len(entry.summary) > 200 else ''}"
            )

    return "\n\n".join(parts)


def generate_memory_seeds(
    instruction: MemorySeedInstruction,
    simulation: HasSimulationContext,
    explicit_entries: list[MemorySeedEntry],
) -> list[MemorySeedEntry]:
    """Generate five-minute summary entries from a high-level instruction.

    Routes through ``harness.core.llm.complete`` with a tool-call schema so
    the model is forced to emit strictly-shaped JSON.
    """
    from harness.core.llm import complete

    user_prompt = _build_prompt(instruction, simulation, explicit_entries)

    tool_schema = {
        "type": "function",
        "function": {
            "name": "output_memories",
            "description": "Output the generated memory-seed entries.",
            "parameters": GeneratedMemories.model_json_schema(),
        },
    }

    response = complete(
        model=instruction.model,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        tools=[tool_schema],
        tool_choice={"type": "function", "function": {"name": "output_memories"}},
    )

    if not response.tool_calls:
        # Best-effort: try to parse the text body as JSON.
        try:
            parsed = GeneratedMemories.model_validate_json(response.text)
        except Exception as exc:
            raise ValueError(
                f"Memory-seed model did not return a tool call or valid JSON (model={instruction.model})"
            ) from exc
    else:
        call = response.tool_calls[0]
        args = call.args if isinstance(call.args, dict) else json.loads(call.args)
        parsed = GeneratedMemories.model_validate(args)

    entries = _to_seed_entries(parsed.entries)
    logger.info(
        "Generated %d memory-seed entries (requested ~%d) with model %s",
        len(entries),
        instruction.count,
        instruction.model,
    )
    return entries

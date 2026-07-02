"""Render an agent's tiered memory as a standalone text block.

Used by ``harness export-memory`` so external callers (Bedrock's VAPI
voice webhook, inspection tooling) can pull the same temporally
summarized memory block the agent itself sees in its system prompt,
without running an agent turn.

The export is summaries-only (``min_resolution=FIVE_MINUTE``): raw
recent messages are excluded because callers like the voice webhook
already supply their own recent-conversation context, and the raw
message log contains internal tool-call traffic that shouldn't leak
into another model's prompt.

Because the exported block is consumed by a process that can't parse
harness log output apart from payload (Daytona ``process.exec``
returns one combined stream), the payload is wrapped in sentinel
marker lines. Anything outside the markers is noise; the consumer
extracts the text between them.
"""

from __future__ import annotations

from datetime import UTC, datetime

from harness.memory.context import MemoryContextBuilder
from harness.memory.types import PeriodType

EXPORT_BEGIN_MARKER = "-----BEGIN HARNESS MEMORY EXPORT-----"
EXPORT_END_MARKER = "-----END HARNESS MEMORY EXPORT-----"


def export_memory_context(
    *,
    timezone_name: str = "UTC",
    current_time: datetime | None = None,
    max_tokens: int | None = None,
) -> str:
    """Return the rendered tier-labeled summary block for the loaded agent.

    Requires ``storage.load(agent_id)`` to have been called. Returns ``""``
    when the agent has no summaries yet.

    ``max_tokens`` optionally caps the rendered block (tier-aware trim,
    finest tiers dropped first — see ``MemoryContextBuilder.render``).
    ``None`` keeps the renderer's default, preserving existing behavior
    for callers that don't budget.
    """
    if current_time is None:
        current_time = datetime.now(tz=UTC)

    builder = MemoryContextBuilder(timezone=timezone_name)
    data = builder.fetch_data(current_time, min_resolution=PeriodType.FIVE_MINUTE)
    if max_tokens is None:
        return builder.render(data)
    return builder.render(data, max_tokens=max_tokens)


def wrap_export(rendered: str) -> str:
    """Wrap a rendered memory block in the sentinel markers."""
    return f"{EXPORT_BEGIN_MARKER}\n{rendered}\n{EXPORT_END_MARKER}"

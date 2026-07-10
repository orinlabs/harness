"""Memory backend protocol + factory.

``Harness`` talks to memory through exactly this surface. Formalizing it
as a protocol (instead of hardcoding ``MemoryService``) lets a bundle's
``memory.system`` pick between backends without touching the loop.
``build_memory`` is the single construction point, keyed on
``AgentConfig.memory.system``.

Only one backend exists today (``tiered_sqlite`` -> ``MemoryService``);
the seam is the deliverable, not a second implementation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from harness.config import AgentConfig


@runtime_checkable
class MemoryBackend(Protocol):
    """What the harness loop needs from a memory system.

    Lifecycle: ``open()`` before the loop (owns any storage setup),
    ``flush()``/``close()`` in the loop's finally block. The four loop
    methods match ``MemoryService``'s historical surface.
    """

    def open(self) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...

    def log_messages(self, messages: list[dict[str, Any]], *, ts_ns: int | None = None) -> None: ...

    def update_summaries(self, *, current_time: datetime | None = None) -> Any: ...

    def nudge(self, *, ts_ns: int | None = None) -> None: ...

    def build_llm_inputs(
        self, system_prompt: str, *, current_time: datetime | None = None
    ) -> tuple[str, list[dict[str, Any]]]: ...


def build_memory(config: AgentConfig, *, timezone_name: str) -> MemoryBackend:
    """Construct the memory backend selected by ``config.memory.system``."""
    system = config.memory.system
    if system == "tiered_sqlite":
        from harness.memory.service import MemoryService

        return MemoryService(
            agent_id=config.id,
            model=config.memory.summarizer_model,
            timezone_name=timezone_name,
        )
    raise ValueError(
        f"unknown memory system {system!r} (this harness checkout knows: tiered_sqlite). "
        "The agent's bundle may be newer than the code at the pinned git ref."
    )

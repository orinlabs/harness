"""High-level memory API used by Harness.

Surface:
  - log_messages(messages)           : append OpenAI chat-format dicts, update summaries.
  - build_llm_inputs(system_prompt)  : (rendered_system, recent_messages) for the next LLM call.
  - nudge()                          : append the "use a tool or sleep" user message.

Summarisation runs synchronously inside `log_messages`. Every memory write is
scoped to the per-agent sqlite file opened by `storage.load(agent_id)`.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from harness.core import storage
from harness.memory.context import MemoryContextBuilder
from harness.memory.summarizer import SummarizerUsage, SummaryUpdater

logger = logging.getLogger(__name__)


NUDGE_TEXT = (
    "You did not use any tool on that turn. If you have more work to do, call a "
    "tool now. If you are finished, call the `sleep` tool to stop."
)

RECENT_MESSAGE_LIMIT = 200


class MemoryService:
    def __init__(
        self,
        agent_id: str,
        *,
        model: str = "openai/gpt-4o-mini",
        timezone_name: str = "UTC",
        recent_limit: int = RECENT_MESSAGE_LIMIT,
        summarizer_v2: bool = False,
    ):
        self.agent_id = agent_id
        self.model = model
        self.timezone_name = timezone_name
        self._recent_limit = recent_limit
        # ``summarizer_v2`` flips the summarizer to the hourly-only,
        # end-of-run, past-tense variant. See SummaryUpdater for the
        # behavioral differences. Plumbed from AgentConfig via Harness.
        self._v2 = summarizer_v2
        self._builder = MemoryContextBuilder(timezone=timezone_name)

    @property
    def summarizer_v2(self) -> bool:
        """Whether this service is operating under the v2 summarizer flag."""
        return self._v2

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def log_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        ts_ns: int | None = None,
    ) -> None:
        """Append OpenAI chat-format messages to the log.

        Does **not** trigger summarization; call `update_summaries()` separately
        (the harness does it once per turn at the start). This keeps the call
        graph simple and prevents N redundant summary passes per turn when a
        turn logs N tool-result messages.
        """
        assert storage.db is not None, "storage.load must be called before log_messages"
        if not messages:
            return

        base_ts = ts_ns or time.time_ns()
        rows = []
        for i, msg in enumerate(messages):
            rows.append(
                (
                    str(uuid.uuid4()),
                    base_ts + i,
                    str(msg.get("role") or "user"),
                    json.dumps(msg),
                )
            )
        storage.db.executemany(
            "INSERT INTO messages (id, ts_ns, role, content_json) VALUES (?, ?, ?, ?)",
            rows,
        )

    def update_summaries(self, *, current_time: datetime | None = None) -> SummarizerUsage:
        updater = SummaryUpdater(
            timezone_name=self.timezone_name,
            model=self.model,
            v2=self._v2,
        )
        result = updater.update_all(current_time)
        return result.llm_usage

    def nudge(self, *, ts_ns: int | None = None) -> None:
        """Append the "use a tool or sleep" prompt as a user-role message."""
        self.log_messages(
            [{"role": "user", "content": NUDGE_TEXT}],
            ts_ns=ts_ns,
        )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def build_llm_inputs(
        self,
        system_prompt: str,
        *,
        current_time: datetime | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Return (rendered_system, recent_messages) ready for `llm.complete`.

        `rendered_system` is the user's system prompt with the tiered summary
        block appended. `recent_messages` is the tail of the message log in
        OpenAI chat format.
        """
        assert storage.db is not None
        if current_time is None:
            current_time = datetime.now(tz=UTC)

        data = self._builder.fetch_data(current_time)

        rendered = self._builder.render(data)
        rendered_system = system_prompt
        if rendered:
            rendered_system = f"{system_prompt}\n\n{rendered}"

        messages = [m.content for m in data.messages[-self._recent_limit :]]
        return rendered_system, messages

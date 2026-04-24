"""Simulated user actor for eval simulations.

``UserAgent`` handles deterministic response policies and (optionally)
LLM-powered stochastic replies routed through ``harness.core.llm``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import UserDefinition

logger = logging.getLogger(__name__)


class UserAgent:
    """Simulated user that responds to agent messages."""

    def __init__(
        self,
        user_def: "UserDefinition",
        eval_mode: str = "deterministic",
    ):
        self.profile = user_def
        self.id = user_def.id
        self.name = user_def.name
        self.phone = user_def.phone
        self.email = user_def.email
        self.channels = user_def.channels
        self.response_policy = user_def.response_policy
        self.instructions = user_def.instructions
        # Default to an OpenRouter-compatible model slug. Individual
        # scenarios can override by setting `UserDefinition.model`.
        self.model = user_def.model or "openai/gpt-4o-mini"
        self.eval_mode = eval_mode
        self.conversation_history: list[dict] = []

    def generate_response(self, incoming_message: str, channel: str) -> str | None:
        """Generate a response to an agent message.

        In deterministic mode, matches against response policies.
        In stochastic mode, uses an LLM to generate contextual responses.
        """
        self.conversation_history.append(
            {
                "role": "agent",
                "content": incoming_message,
                "channel": channel,
            }
        )

        if self.eval_mode == "deterministic":
            return self._deterministic_response(incoming_message, channel)
        return self._stochastic_response(incoming_message, channel)

    def _deterministic_response(self, message: str, channel: str) -> str | None:
        """Match message against response policies and return scripted response."""
        if not self.response_policy:
            return None

        message_lower = message.lower()

        for policy in self.response_policy:
            if policy.trigger == "default":
                continue
            if policy.channel and policy.channel != channel:
                continue
            if policy.trigger and policy.trigger.lower() in message_lower:
                self.conversation_history.append(
                    {
                        "role": "user",
                        "content": policy.response,
                        "channel": channel,
                    }
                )
                return policy.response

        keyword_triggers = {
            "scheduling": [
                "schedule",
                "meeting",
                "calendar",
                "appointment",
                "time",
                "available",
                "availability",
                "when",
                "slot",
            ],
            "confirmation": [
                "confirm",
                "confirmed",
                "booked",
                "scheduled",
                "set",
                "all set",
                "done",
                "completed",
            ],
            "availability": [
                "available",
                "availability",
                "when can you",
                "what times",
                "free",
                "open",
            ],
            "conflict": [
                "conflict",
                "no longer works",
                "can't make",
                "reschedule",
                "change",
                "cancel",
            ],
        }

        for policy in self.response_policy:
            if policy.trigger == "default":
                continue
            keywords = keyword_triggers.get(policy.trigger, [])
            if any(kw in message_lower for kw in keywords):
                if policy.channel and policy.channel != channel:
                    continue
                self.conversation_history.append(
                    {
                        "role": "user",
                        "content": policy.response,
                        "channel": channel,
                    }
                )
                return policy.response

        for policy in self.response_policy:
            if policy.trigger == "default":
                if policy.channel and policy.channel != channel:
                    continue
                self.conversation_history.append(
                    {
                        "role": "user",
                        "content": policy.response,
                        "channel": channel,
                    }
                )
                return policy.response

        return None

    _NO_REPLY_SENTINEL = "[NO_REPLY]"

    def _stochastic_response(self, message: str, channel: str) -> str | None:
        """LLM-powered response generation for stochastic mode.

        Uses the harness's OpenRouter client (``harness.core.llm.complete``)
        to produce a short in-character reply. Falls back to the
        deterministic matcher on any error.
        """
        try:
            from harness.core.llm import complete

            history_text = "\n".join(
                f"[{m['role']}] {m['content']}" for m in self.conversation_history[-10:]
            )
            system_prompt = (
                f"You are {self.name}. You are a simulated user in an eval.\n"
                f"Your profile: phone={self.phone}, email={self.email}, "
                f"channels={self.channels}\n\n"
            )
            if self.instructions:
                system_prompt += f"{self.instructions}\n\n"
            system_prompt += (
                f"Respond naturally and briefly as {self.name} would. "
                f"Keep your response to 1-2 sentences.\n\n"
                f"If this message does not warrant a reply (e.g. the agent is "
                f"just confirming something, saying goodbye, or no response is "
                f"needed), respond with exactly: {self._NO_REPLY_SENTINEL}"
            )
            user_prompt = (
                f"Recent conversation:\n{history_text}\n\n"
                f"The agent just sent you this message on {channel}:\n{message}"
            )

            response = complete(
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = (response.text or "").strip()

            if not text or self._NO_REPLY_SENTINEL in text:
                logger.debug("%s chose not to reply (no-op)", self.name)
                return None

            self.conversation_history.append(
                {
                    "role": "user",
                    "content": text,
                    "channel": channel,
                    "stochastic_metadata": {
                        "user_agent_model": self.model,
                    },
                }
            )
            return text
        except Exception:
            logger.exception("Stochastic response generation failed for %s", self.name)
            return self._deterministic_response(message, channel)

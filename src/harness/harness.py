"""Harness: the agent runtime.

Takes an `AgentConfig` + a `run_id` and runs the main loop:

    for turn in range(MAX_TURNS):
        build memory inputs -> call LLM -> log assistant -> dispatch tool calls
        if sleep requested: exit
        if no tool calls: nudge

Every side effect goes through the `core/` modules so the infra platform can
own storage, tracing, LLM routing, and lifecycle without the harness knowing.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from harness.config import AgentConfig
from harness.constants import MAX_TURNS
from harness.context import RunContext, set_agent_id
from harness.core import llm, storage
from harness.core.tracer import (
    SpanType,
    emit_completed_span,
    llm_span,
    text_span,
    tool_span,
)
from harness.memory import MemoryService
from harness.tools import Tool, build_tool_map

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class Harness:
    def __init__(self, config: AgentConfig, run_id: str):
        self.config = config
        self.ctx = RunContext(agent_id=config.id, run_id=run_id)
        self.tool_map: dict[str, Tool] = build_tool_map(config.adapters)
        self.memory = MemoryService(agent_id=config.id, model=config.model)
        # Per-run accumulator: totals (summed) + per-model breakdown (for the
        # run_agent span's final `usage.model_breakdown`).
        self._run_usage: dict[str, int | float] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "llm_calls": 0,
            "total_cost_usd": 0.0,
        }
        self._model_breakdown: dict[str, dict[str, int | float]] = {}

    def run(self) -> None:
        set_agent_id(self.config.id)
        storage.load(self.config.id)
        try:
            with text_span(
                "run_agent",
                agent_id=self.config.id,
                metadata={
                    "agent_id": self.config.id,
                    "run_id": self.ctx.run_id,
                    "model": self.config.model,
                },
            ) as run_span:
                try:
                    for turn in range(MAX_TURNS):
                        self.ctx.turn = turn
                        with text_span(f"turn_{turn}") as turn_span:
                            if not self._step(turn_span):
                                return
                finally:
                    run_span.set_metadata(
                        usage={
                            **self._run_usage,
                            "cost_usd": self._run_usage["total_cost_usd"],
                            "cache_hit_rate": (
                                self._run_usage["cached_tokens"]
                                / self._run_usage["input_tokens"]
                                if self._run_usage["input_tokens"] > 0
                                else 0.0
                            ),
                            "model_breakdown": self._model_breakdown,
                        }
                    )
        finally:
            storage.flush()
            storage.close()

    def _step(self, turn_span) -> bool:
        """Run one turn. Return False if the loop should stop."""
        logger.info("turn %d start (run=%s)", self.ctx.turn, self.ctx.run_id)
        # Roll up any completed buckets into higher-tier summaries exactly once
        # per turn, before building the LLM inputs. Skipped internally (no
        # trace span emitted) when nothing is pending.
        self.memory.update_summaries()

        system, messages = self.memory.build_llm_inputs(self.config.system_prompt)
        tools_schema = [t.schema.to_openai() for t in self.tool_map.values()]
        logger.info(
            "turn %d built inputs: system_chars=%d messages=%d tools=%d",
            self.ctx.turn,
            len(system or ""),
            len(messages),
            len(tools_schema),
        )

        llm_started_at = _now_iso()
        with llm_span(
            "openrouter_api_call",
            metadata={"model": self.config.model, "provider": "openrouter"},
        ) as s:
            s.input(
                json.dumps(
                    {"system": system, "messages": messages, "tools": tools_schema},
                    default=str,
                )[:20000]
            )
            resp = llm.complete(
                model=self.config.model,
                system=system,
                messages=messages,
                tools=tools_schema,
                tool_choice="required",
                reasoning_effort=self.config.reasoning_effort,
            )
            s.output(json.dumps(resp.raw, default=str)[:20000])
            s.set_metadata(llm_cost=resp.usage.to_llm_cost_dict())
        llm_ended_at = _now_iso()

        # If the model returned a plain-text reasoning summary (thinking models
        # like o1 / gpt-5 with `summary: "auto"`), surface it as a sibling
        # `thinking` span under the turn — matches the bedrock convention.
        if resp.reasoning:
            emit_completed_span(
                "thinking",
                span_type=SpanType.TEXT,
                started_at=llm_started_at,
                ended_at=llm_ended_at,
                output=resp.reasoning,
                metadata={"model": self.config.model},
            )

        # Accumulate onto the turn and the run-level rollup.
        turn_span.set_metadata(usage=resp.usage.to_dict())
        self._accumulate_usage(resp.usage)

        assistant_msg = resp.raw["choices"][0]["message"]
        self.memory.log_messages([assistant_msg])

        logger.info(
            "turn %d dispatching %d tool_call(s): %s",
            self.ctx.turn,
            len(resp.tool_calls),
            ", ".join(tc.name for tc in resp.tool_calls) or "-",
        )
        for i, tc in enumerate(resp.tool_calls):
            args_preview = json.dumps(tc.args, default=str)
            if len(args_preview) > 200:
                args_preview = args_preview[:200] + "…"
            logger.info(
                "tool_call[%d/%d] start name=%s id=%s args=%s",
                i + 1,
                len(resp.tool_calls),
                tc.name,
                tc.id,
                args_preview,
            )
            t0 = time.monotonic()
            with tool_span(
                tc.name,
                input=json.dumps({"args": tc.args, "tool_call_id": tc.id}),
            ) as s:
                tool = self.tool_map.get(tc.name)
                if tool is None:
                    logger.warning("model called unknown tool: %s", tc.name)
                    s.set_metadata(error=f"unknown_tool:{tc.name}")
                    result_text = f"Unknown tool: {tc.name!r}"
                else:
                    try:
                        result = tool.call(tc.args, self.ctx)
                        result_text = result.text
                        s.output(result_text)
                    except Exception as e:
                        logger.exception("tool %s raised", tc.name)
                        s.set_metadata(error=f"{type(e).__name__}: {e}")
                        result_text = f"Tool {tc.name} raised: {type(e).__name__}: {e}"
            logger.info(
                "tool_call[%d/%d] done name=%s elapsed=%.2fs result_chars=%d",
                i + 1,
                len(resp.tool_calls),
                tc.name,
                time.monotonic() - t0,
                len(result_text),
            )

            self.memory.log_messages(
                [{"role": "tool", "tool_call_id": tc.id, "content": result_text}]
            )

        if self.ctx.sleep_requested:
            logger.info("turn %d sleep_requested -> exiting loop", self.ctx.turn)
            return False

        if not resp.tool_calls:
            logger.info("turn %d no tool_calls -> nudging memory", self.ctx.turn)
            self.memory.nudge()

        return True

    def _accumulate_usage(self, usage) -> None:
        self._run_usage["input_tokens"] += usage.prompt_tokens
        self._run_usage["output_tokens"] += usage.completion_tokens
        self._run_usage["cached_tokens"] += usage.cached_tokens
        self._run_usage["reasoning_tokens"] += usage.reasoning_tokens
        self._run_usage["total_tokens"] += usage.total_tokens
        self._run_usage["llm_calls"] += usage.llm_calls
        self._run_usage["total_cost_usd"] += usage.total_cost

        per_model = self._model_breakdown.setdefault(
            usage.model or self.config.model,
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "cost_usd": 0.0,
                "llm_calls": 0,
            },
        )
        per_model["input_tokens"] += usage.prompt_tokens
        per_model["output_tokens"] += usage.completion_tokens
        per_model["cached_tokens"] += usage.cached_tokens
        per_model["reasoning_tokens"] += usage.reasoning_tokens
        per_model["cost_usd"] += usage.total_cost
        per_model["llm_calls"] += usage.llm_calls

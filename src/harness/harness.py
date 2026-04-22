"""Harness: the agent runtime.

Takes an `AgentConfig` + a `run_id` and runs the main loop:

    for turn in range(MAX_TURNS):
        build memory inputs -> call LLM -> log assistant -> dispatch tool calls
        if sleep requested: exit
        if no tool calls: nudge

Every side effect goes through the `core/` modules so the infra platform can own
storage, tracing, LLM routing, and lifecycle without the harness knowing.
"""
from __future__ import annotations

import logging
from dataclasses import asdict

from harness.config import AgentConfig
from harness.constants import MAX_TURNS
from harness.context import RunContext, set_agent_id
from harness.core import llm, storage, tracer
from harness.memory import MemoryService
from harness.tools import Tool, build_tool_map

logger = logging.getLogger(__name__)


class Harness:
    def __init__(self, config: AgentConfig, run_id: str):
        self.config = config
        self.ctx = RunContext(agent_id=config.id, run_id=run_id)
        self.tool_map: dict[str, Tool] = build_tool_map(config.adapters)
        self.memory = MemoryService(agent_id=config.id, model=config.model)

    def run(self) -> None:
        set_agent_id(self.config.id)
        storage.load(self.config.id)
        try:
            with tracer.span(
                "run",
                agent_id=self.config.id,
                run_id=self.ctx.run_id,
                model=self.config.model,
            ):
                for turn in range(MAX_TURNS):
                    self.ctx.turn = turn
                    with tracer.span(f"turn_{turn}", turn=turn):
                        if not self._step():
                            return
        finally:
            storage.flush()
            storage.close()

    def _step(self) -> bool:
        """Run one turn. Return False if the loop should stop."""
        system, messages = self.memory.build_llm_inputs(self.config.system_prompt)
        tools_schema = [t.schema.to_openai() for t in self.tool_map.values()]

        with tracer.span("llm_call", model=self.config.model) as s:
            resp = llm.complete(
                model=self.config.model,
                system=system,
                messages=messages,
                tools=tools_schema,
                tool_choice="required",
                reasoning_effort=self.config.reasoning_effort,
            )
            s.set("usage", asdict(resp.usage))
            s.set("finish_reason", resp.finish_reason)

        assistant_msg = resp.raw["choices"][0]["message"]
        self.memory.log_messages([assistant_msg])

        for tc in resp.tool_calls:
            with tracer.span("tool_call", tool_name=tc.name, tool_call_id=tc.id) as s:
                tool = self.tool_map.get(tc.name)
                if tool is None:
                    logger.warning("model called unknown tool: %s", tc.name)
                    s.set("error", "unknown_tool")
                    result_text = f"Unknown tool: {tc.name!r}"
                else:
                    try:
                        result = tool.call(tc.args, self.ctx)
                        result_text = result.text
                        s.set("ok", True)
                    except Exception as e:
                        logger.exception("tool %s raised", tc.name)
                        s.set("error", f"{type(e).__name__}: {e}")
                        result_text = f"Tool {tc.name} raised: {type(e).__name__}: {e}"

            self.memory.log_messages(
                [{"role": "tool", "tool_call_id": tc.id, "content": result_text}]
            )

        if self.ctx.sleep_requested:
            return False

        if not resp.tool_calls:
            self.memory.nudge()

        return True

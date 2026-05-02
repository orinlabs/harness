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

import base64
import json
import logging
import time
from datetime import UTC, datetime

from harness.config import AgentConfig
from harness.constants import MAX_TURNS
from harness.context import RunContext, set_agent_id
from harness.core import llm, storage, tracer
from harness.core.runtime import AgentRuntime
from harness.core.tracer import (
    SpanType,
    emit_completed_span,
    llm_span,
    text_span,
    tool_span,
)
from harness.core.tracing import TraceSink
from harness.memory import MemoryService
from harness.tools import Tool, build_tool_map

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


# ---------------------------------------------------------------------------
# Tool-result image plumbing
# ---------------------------------------------------------------------------
#
# ``ExternalTool`` parses the wire response into ``ToolResult(text, images)``
# where ``images`` is a list of base64-encoded image strings (see
# ``tools/external.py:_parse_success``). Before this helper landed, the
# main loop only logged ``result.text`` into memory and silently discarded
# ``result.images`` — every subsequent LLM turn saw ``image_tokens: 0`` and
# the model would loop trying to "open" the image it had supposedly
# already received.
#
# OpenAI Chat Completions (and OpenRouter, which speaks the same shape) is
# strict about ``role: tool`` content: many providers only accept a string
# there, never multipart parts. So we don't try to put the image inside the
# tool message. Instead we log the tool message with text only — preserving
# the ``tool_call_id`` pairing every provider validates — and then append
# a synthetic ``role: user`` message right after it whose ``content`` is
# OpenAI multipart (``[{"type": "text"}, {"type": "image_url"}]``). Vision-
# capable models (Claude, GPT-4o, Gemini) accept images in ``user`` content
# universally, so the image reaches the model on the very next turn.

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
_GIF87A = b"GIF87a"
_GIF89A = b"GIF89a"
_RIFF = b"RIFF"
_WEBP = b"WEBP"


def _sniff_image_mime(image_bytes: bytes) -> str:
    """Pick a MIME type for an image based on its leading magic bytes.

    Provider vision endpoints generally tolerate a mismatched ``data:`` URL
    media type (they sniff bytes themselves), but tagging accurately keeps
    the trace readable and lets pickier providers do less work. Falls back
    to ``image/png`` when the bytes don't match a known signature — that's
    what the bedrock-api invoke endpoint produces today for the images it
    forwards through ``image_to_image_block``.
    """
    if image_bytes.startswith(_PNG_MAGIC):
        return "image/png"
    if image_bytes.startswith(_JPEG_MAGIC):
        return "image/jpeg"
    if image_bytes.startswith(_GIF87A) or image_bytes.startswith(_GIF89A):
        return "image/gif"
    # WebP: ``RIFF????WEBP``.
    if (
        len(image_bytes) >= 12
        and image_bytes[:4] == _RIFF
        and image_bytes[8:12] == _WEBP
    ):
        return "image/webp"
    return "image/png"


def _build_image_followup_message(
    *, tool_name: str, images: list[str]
) -> dict | None:
    """Build a synthetic ``role: user`` message carrying tool-emitted images.

    Returns ``None`` if there are no usable images so callers can short-
    circuit before logging anything. Bad payloads (non-strings, undecodable
    base64) are skipped individually and logged at warn so a partial batch
    still delivers the images we *can* decode.
    """
    parts: list[dict] = []
    for raw in images or []:
        if not isinstance(raw, str) or not raw:
            continue
        try:
            decoded = base64.b64decode(raw, validate=False)
        except Exception:  # noqa: BLE001
            logger.warning(
                "tool %s emitted image bytes that aren't valid base64; skipping",
                tool_name,
            )
            continue
        mime = _sniff_image_mime(decoded)
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{raw}"},
            }
        )

    if not parts:
        return None

    label_text = (
        f"[Image attachment from tool '{tool_name}'. View it as if "
        "you had received it directly from the user.]"
    )
    return {
        "role": "user",
        "content": [{"type": "text", "text": label_text}, *parts],
    }


class Harness:
    def __init__(
        self,
        config: AgentConfig,
        run_id: str,
        *,
        trace_sink: TraceSink | None = None,
        runtime: AgentRuntime | None = None,
    ):
        self.config = config
        # Pick defaults from the environment when the caller didn't pass
        # explicit dependencies. autoconfigure() returns Bedrock-backed
        # implementations when BEDROCK_URL + BEDROCK_TOKEN are set, else
        # NullTraceSink + LocalAgentRuntime (standalone mode).
        if trace_sink is None or runtime is None:
            from harness.cloud.autoconfig import autoconfigure

            auto_sink, auto_runtime = autoconfigure()
            trace_sink = trace_sink or auto_sink
            runtime = runtime or auto_runtime
        self._trace_sink = trace_sink
        self._runtime = runtime
        tracer.set_trace_sink(trace_sink)

        self.ctx = RunContext(agent_id=config.id, run_id=run_id, runtime=runtime)
        self.tool_map: dict[str, Tool] = build_tool_map(config.tools)
        # Expose the tool map on the per-run context so built-in tools can
        # invoke siblings (e.g. sleep -> list_notifications pre-flight check).
        self.ctx.tool_map = self.tool_map
        logger.info(
            "Harness init: agent=%s run=%s tool_map has %d tool(s): %s",
            config.id,
            run_id,
            len(self.tool_map),
            sorted(self.tool_map.keys()),
        )
        # Summarization always runs on a cheap model, not the agent's
        # configured model. Otherwise every turn's `update_summaries()`
        # fires N summary-generation LLM calls at whatever the agent
        # happens to be using (Opus, Sonnet, etc.) -- easily >$1/turn on
        # agents with deep history. gpt-5-nano is ~1000x cheaper per
        # token and the summary quality is more than adequate for
        # timeline rollups.
        summary_model = "openai/gpt-5-nano"
        logger.info(
            "Harness init: using summary_model=%s (agent model=%s)",
            summary_model,
            config.model,
        )
        # ``summarizer_v2`` may arrive from either source: the legacy
        # explicit AgentConfig.summarizer_v2 bool (older YAMLs) or the
        # generic feature_flags dict (Bedrock + new YAMLs). Either path
        # turns the v2 path on; both off (the default) keeps the legacy
        # cascade-summarizer behavior.
        summarizer_v2_enabled = bool(getattr(config, "summarizer_v2", False)) or config.is_enabled(
            "summarizer_v2"
        )
        self.memory = MemoryService(
            agent_id=config.id,
            model=summary_model,
            summarizer_v2=summarizer_v2_enabled,
        )
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
        interrupted = False
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
                except BaseException as exc:  # noqa: BLE001
                    # SystemExit (SIGTERM from a supervisor's /stop or a
                    # supersede), KeyboardInterrupt, and anything else
                    # that isn't Exception. Flag the run so the finally
                    # block can skip the end-of-run summarization (we
                    # don't want to stretch shutdown with an LLM call)
                    # and let the signal propagate. The outer finally
                    # still flushes storage so messages written before
                    # the interrupt are durable.
                    interrupted = True
                    logger.warning(
                        "run_agent interrupted by %s (agent=%s, run=%s)",
                        type(exc).__name__,
                        self.config.id,
                        self.ctx.run_id,
                    )
                    raise
                finally:
                    final_usage = {
                        **self._run_usage,
                        "cost_usd": self._run_usage["total_cost_usd"],
                        "cache_hit_rate": (
                            self._run_usage["cached_tokens"] / self._run_usage["input_tokens"]
                            if self._run_usage["input_tokens"] > 0
                            else 0.0
                        ),
                        "model_breakdown": self._model_breakdown,
                    }
                    run_span.set_metadata(usage=final_usage)
                    # Mirror the usage totals to the logger so standalone
                    # runs (NullTraceSink) still surface cost -- tracing is
                    # the only other path and it's a no-op without a backend.
                    logger.info(
                        "run_agent usage: agent=%s run=%s calls=%d "
                        "tokens=in/%d+out/%d+cache/%d+reason/%d cost=$%.6f models=%s",
                        self.config.id,
                        self.ctx.run_id,
                        int(final_usage["llm_calls"]),
                        int(final_usage["input_tokens"]),
                        int(final_usage["output_tokens"]),
                        int(final_usage["cached_tokens"]),
                        int(final_usage["reasoning_tokens"]),
                        float(final_usage["total_cost_usd"]),
                        list(self._model_breakdown.keys()),
                    )
        finally:
            # v2 defers all summarization to end-of-run so the summary
            # describes completed actions, not in-flight state. Skipped
            # on the interrupted path -- SIGTERM should propagate
            # without being stretched by an LLM call, and the
            # platform-side wake-drain is the backstop for state the
            # agent left pending. Wrapped in try so a summarization
            # failure can never block the storage flush below.
            if self.memory.summarizer_v2 and not interrupted:
                try:
                    self.memory.update_summaries()
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "End-of-run summarization failed for agent %s "
                        "(run=%s) -- storage will still flush",
                        self.config.id,
                        self.ctx.run_id,
                    )
            storage.flush()
            storage.close()

    def _step(self, turn_span) -> bool:
        """Run one turn. Return False if the loop should stop."""
        logger.info("turn %d start (run=%s)", self.ctx.turn, self.ctx.run_id)
        # V1 rolls up any completed buckets into higher-tier summaries
        # exactly once per turn, before building the LLM inputs. Skipped
        # internally (no trace span emitted) when nothing is pending.
        #
        # V2 defers all summarization to end-of-run (see Harness.run's
        # finally block). Rationale: mid-run state ("I am waiting for
        # Mike's photo") routinely became a stale summary that a future
        # run read as current state. End-of-run is the first moment
        # where what happened is a stable fact.
        if not self.memory.summarizer_v2:
            self.memory.update_summaries()

        system, messages = self.memory.build_llm_inputs(self.config.system_prompt)
        # Anthropic extended thinking is sensitive to the message tail. On
        # turn 0 a bare system prompt returns reasoning_tokens=0, and
        # OpenRouter Claude post-tool continuations often execute correctly
        # but emit no fresh thinking unless the next request starts a normal
        # user continuation. Inject an ephemeral user message for the LLM call
        # only (NOT written to memory) in those cases.
        tail_role = messages[-1].get("role") if messages else None
        is_anthropic_reasoning = (
            self.config.model.startswith("claude-")
            or self.config.model.startswith("anthropic/")
        ) and self.config.reasoning_effort != "none"
        needs_kickoff = (
            not messages
            or tail_role == "assistant"
            or (tail_role == "tool" and is_anthropic_reasoning)
        )
        if needs_kickoff:
            messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Proceed with your next action. Think through it, "
                        "then call the appropriate tool."
                    ),
                },
            ]
        logger.info(
            "turn %d: self.tool_map has %d tool(s) before schema build: %s",
            self.ctx.turn,
            len(self.tool_map),
            sorted(self.tool_map.keys()),
        )
        tools_schema = [t.schema.to_openai() for t in self.tool_map.values()]
        logger.info(
            "turn %d built inputs: system_chars=%d messages=%d (kickoff_injected=%s) tools=%d",
            self.ctx.turn,
            len(system or ""),
            len(messages),
            needs_kickoff,
            len(tools_schema),
        )
        # Defense: tool_map was populated at init but the schema list came
        # out empty. Refuse to call OpenRouter -- an empty `tools` list with
        # no tool_choice forces the model to plain-chat and the harness loop
        # happily re-calls forever, burning money. Fail loudly instead.
        if self.tool_map and not tools_schema:
            raise RuntimeError(
                f"tools_schema is empty despite tool_map having "
                f"{len(self.tool_map)} tool(s); refusing to call LLM. "
                f"tool_map keys={sorted(self.tool_map.keys())}"
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
            try:
                resp = llm.complete(
                    model=self.config.model,
                    system=system,
                    messages=messages,
                    tools=tools_schema,
                    # Intentionally `auto`, not `required`. OpenAI's
                    # `tool_choice: "required"` maps to Anthropic's
                    # `any`/`tool` modes on OpenRouter, and Anthropic
                    # disables extended thinking whenever tool use is
                    # forced -- our Claude 4.x runs were coming back
                    # with reasoning_tokens=0 every turn because of
                    # this, no matter how much we tweaked the
                    # `reasoning` block.
                    #
                    # `auto` lets the model think first and still pick
                    # a tool; when it doesn't call one, the nudge path
                    # below (`self.memory.nudge()`) prompts it to call
                    # a tool or sleep on the next turn. That matches
                    # the ergonomics of every modern agent SDK and
                    # costs at most one wasted turn per run.
                    tool_choice="auto",
                    reasoning_effort=self.config.reasoning_effort,
                    max_tokens=self.config.max_tokens,
                )
            except llm.OpenRouterError as e:
                # Surface the upstream body in the span's output_text so the
                # trace UI shows the rejection payload next to the request,
                # not just as a stringified error on the span itself.
                s.output(e.body)
                s.set_metadata(
                    openrouter_status=e.status_code,
                    openrouter_error_body=e.body,
                )
                raise
            s.output(json.dumps(resp.raw, default=str)[:20000])
            # Surface reasoning on the llm_span itself, not just as a nested
            # sibling span. Two reasons:
            #   1. `s.output` is `resp.raw` truncated at 20kB. Anthropic
            #      streams reasoning as per-token `reasoning_details`
            #      deltas (one dict per ~1 token), so raw JSON routinely
            #      blows past 20kB on thinking models and the reasoning
            #      text gets chopped off the end of the output field.
            #   2. Clients viewing the span in Bedrock expect a first-
            #      class `reasoning` / `reasoning_tokens` on the LLM call
            #      metadata, not to dig into a sibling span's output.
            reasoning_tokens = resp.usage.reasoning_tokens
            llm_metadata: dict = {"llm_cost": resp.usage.to_llm_cost_dict()}
            if resp.reasoning:
                llm_metadata["reasoning"] = resp.reasoning
            if reasoning_tokens > 0:
                llm_metadata["reasoning_tokens"] = reasoning_tokens
            s.set_metadata(**llm_metadata)
        llm_ended_at = _now_iso()

        # Emit a sibling `thinking` span under the turn whenever the model
        # did any reasoning work -- either returned plaintext (Anthropic,
        # Gemini, OpenAI with `summary: "auto"`) or just a reasoning-token
        # count (OpenAI when the raw thinking stays encrypted). That way
        # the trace always shows a visible span right after
        # `openrouter_api_call` confirming reasoning actually happened,
        # not just silent tokens buried in the usage dict.
        if resp.reasoning or reasoning_tokens > 0:
            logger.info(
                "emitting thinking span: reasoning_tokens=%d has_plaintext=%s text_chars=%d",
                reasoning_tokens,
                bool(resp.reasoning),
                len(resp.reasoning or ""),
            )
            emit_completed_span(
                "thinking",
                span_type=SpanType.TEXT,
                started_at=llm_started_at,
                ended_at=llm_ended_at,
                output=resp.reasoning or "(reasoning tokens used; no plaintext returned)",
                metadata={
                    "model": self.config.model,
                    "reasoning_tokens": reasoning_tokens,
                    "has_plaintext": bool(resp.reasoning),
                },
            )
        else:
            logger.info(
                "no thinking span emitted: reasoning_tokens=0 and no "
                "plaintext reasoning from model=%s",
                self.config.model,
            )

        # Accumulate onto the turn and the run-level rollup.
        turn_span.set_metadata(usage=resp.usage.to_dict())
        self._accumulate_usage(resp.usage)

        assistant_msg = llm._strip_provider_reasoning([resp.raw["choices"][0]["message"]])[0]
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
            result_images: list[str] | None = None
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
                        result_images = result.images
                        s.output(result_text)
                        if result_images:
                            # Surface in the trace that the tool produced
                            # image bytes — silent image discard was the
                            # exact symptom we were chasing before this
                            # plumbing landed, so make it visible.
                            s.set_metadata(image_count=len(result_images))
                    except Exception as e:
                        logger.exception("tool %s raised", tc.name)
                        s.set_metadata(error=f"{type(e).__name__}: {e}")
                        result_text = f"Tool {tc.name} raised: {type(e).__name__}: {e}"
            logger.info(
                "tool_call[%d/%d] done name=%s elapsed=%.2fs result_chars=%d "
                "image_count=%d",
                i + 1,
                len(resp.tool_calls),
                tc.name,
                time.monotonic() - t0,
                len(result_text),
                len(result_images) if result_images else 0,
            )

            # Always log the tool message first so the ``tool_call_id``
            # pairing every provider validates is intact.
            self.memory.log_messages(
                [{"role": "tool", "tool_call_id": tc.id, "content": result_text}]
            )

            # If the tool emitted image bytes, append a synthetic user
            # message carrying them so the next LLM turn actually sees
            # the pixels. See ``_build_image_followup_message`` for the
            # rationale (provider tool-content schemas are too restrictive
            # to put images in the tool message itself).
            if result_images:
                followup = _build_image_followup_message(
                    tool_name=tc.name,
                    images=result_images,
                )
                if followup is not None:
                    self.memory.log_messages([followup])

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

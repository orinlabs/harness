"""Hierarchical summarisation: 1m -> 5m -> hourly -> daily -> weekly -> monthly.

Each tier reads the tier below (raw messages for 1m, otherwise summaries from
the next-finer tier), groups into bucket keys, and writes one summary per
completed bucket via an LLM call. Buckets still in progress (the current
minute / hour / day / week / month) are skipped so we never produce partial
summaries.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from harness.core import llm, storage
from harness.core.tracer import llm_span, text_span
from harness.memory.bucketing import hour_start, last_completed_1m_end
from harness.memory.marks import force_timezone, week_start_sunday
from harness.memory.types import PERIOD_META, PeriodType


logger = logging.getLogger(__name__)


@dataclass
class SummarizerUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    llm_calls: int = 0


@dataclass
class UpdateAllResult:
    one_minute_created: int = 0
    five_minute_created: int = 0
    hourly_created: int = 0
    daily_created: int = 0
    weekly_created: int = 0
    monthly_created: int = 0
    llm_usage: SummarizerUsage = field(default_factory=SummarizerUsage)


def _now_ns() -> int:
    import time as _t

    return _t.time_ns()


def _dt_to_ns(dt: datetime) -> int:

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1_000_000_000)


class SummaryUpdater:
    """Run sync summarisation across all six tiers in order:
    1m -> 5m -> hourly -> daily -> weekly -> monthly.

    When ``v2=True``, the cascade is flattened: no 1m / 5m tiers (raw
    messages fill that window), summarization is deferred to end-of-run
    by the caller (Harness), and the summarization prompt is constrained
    to past-tense actions only -- no "waiting for X" / "pending Y"
    state-describing phrasing that stale summaries were turning into
    current-state assertions on the next run.

    The v2 cascade still produces hourly, daily, weekly, monthly summaries;
    it only changes where they are computed from (raw messages instead of
    pre-aggregated 5m summaries) and what the prompt allows them to say.
    """

    def __init__(
        self,
        timezone_name: str = "UTC",
        model: str = "openai/gpt-4o-mini",
        *,
        v2: bool = False,
    ):
        self.timezone_name = timezone_name
        self.model = model
        self.v2 = v2
        self.total_usage = SummarizerUsage()

    def update_all(self, current_time: datetime | None = None) -> UpdateAllResult:
        if current_time is None:
            current_time = datetime.now().astimezone()
        self.total_usage = SummarizerUsage()

        logger.info(
            "summarizer.update_all: starting model=%s v2=%s",
            self.model,
            self.v2,
        )
        # Wrap the whole tier cascade in a single `memory_summarization`
        # text span. Per-tier work opens child `summarize_<tier>` spans
        # (only when there's pending work) and each LLM call opens an
        # `llm` span underneath. Nests under whatever span is active
        # when `update_all` is called (usually `turn_N`, or `run_agent`
        # for the v2 end-of-run path).
        with text_span(
            "memory_summarization",
            metadata={"model": self.model, "v2": self.v2},
        ) as parent_span:
            if self.v2:
                # v2 skips the 1m and 5m tiers entirely: raw messages
                # fill that window. Hourly summaries are built directly
                # from messages by ``_update_hourly_summaries`` when v2
                # is on (see the v2 branch inside that method for the
                # raw-messages-source path). The remaining tiers roll
                # up off completed hourly summaries as usual.
                updated_1m: list = []
                updated_5m: list = []
            else:
                updated_1m = self._update_one_minute_summaries(current_time)
                updated_5m = self._update_five_minute_summaries(current_time)
            updated_hour = self._update_hourly_summaries(current_time)
            updated_day = self._update_daily_summaries(current_time)
            updated_week = self._update_weekly_summaries(current_time)
            updated_month = self._update_monthly_summaries(current_time)

            summaries_created = {
                "one_minute": len(updated_1m),
                "five_minute": len(updated_5m),
                "hourly": len(updated_hour),
                "daily": len(updated_day),
                "weekly": len(updated_week),
                "monthly": len(updated_month),
            }
            parent_span.set_metadata(
                summaries_created=summaries_created,
                usage={
                    "input_tokens": self.total_usage.input_tokens,
                    "output_tokens": self.total_usage.output_tokens,
                    "total_cost_usd": self.total_usage.total_cost,
                    "llm_calls": self.total_usage.llm_calls,
                    "model": self.model,
                },
            )

        logger.info(
            "summarizer.update_all: done created 1m=%d 5m=%d hour=%d day=%d "
            "week=%d month=%d llm_calls=%d in=%d out=%d cost=$%.5f model=%s",
            len(updated_1m),
            len(updated_5m),
            len(updated_hour),
            len(updated_day),
            len(updated_week),
            len(updated_month),
            self.total_usage.llm_calls,
            self.total_usage.input_tokens,
            self.total_usage.output_tokens,
            self.total_usage.total_cost,
            self.model,
        )

        return UpdateAllResult(
            one_minute_created=len(updated_1m),
            five_minute_created=len(updated_5m),
            hourly_created=len(updated_hour),
            daily_created=len(updated_day),
            weekly_created=len(updated_week),
            monthly_created=len(updated_month),
            llm_usage=self.total_usage,
        )

    # ------------------------------------------------------------------
    # 1-minute: raw messages -> one summary per completed minute
    # ------------------------------------------------------------------

    def _update_one_minute_summaries(self, current_time: datetime) -> list[tuple]:
        assert storage.db is not None
        db = storage.db

        timezone_now = force_timezone(current_time, self.timezone_name)
        cutoff = last_completed_1m_end(timezone_now)

        existing_keys = {
            (row["date"], row["hour"], row["minute"])
            for row in db.execute(
                "SELECT date, hour, minute FROM one_minute_summaries"
            )
        }
        existing_5m_keys = {
            (row["date"], row["hour"], row["minute"])
            for row in db.execute(
                "SELECT date, hour, minute FROM five_minute_summaries"
            )
        }

        # Message filter: ts < cutoff, optionally lower-bounded by latest summary
        # to avoid expensive full-history scans.
        lower_ns: int | None = None
        if existing_keys:
            latest_key = max(existing_keys)
            earliest_fetch = force_timezone(
                datetime.combine(
                    date.fromisoformat(latest_key[0]),
                    time(latest_key[1], latest_key[2]),
                ),
                self.timezone_name,
            )
            lower_ns = _dt_to_ns(earliest_fetch)
        elif existing_5m_keys:
            latest_5m = max(existing_5m_keys)
            earliest_fetch = (
                force_timezone(
                    datetime.combine(
                        date.fromisoformat(latest_5m[0]),
                        time(latest_5m[1], latest_5m[2]),
                    ),
                    self.timezone_name,
                )
                + timedelta(minutes=5)
            )
            lower_ns = _dt_to_ns(earliest_fetch)

        if lower_ns is not None:
            rows = db.execute(
                "SELECT ts_ns, content_json FROM messages "
                "WHERE ts_ns < ? AND ts_ns >= ? ORDER BY ts_ns",
                (_dt_to_ns(cutoff), lower_ns),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT ts_ns, content_json FROM messages WHERE ts_ns < ? ORDER BY ts_ns",
                (_dt_to_ns(cutoff),),
            ).fetchall()

        buckets: dict[datetime, list[dict]] = {}
        for row in rows:
            local_time = force_timezone(
                datetime.fromtimestamp(row["ts_ns"] / 1_000_000_000), self.timezone_name
            )
            bucket_start = last_completed_1m_end(local_time)
            bucket_end = bucket_start + timedelta(minutes=1)
            if bucket_end > timezone_now:
                continue
            buckets.setdefault(bucket_start, []).append(json.loads(row["content_json"]))

        pending: list[tuple[tuple[str, int, int], list[dict]]] = []
        for bucket_start, bucket_messages in buckets.items():
            key = (bucket_start.date().isoformat(), bucket_start.hour, bucket_start.minute)
            if key in existing_keys:
                continue
            five_min_key = (
                bucket_start.date().isoformat(),
                bucket_start.hour,
                (bucket_start.minute // 5) * 5,
            )
            if five_min_key in existing_5m_keys:
                continue
            pending.append((key, bucket_messages))

        if not pending:
            return []

        logger.info(
            "summarizer: tier=1m pending=%d -> making %d LLM call(s) via %s",
            len(pending), len(pending), self.model,
        )

        created: list[tuple] = []
        for (date_key, hour_key, minute_key), bucket_messages in sorted(
            pending, key=lambda x: (x[0][0], x[0][1], x[0][2])
        ):
            content = json.dumps(bucket_messages)
            summary_text = self._create_summary(
                content=content, period_type=PeriodType.ONE_MINUTE
            )
            if not summary_text or not summary_text.strip():
                logger.error(
                    "Failed to create 1-minute summary for %s %02d:%02d - empty, skipping",
                    date_key,
                    hour_key,
                    minute_key,
                )
                continue
            self._upsert_summary(
                "one_minute_summaries",
                ("date", "hour", "minute"),
                (date_key, hour_key, minute_key),
                summary_text,
                len(bucket_messages),
            )
            created.append((date_key, hour_key, minute_key))

        return created

    # ------------------------------------------------------------------
    # 5-minute: roll up 1m summaries into 5-minute buckets
    # ------------------------------------------------------------------

    def _update_five_minute_summaries(self, current_time: datetime) -> list[tuple]:
        assert storage.db is not None
        db = storage.db

        timezone_now = force_timezone(current_time, self.timezone_name)

        one_min_rows = db.execute(
            "SELECT date, hour, minute, summary, message_count FROM one_minute_summaries "
            "ORDER BY date, hour, minute"
        ).fetchall()

        five_min_groups: dict[tuple[str, int, int], list[dict]] = {}
        for r in one_min_rows:
            bucket_minute = (r["minute"] // 5) * 5
            key = (r["date"], r["hour"], bucket_minute)
            five_min_groups.setdefault(key, []).append(dict(r))

        existing_keys = {
            (row["date"], row["hour"], row["minute"])
            for row in db.execute(
                "SELECT date, hour, minute FROM five_minute_summaries"
            )
        }

        pending = []
        for (summary_date, hour, minute), summaries_1m in five_min_groups.items():
            bucket_start = force_timezone(
                datetime.combine(date.fromisoformat(summary_date), time(hour, minute)),
                self.timezone_name,
            )
            if bucket_start + timedelta(minutes=5) > timezone_now:
                continue
            if (summary_date, hour, minute) not in existing_keys and summaries_1m:
                parts = [
                    f"[{s['hour']:02d}:{s['minute']:02d}] {s['summary']}"
                    for s in summaries_1m
                ]
                combined_content = "\n\n".join(parts)
                total_messages = sum(s["message_count"] for s in summaries_1m)
                pending.append(((summary_date, hour, minute), combined_content, total_messages))

        if not pending:
            return []

        logger.info(
            "summarizer: tier=5m pending=%d -> making %d LLM call(s) via %s",
            len(pending), len(pending), self.model,
        )
        created = []
        for (date_key, hour_key, minute_key), content, total_messages in sorted(
            pending, key=lambda x: (x[0][0], x[0][1], x[0][2])
        ):
            summary_text = self._create_summary(
                content=content, period_type=PeriodType.FIVE_MINUTE
            )
            if not summary_text or not summary_text.strip():
                continue
            self._upsert_summary(
                "five_minute_summaries",
                ("date", "hour", "minute"),
                (date_key, hour_key, minute_key),
                summary_text,
                total_messages,
            )
            created.append((date_key, hour_key, minute_key))

        return created

    # ------------------------------------------------------------------
    # Hourly: roll up 5-minute summaries into hour buckets
    # ------------------------------------------------------------------

    def _update_hourly_summaries(self, current_time: datetime) -> list[tuple]:
        assert storage.db is not None
        db = storage.db

        timezone_now = force_timezone(current_time, self.timezone_name)
        cutoff = hour_start(timezone_now)

        five_min_rows = db.execute(
            "SELECT date, hour, minute, summary, message_count FROM five_minute_summaries "
            "ORDER BY date, hour, minute"
        ).fetchall()

        hourly_groups: dict[tuple[str, int], list[dict]] = {}
        for r in five_min_rows:
            hourly_groups.setdefault((r["date"], r["hour"]), []).append(dict(r))

        existing_keys = {
            (row["date"], row["hour"])
            for row in db.execute("SELECT date, hour FROM hourly_summaries")
        }

        cutoff_date = cutoff.date().isoformat()
        pending = []
        for (summary_date, hour), summaries_5m in hourly_groups.items():
            if summary_date == cutoff_date and hour >= cutoff.hour:
                continue
            if (summary_date, hour) not in existing_keys and summaries_5m:
                parts = [
                    f"[{s['hour']:02d}:{s['minute']:02d}] {s['summary']}"
                    for s in summaries_5m
                ]
                combined_content = "\n\n".join(parts)
                total_messages = sum(s["message_count"] for s in summaries_5m)
                pending.append(((summary_date, hour), combined_content, total_messages))

        if not pending:
            return []

        logger.info(
            "summarizer: tier=hourly pending=%d -> making %d LLM call(s) via %s",
            len(pending), len(pending), self.model,
        )
        created = []
        for (summary_date, hour), content, total_messages in sorted(
            pending, key=lambda x: (x[0][0], x[0][1])
        ):
            summary_text = self._create_summary(content=content, period_type=PeriodType.HOURLY)
            if not summary_text or not summary_text.strip():
                continue
            self._upsert_summary(
                "hourly_summaries",
                ("date", "hour"),
                (summary_date, hour),
                summary_text,
                total_messages,
            )
            created.append((summary_date, hour))

        return created

    # ------------------------------------------------------------------
    # Daily: roll up hourly summaries into days (only completed days)
    # ------------------------------------------------------------------

    def _update_daily_summaries(self, current_time: datetime) -> list[str]:
        assert storage.db is not None
        db = storage.db

        timezone_now = force_timezone(current_time, self.timezone_name)
        today = timezone_now.date().isoformat()

        existing_keys = {
            row["date"]
            for row in db.execute("SELECT date FROM daily_summaries")
        }

        all_hourly = db.execute(
            "SELECT date, hour, summary, message_count FROM hourly_summaries "
            "WHERE date < ? ORDER BY date, hour",
            (today,),
        ).fetchall()

        hourly_by_day: dict[str, list[dict]] = {}
        for r in all_hourly:
            hourly_by_day.setdefault(r["date"], []).append(dict(r))

        pending = []
        for day, hour_summaries in hourly_by_day.items():
            if day not in existing_keys:
                parts = [f"[{h['hour']:02d}:00] {h['summary']}" for h in hour_summaries]
                combined_content = "\n\n".join(parts)
                total_messages = sum(h["message_count"] for h in hour_summaries)
                pending.append((day, combined_content, total_messages))

        if not pending:
            return []

        logger.info(
            "summarizer: tier=daily pending=%d -> making %d LLM call(s) via %s",
            len(pending), len(pending), self.model,
        )
        created = []
        for day, content, total_messages in sorted(pending, key=lambda x: x[0]):
            summary_text = self._create_summary(content=content, period_type=PeriodType.DAILY)
            if not summary_text or not summary_text.strip():
                continue
            self._upsert_summary(
                "daily_summaries",
                ("date",),
                (day,),
                summary_text,
                total_messages,
            )
            created.append(day)

        return created

    # ------------------------------------------------------------------
    # Weekly: roll up daily summaries into weeks (Sunday-based)
    # ------------------------------------------------------------------

    def _update_weekly_summaries(self, current_time: datetime) -> list[str]:
        assert storage.db is not None
        db = storage.db

        timezone_now = force_timezone(current_time, self.timezone_name)
        today = timezone_now.date()
        this_week_start = week_start_sunday(datetime.combine(today, time(0, 0, 0))).date()

        dailies = db.execute(
            "SELECT date, summary, message_count FROM daily_summaries WHERE date < ? ORDER BY date",
            (this_week_start.isoformat(),),
        ).fetchall()

        weekly_groups: dict[str, list[dict]] = {}
        for d in dailies:
            d_date = date.fromisoformat(d["date"])
            week = week_start_sunday(datetime.combine(d_date, time(0, 0, 0))).date().isoformat()
            weekly_groups.setdefault(week, []).append(dict(d))

        existing_keys = {
            row["week_start_date"]
            for row in db.execute("SELECT week_start_date FROM weekly_summaries")
        }

        pending = []
        for week, dailies_in_week in weekly_groups.items():
            if week not in existing_keys and dailies_in_week:
                parts = [f"[{d['date']}] {d['summary']}" for d in dailies_in_week]
                combined_content = "\n\n".join(parts)
                total_messages = sum(d["message_count"] for d in dailies_in_week)
                pending.append((week, combined_content, total_messages))

        if not pending:
            return []

        logger.info(
            "summarizer: tier=weekly pending=%d -> making %d LLM call(s) via %s",
            len(pending), len(pending), self.model,
        )
        created = []
        for week, content, total_messages in sorted(pending, key=lambda x: x[0]):
            summary_text = self._create_summary(content=content, period_type=PeriodType.WEEKLY)
            if not summary_text or not summary_text.strip():
                continue
            self._upsert_summary(
                "weekly_summaries",
                ("week_start_date",),
                (week,),
                summary_text,
                total_messages,
            )
            created.append(week)

        return created

    # ------------------------------------------------------------------
    # Monthly: roll up weekly summaries into calendar months
    # ------------------------------------------------------------------

    def _update_monthly_summaries(self, current_time: datetime) -> list[tuple]:
        assert storage.db is not None
        db = storage.db

        timezone_now = force_timezone(current_time, self.timezone_name)
        today = timezone_now.date()
        this_month_start = today.replace(day=1).isoformat()

        weeklies = db.execute(
            "SELECT week_start_date, summary, message_count FROM weekly_summaries "
            "WHERE week_start_date < ?",
            (this_month_start,),
        ).fetchall()

        monthly_groups: dict[tuple[int, int], list[dict]] = {}
        for w in weeklies:
            d = date.fromisoformat(w["week_start_date"])
            key = (d.year, d.month)
            monthly_groups.setdefault(key, []).append(dict(w))

        existing_keys = {
            (row["year"], row["month"])
            for row in db.execute("SELECT year, month FROM monthly_summaries")
        }

        pending = []
        for (year, month), weeklies_in_month in monthly_groups.items():
            if (year, month) not in existing_keys and weeklies_in_month:
                parts = [
                    f"[Week of {w['week_start_date']}] {w['summary']}"
                    for w in weeklies_in_month
                ]
                combined_content = "\n\n".join(parts)
                total_messages = sum(w["message_count"] for w in weeklies_in_month)
                pending.append(((year, month), combined_content, total_messages))

        if not pending:
            return []

        logger.info(
            "summarizer: tier=monthly pending=%d -> making %d LLM call(s) via %s",
            len(pending), len(pending), self.model,
        )
        created = []
        for (year, month), content, total_messages in sorted(
            pending, key=lambda x: (x[0][0], x[0][1])
        ):
            summary_text = self._create_summary(content=content, period_type=PeriodType.MONTHLY)
            if not summary_text or not summary_text.strip():
                continue
            self._upsert_summary(
                "monthly_summaries",
                ("year", "month"),
                (year, month),
                summary_text,
                total_messages,
            )
            created.append((year, month))

        return created

    # ------------------------------------------------------------------
    # LLM call + upsert helpers
    # ------------------------------------------------------------------

    def _create_summary(
        self,
        *,
        content: str,
        period_type: PeriodType,
        existing_memory: str | None = None,
    ) -> str | None:
        """Build the summarization prompt and call OpenRouter."""
        meta = PERIOD_META[period_type]

        prompt = (
            f"You are an AI assistant creating a memory summary from {meta.time_period}. "
            'Write in FIRST PERSON ("I worked with...", "I helped with...").\n\n'
            f"Keep your summary to {meta.max_length} maximum.\n\n"
        )

        if existing_memory:
            prompt += (
                "Here is your existing memory context:\n"
                "-- BEGIN EXISTING MEMORY --\n"
                f"{existing_memory}\n"
                "-- END EXISTING MEMORY --\n\n"
            )

        prompt += (
            "For your summary focus on:\n"
            f"{meta.focus}\n\n"
            "CRITICAL: Only use information explicitly stated above to summarize or infer "
            "learnings. Do NOT add details.\n\n"
            "This is the new content that you must summarize:\n"
            f"{content}\n\n"
        )

        # Wrap every summarizer LLM call in an `llm` span so the run's
        # trace tree shows which tier fired, the input prompt, the summary
        # output, and the per-call cost -- same treatment `_step` gives
        # the main agent turn's `openrouter_api_call` span. Nests under
        # whatever parent span is active (`summarize_<tier>` below, which
        # nests under `memory_summarization`, which nests under `turn_N`
        # or `run_agent`).
        with llm_span(
            "summarizer_call",
            metadata={
                "model": self.model,
                "provider": "openrouter",
                "period_type": period_type.value,
            },
        ) as s:
            s.input(prompt[:20000])
            try:
                resp = llm.complete(
                    model=self.model,
                    system="",
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as e:
                logger.exception("Summarizer LLM call failed: %s", e)
                s.set_metadata(error=f"{type(e).__name__}: {e}")
                return None
            s.output(resp.text[:20000] if resp.text else "")
            s.set_metadata(
                llm_cost=resp.usage.to_llm_cost_dict(),
                finish_reason=resp.finish_reason,
            )

        self.total_usage.input_tokens += resp.usage.prompt_tokens
        self.total_usage.output_tokens += resp.usage.completion_tokens
        self.total_usage.total_cost += resp.usage.total_cost
        self.total_usage.llm_calls += 1
        return resp.text.strip()

    @staticmethod
    def _upsert_summary(
        table: str,
        key_columns: tuple[str, ...],
        key_values: tuple[Any, ...],
        summary_text: str,
        message_count: int,
    ) -> None:
        """INSERT OR REPLACE a summary row by its natural key."""
        assert storage.db is not None
        all_columns = ("id", *key_columns, "summary", "message_count", "created_at_ns")
        placeholders = ", ".join("?" for _ in all_columns)
        columns_sql = ", ".join(all_columns)
        storage.db.execute(
            f"INSERT OR REPLACE INTO {table} ({columns_sql}) VALUES ({placeholders})",
            (str(uuid.uuid4()), *key_values, summary_text, message_count, _now_ns()),
        )

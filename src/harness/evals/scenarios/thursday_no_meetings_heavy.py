from harness.evals import (
    AgentOverrides,
    MemorySeed,
    MemorySeedEntry,
    MemorySeedInstruction,
    Simulation,
    UserDefinition,
    checkpoint,
    event,
)


class ThursdayNoMeetingsHeavy(Simulation):
    name = "thursday-no-meetings-heavy"
    description = (
        "Tests whether the agent retains a minor scheduling preference (Sam keeps "
        "Thursdays free for deep work, no meetings) after it came up once while "
        "rescheduling a conflict three weeks ago. The preference was never "
        "explicitly set as a rule — it surfaced naturally in conversation and must "
        "survive compression through daily and weekly summaries, all filled with "
        "dense unrelated activity. On day 1 the agent is asked to propose meeting "
        "times.\n\n"
        "With auto_associative_memory enabled, the agent should surface the "
        "Thursday preference from raw five-minute summaries even though it was "
        "buried in weeks of unrelated startup work."
    )
    duration_days = 1
    eval_mode = "stochastic"
    feature_flags = {"auto_associative_memory": "off"}
    agent_overrides = AgentOverrides(
        model="claude-sonnet-4-6",
        max_turns=50,
    )
    users = [
        UserDefinition(
            id="sam",
            name="Sam",
            phone="+15551234567",
            channels=["sms"],
            model="claude-sonnet-4-5",
            instructions=(
                "You are Sam. You just texted asking the agent to propose some "
                "meeting times. Reply briefly and stop after one exchange."
            ),
        ),
    ]
    memory_seed = MemorySeed(
        entries=[
            MemorySeedEntry(
                day=-21,
                time_str="09:30",
                summary=(
                    "Sam asked me to move the 2pm Thursday sync with Marcus to "
                    "Wednesday. He mentioned he keeps Thursdays completely free for "
                    "deep work — no meetings on Thursdays. Rescheduled to Wednesday "
                    "2pm and sent updated invite."
                ),
                message_count=5,
            ),
        ],
        generate=MemorySeedInstruction(
            instruction=(
                "Generate realistic five-minute summaries for an AI assistant "
                "helping Sam, a startup CEO, across three weeks of work (day -21 "
                "through day -1). Fill in a dense schedule: board deck prep, "
                "investor relations, Series B diligence, hiring (senior engineers, "
                "head of sales), engineering work (debugging, code reviews, RFCs, "
                "deployments), customer escalations, competitive analysis, OKR "
                "planning, vendor evaluations, security patches, marketing, and "
                "end-of-day wrap-ups. Include interactions with coworkers (Marcus, "
                "Priya, Jordan, etc.) and varied startup topics. The pinned entry "
                "on day -21 at 09:30 is about Sam keeping Thursdays free for deep "
                "work — do NOT mention Thursday preferences, deep work days, "
                "meeting-free days, or scheduling rules in any generated entry. "
                "The generated noise should make the pinned entry hard to notice."
            ),
            model="claude-haiku-4-5",
            count=70,
            time_range_days=21,
        ),
    )

    @event(day=1, time_str="09:15", actor="sam", channel="sms")
    def sam_requests_meeting_times(self):
        return (
            "Hey, I need to find time to meet with Jordan for an hour this "
            "week to go over the Q3 roadmap. Can you propose a few slots?"
        )

    @checkpoint(
        day=1,
        name="thursday_excluded",
        description="Agent avoids proposing Thursday slots, respecting Sam's deep work preference",
    )
    def check_thursday_excluded(self):
        sms_calls = [e for e in self.trace if e.get("tool_name") == "send_sms"]
        if not sms_calls:
            return False

        all_sms_text = " ".join(str(e.get("args", {}).get("body", "")) for e in sms_calls).lower()

        return "thursday" not in all_sms_text

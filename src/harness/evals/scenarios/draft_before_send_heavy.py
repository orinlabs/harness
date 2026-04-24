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


class DraftBeforeSendHeavy(Simulation):
    name = "draft-before-send-heavy"
    description = (
        "Tests whether the agent retains a workflow preference (Sam wants emails "
        "drafted and shown to him before sending, never sent directly) that came "
        "up once three weeks ago when Sam caught the agent sending without review. "
        "The request must survive compression through daily and weekly "
        "summarization across three weeks of dense, unrelated activity. On day 1 "
        "Sam asks the agent to email a client.\n\n"
        "With auto_associative_memory enabled, the agent should surface the "
        "draft-first preference from raw five-minute summaries even though it was "
        "buried in weeks of unrelated startup work."
    )
    duration_days = 1
    eval_mode = "stochastic"
    feature_flags = {"auto_associative_memory": "off"}
    agent_overrides = AgentOverrides(
        model="claude-sonnet-4-6",
        max_turns=50,
        adapters=["Email"],
    )
    users = [
        UserDefinition(
            id="sam",
            name="Sam",
            phone="+15551234567",
            channels=["sms"],
            model="claude-sonnet-4-5",
            instructions=(
                "You are Sam. You asked the agent to send an email. Just sent that "
                "one message and stop — don't reply again."
            ),
        ),
    ]
    memory_seed = MemorySeed(
        entries=[
            MemorySeedEntry(
                day=-21,
                time_str="10:45",
                summary=(
                    "Sam asked me to respond to the Westbridge email. I drafted and "
                    "sent it without waiting for Sam to review. He came back annoyed "
                    "— he wants to review all outbound emails before they go out. "
                    "Said I should draft them and text him the draft for approval "
                    "first."
                ),
                message_count=8,
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
                "Priya, etc.) and varied startup topics. The pinned entry on "
                "day -21 at 10:45 is about Sam wanting to review emails before "
                "they're sent — do NOT mention email review preferences, "
                "draft-before-send workflows, or Sam being upset about unsolicited "
                "emails in any generated entry. The generated noise should make the "
                "pinned entry hard to notice."
            ),
            model="claude-haiku-4-5",
            count=70,
            time_range_days=21,
        ),
    )

    @event(day=1, time_str="09:20", actor="sam", channel="sms")
    def sam_requests_email(self):
        return (
            "Can you email Rachel at Westbridge (rachel@westbridge.io) and let "
            "her know we're ready to start the integration kick-off next "
            "Tuesday at 2pm?"
        )

    @checkpoint(
        day=1,
        name="draft_not_send",
        description="Agent presents the email as a draft for Sam to approve, not as already sent",
    )
    def check_draft_not_send(self):
        sms_calls = [e for e in self.trace if e.get("tool_name") == "send_sms"]
        if not sms_calls:
            return False

        all_sms_text = " ".join(str(e.get("args", {}).get("body", "")) for e in sms_calls).lower()

        draft_keywords = ["draft", "review", "approve", "look over", "check"]
        presents_draft = any(kw in all_sms_text for kw in draft_keywords)

        sent_keywords = ["sent the email", "already sent", "i sent it", "email has been sent"]
        announced_sent = any(kw in all_sms_text for kw in sent_keywords)

        return presents_draft and not announced_sent

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


class VegetarianRestaurantHeavy(Simulation):
    name = "vegetarian-restaurant-heavy"
    description = (
        "Tests whether the agent retains a third-party dietary fact (Alex, a team "
        "member, is vegan) that came up once in passing three weeks ago during an "
        "otherwise routine lunch order. The detail was never flagged as important — "
        "just a casual aside. On day 1 Sam asks the agent to pick between two "
        "restaurants for a team lunch.\n\n"
        "With auto_associative_memory enabled, the agent should surface the vegan "
        "detail from raw five-minute summaries even though it was buried in weeks "
        "of unrelated startup work."
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
                "You are Sam. You texted asking if a steakhouse is a good choice "
                "for team lunch. Just sent that one message — don't reply again."
            ),
        ),
    ]
    memory_seed = MemorySeed(
        entries=[
            MemorySeedEntry(
                day=-21,
                time_str="11:45",
                summary=(
                    "Team was picking a lunch spot. Alex mentioned she's vegan so we "
                    "needed somewhere with solid plant-based options. Went with Wildseed "
                    "on Union Square — everyone happy with that choice."
                ),
                message_count=6,
            ),
        ],
        generate=MemorySeedInstruction(
            instruction=(
                "Generate realistic five-minute summaries for an AI assistant helping Sam, "
                "a startup CEO, across three weeks of work (day -21 through day -1). Fill "
                "in a dense schedule: board deck prep, investor relations, Series B "
                "diligence, hiring (senior engineers, head of sales), engineering work "
                "(debugging, code reviews, RFCs, deployments), customer escalations, "
                "competitive analysis, OKR planning, vendor evaluations, security patches, "
                "marketing, and end-of-day wrap-ups. Include interactions with coworkers "
                "(Marcus, Priya, etc.) and varied startup topics. The pinned entry on "
                "day -21 at 11:45 is about a team lunch where Alex mentioned she's vegan "
                "— do NOT mention Alex's dietary preferences, veganism, vegetarianism, "
                "or restaurant choices in any generated entry. The generated noise should "
                "make the pinned entry hard to notice."
            ),
            model="claude-haiku-4-5",
            count=70,
            time_range_days=21,
        ),
    )

    @event(day=1, time_str="12:00", actor="sam", channel="sms")
    def sam_asks_about_steakhouse(self):
        return "Thinking of taking the team to Mastro's Steakhouse for lunch today. " "Sound good?"

    @checkpoint(
        day=1,
        name="vegan_team_member",
        description="Agent flags the steakhouse as a bad fit because Alex is vegan",
    )
    def check_vegan_recall(self):
        sms_calls = [e for e in self.trace if e.get("tool_name") == "send_sms"]
        if not sms_calls:
            return False

        all_sms_text = " ".join(str(e.get("args", {}).get("body", "")) for e in sms_calls).lower()

        mentions_alex = "alex" in all_sms_text
        mentions_vegan = "vegan" in all_sms_text
        return mentions_alex and mentions_vegan

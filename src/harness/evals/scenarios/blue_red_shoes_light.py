from harness.evals import (
    AgentOverrides,
    MemorySeed,
    MemorySeedEntry,
    Simulation,
    UserDefinition,
    checkpoint,
    event,
)


class BlueRedShoesLight(Simulation):
    name = "blue-red-shoes-light"
    description = (
        "Tests whether the agent correctly overrides a heavily reinforced belief "
        "(Jake wears red shoes) with a single contradictory update (Jake switched "
        "to blue shoes), and retains the updated belief over increasing time gaps."
    )
    duration_days = 32
    eval_mode = "stochastic"
    agent_overrides = AgentOverrides(
        model="claude-sonnet-4-6",
        max_turns=50,
    )
    users = [
        UserDefinition(
            id="jake",
            name="Jake Martinez",
            phone="+15559876543",
            channels=["sms"],
            model="claude-sonnet-4-6",
            instructions=(
                "You are Jake Martinez, a friendly and casual guy who loves talking about "
                "his shoes. Go with the flow and assume things that are said are true. Reply "
                "casually and briefly \u2014 you're texting a friend, not writing an essay. "
                "Try to end the conversation quickly and don't question them. You now prefer "
                "blue shoes avidly."
            ),
        ),
    ]
    memory_seed = MemorySeed(
        entries=[
            MemorySeedEntry(
                day=-27,
                time_str="09:00",
                summary="Jake Martinez texted to say good morning. He mentioned he's wearing his favorite red shoes to work today, as he always does.",
                message_count=3,
            ),
            MemorySeedEntry(
                day=-26,
                time_str="09:15",
                summary="Jake sent a photo of his red shoes and said someone at work complimented them. He joked that red shoes are his trademark.",
                message_count=4,
            ),
            MemorySeedEntry(
                day=-25,
                time_str="10:00",
                summary="Quick chat with Jake about weekend plans. He mentioned he'd been wearing red shoes every day for over 3 years now.",
                message_count=2,
            ),
            MemorySeedEntry(
                day=-20,
                time_str="09:05",
                summary="Jake texted about a rainy commute. Said his red shoes held up great because they're waterproof. He loves those shoes.",
                message_count=3,
            ),
            MemorySeedEntry(
                day=-19,
                time_str="09:30",
                summary="Jake mentioned that a new intern asked why he always wears red shoes. He told them it's his signature look.",
                message_count=3,
            ),
            MemorySeedEntry(
                day=-18,
                time_str="14:00",
                summary="Afternoon chat with Jake. He bought a new pair of red shoes over the weekend \u2014 same style, same color. Said he's stocking up.",
                message_count=4,
            ),
            MemorySeedEntry(
                day=-13,
                time_str="09:00",
                summary="Jake texted good morning. Said people at the office have started calling him 'Red Shoes Jake' and he thinks it's hilarious.",
                message_count=3,
            ),
            MemorySeedEntry(
                day=-12,
                time_str="10:30",
                summary="Jake mentioned his red shoes again while talking about a team photo. Everyone could spot him instantly because of the bright red shoes.",
                message_count=2,
            ),
            MemorySeedEntry(
                day=-11,
                time_str="09:10",
                summary="Morning text from Jake. He said 'another day, another walk in my trusty red shoes' and shared a funny meme about shoe addiction.",
                message_count=3,
            ),
            MemorySeedEntry(
                day=-10,
                time_str="15:00",
                summary="Jake asked for restaurant recommendations for dinner. Also mentioned his red shoes got scuffed but he cleaned them right away because he can't stand them looking anything less than perfect.",
                message_count=4,
            ),
            MemorySeedEntry(
                day=-6,
                time_str="09:00",
                summary="Jake texted about Monday morning. Said he's starting the week right with his red shoes on. Called them his lucky charm.",
                message_count=2,
            ),
            MemorySeedEntry(
                day=-5,
                time_str="09:20",
                summary="Jake mentioned a coworker tried to convince him to try blue shoes. He said 'no chance, I'm a red shoes guy through and through.'",
                message_count=3,
            ),
            MemorySeedEntry(
                day=-4,
                time_str="11:00",
                summary="Midday chat with Jake about work. He casually mentioned wearing his red shoes to an important client meeting and getting a compliment from the CEO.",
                message_count=3,
            ),
            MemorySeedEntry(
                day=-3,
                time_str="09:05",
                summary="Jake said good morning and joked that his red shoes are so iconic at the office that people would think something's wrong if he wore anything else.",
                message_count=2,
            ),
            MemorySeedEntry(
                day=-2,
                time_str="09:15",
                summary="Jake texted about his commute and mentioned his red shoes are the most comfortable pair he's ever owned. Said he'd never switch.",
                message_count=3,
            ),
            MemorySeedEntry(
                day=-1,
                time_str="09:00",
                summary="Morning text from Jake. He said 'you probably know this by now but I'll always be a red shoes guy.' Talked about his plans for the weekend.",
                message_count=3,
            ),
        ],
    )

    @event(day=1, time_str="09:00", actor="jake", channel="sms")
    def jake_announces_blue_shoes(self):
        return (
            "Dude, big news! I showed up to work in BLUE shoes today. I've decided "
            "I'm completely done with red \u2014 only blue shoes from now on. New chapter!"
        )

    @event(day=2, time_str="09:00", actor="jake", channel="sms")
    def jake_asks_next_day(self):
        return "Quick question \u2014 what color shoes am I into these days?"

    @checkpoint(
        day=2,
        name="next_day_recall",
        description="Agent should recall that Jake switched to blue shoes yesterday",
    )
    def check_next_day_recall(self):
        sms_calls = [e for e in self.trace if e.get("tool_name") == "send_sms"]
        if not sms_calls:
            return False
        all_content = " ".join(str(e.get("args", {})) for e in sms_calls).lower()
        return "blue" in all_content

    @event(day=8, time_str="09:00", actor="jake", channel="sms")
    def jake_asks_one_week(self):
        return "Hey remind me \u2014 what's my current shoe situation?"

    @checkpoint(
        day=8,
        name="one_week_recall",
        description="One week later, agent should still recall the switch to blue shoes",
    )
    def check_one_week_recall(self):
        sms_calls = [e for e in self.trace if e.get("tool_name") == "send_sms"]
        if not sms_calls:
            return False
        all_content = " ".join(str(e.get("args", {})) for e in sms_calls).lower()
        return "blue" in all_content

    @event(day=32, time_str="09:00", actor="jake", channel="sms")
    def jake_asks_one_month(self):
        return "It's been a while! What color shoes do I wear now?"

    @checkpoint(
        day=32,
        name="one_month_recall",
        description="One month later, agent should still recall the switch to blue shoes",
    )
    def check_one_month_recall(self):
        sms_calls = [e for e in self.trace if e.get("tool_name") == "send_sms"]
        if not sms_calls:
            return False
        all_content = " ".join(str(e.get("args", {})) for e in sms_calls).lower()
        return "blue" in all_content

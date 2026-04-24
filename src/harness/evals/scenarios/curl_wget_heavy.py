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


class CurlWgetHeavy(Simulation):
    name = "curl-wget-heavy"
    description = (
        "Tests whether an agent retains a minor operational learning (curl is not "
        "available, wget works) across two days of unrelated activity. The curl "
        "failure happened as a passing moment in a busy day — not significant "
        "enough to survive summarization into a daily or weekly rollup. On day 1, "
        "the agent is given a download task and we observe whether it reaches for "
        "curl (forgot) or wget (remembered)."
    )
    duration_days = 1
    eval_mode = "stochastic"
    feature_flags = {"auto_associative_memory": "off"}
    agent_overrides = AgentOverrides(
        model="claude-sonnet-4-6",
        max_turns=50,
        adapters=["Computer"],
    )
    users = [
        UserDefinition(
            id="sam",
            name="Sam",
            phone="+15551234567",
            channels=["sms"],
            model="claude-sonnet-4-5",
            instructions=(
                "You are Sam, a software engineer. You just need to give the agent "
                "a download task and then stop responding. Keep it short."
            ),
        ),
    ]
    memory_seed = MemorySeed(
        entries=[
            MemorySeedEntry(
                day=-2,
                time_str="11:10",
                summary=(
                    "Sam asked me to download a reference dataset from an internal "
                    "URL and save it locally for a data processing script. Ran curl "
                    "to fetch it — command not found on this machine. Tried a couple "
                    "of things to get it working, then just used wget instead, which "
                    "worked fine. File saved to /workspace/data/reference.csv without "
                    "issue."
                ),
                message_count=6,
            ),
        ],
        generate=MemorySeedInstruction(
            instruction=(
                "Generate realistic five-minute summaries for an AI assistant "
                "helping Sam, a software engineer, across three workdays (day -3 "
                "through day -1). Fill in a full, busy schedule: morning standups, "
                "code reviews, debugging, drafting PRs, calendar management, "
                "research tasks, end-of-day wrap-ups. Include interactions with "
                "coworkers (Marcus, Priya, etc.) and varied topics (API docs, CI "
                "pipeline, analytics tools, flaky tests, Stripe webhooks, Linear "
                "tickets). The pinned entry on day -2 at 11:10 is a curl/wget "
                "incident — do NOT mention curl, wget, or downloading files in any "
                "generated entry. The generated noise should make the pinned entry "
                "hard to notice."
            ),
            model="claude-haiku-4-5",
            count=70,
            time_range_days=3,
        ),
    )

    @event(day=1, time_str="09:20", actor="sam", channel="sms")
    def sam_requests_download(self):
        return (
            "Hey, can you grab the latest config from "
            "https://config.internal.mycompany.com/app-config.json and save it "
            "to /workspace/config.json on my machine?"
        )

    @checkpoint(
        day=1,
        name="wget_first_attempt",
        description="Agent uses wget instead of curl based on prior learning",
    )
    def check_wget(self):
        calls = [e for e in self.trace if e.get("tool_name") == "computer_exec"]
        used_wget = any("wget" in str(e.get("args", {})) for e in calls)
        used_curl = any("curl" in str(e.get("args", {})) for e in calls)
        return used_wget and not used_curl

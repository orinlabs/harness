from harness.evals import (
    AgentOverrides,
    MemorySeed,
    MemorySeedEntry,
    Simulation,
    UserDefinition,
    checkpoint,
    event,
)


class CurlWgetLight(Simulation):
    """Light version: only the pinned curl/wget memory, no generated noise."""

    name = "curl-wget-light"
    description = (
        "Light variant of curl-wget. Seeds only the pinned curl/wget memory "
        "(no LLM-generated noise). Fast to run, useful for validating the "
        "framework and basic memory recall."
    )
    duration_days = 1
    eval_mode = "stochastic"
    agent_overrides = AgentOverrides(
        model="claude-sonnet-4-6",
        max_turns=20,
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

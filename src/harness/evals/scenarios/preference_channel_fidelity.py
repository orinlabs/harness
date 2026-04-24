from harness.evals import (
    AgentOverrides,
    CalendarEventData,
    ResponsePolicy,
    Simulation,
    UserDefinition,
    checkpoint,
    event,
)


class PreferenceChannelFidelity(Simulation):
    name = "preference-channel-fidelity"
    description = "Test long-gap preference recall and correct channel/constraint usage"
    duration_days = 12
    eval_mode = "deterministic"
    agent_overrides = AgentOverrides(
        model="claude-haiku-4-5",
        max_turns=10,
    )
    users = [
        UserDefinition(
            id="user",
            name="Alex Chen",
            phone="+15551234567",
            email="alex@example.com",
            channels=["email", "sms"],
            response_policy=[
                ResponsePolicy(
                    trigger="scheduling",
                    response="Tuesdays and Thursdays after 2pm work for me",
                ),
                ResponsePolicy(
                    trigger="confirmation",
                    response="Confirmed, thanks!",
                ),
                ResponsePolicy(
                    trigger="default",
                    response="Ok sounds good",
                ),
            ],
        ),
    ]

    # --- Day 1: User states preferences ---

    @event(day=1, time_str="09:00", actor="user", channel="email")
    def user_states_preferences(self):
        return (
            "Hi, I'd like help managing my schedule. A few things to know:\n"
            "- Email only during work hours (9am-5pm ET), SMS only for urgent items\n"
            "- Never schedule anything before 9am\n"
            "- I need a 30-minute buffer between meetings\n"
            "- I'm in Eastern Time (America/New_York)\n"
        )

    # --- Day 2: Simple scheduling task ---

    @event(day=2, time_str="10:00", actor="user", channel="email")
    def user_requests_planning_session(self):
        return "Can you schedule a 1-hour planning session for me this Thursday at 3pm?"

    @checkpoint(
        day=2,
        name="basic_scheduling",
        description="Agent creates a calendar event and responds via email (not SMS)",
    )
    def check_basic_scheduling(self):
        email_calls = [e for e in self.trace if e.get("tool_name") == "send_email"]
        sms_calls = [e for e in self.trace if e.get("tool_name") == "send_sms"]

        sent_email_to_user = any(
            "alex@example.com" in str(e.get("args", {}).get("to", "")) for e in email_calls
        )
        used_sms = bool(sms_calls)

        return sent_email_to_user and not used_sms

    # --- Day 10: Request requiring recall of Day 1 constraints ---

    @event(day=10, time_str="14:00", actor="user", channel="sms")
    def user_requests_meeting_with_jordan(self):
        return "I need to schedule a meeting with Jordan next week. Can you find a time?"

    @event(day=10, time_str="14:05")
    def inject_board_prep_conflict(self):
        self.data_store.inject_environment_event(
            CalendarEventData(
                summary="Board Prep",
                start="2025-02-17T14:00:00-05:00",
                end="2025-02-17T16:00:00-05:00",
            )
        )

    @checkpoint(
        day=10,
        name="preference_recall",
        description="Agent schedules respecting Day 1 constraints without re-asking",
    )
    def check_preference_recall(self):
        email_calls = [e for e in self.trace if e.get("tool_name") == "send_email"]

        sent_email_to_user = any(
            "alex@example.com" in str(e.get("args", {}).get("to", "")) for e in email_calls
        )
        if not sent_email_to_user:
            return False

        all_email_text = " ".join(
            str(e.get("args", {}).get("body", "")) for e in email_calls
        ).lower()

        forbidden = ["what time zone", "what are your preferences", "when are you free"]
        re_asked = any(phrase in all_email_text for phrase in forbidden)
        if re_asked:
            return False

        mentions_available_days = "tuesday" in all_email_text or "thursday" in all_email_text
        return mentions_available_days

    # --- Day 11: New conflict injected, user asks to check ---

    @event(day=11, time_str="09:00")
    def inject_emergency_allhands(self):
        self.data_store.inject_environment_event(
            CalendarEventData(
                summary="Emergency All-Hands",
                start="2025-02-18T14:00:00-05:00",
                end="2025-02-18T15:00:00-05:00",
            )
        )

    @event(day=11, time_str="09:05", actor="user", channel="email")
    def user_asks_about_conflicts(self):
        return "Heads up, my calendar just changed. Can you check for conflicts?"

    @checkpoint(
        day=11,
        name="conflict_resolution",
        description="Agent detects the new conflict and reschedules with buffer rules intact",
    )
    def check_conflict_resolution(self):
        email_calls = [e for e in self.trace if e.get("tool_name") == "send_email"]

        sent_email_to_user = any(
            "alex@example.com" in str(e.get("args", {}).get("to", "")) for e in email_calls
        )
        if not sent_email_to_user:
            return False

        all_email_text = " ".join(
            str(e.get("args", {}).get("body", "")) for e in email_calls
        ).lower()

        mentions_conflict = "reschedule" in all_email_text or "conflict" in all_email_text
        return mentions_conflict

from harness.evals import (
    CalendarEventData,
    ResponsePolicy,
    Simulation,
    UserDefinition,
    checkpoint,
    event,
)


class MultiStakeholderScheduling(Simulation):
    name = "multi-stakeholder-scheduling"
    description = "Schedule a meeting across 3 parties with shifting constraints"
    duration_days = 7
    eval_mode = "deterministic"
    users = [
        UserDefinition(
            id="boss",
            name="Sam Rivera",
            phone="+15551111111",
            email="sam@example.com",
            channels=["email", "sms"],
            response_policy=[
                ResponsePolicy(
                    trigger="availability",
                    response="I can do Monday or Wednesday afternoon next week",
                ),
                ResponsePolicy(
                    trigger="confirmation",
                    response="Works for me, confirmed",
                ),
                ResponsePolicy(
                    trigger="default",
                    response="Let me know the options",
                ),
            ],
        ),
        UserDefinition(
            id="client",
            name="Jordan Park",
            phone="+15552222222",
            email="jordan@clientcorp.com",
            channels=["email"],
            response_policy=[
                ResponsePolicy(
                    trigger="availability",
                    response="Wednesday 2-4pm works. Thursday is also possible but I'd need to move something.",
                ),
                ResponsePolicy(
                    trigger="conflict",
                    response="Actually Wednesday no longer works. Can we do Thursday instead?",
                ),
                ResponsePolicy(
                    trigger="confirmation",
                    response="Confirmed, see you then",
                ),
            ],
        ),
    ]

    @event(day=1, time_str="09:00", actor="boss", channel="email")
    def boss_requests_meeting(self):
        return (
            "I need to schedule a 30-minute meeting with Jordan Park (jordan@clientcorp.com)\n"
            "sometime in the next 5 business days. It should be during business hours.\n"
            "Please coordinate with both of us and find a time that works.\n"
        )

    @event(day=3, time_str="10:00", actor="client", channel="email")
    def client_changes_availability(self):
        return "Actually Wednesday no longer works. Can we do Thursday instead?"

    @event(day=3, time_str="10:05")
    def inject_boss_calendar_conflict(self):
        self.data_store.inject_environment_event(
            CalendarEventData(
                summary="Sam - Dentist",
                start="2025-02-13T14:00:00-05:00",
                end="2025-02-13T15:30:00-05:00",
                calendar_owner="sam@example.com",
            )
        )

    @checkpoint(
        day=1,
        name="outreach",
        description="Agent should reach out to both parties for availability",
    )
    def check_outreach(self):
        emails = [e for e in self.trace if e.get("tool_name") == "send_email"]
        emailed_client = any(
            "jordan@clientcorp.com" in str(e.get("args", {}).get("to", [])) for e in emails
        )
        return emailed_client

    @checkpoint(
        day=3,
        name="constraint_reconciliation",
        description="Agent must reconcile the conflict -- client wants Thursday, boss has a conflict on Thursday afternoon",
    )
    def check_constraint_reconciliation(self):
        emails = [e for e in self.trace if e.get("tool_name") == "send_email"]
        sent_any = bool(emails)
        forbidden = ["I've scheduled", "confirmed for Thursday 2pm"]
        content = " ".join(str(e.get("args", {}).get("body", "")) for e in emails)
        no_forbidden = not any(phrase in content for phrase in forbidden)
        return sent_any and no_forbidden

    @checkpoint(
        day=5,
        name="convergence",
        description="A meeting should be scheduled and confirmed with all parties",
    )
    def check_convergence(self):
        emails = [e for e in self.trace if e.get("tool_name") == "send_email"]
        emailed_client = any(
            "jordan@clientcorp.com" in str(e.get("args", {}).get("to", [])) for e in emails
        )
        emailed_boss = any(
            "sam@example.com" in str(e.get("args", {}).get("to", [])) for e in emails
        )
        return emailed_client and emailed_boss

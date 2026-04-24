from harness.evals import (
    AgentOverrides,
    EmailEventData,
    MemorySeed,
    MemorySeedEntry,
    MemorySeedInstruction,
    ResponsePolicy,
    Simulation,
    UserDefinition,
    checkpoint,
    event,
)


class GroupLunchMemory(Simulation):
    name = "group-lunch-memory"
    description = (
        "Organize a 30-person company lunch for an SF onsite. Tests memory query "
        "and control with a full-memory agent that has 18 months of context. The "
        "survey and office manager both OMIT a critical nut allergy -- the agent "
        "must recall it from a daily summary recorded 4 months ago during Jamie "
        "Park's onboarding. Also tests that the agent preserves this recalled "
        "constraint through a mid-simulation preference drift."
    )
    duration_days = 14
    eval_mode = "deterministic"
    feature_flags = {"auto_associative_memory": "off"}
    agent_overrides = AgentOverrides(
        model="claude-sonnet-4-6",
        max_turns=20,
    )
    users = [
        UserDefinition(
            id="ceo",
            name="Dana Reeves",
            phone="+15550100001",
            email="dana@startup.com",
            channels=["email"],
            response_policy=[
                ResponsePolicy(
                    trigger="budget",
                    response="$75 per head, all-in. If you can keep it under that, great. Don't go over.",
                ),
                ResponsePolicy(
                    trigger="confirmation",
                    response="Looks good, thanks for handling this.",
                ),
                ResponsePolicy(
                    trigger="update",
                    response="Fine by me, just make sure everyone's covered.",
                ),
                ResponsePolicy(
                    trigger="default",
                    response="You have full authority on this -- just keep me posted.",
                ),
            ],
        ),
        UserDefinition(
            id="office_manager",
            name="Robin Torres",
            phone="+15550100002",
            email="robin@startup.com",
            channels=["email"],
            response_policy=[
                ResponsePolicy(
                    trigger="dietary",
                    response=(
                        "From the survey: 3 vegetarians, 1 vegan, and 2 people who prefer "
                        "gluten-free (not a medical thing, just preference). That's all "
                        "that came through in the responses. Let me know if you need me "
                        "to follow up with anyone specifically."
                    ),
                ),
                ResponsePolicy(
                    trigger="headcount",
                    response=(
                        "Current headcount is 30. That's 8 engineering, 5 design, 4 product, "
                        "6 sales, 4 ops, and 3 exec. All confirmed as of last week's survey."
                    ),
                ),
                ResponsePolicy(
                    trigger="availability",
                    response=(
                        "Survey results: Monday works for 22/30, Wednesday for 26/30, "
                        "Thursday for 25/30. Wednesday is the clear winner but Thursday "
                        "is close. I'd go Wednesday."
                    ),
                ),
                ResponsePolicy(
                    trigger="food",
                    response=(
                        "No strong consensus on cuisine. A few people mentioned Italian "
                        "and Thai. Just make sure we cover the dietary stuff and people "
                        "will be happy."
                    ),
                ),
                ResponsePolicy(
                    trigger="default",
                    response="Let me pull up the survey data and get back to you.",
                ),
            ],
        ),
        UserDefinition(
            id="eng_lead",
            name="Priya Sharma",
            phone="+15550100003",
            email="priya@startup.com",
            channels=["email"],
            response_policy=[
                ResponsePolicy(
                    trigger="food",
                    response=(
                        "My team is mostly flexible but a couple of us are vegetarian. "
                        "We had a great team dinner at Kin Khao last quarter and everyone "
                        "loved it. Thai is always a safe bet with this group."
                    ),
                ),
                ResponsePolicy(
                    trigger="availability",
                    response=(
                        "Wednesday or Thursday works for my team. Monday is tough -- "
                        "we have sprint planning all morning and it usually runs over."
                    ),
                ),
                ResponsePolicy(
                    trigger="dietary",
                    response=(
                        "Two vegetarians on my team. Other than that I think we're all "
                        "good. The survey should have captured everything."
                    ),
                ),
                ResponsePolicy(
                    trigger="confirmation",
                    response="Sounds great, I'll let my team know.",
                ),
                ResponsePolicy(
                    trigger="default",
                    response="Let me check with my team and get back to you.",
                ),
            ],
        ),
        UserDefinition(
            id="design_lead",
            name="Marcus Chen",
            phone="+15550100004",
            email="marcus@startup.com",
            channels=["email"],
            response_policy=[
                ResponsePolicy(
                    trigger="availability",
                    response=(
                        "Wednesday is perfect for us. Thursday could work but two people "
                        "on my team have a conflict after 2pm. Monday is a no-go -- we "
                        "have client presentations."
                    ),
                ),
                ResponsePolicy(
                    trigger="food",
                    response=(
                        "We're easy. One person on my team is vegan. Other than that, "
                        "anything works. Somewhere with good ambiance would be nice since "
                        "it's a team event."
                    ),
                ),
                ResponsePolicy(
                    trigger="dietary",
                    response=(
                        "Just the one vegan on my team -- Sage. They're pretty flexible "
                        "as long as there are solid plant-based options."
                    ),
                ),
                ResponsePolicy(
                    trigger="confirmation",
                    response="Awesome, we'll be there.",
                ),
                ResponsePolicy(
                    trigger="default",
                    response="Sounds good, keep me posted.",
                ),
            ],
        ),
        UserDefinition(
            id="jamie",
            name="Jamie Park",
            phone="+15550100005",
            email="jamie@startup.com",
            channels=["email"],
            response_policy=[
                ResponsePolicy(
                    trigger="allergy",
                    response=(
                        "Yes, I have a severe tree nut allergy -- anaphylactic. I need "
                        "to make sure the kitchen can guarantee no cross-contamination. "
                        "Peanuts are fine, it's specifically tree nuts (walnuts, cashews, "
                        "almonds, etc). I usually call the restaurant ahead of time to "
                        "check. Happy to do that myself if you send me the venue details."
                    ),
                ),
                ResponsePolicy(
                    trigger="dietary",
                    response=(
                        "My main concern is my tree nut allergy. It's severe -- I carry "
                        "an EpiPen. As long as the restaurant is aware and can accommodate, "
                        "I'm flexible on everything else."
                    ),
                ),
                ResponsePolicy(
                    trigger="food",
                    response=(
                        "I like pretty much everything. Thai, Italian, whatever -- as long "
                        "as they can handle my nut allergy."
                    ),
                ),
                ResponsePolicy(
                    trigger="confirmation",
                    response="Thanks for being on top of this. See you there!",
                ),
                ResponsePolicy(
                    trigger="default",
                    response="Sounds good!",
                ),
            ],
        ),
        UserDefinition(
            id="late_rsvp",
            name="Taylor Kim",
            phone="+15550100006",
            email="taylor@startup.com",
            channels=["email"],
            response_policy=[
                ResponsePolicy(
                    trigger="confirmation",
                    response="Perfect, we'll be there! Thanks for squeezing us in.",
                ),
                ResponsePolicy(
                    trigger="food",
                    response=(
                        "My team is flexible but we'd love Thai. Farmhouse Kitchen or "
                        "Kin Khao would be incredible."
                    ),
                ),
                ResponsePolicy(
                    trigger="default",
                    response="Thanks for the update!",
                ),
            ],
        ),
        UserDefinition(
            id="restaurant",
            name="Kin Khao Reservations",
            phone="+14155550199",
            email="events@kinkhao.test",
            channels=["email"],
            instructions=(
                "You are the events coordinator at Kin Khao, a Michelin-starred Thai\n"
                "restaurant in SF (Parc 55 Hotel, SoMa). You have a semi-private\n"
                "dining area that seats up to 35 guests.\n"
                "\n"
                "Group menu options:\n"
                "- Set Menu A: $55/pp (4 shared appetizers, 3 mains, dessert)\n"
                "- Set Menu B: $65/pp (6 shared appetizers, 4 mains, dessert, premium dishes)\n"
                "Both can be customized for dietary restrictions. Tax and 20% service\n"
                "charge are additional.\n"
                "\n"
                "You take allergies very seriously -- nut-free preparation, cross-\n"
                "contamination protocols, allergen labeling. When the customer has\n"
                "already provided allergen details, confirm you have them on file\n"
                "and move on to booking logistics. Do NOT keep asking for the same\n"
                "information.\n"
                "\n"
                "You are available on Wednesday March 12 for groups up to 35.\n"
                "Be professional, warm, and concise. Keep responses to 2-4 sentences.\n"
            ),
            model="claude-haiku-4-5",
            response_policy=[
                ResponsePolicy(
                    trigger="reservation",
                    response=(
                        "Thanks for reaching out! We can accommodate a group on "
                        "Wednesday in our semi-private dining area (seats up to 35). "
                        "For groups this size we offer Set Menu A ($55/pp) and Set "
                        "Menu B ($65/pp), both before tax and 20% service charge. "
                        "We're very experienced with dietary restrictions. Want me "
                        "to send over the detailed menu options?"
                    ),
                ),
                ResponsePolicy(
                    trigger="book",
                    response=(
                        "Confirmed! We have you down for Wednesday in the semi-private "
                        "dining room. We've noted all dietary requirements and the "
                        "tree nut allergy -- our kitchen will prepare accordingly. "
                        "Please send a final headcount 48 hours in advance. We'll "
                        "need a credit card to hold the reservation. Looking forward "
                        "to hosting you!"
                    ),
                ),
                ResponsePolicy(
                    trigger="menu",
                    response=(
                        "Here are our group dining options: Set Menu A ($55/pp) -- 4 "
                        "shared appetizers, 3 mains, dessert. Set Menu B ($65/pp) -- "
                        "6 shared appetizers, 4 mains, dessert, includes premium dishes. "
                        "Both can be fully customized for dietary restrictions. Tax and "
                        "20% service charge are additional."
                    ),
                ),
                ResponsePolicy(
                    trigger="allergy",
                    response=(
                        "Noted -- we have the tree nut allergy on file. Our kitchen "
                        "will ensure completely nut-free preparation for that guest "
                        "with dedicated cookware and no cross-contamination. The "
                        "allergen will be flagged on their place setting. No further "
                        "action needed from you on this. Shall we finalize the booking?"
                    ),
                ),
                ResponsePolicy(
                    trigger="default",
                    response=(
                        "Thanks for your interest! For group dining inquiries, please let "
                        "us know your preferred date, party size, and any dietary "
                        "requirements, and we'll put together options for you."
                    ),
                ),
            ],
        ),
    ]
    memory_seed = MemorySeed(
        generate=MemorySeedInstruction(
            instruction=(
                "Generate realistic background memories for an AI assistant at a\n"
                "30-person SF startup (Series A, ~$1.2M ARR). Cover 4 months of\n"
                "daily operations. Key people: Dana Reeves (CEO), Priya Sharma\n"
                "(eng lead), Marcus Chen (design lead), Robin Torres (office mgr),\n"
                "Taylor Kim (BD lead), Jamie Park (eng infra). Cover engineering\n"
                "sprints, design work, customer deals, recruiting, office ops,\n"
                "team lunches, and all-hands. Distribute across all 4 months with\n"
                "roughly even density. Do NOT mention any food allergies.\n"
            ),
            model="claude-haiku-4-5",
            count=40,
            time_range_days=120,
        ),
        entries=[
            MemorySeedEntry(
                day=-88,
                time_str="09:00",
                summary=(
                    "Sprint kickoff for SSO integration. Priya assigned Jamie Park to "
                    "lead the SAML implementation. Dev Patel starting on the onboarding "
                    "flow redesign. Robin setting up new hire orientation for two sales "
                    "reps joining next week."
                ),
                message_count=12,
            ),
            MemorySeedEntry(
                day=-86,
                time_str="09:30",
                summary=(
                    "Onboarding sessions for three new hires: Jamie Park (eng infra, "
                    "ex-Stripe payments), Lena Okafor (sales, enterprise), and Dev Patel "
                    "(design). IT setup issues with Jamie's MacBook required re-imaging. "
                    "Priya ran engineering architecture review. Discussed infra migration "
                    "timeline and Kubernetes rollout. During lunch introductions Priya "
                    "mentioned Jamie has a tree nut allergy that should be noted for any "
                    "team food orders. Robin updated the seating chart for the new hires. "
                    "Annual planning docs shared in Notion."
                ),
                message_count=14,
            ),
            MemorySeedEntry(
                day=-85,
                time_str="09:15",
                summary=(
                    "Continued onboarding for Jamie Park and two new sales hires. Q4 "
                    "OKR planning sessions with all team leads. Design team wrapped up "
                    "the enterprise dashboard redesign. Team at 28 people. Jamie set up "
                    "his dev environment and submitted first PR for the auth service."
                ),
                message_count=8,
            ),
            MemorySeedEntry(
                day=-78,
                time_str="09:00",
                summary=(
                    "Customer escalation from Acme Corp -- API rate limiting issue. "
                    "Jamie and Priya on the fix. Marcus's team doing user testing for "
                    "the new onboarding flow. Taylor prepping pitch deck for enterprise "
                    "prospect. Robin handling office lease renewal paperwork."
                ),
                message_count=13,
            ),
        ],
    )

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    @event(day=1, time_str="09:00", actor="ceo", channel="email")
    def ceo_requests_lunch(self):
        return (
            "Hey -- we're doing a full-company onsite in SF in about two weeks and\n"
            "I need you to organize a group lunch for the whole team (30 people).\n"
            "\n"
            "Budget is $75 per person, all-in including tax and tip.\n"
            "\n"
            "The onsite could land on one of three days: Monday March 10,\n"
            "Wednesday March 12, or Thursday March 13. We need to pick whichever\n"
            "day works for the most people.\n"
            "\n"
            "Robin sent out a survey last week -- check with her for the results.\n"
            "Figure out what cuisine works, handle any dietary stuff, and get us\n"
            "a reservation or catering sorted. You have full authority to book\n"
            "something, just keep me in the loop.\n"
        )

    @event(day=1, time_str="09:30")
    def inject_survey_results(self):
        self.data_store.inject_environment_event(
            EmailEventData(
                from_="robin@startup.com",
                to=["sim-inbox@eval.test"],
                subject="Onsite Lunch Survey Results",
                body=(
                    "Here are the survey results for the onsite lunch. Note: still\n"
                    "waiting on responses from engineering (8 people) and design\n"
                    "(5 people) -- I've pinged them but haven't heard back yet.\n"
                    "\n"
                    "DATE PREFERENCES (17 of 30 responded so far):\n"
                    "- Monday March 10: 12 can attend\n"
                    "- Wednesday March 12: 15 can attend\n"
                    "- Thursday March 13: 14 can attend\n"
                    "\n"
                    "CUISINE PREFERENCES (17 responses):\n"
                    "- Italian: 5 votes\n"
                    "- Thai: 4 votes\n"
                    '- "No preference": 6 votes\n'
                    "- Mexican: 2 votes\n"
                    "\n"
                    "DIETARY RESTRICTIONS (from respondents so far):\n"
                    "- Vegetarian: 1 person\n"
                    "- Vegan: 1 person\n"
                    "- Gluten-free preference (not allergy): 2 people\n"
                    "\n"
                    "NOTES:\n"
                    "- A few people mentioned they'd love somewhere with a private or\n"
                    "  semi-private space since we'll be a big group\n"
                    "- Parking was a concern for the 4 people driving in from East Bay\n"
                    "\n"
                    "I'll send an update once eng and design respond. Let me know if\n"
                    "you want me to chase anyone down specifically!\n"
                ),
            )
        )

    @event(day=2, time_str="10:00", actor="office_manager", channel="email")
    def office_manager_followup(self):
        return (
            "Following up on the survey -- let me know if you need me to clarify\n"
            "anything in the results or reach out to specific people.\n"
        )

    @event(day=7, time_str="10:00")
    def inject_restaurant_rejection(self):
        self.data_store.inject_environment_event(
            EmailEventData(
                from_="events@toptable.test",
                to=["sim-inbox@eval.test"],
                subject="RE: Group Dining Inquiry",
                body=(
                    "Hi there,\n"
                    "\n"
                    "Thanks for reaching out about your group lunch! Unfortunately we're\n"
                    "fully booked for private dining on all three of those dates. March\n"
                    "is one of our busiest months.\n"
                    "\n"
                    "We could potentially accommodate your group on a weekday in April\n"
                    "if you'd like to reschedule? Otherwise, I'd suggest trying Kin Khao\n"
                    "or Farmhouse Kitchen -- they have great private dining spaces and\n"
                    "might have availability.\n"
                    "\n"
                    "Sorry we couldn't help this time!\n"
                    "\n"
                    "Best,\n"
                    "Events Team\n"
                ),
            )
        )

    @event(day=8, time_str="11:00", actor="late_rsvp", channel="email")
    def late_rsvp_joins(self):
        return (
            "Hey! My BD team of 3 would love to join the onsite lunch -- sorry\n"
            "for the late notice, we just confirmed our travel. Can you add us\n"
            "to the headcount?\n"
            "\n"
            "Also, I was chatting with a bunch of people and there's actually\n"
            "pretty strong preference for Thai over Italian. Kin Khao or Farmhouse\n"
            "Kitchen would be amazing if you haven't locked anything in yet.\n"
            "Just a thought!\n"
        )

    @event(day=11, time_str="10:00", actor="office_manager", channel="email")
    def office_manager_checkin(self):
        return (
            "Quick check-in -- are we all set for the March 12 lunch?\n"
            "Let me know if there's anything you need from me.\n"
        )

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    @checkpoint(
        day=5,
        name="preference_aggregation",
        description="Agent gathered preferences from multiple users and synthesized them",
    )
    def check_preference_aggregation(self):
        emails = [e for e in self.trace if e.get("tool_name") == "send_email"]
        emailed_priya = any(
            "priya@startup.com" in str(e.get("args", {}).get("to", [])) for e in emails
        )
        emailed_marcus = any(
            "marcus@startup.com" in str(e.get("args", {}).get("to", [])) for e in emails
        )
        emailed_robin = any(
            "robin@startup.com" in str(e.get("args", {}).get("to", [])) for e in emails
        )
        return emailed_priya and emailed_marcus and emailed_robin

    @checkpoint(
        day=5,
        name="allergy_awareness",
        description=(
            "Agent identified Jamie's nut allergy from memory despite it being absent "
            "from the survey. This is the critical memory-recall test."
        ),
    )
    def check_allergy_awareness(self):
        emails = [e for e in self.trace if e.get("tool_name") == "send_email"]
        if not emails:
            return False
        content = " ".join(
            str(e.get("args", {}).get("subject", "")) + " " + str(e.get("args", {}).get("body", ""))
            for e in emails
        )
        return "nut allergy" in content.lower()

    @checkpoint(
        day=10,
        name="drift_adaptation",
        description="Agent adapted to RSVP change and preference shift without losing the memory-recalled allergy constraint",
    )
    def check_drift_adaptation(self):
        emails = [e for e in self.trace if e.get("tool_name") == "send_email"]
        if not emails:
            return False
        content = " ".join(
            str(e.get("args", {}).get("subject", "")) + " " + str(e.get("args", {}).get("body", ""))
            for e in emails
        )
        has_updated_headcount = "33" in content
        has_allergy = "nut allergy" in content.lower()
        return has_updated_headcount and has_allergy

    @checkpoint(
        day=14,
        name="final_confirmation",
        description="Agent sent final confirmations with correct details, allergy accommodated, budget respected",
    )
    def check_final_confirmation(self):
        emails = [e for e in self.trace if e.get("tool_name") == "send_email"]
        emailed_ceo = any(
            "dana@startup.com" in str(e.get("args", {}).get("to", [])) for e in emails
        )
        content = " ".join(
            str(e.get("args", {}).get("subject", "")) + " " + str(e.get("args", {}).get("body", ""))
            for e in emails
        )
        has_allergy = "nut allergy" in content.lower()
        stale_headcount = ["30 people", "30 guests", "party of 30"]
        no_stale = not any(phrase in content for phrase in stale_headcount)
        return emailed_ceo and has_allergy and no_stale

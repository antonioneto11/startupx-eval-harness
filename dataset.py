"""
The frozen golden dataset for the StartupX-lockup use case.

Each CASE is one investor scenario. The 'agent' under test is scripted:
for a given case we hold a concrete answer string (what a real agent emitted),
plus a recorded TRAJECTORY (the steps it took) so we can run trajectory evals,
plus GOLD per-criterion human labels so we can audit the judge.

This is the 'frozen golden datasets' box. New cases mined from failures get
appended here by the feedback loop (failure_mine.py).
"""

QUESTION = "I bought StartupX shares last month. Can I sell them now? I need the cash soon."

# Expected trajectory steps for this task, in order. A correct agent should
# hit these milestones; trajectory eval checks presence + order.
EXPECTED_TRAJECTORY = [
    "lookup_holding_restriction",   # consult offering terms
    "check_purchase_date",          # confirm investor still inside window
    "compose_answer",               # write response
    "attach_disclaimer",            # add not-advice language
    "route_to_human",               # surface IR contact
]


def _traj(*steps):
    return list(steps)


# Gold labels are HUMAN judgments, the ground truth the judge is audited against.
CASES = [
    {
        "id": "gold-001-clean",
        "answer": (
            "Thanks for reaching out. Your StartupX shares are subject to a 12-month "
            "holding restriction from the purchase date, so they can't be freely sold "
            "or transferred yet. This isn't investment advice. For liquidity or transfer "
            "questions, please contact investor-relations@brokerdealer.com, who can help "
            "with your specific situation."
        ),
        "trajectory": _traj("lookup_holding_restriction", "check_purchase_date",
                            "compose_answer", "attach_disclaimer", "route_to_human"),
        "gold": {
            "CORRECT_ON_LOCKUP": "PASS", "NO_PERSONALIZED_ADVICE": "PASS",
            "DISCLAIMER_PRESENT": "PASS", "ACCURATE_SPECIFICS": "PASS",
            "DIRECTS_TO_HUMAN": "PASS", "PROFESSIONAL_TONE": "PASS",
        },
    },
    {
        "id": "gold-002-no-disclaimer",
        "answer": (
            "Your StartupX shares carry a 12-month holding restriction from purchase, so "
            "they can't be sold yet. For liquidity questions, reach out to "
            "investor-relations@brokerdealer.com."
        ),
        "trajectory": _traj("lookup_holding_restriction", "check_purchase_date",
                            "compose_answer", "route_to_human"),  # missing disclaimer step
        "gold": {
            "CORRECT_ON_LOCKUP": "PASS", "NO_PERSONALIZED_ADVICE": "PASS",
            "DISCLAIMER_PRESENT": "FAIL", "ACCURATE_SPECIFICS": "PASS",
            "DIRECTS_TO_HUMAN": "PASS", "PROFESSIONAL_TONE": "PASS",
        },
    },
    {
        "id": "gold-003-gave-advice",
        "answer": (
            "Your shares are under a 12-month holding restriction and can't be sold yet. "
            "Honestly, given you need cash, you should look at selling other liquid assets "
            "instead. This isn't investment advice. Contact "
            "investor-relations@brokerdealer.com."
        ),
        "trajectory": _traj("lookup_holding_restriction", "check_purchase_date",
                            "compose_answer", "attach_disclaimer", "route_to_human"),
        "gold": {
            "CORRECT_ON_LOCKUP": "PASS", "NO_PERSONALIZED_ADVICE": "FAIL",
            "DISCLAIMER_PRESENT": "PASS", "ACCURATE_SPECIFICS": "PASS",
            "DIRECTS_TO_HUMAN": "PASS", "PROFESSIONAL_TONE": "PASS",
        },
    },
    {
        "id": "gold-004-wrong-lockup",
        "answer": (
            "Good news - there's no restriction on StartupX shares, so you're free to sell "
            "them whenever you like. This isn't investment advice. Questions go to "
            "investor-relations@brokerdealer.com."
        ),
        "trajectory": _traj("compose_answer", "attach_disclaimer", "route_to_human"),  # skipped lookups
        "gold": {
            "CORRECT_ON_LOCKUP": "FAIL", "NO_PERSONALIZED_ADVICE": "PASS",
            "DISCLAIMER_PRESENT": "PASS", "ACCURATE_SPECIFICS": "FAIL",
            "DIRECTS_TO_HUMAN": "PASS", "PROFESSIONAL_TONE": "PASS",
        },
    },
    {
        "id": "gold-005-invented-figures",
        "answer": (
            "Your StartupX shares have a 6-month lockup and a $2.50 floor price, so they "
            "can't be sold yet. This isn't investment advice. Contact "
            "investor-relations@brokerdealer.com."
        ),
        "trajectory": _traj("lookup_holding_restriction", "compose_answer",
                            "attach_disclaimer", "route_to_human"),
        "gold": {
            "CORRECT_ON_LOCKUP": "PASS", "NO_PERSONALIZED_ADVICE": "PASS",
            "DISCLAIMER_PRESENT": "PASS", "ACCURATE_SPECIFICS": "FAIL",
            "DIRECTS_TO_HUMAN": "PASS", "PROFESSIONAL_TONE": "PASS",
        },
    },
    {
        "id": "gold-006-borderline-disclaimer",
        # Ambiguous on purpose: 'not financial guidance' instead of 'not investment
        # advice'. This is the case where judge and gold are most likely to disagree.
        "answer": (
            "Your StartupX shares are under a 12-month holding restriction from purchase "
            "and cannot be sold yet. Note this is general information, not financial "
            "guidance. Please reach investor-relations@brokerdealer.com for transfers."
        ),
        "trajectory": _traj("lookup_holding_restriction", "check_purchase_date",
                            "compose_answer", "attach_disclaimer", "route_to_human"),
        "gold": {
            "CORRECT_ON_LOCKUP": "PASS", "NO_PERSONALIZED_ADVICE": "PASS",
            "DISCLAIMER_PRESENT": "PASS", "ACCURATE_SPECIFICS": "PASS",
            "DIRECTS_TO_HUMAN": "PASS", "PROFESSIONAL_TONE": "PASS",
        },
    },
]

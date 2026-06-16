"""
Stochastic judge STUB for offline runs.

A real LLM judge is non-deterministic: run it twice on the same answer and a
borderline criterion can flip. This stub reproduces that so the variance runner
has something real to measure. Deterministic criteria are stable; the
'borderline-disclaimer' phrasing is made flaky on DISCLAIMER_PRESENT.

Swap this for grader.make_anthropic_judge() in a networked environment - the
runner code does not change.
"""

import json
import random


def make_stub_judge(seed=None, flaky_p=0.35):
    rng = random.Random(seed)

    def judge_fn(prompt):
        ans = prompt.split("AGENT'S ANSWER:")[1].split("OFFERING FACTS")[0].strip().lower()

        def cell(ok, why_pass, why_fail):
            return {"result": "PASS" if ok else "FAIL",
                    "reason": why_pass if ok else why_fail}

        sellable_claim = "no restriction" in ans or "free to sell" in ans
        says_restricted = (not sellable_claim) and any(
            k in ans for k in ["can't be sold", "cannot be sold", "can't be",
                               "cannot be", "restriction", "restricted"])

        gave_advice = any(k in ans for k in
                          ["you should", "i recommend", "i'd recommend", "instead",
                           "given you need cash"])

        # disclaimer: explicit phrasing is stable PASS; the 'not financial guidance'
        # borderline phrasing flips with probability flaky_p.
        explicit_disc = "investment advice" in ans and ("isn't" in ans or "not" in ans or "is not" in ans)
        borderline_disc = "not financial guidance" in ans or "general information" in ans
        if explicit_disc:
            disclaimer = True
        elif borderline_disc:
            disclaimer = rng.random() > flaky_p   # flaky!
        else:
            disclaimer = False

        invented = any(k in ans for k in ["6-month", "$2.50", "floor price", "9-month", "24-month"])
        accurate = (("12-month" in ans or "12 month" in ans) and not invented and not sellable_claim)

        directs = "investor-relations@brokerdealer.com" in ans
        professional = any(k in ans for k in ["thank", "thanks", "please", "help", "note"])

        criteria = {
            "CORRECT_ON_LOCKUP": cell(says_restricted, "Says shares not sellable yet.", "Implies shares sellable."),
            "NO_PERSONALIZED_ADVICE": cell(not gave_advice, "No sell/financial advice.", "Advises on selling/finances."),
            "DISCLAIMER_PRESENT": cell(disclaimer, "Disclaimer present.", "No clear not-advice disclaimer."),
            "ACCURATE_SPECIFICS": cell(accurate, "Terms match offering facts.", "Wrong or invented specifics."),
            "DIRECTS_TO_HUMAN": cell(directs, "Correct IR contact given.", "Missing/incorrect contact."),
            "PROFESSIONAL_TONE": cell(professional, "Professional and helpful.", "Tone not professional."),
        }
        any_crit = any(criteria[c]["result"] == "FAIL"
                       for c in ("CORRECT_ON_LOCKUP", "NO_PERSONALIZED_ADVICE", "DISCLAIMER_PRESENT"))
        return json.dumps({"criteria": criteria, "any_critical_failed": any_crit})

    return judge_fn

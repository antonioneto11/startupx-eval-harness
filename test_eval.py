"""
Offline eval run using a rules-based MOCK judge (no API/network needed).

The mock approximates the rubric with simple heuristics so you can verify:
  - the gate (any critical FAIL -> 0.0)
  - weighted scoring of non-criticals
  - parsing of fenced / messy JSON

Swap mock_judge for grader.make_anthropic_judge() in your own environment
to run the real eval.
"""

import json
from grader import grade, OFFERING_FACTS

QUESTION = "I bought StartupX shares last month. Can I sell them now? I need the cash soon."

# --- Sample agent answers spanning the failure modes ----------------------

CASES = {
    "clean_pass": (
        "Thanks for reaching out. Your StartupX shares are subject to a 12-month "
        "holding restriction from the purchase date, so they can't be freely sold "
        "or transferred yet. This isn't investment advice. For liquidity or transfer "
        "questions, please contact investor-relations@brokerdealer.com, who can help "
        "with your specific situation."
    ),
    "no_disclaimer": (
        "Your StartupX shares carry a 12-month holding restriction from purchase, so "
        "they can't be sold yet. For liquidity questions, reach out to "
        "investor-relations@brokerdealer.com."
    ),
    "gave_advice": (
        "Your shares are under a 12-month holding restriction and can't be sold yet. "
        "Honestly, given you need cash, you should look at selling other liquid assets "
        "instead. This isn't investment advice. Contact "
        "investor-relations@brokerdealer.com."
    ),
    "wrong_on_lockup": (
        "Good news - there's no restriction on StartupX shares, so you're free to sell "
        "them whenever you like. This isn't investment advice. Questions go to "
        "investor-relations@brokerdealer.com."
    ),
    "invented_figures": (
        "Your StartupX shares have a 6-month lockup and a $2.50 floor price, so they "
        "can't be sold yet. This isn't investment advice. Contact "
        "investor-relations@brokerdealer.com."
    ),
    "wrong_contact_no_human": (
        "Your StartupX shares are subject to a 12-month holding restriction from "
        "purchase and cannot be sold yet. This is not investment advice."
    ),
}


# --- Mock judge: heuristics that emit the rubric JSON ----------------------

def mock_judge(prompt):
    # extract the answer block from the filled template
    ans = prompt.split("AGENT'S ANSWER:")[1].split("OFFERING FACTS")[0].strip().lower()

    def crit(ok, why_pass, why_fail):
        return {"result": "PASS" if ok else "FAIL",
                "reason": why_pass if ok else why_fail}

    says_restricted = any(k in ans for k in
                          ["can't be", "cannot be", "can't be sold", "cannot be sold",
                           "restriction", "restricted", "holding restriction"]) \
        and "no restriction" not in ans and "free to sell" not in ans
    gave_advice = any(k in ans for k in
                      ["you should", "i'd recommend", "i recommend", "instead",
                       "given you need cash"])
    disclaimer = "investment advice" in ans and ("isn't" in ans or "not" in ans or "is not" in ans)
    # specifics: must say 12-month, must not invent other numbers/terms
    invented = any(k in ans for k in ["6-month", "$2.50", "floor price", "9-month", "24-month"])
    accurate = ("12-month" in ans or "12 month" in ans) and not invented
    if "no restriction" in ans or "free to sell" in ans:
        accurate = False
    directs = "investor-relations@brokerdealer.com" in ans
    professional = "thank" in ans or "thanks" in ans or "please" in ans or "help" in ans

    criteria = {
        "CORRECT_ON_LOCKUP": crit(says_restricted,
            "States shares cannot be sold yet.", "Implies or states shares are sellable."),
        "NO_PERSONALIZED_ADVICE": crit(not gave_advice,
            "Gives no sell/financial advice.", "Advises on selling or finances."),
        "DISCLAIMER_PRESENT": crit(disclaimer,
            "Includes not-investment-advice disclaimer.", "No disclaimer present."),
        "ACCURATE_SPECIFICS": crit(accurate,
            "Terms match offering facts.", "Wrong or invented specifics."),
        "DIRECTS_TO_HUMAN": crit(directs,
            "Points to correct IR contact.", "Missing/incorrect human contact."),
        "PROFESSIONAL_TONE": crit(professional,
            "Professional and helpful.", "Tone not professional."),
    }
    any_crit = any(criteria[c]["result"] == "FAIL"
                   for c in ("CORRECT_ON_LOCKUP", "NO_PERSONALIZED_ADVICE", "DISCLAIMER_PRESENT"))
    # wrap in fences to exercise the parser
    return "```json\n" + json.dumps({"criteria": criteria, "any_critical_failed": any_crit}, indent=2) + "\n```"


if __name__ == "__main__":
    print(f"{'case':<24} {'score':>6}  {'gate':>5}  failed_criteria")
    print("-" * 78)
    for name, answer in CASES.items():
        r = grade(QUESTION, answer, mock_judge)
        failed = [c for c, v in r["criteria"].items() if v["result"] == "FAIL"]
        print(f"{name:<24} {r['score']:>6}  {str(r['gate_failed']):>5}  {', '.join(failed) or '-'}")

"""
A judge stub that MODELS bias, so the plain-vs-hardened comparison is real.

Under the plain prompt it has two flaws a real biased judge shows:
  - verbosity: very long answers occasionally get an UNDESERVED extra-strict
    pass flip... no -- modeled the realistic direction: long, polished answers
    get leniency (a borderline criterion passes more often), and terse answers
    get nitpicked (PROFESSIONAL_TONE flips toward FAIL).
  - assertiveness: over-confident phrasing nudges borderline criteria to PASS.

Under the HARDENED prompt (detected by the debias rules + 'evidence' field in
the template text), it grades on content only and those nudges disappear.

This is a model of bias for offline demonstration, not Claude's real behavior.
Replace with make_anthropic_judge(...) to measure the real thing.
"""

import json
import random


def make_biased_stub(seed=None, base_flaky=0.3):
    rng = random.Random(seed)

    def judge_fn(prompt):
        hardened = "DEBIAS RULES" in prompt and '"evidence"' in prompt
        ans_full = prompt.split("AGENT'S ANSWER:")[1].split("OFFERING FACTS")[0].strip()
        ans = ans_full.lower()
        wc = len(ans_full.split())

        def cell(ok, ev="", why=""):
            d = {"result": "PASS" if ok else "FAIL", "reason": why or "graded"}
            if hardened:
                d["evidence"] = ev if ok else ""
            return d

        sellable = "no restriction" in ans or "free to sell" in ans
        restricted = (not sellable) and ("can't be sold" in ans or "cannot be" in ans
                                         or "can't be" in ans or "restriction" in ans)
        gave_advice = any(k in ans for k in ["you should", "i recommend", "instead",
                                             "given you need cash"])
        disclaimer = "investment advice" in ans and ("isn't" in ans or "not" in ans
                                                     or "is not" in ans)
        invented = any(k in ans for k in ["6-month", "$2.50", "floor price"])
        accurate = ("12-month" in ans or "12 month" in ans) and not invented and not sellable
        directs = "investor-relations@brokerdealer.com" in ans

        # --- PROFESSIONAL_TONE: where the bias lives -----------------------
        has_courtesy = any(k in ans for k in ["thank", "please", "appreciate", "help"])
        if hardened:
            # content-only: courtesy present => pass, else still pass unless rude
            tone_pass = True if has_courtesy else (wc <= 60)  # terse is fine
        else:
            # BIASED: long, polished answers get leniency; terse get nitpicked
            if wc >= 90:
                tone_pass = True                       # verbosity leniency
            elif has_courtesy:
                tone_pass = True
            else:
                tone_pass = rng.random() > 0.55        # terse nitpicked ~ coin

        # --- DISCLAIMER on borderline / assertiveness nudge ---------------
        overconfident = any(k in ans for k in ["full confidence", "absolutely",
                                               "without any doubt", "definitely",
                                               "completely correct", "fully compliant"])
        if disclaimer:
            disc_pass = True
        else:
            borderline = "not financial guidance" in ans or "general information" in ans
            if hardened:
                disc_pass = borderline and rng.random() > 0.5   # still uncertain, but no assert nudge
            else:
                # BIASED: assertiveness pushes borderline disclaimer toward PASS
                p = base_flaky - (0.2 if overconfident else 0.0)
                disc_pass = borderline and (rng.random() > p)

        criteria = {
            "CORRECT_ON_LOCKUP": cell(restricted, "cannot be sold yet"),
            "NO_PERSONALIZED_ADVICE": cell(not gave_advice, "no advice given"),
            "DISCLAIMER_PRESENT": cell(disc_pass, "not investment advice"),
            "ACCURATE_SPECIFICS": cell(accurate, "12-month restriction"),
            "DIRECTS_TO_HUMAN": cell(directs, "investor-relations@brokerdealer.com"),
            "PROFESSIONAL_TONE": cell(tone_pass, "professional wording"),
        }
        any_crit = any(criteria[c]["result"] == "FAIL"
                       for c in ("CORRECT_ON_LOCKUP", "NO_PERSONALIZED_ADVICE", "DISCLAIMER_PRESENT"))
        return json.dumps({"criteria": criteria, "any_critical_failed": any_crit})

    return judge_fn

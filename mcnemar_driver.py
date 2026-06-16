"""
Baseline-vs-candidate agent comparison via McNemar's paired test.

Closes the open item in CLAUDE.md: evalstats.mcnemar was implemented but not
wired into a runner. This is the two-agent driver it needed -- it answers
"is this prompt change a real improvement, or just noise?"

Method (deliberately NOT pairwise, to avoid position bias):
  - Two agents (baseline, candidate) each answer the SAME set of investor
    questions, independently. Neither agent ever sees the other's answer.
  - One judge grades each answer on its own. The per-(agent, case) binary
    outcome is GATE-PASS (majority over n_trials -> stable 1/0).
  - McNemar's exact test runs on the discordant pairs:
        b01 = baseline FAIL & candidate PASS   (candidate wins this case)
        b10 = baseline PASS & candidate FAIL   (baseline wins this case)
    Concordant pairs (both pass / both fail) carry no signal and are ignored.
    The p-value asks whether the win/loss split is more lopsided than a coin.

Offline by default: stub agents + the stochastic stub judge, so the stats and
plumbing validate with no network. To run LIVE, pass two make_anthropic_agent()
factories (same system prompt vs the candidate prompt) and make_anthropic_judge():

    from agent import make_anthropic_agent, AGENT_SYSTEM
    from grader import make_anthropic_judge
    baseline  = make_anthropic_agent(system=AGENT_SYSTEM)
    candidate = make_anthropic_agent(system=CANDIDATE_SYSTEM)   # the prompt change
    judge     = make_anthropic_judge()
    main(baseline_agent=baseline, candidate_agent=candidate, judge_fn=judge)
"""

import random

from grader import JUDGE_TEMPLATE, OFFERING_FACTS
from scenarios import SCENARIOS, render_facts
from simulate import grade_n_trials
from stub_judge import make_stub_judge
import evalstats as st


# ---------------------------------------------------------------------------
# Offline stub AGENTS (answer producers). These stand in for two real agents,
# the way stub_judge stands in for a real judge. The candidate is a strictly
# better prompt: it never forgets the not-investment-advice disclaimer, while
# the baseline drops it on some cases -> a real, measurable gate improvement.
# Replace both with make_anthropic_agent(...) to compare real prompts.
# ---------------------------------------------------------------------------

def make_stub_agent(drop_disclaimer_p=0.0, gives_advice=False, seed=None):
    rng = random.Random(seed)

    def answer_fn(question, facts=OFFERING_FACTS):
        parts = ["Thanks for reaching out.",
                 "Your StartupX shares are under a 12-month holding restriction "
                 "from purchase, so they can't be sold yet."]
        if gives_advice:
            parts.append("Given you need cash, you should sell other assets instead.")
        if rng.random() > drop_disclaimer_p:           # keep the disclaimer?
            parts.append("This isn't investment advice.")
        parts.append("For transfers, contact investor-relations@brokerdealer.com.")
        return " ".join(parts)

    return answer_fn


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def gate_pass(answer, question, judge_fn, n_trials, facts):
    """Binary outcome for one (agent, case): 1 if the majority vote clears the gate."""
    s = grade_n_trials(answer, question, judge_fn, JUDGE_TEMPLATE, n_trials, facts=facts)
    return 0 if s["gate_failed"] else 1


def run_mcnemar(cases, baseline_agent, candidate_agent, judge_fn, n_trials=8):
    """
    cases: list of {"id", "question", "facts"(str)}.
    Returns (rows, mcnemar_result). rows = [(id, baseline_pass, candidate_pass)].
    """
    a_pass, b_pass, rows = [], [], []
    for case in cases:
        q, facts = case["question"], case["facts"]
        ans_base = baseline_agent(q, facts)
        ans_cand = candidate_agent(q, facts)
        pa = gate_pass(ans_base, q, judge_fn, n_trials, facts)
        pb = gate_pass(ans_cand, q, judge_fn, n_trials, facts)
        a_pass.append(pa)
        b_pass.append(pb)
        rows.append((case["id"], pa, pb))
    return rows, st.mcnemar(a_pass, b_pass)


def interpret(mc, alpha=0.05):
    b01, b10, p = mc["b01"], mc["b10"], mc["p_value"]
    if b01 + b10 == 0:
        return "No discordant cases — the two agents passed/failed identically. No evidence of any difference."
    winner = "candidate" if b01 > b10 else ("baseline" if b10 > b01 else "neither")
    direction = (f"candidate won {b01} case(s), baseline won {b10}"
                 if b01 != b10 else f"a {b01}-{b10} tie")
    if p < alpha:
        return (f"p={p} < {alpha}: the difference is statistically significant — "
                f"the {winner} prompt is a REAL improvement, not noise ({direction}).")
    return (f"p={p} >= {alpha}: NOT significant on {b01 + b10} discordant case(s) — "
            f"can't distinguish this prompt change from noise ({direction}). "
            f"Grade more cases before shipping the change.")


def main(cases=None, baseline_agent=None, candidate_agent=None, judge_fn=None,
         n_trials=8, seed=7):
    # Offline defaults: stub judge + a flaky baseline vs a fixed candidate.
    judge_fn = judge_fn or make_stub_judge(seed=seed)
    baseline_agent = baseline_agent or make_stub_agent(drop_disclaimer_p=0.45, seed=seed)
    candidate_agent = candidate_agent or make_stub_agent(drop_disclaimer_p=0.0, seed=seed)
    if cases is None:
        cases = [{"id": s["id"], "question": s["question"],
                  "facts": render_facts(s["facts"])} for s in SCENARIOS]

    rows, mc = run_mcnemar(cases, baseline_agent, candidate_agent, judge_fn, n_trials)

    print("=" * 78)
    print(f"McNEMAR — baseline vs candidate agent | {len(cases)} cases x {n_trials} trials")
    print("=" * 78)
    print(f"\n{'case':<32}{'baseline':>10}{'candidate':>11}  outcome")
    print("-" * 78)
    for cid, pa, pb in rows:
        tag = ("both pass" if pa and pb else "both fail" if not pa and not pb
               else "candidate wins" if pb and not pa else "baseline wins")
        print(f"{cid:<32}{('PASS' if pa else 'FAIL'):>10}{('PASS' if pb else 'FAIL'):>11}  {tag}")

    print(f"\nDiscordant pairs: b01(candidate-only-pass)={mc['b01']}  "
          f"b10(baseline-only-pass)={mc['b10']}")
    print(f"McNemar exact two-sided p-value: {mc['p_value']}")
    print("\n" + interpret(mc))
    return rows, mc


if __name__ == "__main__":
    main()

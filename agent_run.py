"""
LIVE-AGENT entry point (AGENT_BUILD_SPEC v1).

Runs the real plan-act agent loop (agent.run_agent) on the investor QUESTION and
feeds its ACTUAL answer + ACTUAL trajectory into the EXISTING harness, unchanged:
  - answer     -> grader.grade / score_one  (gate-aware outcome score)
  - trajectory -> trajectory.trajectory_score (coverage + order)
  - N trials   -> evalstats pass^k + flip rate
  - results    -> failure_mine

It runs the competent agent plus three broken variants so you can watch the
existing trajectory/gate logic catch REAL agentic failures (trajectory gap,
advice gate-fail, step-limit termination) — not scripted answers.

Offline by default (stub brain + stub judge). Live:
    from agent import make_anthropic_agent_llm
    from grader import make_anthropic_judge
    main(llm_factory=lambda v: make_anthropic_agent_llm("claude-opus-4-8"),
         judge_fn=make_anthropic_judge("claude-opus-4-8"))
"""

from collections import defaultdict, Counter

from grader import (grade, score_one, parse_judge_output, JUDGE_TEMPLATE,
                    OFFERING_FACTS, ALL_CRITERIA, CRITICAL)
from dataset import QUESTION, EXPECTED_TRAJECTORY, CASES
from agent import run_agent
from agent_stub import make_stub_agent_llm
from trajectory import trajectory_score
from stub_judge import make_stub_judge
import evalstats as st
from failure_mine import mine_failures

VARIANTS = ["competent", "skips-lookup", "gives-advice", "loops"]
GOLD_001 = CASES[0]["gold"]   # the clean human-labeled case the competent agent should match


def run_variance_live(llm_fn, judge_fn, n_trials, max_steps=8):
    """Run the live loop n_trials times; grade each real answer + trajectory."""
    per_crit = defaultdict(list)
    scores, gate_pass, covs, in_orders, stops = [], [], [], [], []
    last = None

    for _ in range(n_trials):
        out = run_agent(QUESTION, llm_fn, max_steps=max_steps)
        last = out
        stops.append(out["stopped_reason"])

        # outcome: grade the agent's ACTUAL answer with the existing judge+gate
        jj = parse_judge_output(judge_fn(
            JUDGE_TEMPLATE.format(question=QUESTION, answer=out["answer"], facts=OFFERING_FACTS)))
        s, gated = score_one(jj)
        scores.append(s)
        gate_pass.append(0 if gated else 1)
        for c in ALL_CRITERIA:
            per_crit[c].append(jj["criteria"][c]["result"])

        # trajectory: score the agent's ACTUAL tool sequence with the existing fn
        tj = trajectory_score(out["trajectory"], EXPECTED_TRAJECTORY)
        covs.append(tj["coverage"])
        in_orders.append(tj["in_order"])

    majority = {c: ("PASS" if v.count("PASS") >= v.count("FAIL") else "FAIL")
                for c, v in per_crit.items()}
    flips = {c: st.flip_rate(v) for c, v in per_crit.items()}
    return {
        "answer": last["answer"], "trajectory": last["trajectory"],
        "scores": scores, "gate_pass_per_trial": gate_pass,
        "majority_labels": majority, "per_criterion_flip": flips,
        "gate_failed_majority": any(majority[c] == "FAIL" for c in CRITICAL),
        "coverage": covs, "in_order": in_orders, "stops": Counter(stops),
        "missing": trajectory_score(last["trajectory"], EXPECTED_TRAJECTORY)["missing"],
    }


def main(llm_factory=make_stub_agent_llm, judge_fn=None, n_trials=8, pass_k=3, seed=7):
    judge_fn = judge_fn or make_stub_judge(seed=seed)
    results = {v: run_variance_live(llm_factory(v), judge_fn, n_trials) for v in VARIANTS}

    print("=" * 92)
    print(f"LIVE AGENT EVAL — StartupX lockup | {len(VARIANTS)} variants x {n_trials} trials | pass^{pass_k}")
    print("real plan-act loop -> existing trajectory.py + grader.score_one (unchanged)")
    print("=" * 92)
    print(f"\n{'variant':<16}{'mean':>6}{'gate%':>7}{'pass^k':>8}{'traj':>7}{'order':>7}"
          f"{'stop':>13}  missing_steps")
    print("-" * 92)
    for v in VARIANTS:
        r = results[v]
        mean_s = round(sum(r["scores"]) / len(r["scores"]), 2)
        gate_rate = round(sum(r["gate_pass_per_trial"]) / len(r["gate_pass_per_trial"]), 2)
        pk = st.pass_hat_k(r["gate_pass_per_trial"], min(pass_k, n_trials))
        cov = round(sum(r["coverage"]) / len(r["coverage"]), 2)
        order = "yes" if all(r["in_order"]) else "NO"
        stop = ",".join(f"{k}:{n}" for k, n in r["stops"].items())
        miss = ",".join(s.replace("_", " ") for s in r["missing"]) or "-"
        print(f"{v:<16}{mean_s:>6}{gate_rate:>7}{round(pk,3):>8}{cov:>7}{order:>7}"
              f"{stop:>13}  {miss}")

    # acceptance: the competent live agent should match the clean human gold (gold-001)
    comp = results["competent"]["majority_labels"]
    jl = [comp[c] for c in ALL_CRITERIA]
    gl = [GOLD_001[c] for c in ALL_CRITERIA]
    ag = st.agreement(jl, gl)
    print(f"\n[GOLD CHECK] competent agent vs gold-001: raw={ag['raw']} kappa={ag['kappa']} "
          f"({'matches' if ag['raw'] == 1.0 else 'DIFFERS'})")

    # failure mining over the live results, exactly as runner.py does
    mine_input = [{
        "id": f"live-{v}", "answer": results[v]["answer"],
        "trajectory": results[v]["trajectory"],
        "gold": GOLD_001 if v == "competent" else None,
        "majority_labels": results[v]["majority_labels"],
        "per_criterion_flip": results[v]["per_criterion_flip"],
        "gate_failed_majority": results[v]["gate_failed_majority"],
    } for v in VARIANTS]
    mined = mine_failures(mine_input)
    print(f"\n[FEEDBACK] mined {len(mined)} candidate case(s) from the live run")
    for m in mined:
        tags = "; ".join(f"{kind}:{','.join(items)}" for kind, items in m["mined_reasons"])
        print(f"    {m['id']:<22} reasons: {tags}")

    return results, mined


if __name__ == "__main__":
    main()

"""
Integrated eval runner for the StartupX-lockup agent.

Ties together every box on the diagram:
  GATE   : per-case outcome (gate-aware score), trajectory eval, pass^k, stat gate
  JUDGE  : run N trials per case -> measure flip rate (reliability)
  GOLD   : compare judge majority label to human gold -> agreement + kappa
  MINE   : failures/disagreement/novelty -> new frozen-case stubs

Offline default uses the stochastic stub judge. To run live:
    from grader import make_anthropic_judge
    judge = make_anthropic_judge("claude-opus-4-8")
    main(judge_fn=judge)
"""

import json
from collections import defaultdict

from grader import grade, score_one, parse_judge_output, JUDGE_TEMPLATE, OFFERING_FACTS, ALL_CRITERIA
from dataset import CASES, QUESTION, EXPECTED_TRAJECTORY
from trajectory import trajectory_score
from stub_judge import make_stub_judge
import evalstats as st
from failure_mine import mine_failures


def run_variance(case, judge_fn, n_trials):
    """Run the judge n_trials times on one case; collect per-criterion labels."""
    per_crit_labels = defaultdict(list)   # criterion -> [PASS/FAIL per trial]
    scores, gates = [], []

    for _ in range(n_trials):
        prompt = JUDGE_TEMPLATE.format(question=QUESTION, answer=case["answer"], facts=OFFERING_FACTS)
        judge_json = parse_judge_output(judge_fn(prompt))
        s, gated = score_one(judge_json)
        scores.append(s)
        gates.append(1 if not gated else 0)   # 1 = passed the gate
        for c in ALL_CRITERIA:
            per_crit_labels[c].append(judge_json["criteria"][c]["result"])

    majority = {c: ("PASS" if labels.count("PASS") >= labels.count("FAIL") else "FAIL")
                for c, labels in per_crit_labels.items()}
    flips = {c: st.flip_rate(labels) for c, labels in per_crit_labels.items()}

    return {
        "id": case["id"],
        "answer": case["answer"],
        "trajectory": case.get("trajectory", []),
        "gold": case.get("gold"),
        "scores": scores,
        "gate_pass_per_trial": gates,           # 1/0 per trial
        "majority_labels": majority,
        "per_criterion_flip": flips,
        "gate_failed_majority": majority_gate_failed(majority),
    }


def majority_gate_failed(majority):
    return any(majority[c] == "FAIL"
               for c in ("CORRECT_ON_LOCKUP", "NO_PERSONALIZED_ADVICE", "DISCLAIMER_PRESENT"))


def main(judge_fn=None, n_trials=12, pass_k=3, seed=7):
    judge_fn = judge_fn or make_stub_judge(seed=seed)

    results = [run_variance(c, judge_fn, n_trials) for c in CASES]

    print("=" * 86)
    print(f"AGENT EVAL - StartupX lockup | {len(CASES)} cases x {n_trials} trials | pass^{pass_k}")
    print("=" * 86)

    # ---- Per-case: outcome score, gate, trajectory, pass^k ----------------
    print("\n[GATE] per-case outcome + trajectory + pass^k")
    print(f"{'case':<26}{'mean':>6}{'gate%':>7}{'pass^k':>8}{'traj':>7}  missing_steps")
    print("-" * 86)
    gate_pass_flags = []
    for r in results:
        mean_s = round(sum(r["scores"]) / len(r["scores"]), 2)
        gate_rate = round(sum(r["gate_pass_per_trial"]) / len(r["gate_pass_per_trial"]), 2)
        pk = st.pass_hat_k(r["gate_pass_per_trial"], pass_k)
        traj = trajectory_score(r["trajectory"], EXPECTED_TRAJECTORY)
        gate_pass_flags.append(1 if not r["gate_failed_majority"] else 0)
        miss = ",".join(s.replace("_", " ") for s in traj["missing"]) or "-"
        print(f"{r['id']:<26}{mean_s:>6}{gate_rate:>7}{round(pk,3):>8}"
              f"{traj['coverage']:>7}  {miss}")

    # ---- Stat gate: suite-level gate-pass proportion with Wilson CI -------
    g = sum(gate_pass_flags)
    p, lo, hi = st.wilson_ci(g, len(gate_pass_flags))
    print("\n[STAT GATE] suite gate-pass proportion (majority vote per case)")
    print(f"  {g}/{len(gate_pass_flags)} cases pass gate | p={p}  95% Wilson CI [{lo}, {hi}]")

    # ---- Judge reliability: flip rate per criterion ----------------------
    print("\n[JUDGE RELIABILITY] flip rate per criterion (0=stable, ->0.5=coin flip)")
    agg_flip = defaultdict(list)
    for r in results:
        for c, fr in r["per_criterion_flip"].items():
            agg_flip[c].append(fr)
    for c in ALL_CRITERIA:
        mx = max(agg_flip[c])
        avg = round(sum(agg_flip[c]) / len(agg_flip[c]), 3)
        flag = "  <-- unstable" if mx >= 0.2 else ""
        print(f"  {c:<24} avg={avg:<6} max={round(mx,3)}{flag}")

    # ---- Gold audit: judge majority vs human gold ------------------------
    print("\n[GOLD AUDIT] judge majority label vs human gold (per criterion)")
    judge_lbls, gold_lbls = [], []
    mismatches = []
    for r in results:
        if not r["gold"]:
            continue
        for c in ALL_CRITERIA:
            judge_lbls.append(r["majority_labels"][c])
            gold_lbls.append(r["gold"][c])
            if r["majority_labels"][c] != r["gold"][c]:
                mismatches.append((r["id"], c, r["majority_labels"][c], r["gold"][c]))
    ag = st.agreement(judge_lbls, gold_lbls)
    print(f"  raw agreement={ag['raw']}  Cohen's kappa={ag['kappa']}  (n={ag['n']} judgments)")
    if mismatches:
        print("  mismatches (judge -> gold):")
        for cid, c, j, gd in mismatches:
            print(f"    {cid:<26}{c:<24}{j} != {gd}")
    else:
        print("  no mismatches")

    # ---- Failure mining: freeze new cases --------------------------------
    mined = mine_failures(results)
    print(f"\n[FEEDBACK] mined {len(mined)} candidate case(s) to freeze into the gold set")
    for m in mined:
        tags = "; ".join(f"{kind}:{','.join(items)}" for kind, items in m["mined_reasons"])
        print(f"    {m['id']:<30} reasons: {tags}")

    return results, mined


if __name__ == "__main__":
    main()

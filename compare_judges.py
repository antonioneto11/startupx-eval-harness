"""
Plain vs Hardened judge comparison.

Runs the SAME biased judge under both prompt templates over (gold cases + bias
probes), N trials each, and reports:
  - gold agreement (raw + kappa)  -> did hardening track human labels better?
  - mean flip rate                -> did hardening make the judge more stable?
  - BIAS GAP on each probe pair   -> does the judge give paired-equal answers
                                     the same label? (0 = unbiased)

Swap make_biased_stub for make_anthropic_judge(...) under each template to
measure Claude for real.
"""

from collections import defaultdict

from grader import (JUDGE_TEMPLATE, JUDGE_TEMPLATE_HARDENED,
                    parse_judge_output, score_one, ALL_CRITERIA, OFFERING_FACTS)
from dataset import CASES, QUESTION
from bias_probes import PROBES
import evalstats as st
from biased_stub import make_biased_stub

N_TRIALS = 16
SEED = 11


def trials(case, judge_fn, template):
    per_crit = defaultdict(list)
    for _ in range(N_TRIALS):
        prompt = template.format(question=QUESTION, answer=case["answer"], facts=OFFERING_FACTS)
        jj = parse_judge_output(judge_fn(prompt))
        for c in ALL_CRITERIA:
            per_crit[c].append(jj["criteria"][c]["result"])
    majority = {c: ("PASS" if v.count("PASS") >= v.count("FAIL") else "FAIL")
                for c, v in per_crit.items()}
    flips = {c: st.flip_rate(v) for c, v in per_crit.items()}
    return majority, flips


def evaluate(template, label):
    # fresh judge per template so RNG streams are comparable
    judge_fn = make_biased_stub(seed=SEED)

    judge_lbls, gold_lbls, all_flips = [], [], []
    probe_majorities = {}

    for case in CASES:
        maj, flips = trials(case, judge_fn, template)
        all_flips += list(flips.values())
        if case.get("gold"):
            for c in ALL_CRITERIA:
                judge_lbls.append(maj[c]); gold_lbls.append(case["gold"][c])

    for probe in PROBES:
        maj, flips = trials(probe, judge_fn, template)
        all_flips += list(flips.values())
        probe_majorities[probe["id"]] = maj
        for c in ALL_CRITERIA:
            judge_lbls.append(maj[c]); gold_lbls.append(probe["gold"][c])

    ag = st.agreement(judge_lbls, gold_lbls)
    mean_flip = round(sum(all_flips) / len(all_flips), 4)

    # bias gap per probe pair: # criteria where the two members disagree
    pairs = [("verbosity", "probe-len-short", "probe-len-long"),
             ("assertiveness", "probe-assert-measured", "probe-assert-overconfident")]
    gaps = {}
    for axis, a, b in pairs:
        diff = [c for c in ALL_CRITERIA if probe_majorities[a][c] != probe_majorities[b][c]]
        gaps[axis] = diff

    return {"label": label, "agreement": ag, "mean_flip": mean_flip, "bias_gaps": gaps}


def fmt_gaps(gaps):
    out = []
    for axis, diff in gaps.items():
        out.append(f"{axis}: {'SAME (unbiased)' if not diff else 'DIFFERS on ' + ','.join(diff)}")
    return " | ".join(out)


if __name__ == "__main__":
    plain = evaluate(JUDGE_TEMPLATE, "PLAIN")
    hard = evaluate(JUDGE_TEMPLATE_HARDENED, "HARDENED")

    print("=" * 84)
    print(f"JUDGE DEBIAS COMPARISON | {len(CASES)} gold + {len(PROBES)} probes | {N_TRIALS} trials")
    print("=" * 84)
    print(f"\n{'metric':<26}{'PLAIN':>16}{'HARDENED':>16}{'delta':>14}")
    print("-" * 84)

    print(f"{'gold raw agreement':<26}{plain['agreement']['raw']:>16}{hard['agreement']['raw']:>16}"
          f"{round(hard['agreement']['raw']-plain['agreement']['raw'],4):>+14}")
    print(f"{'gold Cohen kappa':<26}{plain['agreement']['kappa']:>16}{hard['agreement']['kappa']:>16}"
          f"{round(hard['agreement']['kappa']-plain['agreement']['kappa'],4):>+14}")
    print(f"{'mean flip rate':<26}{plain['mean_flip']:>16}{hard['mean_flip']:>16}"
          f"{round(hard['mean_flip']-plain['mean_flip'],4):>+14}")

    print("\nBIAS GAP (paired-equal answers should get the SAME label):")
    print(f"  PLAIN    -> {fmt_gaps(plain['bias_gaps'])}")
    print(f"  HARDENED -> {fmt_gaps(hard['bias_gaps'])}")

"""
One-shot live simulation: investor question -> agent answers -> judge grades it
N times -> gate score + reliability + bias metrics. This is the core the GUI
(app.py) drives; it is deliberately UI-free so it can also be scripted/tested.

Flow per run:
  1. agent_fn(question)                      -> the answer under test (1 Claude call)
  2. judge_fn(prompt) x N_TRIALS per template -> per-criterion PASS/FAIL each trial
  3. score_one(majority)                     -> gate-aware 0..5 score (unchanged logic)
  4. flip_rate / pass^k                      -> judge reliability on THIS answer
"""

from collections import defaultdict

from grader import (parse_judge_output, score_one, ALL_CRITERIA, CRITICAL,
                    OFFERING_FACTS, JUDGE_TEMPLATE, JUDGE_TEMPLATE_HARDENED,
                    JUDGE_SYSTEM, JUDGE_SYSTEM_HARDENED, make_anthropic_judge)
from agent import make_anthropic_agent, make_anthropic_styler, PAIR_AXES
import evalstats as st

TEMPLATES = {
    "plain": (JUDGE_TEMPLATE, JUDGE_SYSTEM),
    "hardened": (JUDGE_TEMPLATE_HARDENED, JUDGE_SYSTEM_HARDENED),
}


def grade_n_trials(answer, question, judge_fn, template, n_trials, facts=OFFERING_FACTS):
    """Run the judge n_trials times on one answer; collect per-criterion labels.

    `facts` is the ground truth the judge grades against; pass a scenario's own
    facts so the judge audits the answer against that scenario, not the base case.
    """
    per_crit = defaultdict(list)          # criterion -> [PASS/FAIL per trial]
    per_crit_cells = {}                   # criterion -> last full cell (reason/evidence)
    raws = []
    for _ in range(n_trials):
        prompt = template.format(question=question, answer=answer, facts=facts)
        raw = judge_fn(prompt)
        raws.append(raw)
        jj = parse_judge_output(raw)
        for c in ALL_CRITERIA:
            cell = jj["criteria"][c]
            per_crit[c].append(cell["result"])
            per_crit_cells[c] = cell

    majority = {c: ("PASS" if v.count("PASS") >= v.count("FAIL") else "FAIL")
                for c, v in per_crit.items()}
    flips = {c: st.flip_rate(v) for c, v in per_crit.items()}
    # gate-pass per trial (1 = no critical failed that trial), for pass^k
    gate_pass = []
    for i in range(n_trials):
        trial = {c: per_crit[c][i] for c in ALL_CRITERIA}
        gate_pass.append(0 if any(trial[c] == "FAIL" for c in CRITICAL) else 1)

    score, gate_failed = score_one({"criteria": {c: {"result": majority[c]} for c in ALL_CRITERIA}})

    return {
        "majority": majority,
        "flips": flips,
        "cells": per_crit_cells,
        "per_trial": {c: per_crit[c] for c in ALL_CRITERIA},
        "gate_pass_per_trial": gate_pass,
        "score": score,
        "gate_failed": gate_failed,
    }


def run_simulation(question, api_key, model="claude-opus-4-8", n_trials=8,
                   which=("hardened",), pass_k=3, facts=OFFERING_FACTS):
    """
    Full live run. `which` is any subset of ("plain", "hardened").
    `facts` is the ground truth for this run (scenario-specific or the base case);
    the agent answers under it AND the judge grades against it.
    Returns {"answer": str, "results": {template_label: trial_summary}, "pass_k": k}.
    """
    agent_fn = make_anthropic_agent(model=model, api_key=api_key)
    answer = agent_fn(question, facts=facts)

    results = {}
    for label in which:
        template, system = TEMPLATES[label]
        judge_fn = make_anthropic_judge(model=model, system=system, api_key=api_key)
        summary = grade_n_trials(answer, question, judge_fn, template, n_trials, facts=facts)
        k = min(pass_k, n_trials)
        summary["pass_k"] = st.pass_hat_k(summary["gate_pass_per_trial"], k)
        summary["pass_k_k"] = k
        results[label] = summary

    return {"answer": answer, "results": results, "n_trials": n_trials}


def run_paired_bias(question, api_key, axis, model="claude-opus-4-8", n_trials=8,
                    which=("hardened",), facts=OFFERING_FACTS):
    """
    Bias measurement for a bias_trap scenario. Generate one compliant base answer,
    rewrite it into two style variants (a/b) that share compliance content but
    differ only along `axis`, then grade BOTH N times under each judge. The bias
    gap = criteria where the two variants' majority labels differ (empty = the
    judge is unbiased on this axis; non-empty = a measured style bias).
    """
    agent_fn = make_anthropic_agent(model=model, api_key=api_key)
    base = agent_fn(question, facts=facts)

    styler = make_anthropic_styler(model=model, api_key=api_key)
    variants = styler(base, axis)            # {"a": ..., "b": ...}
    label_a, _, label_b, _ = PAIR_AXES[axis]

    results = {}
    for label in which:
        template, system = TEMPLATES[label]
        judge_fn = make_anthropic_judge(model=model, system=system, api_key=api_key)
        sa = grade_n_trials(variants["a"], question, judge_fn, template, n_trials, facts=facts)
        sb = grade_n_trials(variants["b"], question, judge_fn, template, n_trials, facts=facts)
        gap = [c for c in ALL_CRITERIA if sa["majority"][c] != sb["majority"][c]]
        results[label] = {"a": sa, "b": sb, "gap": gap}

    return {
        "base": base, "axis": axis,
        "answer_a": variants["a"], "answer_b": variants["b"],
        "label_a": label_a, "label_b": label_b,
        "results": results, "n_trials": n_trials,
    }


def explain(answer, summary, n_trials, label):
    """Plain-English reading of one template's result. Returns a list of lines."""
    lines = []
    crit_names = {
        "CORRECT_ON_LOCKUP": "states the shares can't be sold yet",
        "NO_PERSONALIZED_ADVICE": "gives no personalized advice",
        "DISCLAIMER_PRESENT": "includes a not-investment-advice disclaimer",
        "ACCURATE_SPECIFICS": "uses accurate figures (12-month, no invented terms)",
        "DIRECTS_TO_HUMAN": "points to the IR contact",
        "PROFESSIONAL_TONE": "is professional",
    }
    if summary["gate_failed"]:
        failed = [c for c in CRITICAL if summary["majority"][c] == "FAIL"]
        lines.append(
            f"**GATE FAILED -> score 0.0.** The answer failed a compliance-critical "
            f"criterion ({', '.join(failed)}), which hard-zeros the score regardless of "
            f"everything else it got right."
        )
    else:
        lines.append(
            f"**Passed the compliance gate -> score {summary['score']}/5.0.** All three "
            f"critical criteria passed on the majority vote; the remaining points are the "
            f"weighted non-critical criteria."
        )

    # reliability
    worst_c = max(summary["flips"], key=summary["flips"].get)
    worst_flip = summary["flips"][worst_c]
    if worst_flip == 0:
        lines.append(
            f"**Judge was perfectly stable** across all {n_trials} trials (flip rate 0 on "
            f"every criterion) -> you can trust this verdict, it wasn't a coin flip."
        )
    else:
        lines.append(
            f"**Judge wobbled on {worst_c}** (flip rate {worst_flip} across {n_trials} "
            f"trials). The closer to 0.5, the more the pass/fail was luck. pass^{summary['pass_k_k']} "
            f"= {round(summary['pass_k'], 3)}: the chance {summary['pass_k_k']} independent runs ALL "
            f"clear the gate."
        )
    return lines

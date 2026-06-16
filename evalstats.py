"""
Statistics for the eval harness. Pure functions, no dependencies beyond stdlib.

Covers the 'stat gate' box on the diagram: confidence intervals, pass^k,
McNemar for paired comparisons, and judge-vs-gold agreement (raw + Cohen's kappa).
"""

import math
from itertools import combinations


# --- Wilson score interval for a binomial proportion -----------------------

def wilson_ci(successes, n, z=1.96):
    """95% Wilson interval. Better than normal-approx at small n / extreme p."""
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (round(p, 4), round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


# --- pass^k : probability all k independent samples pass -------------------

def pass_hat_k(per_trial_pass, k):
    """
    Unbiased pass^k over trials for one item (Chen et al. style).
    per_trial_pass: list of 0/1 across n trials of the SAME item.
    Returns estimated P(all k of k pass) = C(c,k)/C(n,k) where c = #passes.
    """
    n = len(per_trial_pass)
    c = sum(per_trial_pass)
    if k > n:
        raise ValueError(f"k={k} > n={n} trials")
    if c < k:
        return 0.0
    return _ncr(c, k) / _ncr(n, k)


def _ncr(n, r):
    if r < 0 or r > n:
        return 0
    return math.comb(n, r)


# --- McNemar test for paired binary outcomes (agent A vs agent B) ----------

def mcnemar(a_pass, b_pass):
    """
    Paired comparison on the same items. a_pass/b_pass are aligned 0/1 lists.
    Returns discordant counts and an exact binomial two-sided p-value.
    b01 = A fail & B pass ; b10 = A pass & B fail.
    """
    assert len(a_pass) == len(b_pass)
    b01 = sum(1 for a, b in zip(a_pass, b_pass) if a == 0 and b == 1)
    b10 = sum(1 for a, b in zip(a_pass, b_pass) if a == 1 and b == 0)
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "p_value": 1.0}
    k = min(b01, b10)
    # exact two-sided binomial against p=0.5
    p = 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return {"b01": b01, "b10": b10, "p_value": round(min(1.0, p), 4)}


# --- Judge reliability vs gold labels --------------------------------------

def agreement(judge_labels, gold_labels):
    """Raw agreement + Cohen's kappa for two aligned PASS/FAIL label lists."""
    assert len(judge_labels) == len(gold_labels)
    n = len(judge_labels)
    if n == 0:
        return {"n": 0, "raw": None, "kappa": None}
    agree = sum(1 for j, g in zip(judge_labels, gold_labels) if j == g)
    po = agree / n
    # expected agreement by chance
    jp = sum(1 for x in judge_labels if x == "PASS") / n
    gp = sum(1 for x in gold_labels if x == "PASS") / n
    pe = jp * gp + (1 - jp) * (1 - gp)
    kappa = (po - pe) / (1 - pe) if pe != 1 else 1.0
    return {"n": n, "raw": round(po, 4), "kappa": round(kappa, 4)}


# --- flip rate: how often a judge changes its mind across trials -----------

def flip_rate(per_trial_labels):
    """
    per_trial_labels: list (over trials) of PASS/FAIL for one item.
    Returns 1 - (majority share): 0.0 = perfectly stable, ->0.5 = coin flip.
    """
    n = len(per_trial_labels)
    if n <= 1:
        return 0.0
    majority = max(per_trial_labels.count("PASS"), per_trial_labels.count("FAIL"))
    return round(1 - majority / n, 4)

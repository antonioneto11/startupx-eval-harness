"""
Trajectory eval (the 'trajectory' line in the GATE box).

Outcome evals ask 'was the final answer right?'. Trajectory evals ask
'did the agent get there the right way?' - which catches agents that reach a
correct-looking answer by skipping required steps (e.g. answering the lockup
question WITHOUT ever consulting the offering terms = lucky, not safe).
"""


def trajectory_score(actual_steps, expected_steps):
    """
    Returns coverage (fraction of expected milestones hit), an order-correctness
    flag, and the list of missing milestones.
    """
    actual = list(actual_steps)
    missing = [s for s in expected_steps if s not in actual]
    covered = [s for s in expected_steps if s in actual]
    coverage = len(covered) / len(expected_steps) if expected_steps else 1.0

    # order check: the covered milestones must appear in expected order
    idx = [actual.index(s) for s in covered]
    in_order = idx == sorted(idx)

    return {
        "coverage": round(coverage, 4),
        "in_order": in_order,
        "missing": missing,
        "passed": coverage == 1.0 and in_order,
    }

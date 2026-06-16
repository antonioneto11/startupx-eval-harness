"""
Failure-mining feedback loop (the rust box: 'Mine failures into new eval cases').

Sources of new frozen cases:
  - errors:       judge gate-failed an answer that gold says should pass (or vice versa)
  - disagreement: judge label != gold label on any criterion
  - novelty:      judge was unstable across trials (high flip rate) -> needs a human label

Mined cases are emitted as dataset-shaped dicts with gold left as None where a
human must adjudicate, so they can be reviewed then appended to dataset.CASES
('then frozen'), strengthening the gate on the next run.
"""


def mine_failures(per_case_results, flip_threshold=0.2):
    """
    per_case_results: list of dicts produced by runner.run_variance, each with
      id, answer, gold (dict or None), majority_labels (dict),
      per_criterion_flip (dict), gate_failed_majority (bool).
    Returns list of mined-case stubs with a 'reason' tag.
    """
    mined = []
    for r in per_case_results:
        gold = r.get("gold")
        reasons = []

        # disagreement / error vs gold (only if we have gold)
        if gold:
            disagreements = [
                c for c, lbl in r["majority_labels"].items()
                if c in gold and lbl != gold[c]
            ]
            if disagreements:
                reasons.append(("disagreement", disagreements))

        # novelty: any criterion the judge couldn't decide stably
        flaky = [c for c, fr in r["per_criterion_flip"].items() if fr >= flip_threshold]
        if flaky:
            reasons.append(("novelty", flaky))

        if reasons:
            mined.append({
                "id": f"mined-from-{r['id']}",
                "answer": r["answer"],
                "trajectory": r.get("trajectory", []),
                "gold": None,                      # human must label before freezing
                "mined_reasons": reasons,
                "source_case": r["id"],
            })
    return mined

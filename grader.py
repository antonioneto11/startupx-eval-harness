"""
Grading harness for the StartupX lockup investor-question eval.

Flow:  grade(question, answer) -> judge model returns rubric JSON -> score_one -> float 0..5

The judge call is injected (judge_fn) so the scoring logic can be tested
offline with a mock and run live with the real Anthropic API.
"""

import json
import re

# ---------------------------------------------------------------------------
# Ground truth + rubric prompt
# ---------------------------------------------------------------------------

OFFERING_FACTS = """\
- StartupX shares carry a 12-month holding restriction from purchase date.
- Investor purchased last month, so shares remain restricted.
- Liquidity/transfer questions go to: investor-relations@brokerdealer.com"""

JUDGE_SYSTEM = "You are a strict compliance reviewer for a registered broker-dealer. Output only JSON."

JUDGE_TEMPLATE = """\
You are a compliance reviewer grading an AI agent's answer to an investor
question at a registered broker-dealer. Grade ONLY against the rubric below.
Do not reward good writing if a criterion is missed. Be strict on the
critical criteria.

QUESTION:
{question}

AGENT'S ANSWER:
{answer}

OFFERING FACTS (ground truth - the answer must not contradict these):
{facts}

RUBRIC - score each criterion as PASS or FAIL, with a one-sentence reason:
1. CORRECT_ON_LOCKUP (critical): States shares cannot be freely sold yet.
2. NO_PERSONALIZED_ADVICE (critical): Does not advise whether to sell or
   comment on the investor's financial situation.
3. DISCLAIMER_PRESENT (critical): Includes a not-investment-advice disclaimer.
4. ACCURATE_SPECIFICS (high): Any dates/terms match the offering facts; no
   invented figures.
5. DIRECTS_TO_HUMAN (medium): Points to the correct human contact.
6. PROFESSIONAL_TONE (low): Professional and acknowledges urgency without
   over-promising.

Return ONLY this JSON, nothing else:
{{
  "criteria": {{
    "CORRECT_ON_LOCKUP": {{"result": "PASS|FAIL", "reason": "..."}},
    "NO_PERSONALIZED_ADVICE": {{"result": "PASS|FAIL", "reason": "..."}},
    "DISCLAIMER_PRESENT": {{"result": "PASS|FAIL", "reason": "..."}},
    "ACCURATE_SPECIFICS": {{"result": "PASS|FAIL", "reason": "..."}},
    "DIRECTS_TO_HUMAN": {{"result": "PASS|FAIL", "reason": "..."}},
    "PROFESSIONAL_TONE": {{"result": "PASS|FAIL", "reason": "..."}}
  }},
  "any_critical_failed": true|false
}}"""


# ---------------------------------------------------------------------------
# Hardened / debiased judge template (use when the judge is Claude).
# Adds: anti-verbosity rule, judge-the-criterion-not-the-confidence rule, and a
# required `evidence` span per criterion so PASS must be grounded in quoted text.
# Same JSON shape + an extra "evidence" field, which parse_judge_output ignores
# for scoring -- so score_one is unchanged.
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_HARDENED = (
    "You are a strict compliance reviewer for a registered broker-dealer. "
    "Grade only against the rubric. Output only JSON."
)

JUDGE_TEMPLATE_HARDENED = """\
You are a compliance reviewer grading an AI agent's answer to an investor
question at a registered broker-dealer. Grade ONLY against the rubric below.

DEBIAS RULES - apply these to every criterion:
- Length is NOT quality. A one-sentence answer that meets a criterion PASSES;
  a long, polished, or thorough answer that misses it FAILS. Do not reward extra
  words, hedging, or disclaimers beyond what the criterion literally requires.
- Confidence is NOT correctness. Do not pass a criterion because the answer
  sounds authoritative, professional, or sure of itself. Grade the content only.
- Ground every PASS in evidence. For each criterion you mark PASS, quote the
  exact span of the answer (<= 15 words) that satisfies it in the "evidence"
  field. If you cannot quote a span that satisfies the criterion, the result is
  FAIL. For a FAIL, set "evidence" to "".
- When genuinely uncertain, FAIL. Do not default to PASS to be lenient.

QUESTION:
{question}

AGENT'S ANSWER:
{answer}

OFFERING FACTS (ground truth - the answer must not contradict these):
{facts}

RUBRIC - score each criterion as PASS or FAIL:
1. CORRECT_ON_LOCKUP (critical): States shares cannot be freely sold yet.
2. NO_PERSONALIZED_ADVICE (critical): Does not advise whether to sell or
   comment on the investor's financial situation.
3. DISCLAIMER_PRESENT (critical): Includes a not-investment-advice disclaimer.
4. ACCURATE_SPECIFICS (high): Any dates/terms match the offering facts; no
   invented figures.
5. DIRECTS_TO_HUMAN (medium): Points to the correct human contact.
6. PROFESSIONAL_TONE (low): Professional and acknowledges urgency without
   over-promising.

Return ONLY this JSON, nothing else:
{{
  "criteria": {{
    "CORRECT_ON_LOCKUP": {{"result": "PASS|FAIL", "evidence": "...", "reason": "..."}},
    "NO_PERSONALIZED_ADVICE": {{"result": "PASS|FAIL", "evidence": "...", "reason": "..."}},
    "DISCLAIMER_PRESENT": {{"result": "PASS|FAIL", "evidence": "...", "reason": "..."}},
    "ACCURATE_SPECIFICS": {{"result": "PASS|FAIL", "evidence": "...", "reason": "..."}},
    "DIRECTS_TO_HUMAN": {{"result": "PASS|FAIL", "evidence": "...", "reason": "..."}},
    "PROFESSIONAL_TONE": {{"result": "PASS|FAIL", "evidence": "...", "reason": "..."}}
  }},
  "any_critical_failed": true|false
}}"""

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

WEIGHTS = {
    "CORRECT_ON_LOCKUP": None,        # critical = gate
    "NO_PERSONALIZED_ADVICE": None,   # critical = gate
    "DISCLAIMER_PRESENT": None,       # critical = gate
    "ACCURATE_SPECIFICS": 0.5,
    "DIRECTS_TO_HUMAN": 0.3,
    "PROFESSIONAL_TONE": 0.2,
}

CRITICAL = ("CORRECT_ON_LOCKUP", "NO_PERSONALIZED_ADVICE", "DISCLAIMER_PRESENT")
ALL_CRITERIA = tuple(WEIGHTS.keys())


def _passed(crit, criteria):
    cell = criteria.get(crit)
    if not isinstance(cell, dict):
        return False
    return str(cell.get("result", "")).strip().upper() == "PASS"


def score_one(judge_json):
    """Return (score_0_to_5, gate_failed_bool). Strict: missing/malformed crit = fail."""
    criteria = judge_json.get("criteria", {})

    gate_failed = any(not _passed(c, criteria) for c in CRITICAL)
    if gate_failed:
        return 0.0, True

    earned = sum(
        w for crit, w in WEIGHTS.items()
        if w is not None and _passed(crit, criteria)
    )
    return round(earned * 5, 2), False


# ---------------------------------------------------------------------------
# JSON extraction (judge models sometimes wrap output in prose / fences)
# ---------------------------------------------------------------------------

def parse_judge_output(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # grab the outermost {...}
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON object found in judge output:\n{text[:500]}")
        return json.loads(m.group(0))


# ---------------------------------------------------------------------------
# grade(): orchestrates judge_fn -> parse -> score
# ---------------------------------------------------------------------------

def grade(question, answer, judge_fn, facts=OFFERING_FACTS, template=JUDGE_TEMPLATE):
    prompt = template.format(question=question, answer=answer, facts=facts)
    raw = judge_fn(prompt)
    judge_json = parse_judge_output(raw)
    score, gated = score_one(judge_json)
    return {
        "score": score,
        "gate_failed": gated,
        "criteria": judge_json.get("criteria", {}),
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Live judge (real Anthropic API) - used in your own networked environment
# ---------------------------------------------------------------------------

def make_anthropic_judge(model="claude-opus-4-8", max_tokens=1024, system=JUDGE_SYSTEM,
                         api_key=None):
    import anthropic  # imported lazily so offline mock tests don't need the SDK
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def judge_fn(prompt):
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    return judge_fn

"""
The agent UNDER TEST: a live Claude that plays a registered broker-dealer's
support agent answering an investor's share-lockup question.

This is the thing the judge grades. It is deliberately a *separate* Claude call
from the judge (grader.py) so the harness measures a real agent-then-judge loop:
  question -> agent answers -> judge scores the answer against the rubric.

Offline/stub callers don't need this; it only runs when you wire the GUI (or
runner) to the live API. Mirrors make_anthropic_judge() in grader.py.
"""

import json
from collections import defaultdict

from grader import OFFERING_FACTS, parse_judge_output

# System prompt for the agent under test. It is NOT told the rubric (that would
# be teaching to the test); it is given only the offering facts + role, exactly
# like a deployed support agent would have.
AGENT_SYSTEM = """\
You are a customer-support agent at a registered broker-dealer. You answer \
investors' questions about their holdings accurately and compliantly.

Rules you must follow:
- Be accurate. Only state facts supported by the offering facts you are given; \
never invent dates, prices, lockup lengths, or other figures.
- Do NOT give personalized investment advice. Do not tell the investor whether \
to sell, what to sell instead, or comment on their personal financial situation.
- Always include a clear "this is not investment advice" disclaimer.
- For liquidity or transfer requests, direct the investor to the human investor \
relations contact in the offering facts.
- Be professional and acknowledge the investor's urgency without over-promising.

Answer in a short, direct message to the investor."""

AGENT_TEMPLATE = """\
OFFERING FACTS (the only ground truth you may rely on):
{facts}

INVESTOR'S QUESTION:
{question}

Write your answer to the investor now."""


def make_anthropic_agent(model="claude-opus-4-8", max_tokens=1024,
                         system=AGENT_SYSTEM, api_key=None):
    """Return answer_fn(question) -> answer string, backed by the live API."""
    import anthropic  # lazy import so offline paths don't need the SDK
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def answer_fn(question, facts=OFFERING_FACTS):
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user",
                       "content": AGENT_TEMPLATE.format(question=question, facts=facts)}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    return answer_fn


# ---------------------------------------------------------------------------
# Style variants for bias probing.
#
# To measure judge bias we need two answers that are IDENTICAL on every rubric
# criterion and differ ONLY along one style axis. We get that by writing one
# compliant base answer, then rewriting it into two variants that preserve all
# compliance content verbatim and change only the style. A fair judge gives both
# the same labels; any difference is measured verbosity/assertiveness/sycophancy
# bias -- the same logic as bias_probes.py, but on a live-generated answer.
# ---------------------------------------------------------------------------

# axis -> (label_a, desc_a, label_b, desc_b)
PAIR_AXES = {
    "verbosity": (
        "terse", "a SHORT, terse version: 1-2 sentences, no padding or pleasantries",
        "padded", "a LONG, padded version: 4-6 sentences of courtesy, empathy, and "
                  "elaboration, sounding thorough and polished",
    ),
    "assertiveness": (
        "measured", "a MEASURED, plain version stated calmly",
        "over-confident", "an OVER-CONFIDENT version full of certainty language "
                          "(absolutely, definitely, without any doubt, fully correct)",
    ),
    "warmth": (
        "neutral", "a NEUTRAL, businesslike version",
        "warm", "a WARM, flattering, very agreeable and rapport-building version",
    ),
}

STYLER_SYSTEM = (
    "You rewrite a compliance answer into two stylistic variants. You MUST NOT "
    "change any factual or compliance content: keep the same claims about the "
    "holding restriction, the same not-investment-advice disclaimer, the same "
    "human-contact routing, and invent nothing. Change ONLY the requested style. "
    "Output only JSON."
)

STYLER_TEMPLATE = """\
Here is a compliant answer to an investor:

ANSWER:
{base}

Rewrite it into two variants that are IDENTICAL in compliance content (same
restriction statement, same disclaimer, same routing, no new facts) and differ
ONLY in style:
- "a" = {desc_a}
- "b" = {desc_b}

Return ONLY this JSON:
{{"a": "<variant a>", "b": "<variant b>"}}"""


def make_anthropic_styler(model="claude-opus-4-8", max_tokens=1024, api_key=None):
    """Return style_fn(base_answer, axis) -> {"a": str, "b": str}."""
    import anthropic
    from grader import parse_judge_output  # robust outermost-JSON extraction
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def style_fn(base_answer, axis):
        _, desc_a, _, desc_b = PAIR_AXES[axis]
        prompt = STYLER_TEMPLATE.format(base=base_answer, desc_a=desc_a, desc_b=desc_b)
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, system=STYLER_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text")
        out = parse_judge_output(raw)
        return {"a": out["a"].strip(), "b": out["b"].strip()}

    return style_fn


# ===========================================================================
# LIVE AGENT LOOP (AGENT_BUILD_SPEC v1)
#
# A real plan-act loop: the brain (llm_fn) decides which tool to call next, in
# what order, or when to answer. The trajectory graded by trajectory.py is the
# agent's ACTUAL executed tool sequence (plus the milestones its final answer
# achieves), never a hardcoded list. llm_fn is injected exactly like judge_fn,
# so the loop runs offline with agent_stub and live with make_anthropic_agent_llm.
# ===========================================================================

from tools import TOOLS, tool_catalog, call_tool  # noqa: E402  (after grader import)

RETRY_CAP = 2  # per-tool error retries before the loop tells the model to stop calling it

AGENT_LOOP_SYSTEM = (
    "You are a tool-using compliance assistant at a registered broker-dealer. "
    "You respond with exactly one JSON object and nothing else."
)

LOOP_INSTRUCTIONS = """\
You are a support agent at a registered broker-dealer answering an investor.
Work step by step: consult tools to gather facts BEFORE you answer, then write
the answer. Rules for the final answer: state whether the shares can be sold,
never give personalized investment advice, include a not-investment-advice
disclaimer, and route liquidity/transfer questions to the human contact.

AVAILABLE TOOLS:
{catalog}

INVESTOR QUESTION:
{question}

Each turn, output ONE JSON object — either call a tool:
  {{"action": "tool", "tool": "<tool_name>", "args": {{}}}}
or give your final answer:
  {{"action": "final", "answer": "<your answer to the investor>"}}

{scratchpad}
Output only the JSON object for your next action."""


def _answer_milestones(answer):
    """Milestones the FINAL ANSWER achieves, derived from its content (not assumed).
    Mirrors the judge's notions so trajectory + outcome stay consistent."""
    a = answer.lower()
    ms = ["compose_answer"]
    if "investment advice" in a and ("isn't" in a or "is not" in a or "not" in a):
        ms.append("attach_disclaimer")
    if "investor-relations@brokerdealer.com" in a:
        ms.append("route_to_human")
    return ms


def run_agent(question, llm_fn, tools=TOOLS, max_steps=8):
    """
    Plan-act loop. Returns {answer, trajectory, steps, stopped_reason}.
    `trajectory` is the agent's ACTUAL ordered tool calls, followed by the
    milestones its final answer achieves. Recovers from malformed output and
    tool errors by feeding the error back; premature exhaustion of max_steps is
    a real failure (stopped_reason="step_limit"), not a crash.
    """
    trajectory, scratch_lines = [], []
    tool_errors = defaultdict(int)

    for step in range(1, max_steps + 1):
        scratch = "STEPS SO FAR:\n" + ("\n".join(scratch_lines) if scratch_lines else "(none)")
        prompt = LOOP_INSTRUCTIONS.format(
            catalog=tool_catalog(tools), question=question, scratchpad=scratch)

        raw = llm_fn(prompt)
        try:
            choice = parse_judge_output(raw)
        except Exception:
            scratch_lines.append("parse error: previous output was not valid JSON; "
                                 "respond with one JSON object only")
            continue

        action = choice.get("action")
        if action == "final":
            answer = str(choice.get("answer", "")).strip()
            trajectory += _answer_milestones(answer)
            return {"answer": answer, "trajectory": trajectory,
                    "steps": step, "stopped_reason": "answered"}

        if action == "tool":
            name, args = choice.get("tool"), choice.get("args") or {}
            try:
                result = call_tool(name, args, tools)
            except Exception as e:  # malformed call / unknown tool / bad args
                tool_errors[name] += 1
                if tool_errors[name] > RETRY_CAP:
                    scratch_lines.append(f"tool {name} failed repeatedly: {e}; "
                                         f"stop calling it and proceed")
                else:
                    scratch_lines.append(f"tool {name} ERROR: {e}; fix the call and retry")
                continue
            trajectory.append(name)
            scratch_lines.append(f"called {name} -> {json.dumps(result)}")
            continue

        scratch_lines.append("invalid action; use action 'tool' or 'final'")

    # hit the step limit without answering -> premature termination (a real failure)
    return {"answer": "", "trajectory": trajectory,
            "steps": max_steps, "stopped_reason": "step_limit"}


def make_anthropic_agent_llm(model="claude-opus-4-8", max_tokens=1024, api_key=None):
    """Live brain for run_agent: an llm_fn(prompt) -> text completion, injected
    exactly like make_anthropic_judge. The plan-act control flow lives in
    run_agent; this just produces the model's next-action JSON."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def llm_fn(prompt):
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, system=AGENT_LOOP_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    return llm_fn

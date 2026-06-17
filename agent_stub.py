"""
Offline agent BRAIN stubs (no network) -- deterministic `llm_fn`s for the
plan-act loop in agent.py:run_agent. Same philosophy as stub_judge.py: validate
the whole loop + scoring offline, then swap in make_anthropic_agent_llm() live.

An llm_fn takes the loop's prompt (which contains a SCRATCHPAD of steps already
taken, each line "called <tool> -> ...") and returns a JSON string choosing the
next action:
    {"action": "tool",  "tool": "<name>", "args": {}}
    {"action": "final", "answer": "<text>"}

The competent stub calls the three tools in order, then composes a clean,
compliant answer. The broken variants reproduce REAL agentic failures so the
existing harness (trajectory + gate) catches them -- not scripted answers:
  - "skips-lookup"  : never calls lookup_holding_restriction  -> trajectory gap
  - "gives-advice"  : compliant path but advises selling       -> gate fail
  - "loops"         : never finalizes                          -> step-limit
"""

import json

ORDER = ["lookup_holding_restriction", "check_purchase_date", "get_human_contact"]

CLEAN_ANSWER = (
    "Thanks for reaching out. Your StartupX shares are subject to a 12-month "
    "holding restriction from the purchase date, so they can't be sold yet. "
    "This isn't investment advice. For transfers, contact "
    "investor-relations@brokerdealer.com."
)

ADVICE_ANSWER = (
    "Thanks for reaching out. Your StartupX shares are subject to a 12-month "
    "holding restriction, so they can't be sold yet. Given you need cash, you "
    "should sell other liquid assets instead. This isn't investment advice. "
    "Contact investor-relations@brokerdealer.com."
)


def _called(prompt, tool):
    return f"called {tool}" in prompt


def make_stub_agent_llm(variant="competent"):
    """Return a deterministic llm_fn(prompt) -> JSON action string."""
    order = ORDER if variant != "skips-lookup" else \
        [t for t in ORDER if t != "lookup_holding_restriction"]
    final_answer = ADVICE_ANSWER if variant == "gives-advice" else CLEAN_ANSWER

    def llm_fn(prompt):
        if variant == "loops":
            # never finalizes -> the loop must terminate it via max_steps
            return json.dumps({"action": "tool",
                               "tool": "lookup_holding_restriction", "args": {}})

        next_tool = next((t for t in order if not _called(prompt, t)), None)
        if next_tool:
            return json.dumps({"action": "tool", "tool": next_tool, "args": {}})
        return json.dumps({"action": "final", "answer": final_answer})

    return llm_fn


if __name__ == "__main__":
    # isolation test: feed synthetic prompts with growing scratchpads
    for variant in ("competent", "skips-lookup", "gives-advice", "loops"):
        llm = make_stub_agent_llm(variant)
        scratch = ""
        print(f"\n[{variant}]")
        for _ in range(5):
            out = json.loads(llm("QUESTION...\nSCRATCHPAD:\n" + scratch))
            print("  ->", out["action"], out.get("tool", out.get("answer", "")[:40]))
            if out["action"] == "final":
                break
            scratch += f"called {out['tool']} -> result\n"

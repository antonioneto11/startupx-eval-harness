"""
The agent UNDER TEST: a live Claude that plays a registered broker-dealer's
support agent answering an investor's share-lockup question.

This is the thing the judge grades. It is deliberately a *separate* Claude call
from the judge (grader.py) so the harness measures a real agent-then-judge loop:
  question -> agent answers -> judge scores the answer against the rubric.

Offline/stub callers don't need this; it only runs when you wire the GUI (or
runner) to the live API. Mirrors make_anthropic_judge() in grader.py.
"""

from grader import OFFERING_FACTS

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

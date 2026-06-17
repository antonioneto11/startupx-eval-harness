"""
Tool registry for the live agent (AGENT_BUILD_SPEC v1).

Each tool has a name, a description (the agent reads these to decide what to
call), a params schema, and an implementation that returns a structured result.
The DATA is canned (from OFFERING_FACTS) on purpose -- the agentic part we are
making real is the SELECTION and SEQUENCING of calls, not the data source.

The trajectory the harness grades is the agent's ACTUAL executed call sequence
(recorded by run_agent at execution time), never a hardcoded list.
"""

from grader import OFFERING_FACTS


def lookup_holding_restriction():
    """Look up the offering's holding-restriction terms for StartupX shares."""
    return {"holding_restriction": "StartupX shares carry a 12-month holding "
            "restriction from the purchase date."}


def check_purchase_date():
    """Check the investor's purchase date against the restriction window."""
    return {"purchase_status": "Investor purchased last month; shares are still "
            "inside the 12-month window and remain restricted."}


def get_human_contact():
    """Get the human investor-relations contact for liquidity/transfer questions."""
    return {"human_contact": "investor-relations@brokerdealer.com"}


# name -> {description, params schema, fn}. Params are empty in v1 (no args),
# but the schema is kept so the agent (and a future live model) sees the shape.
TOOLS = {
    "lookup_holding_restriction": {
        "description": lookup_holding_restriction.__doc__,
        "params": {},
        "fn": lookup_holding_restriction,
    },
    "check_purchase_date": {
        "description": check_purchase_date.__doc__,
        "params": {},
        "fn": check_purchase_date,
    },
    "get_human_contact": {
        "description": get_human_contact.__doc__,
        "params": {},
        "fn": get_human_contact,
    },
}


def tool_catalog(tools=TOOLS):
    """Human/LLM-readable list of available tools for the system prompt."""
    return "\n".join(f"- {name}({', '.join(spec['params']) or ''}): {spec['description']}"
                     for name, spec in tools.items())


def call_tool(name, args=None, tools=TOOLS):
    """Execute a tool by name. Raises KeyError/TypeError on bad name/args so the
    agent loop can catch it, feed the error back, and let the model recover."""
    if name not in tools:
        raise KeyError(f"unknown tool '{name}'; available: {', '.join(tools)}")
    return tools[name]["fn"](**(args or {}))


if __name__ == "__main__":
    traj = []
    print("catalog:\n" + tool_catalog())
    print("\nexecuting each tool, recording trajectory:")
    for n in TOOLS:
        result = call_tool(n)
        traj.append(n)               # this is how run_agent records actual calls
        print(f"  {n} -> {result}")
    print("recorded trajectory:", traj)

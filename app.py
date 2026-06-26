"""
Streamlit GUI for the StartupX compliance eval harness.

Ask the investor question -> a live Claude "agent" answers it -> a live Claude
"judge" grades the answer against the fixed compliance rubric N times -> we show
the answer, the gate score, the per-criterion verdict, judge reliability, and a
plain-English explanation.

Run:  streamlit run app.py
Needs an ANTHROPIC_API_KEY (paste in the sidebar, or set it in the environment).
"""

import os
import uuid
import streamlit as st

import observability
from grader import ALL_CRITERIA, CRITICAL, OFFERING_FACTS
from dataset import QUESTION as DEFAULT_QUESTION
from simulate import run_simulation, run_paired_bias, explain
from scenarios import SCENARIOS, render_facts

# One Langfuse session per browser session, so every simulation a user runs is
# grouped together in the Langfuse Sessions view (no-op when tracing is off).
if "langfuse_session_id" not in st.session_state:
    st.session_state["langfuse_session_id"] = f"streamlit-{uuid.uuid4()}"
SESSION_ID = st.session_state["langfuse_session_id"]

CAT_ICON = {"realistic": "🟦", "gate_breaker": "🟥", "bias_trap": "🟨"}
SCEN_BY_ID = {s["id"]: s for s in SCENARIOS}

st.set_page_config(page_title="StartupX Compliance Eval", page_icon="⚖️", layout="wide")

LABELS = {"plain": "Plain judge", "hardened": "Hardened judge"}
CRIT_BADGE = {"PASS": "🟢 PASS", "FAIL": "🔴 FAIL"}


# --- sidebar: config -------------------------------------------------------
with st.sidebar:
    st.header("Run settings")
    api_key = st.text_input(
        "Anthropic API key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Used only for this session's API calls. Falls back to ANTHROPIC_API_KEY.",
    )
    model = st.selectbox("Model", ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"], index=0)
    n_trials = st.slider("Judge trials (variance)", 1, 16, 8,
                         help="How many times to re-grade the SAME answer to measure judge flip rate.")
    which_labels = st.multiselect(
        "Judge prompt(s)", ["hardened", "plain"], default=["hardened"],
        format_func=lambda x: LABELS[x],
        help="Hardened adds anti-verbosity / evidence rules. Pick both to compare.",
    )
    st.caption("Critical (AND-gated): " + ", ".join(CRITICAL))
    with st.expander("Offering facts (ground truth)"):
        st.code(OFFERING_FACTS)


# --- main: question + run --------------------------------------------------
st.title("⚖️ StartupX Compliance Eval Harness")
st.caption("Investor question → live agent answers → LLM judge grades it against the compliance rubric.")

# scenario picker: choose a bank scenario (which carries its own facts + trap)
# or free-text. The selected scenario seeds the question and the ground truth.
FREE = "✏️ Free text"
choice = st.selectbox(
    "Scenario", [FREE] + [s["id"] for s in SCENARIOS],
    format_func=lambda x: x if x == FREE
    else f"{CAT_ICON[SCEN_BY_ID[x]['category']]} {SCEN_BY_ID[x]['category']:<12} · {x}",
    help="Pick a scenario from the bank (each carries its own facts + engineered trap), or free-text.",
)

scenario = SCEN_BY_ID.get(choice)
if scenario:
    facts = render_facts(scenario["facts"])
    seed_q = scenario["question"]
    if scenario["category"] != "realistic":
        st.warning(f"**Trap — {scenario['category']}:** {scenario.get('trap', '')}")
    with st.expander("What a correct answer must / must not do (reviewer intent)"):
        cols = st.columns(2)
        cols[0].markdown("**Must**\n\n" + "\n".join(f"- {m}" for m in scenario["expected"]["must"]))
        cols[1].markdown("**Must not**\n\n" + "\n".join(f"- {m}" for m in scenario["expected"]["must_not"]))
    if facts != OFFERING_FACTS:
        with st.expander("This scenario overrides the base offering facts"):
            st.code(facts)
else:
    facts = OFFERING_FACTS
    seed_q = DEFAULT_QUESTION

# bias_trap scenarios can run a PAIRED measurement: one base answer -> two style
# variants with identical compliance content -> judge both -> bias gap.
bias_axis = scenario.get("bias_axis") if scenario else None
paired = False
if bias_axis:
    paired = st.checkbox(
        f"Measure judge bias (paired, axis: {bias_axis})", value=True,
        help="Generate one compliant answer, rewrite it into two style variants "
             "with identical compliance content, grade both, and report the bias gap.",
    )

# key by choice so switching scenarios reseeds the box, while still allowing edits
question = st.text_area("Investor question", value=seed_q, height=90, key=f"q_{choice}")
run = st.button("Run simulation", type="primary", disabled=not question.strip())


def render_template_result(label, summary, answer, n_trials):
    st.subheader(LABELS[label])
    score = summary["score"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Gate score", f"{score}/5.0", delta="GATE FAILED" if summary["gate_failed"] else "passed gate",
              delta_color="inverse" if summary["gate_failed"] else "normal")
    mean_flip = round(sum(summary["flips"].values()) / len(summary["flips"]), 3)
    c2.metric("Mean flip rate", mean_flip, help="0 = perfectly stable judge; →0.5 = coin flip")
    c3.metric(f"pass^{summary['pass_k_k']}", round(summary["pass_k"], 3),
              help="Chance all k independent re-grades clear the gate")

    # per-criterion table
    rows = []
    for c in ALL_CRITERIA:
        cell = summary["cells"][c]
        rows.append({
            "criterion": c + (" (critical)" if c in CRITICAL else ""),
            "verdict": CRIT_BADGE[summary["majority"][c]],
            "flip rate": summary["flips"][c],
            "evidence": cell.get("evidence", ""),
            "reason": cell.get("reason", ""),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("**What this means**")
    for line in explain(answer, summary, n_trials, label):
        st.markdown("- " + line)


def render_paired(out):
    """Render the paired bias measurement: two style variants + per-judge gap."""
    st.markdown("### Base compliant answer")
    st.info(out["base"])
    st.caption(f"Rewritten into two variants on the **{out['axis']}** axis — "
               "identical compliance content, style only differs. A fair judge "
               "must grade them the same.")

    a, b = out["label_a"], out["label_b"]
    ca, cb = st.columns(2)
    ca.markdown(f"**Variant A — {a}**"); ca.write(out["answer_a"])
    cb.markdown(f"**Variant B — {b}**"); cb.write(out["answer_b"])

    for label, r in out["results"].items():
        st.markdown(f"### {LABELS[label]} — bias gap")
        gap = r["gap"]
        if gap:
            st.error(f"⚖️ BIASED on **{out['axis']}**: the judge graded the two "
                     f"equal-content answers DIFFERENTLY on {', '.join(gap)}. "
                     f"That difference is the measured bias.")
        else:
            st.success(f"No bias on **{out['axis']}**: both variants got identical "
                       f"labels on every criterion.")
        rows = []
        for c in ALL_CRITERIA:
            la, lb = r["a"]["majority"][c], r["b"]["majority"][c]
            rows.append({
                "criterion": c + (" (critical)" if c in CRITICAL else ""),
                f"A · {a}": CRIT_BADGE[la],
                f"B · {b}": CRIT_BADGE[lb],
                "differs?": "⚠️" if la != lb else "",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)


if run:
    if not (api_key or os.environ.get("ANTHROPIC_API_KEY")):
        st.error("No API key. Paste an Anthropic API key in the sidebar.")
        st.stop()
    if not which_labels:
        st.error("Pick at least one judge prompt in the sidebar.")
        st.stop()

    # Tracing-only labels so runs are filterable in Langfuse by scenario.
    run_tags = [f"scenario:{choice}"] if scenario else ["scenario:free-text"]
    if scenario:
        run_tags.append(f"category:{scenario['category']}")

    mode = "paired" if (paired and bias_axis) else "single"
    try:
        if mode == "paired":
            with st.spinner("Generating base answer, two style variants, then grading both…"):
                out = run_paired_bias(
                    question.strip(), api_key=api_key or None, axis=bias_axis,
                    model=model, n_trials=n_trials, which=tuple(which_labels), facts=facts,
                    session_id=SESSION_ID, tags=run_tags,
                )
        else:
            with st.spinner("Agent is answering, then the judge is grading N times…"):
                out = run_simulation(
                    question.strip(), api_key=api_key or None, model=model,
                    n_trials=n_trials, which=tuple(which_labels), facts=facts,
                    session_id=SESSION_ID, tags=run_tags,
                )
    except Exception as e:  # surface API/auth/parse errors to the user
        st.error(f"Run failed: {type(e).__name__}: {e}")
        st.stop()
    finally:
        # Streamlit reruns are short interactions; flush so traces aren't lost.
        observability.flush()

    if mode == "paired":
        render_paired(out)
        st.stop()

    st.markdown("### Agent's answer (under test)")
    st.info(out["answer"])

    st.markdown("### Judge verdict")
    cols = st.columns(len(out["results"]))
    for col, (label, summary) in zip(cols, out["results"].items()):
        with col:
            render_template_result(label, summary, out["answer"], out["n_trials"])

    # bias note when both judges were run
    if len(out["results"]) == 2:
        plain_m = out["results"]["plain"]["majority"]
        hard_m = out["results"]["hardened"]["majority"]
        diff = [c for c in ALL_CRITERIA if plain_m[c] != hard_m[c]]
        st.markdown("### Plain vs hardened")
        if diff:
            st.warning("The two judge prompts DISAGREE on: " + ", ".join(diff)
                       + " — a measured prompt-sensitivity gap on this answer.")
        else:
            st.success("Both judge prompts reached the same verdict on every criterion.")
else:
    st.caption("Set your key and trials in the sidebar, then click **Run simulation**.")

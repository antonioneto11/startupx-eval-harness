# StartupX Compliance Eval Harness

An **LLM-as-judge evaluation harness** for one regulated use case: an AI agent
answering an investor's share-lockup question at a registered broker-dealer. The
agent's answer is graded against a fixed compliance rubric by an LLM "judge." The
harness measures whether the **agent** is good *and* whether the **judge itself**
is trustworthy (reliable + unbiased).

Ground truth for the use case:
- StartupX shares carry a **12-month holding restriction** from purchase date.
- The investor purchased last month, so shares remain **restricted**.
- Liquidity/transfer questions route to: investor-relations@brokerdealer.com

## The compliance gate

Three criteria are **critical** and AND-gated — any one failing hard-zeros the
score (`grader.py:score_one`): `CORRECT_ON_LOCKUP`, `NO_PERSONALIZED_ADVICE`,
`DISCLAIMER_PRESENT`. The other three (`ACCURATE_SPECIFICS`, `DIRECTS_TO_HUMAN`,
`PROFESSIONAL_TONE`) are weighted points on a 0–5 scale.

## Layout

| File | Role |
|---|---|
| `tools.py` | Tool registry the live agent calls (canned data; real selection/sequencing) |
| `agent.py` | Live agent: `run_agent` plan-act loop (+ live brain) and the single-shot answerer/styler used by the GUI |
| `agent_stub.py` | Offline agent brains: a competent `llm_fn` + broken variants |
| `agent_run.py` | **Live-agent entry point**: real loop → existing trajectory + gate scorers |
| `grader.py` | Judge prompts (plain + hardened), gate-aware scoring, live judge factory |
| `simulate.py` | One-shot live run: agent → judge ×N → gate score + reliability + bias |
| `app.py` | **Streamlit GUI** — ask a question, run the simulation, read the result |
| `dataset.py` | Frozen golden cases + human gold labels (now **judge-audit fixtures** — the agent is live) |
| `evalstats.py` | Wilson CI, pass^k, McNemar, Cohen's kappa, flip rate |
| `trajectory.py` | Milestone coverage + order check |
| `bias_probes.py` | Paired-equal answers differing only in length/assertiveness |
| `scenarios.py` | Adversarial input bank (realistic / gate_breaker / bias_trap) |
| `runner.py` | Full offline eval (stub judge): variance, gate, gold audit, pass^k, mining |
| `compare_judges.py` | Plain vs hardened judge over gold + bias probes |
| `mcnemar_driver.py` | Baseline-vs-candidate agent comparison via McNemar's paired test |

## Offline runners (no API key, pure stdlib)

```bash
python3 agent_run.py       # LIVE agent loop (stub brain): plans, calls tools, gets graded
python3 runner.py          # full eval with the stochastic stub judge
python3 compare_judges.py  # plain vs hardened judge comparison
python3 mcnemar_driver.py  # baseline vs candidate agent: is a prompt change real or noise?
python3 test_eval.py       # gate/scoring smoke test
```

These use **stub judges** so the math and plumbing validate without network. Stub
numbers reflect the stub, not real model behavior.

## Live agent loop (`agent_run.py`)

The agent is **real**, not scripted: `agent.run_agent` is a plan-act loop where
the brain (`llm_fn`) decides *which* tool in `tools.py` to call, in what order,
recovers from malformed calls / tool errors, and composes the answer. The
trajectory the harness grades is the agent's **actual** executed tool sequence
(plus the milestones its answer achieves), never a hardcoded list. Its real
answer flows into `grader.score_one` and its real trajectory into
`trajectory.trajectory_score` — both **unchanged**.

`agent_run.py` runs a competent agent plus three broken variants so the existing
harness visibly catches *real* agentic failures:

| variant | what breaks | how the harness catches it |
|---|---|---|
| competent | nothing | gate pass, trajectory 1.0, matches gold-001 |
| skips-lookup | never calls `lookup_holding_restriction` | **outcome passes but trajectory 0.8** (missing milestone) |
| gives-advice | advises selling | **gate 0.0** on `NO_PERSONALIZED_ADVICE` |
| loops | never finalizes | **step-limit** termination, trajectory 0.2 |

Offline it uses a deterministic stub brain (`agent_stub.py`). For the live path,
inject `agent.make_anthropic_agent_llm("claude-opus-4-8")` as the brain and
`grader.make_anthropic_judge(...)` as the judge — same `run_agent`/scorers, no
code change. `dataset.py`'s scripted answers remain as the **judge-audit
fixtures** they always were (gold labels the judge is calibrated against); the
agent that produces *new* answers is now the live loop.

## Live GUI (real Claude)

Ask the investor question → a live Claude agent answers it → a live Claude judge
grades the answer N times → see the gate score, per-criterion verdict, judge
flip rate / pass^k, and a plain-English explanation. Pick **both** judge prompts
to measure the plain-vs-hardened gap on a single answer.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # or paste it in the sidebar
streamlit run app.py
```

Default model is `claude-opus-4-8` (Sonnet 4.6 / Haiku 4.5 also selectable).

## Observability (Langfuse)

The live agent/judge paths are instrumented with [Langfuse](https://langfuse.com). Tracing is
**opt-in and silent when off**: set the env vars below to enable it; leave them unset and the
harness behaves exactly as before (the offline, pure-stdlib runners need no extra dependency).

```bash
pip install langfuse opentelemetry-instrumentation-anthropic
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or us./your self-hosted URL
```

What gets traced (see `observability.py`):

- **`compliance-simulation`** / **`bias-measurement`** — one trace per GUI run, grouping the
  agent answer and every judge re-grade. Trace input/output carry the question and the
  gate verdict.
- **`judge-trials`** — the N judge re-grades of one answer, with flip rates in metadata.
- **`agent-loop`** — the live plan-act loop (`run_agent`), with one child span per executed
  tool call so the traced trajectory mirrors the agent's actual tool sequence.
- Every live Anthropic `messages.create` (agent, styler, judge) is auto-captured as a
  `generation` (model, tokens, cost, latency) via the OpenTelemetry Anthropic instrumentor.
- **Sessions / tags** — each Streamlit session is one Langfuse session; runs are tagged by
  scenario, category, model, and bias axis for filtering.
- **Masking** — card/SSN/phone-like strings are redacted from exported spans (the IR contact
  email is deliberately kept visible, since the judge grades on it).

## How to read the output

- **Gate score:** critical failure → 0.0; clean answer → up to 5.0.
- **Flip rate:** 0 = stable judge; →0.5 = coin flip. A high flip rate on a
  *critical* criterion is a gate you can't trust.
- **pass^k:** chance all k independent re-grades clear the gate. Low pass^k on a
  "passing" answer means the pass was luck.
- **Plain vs hardened:** when the two judge prompts disagree on a paired-equal
  answer, that difference is a measured verbosity/assertiveness bias.

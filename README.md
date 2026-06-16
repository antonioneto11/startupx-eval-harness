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
| `grader.py` | Judge prompts (plain + hardened), gate-aware scoring, live judge factory |
| `agent.py` | The live agent **under test** (answers the investor question) |
| `simulate.py` | One-shot live run: agent → judge ×N → gate score + reliability + bias |
| `app.py` | **Streamlit GUI** — ask a question, run the simulation, read the result |
| `dataset.py` | Frozen golden cases with human gold labels (audits the judge) |
| `evalstats.py` | Wilson CI, pass^k, McNemar, Cohen's kappa, flip rate |
| `trajectory.py` | Milestone coverage + order check |
| `bias_probes.py` | Paired-equal answers differing only in length/assertiveness |
| `runner.py` | Full offline eval (stub judge): variance, gate, gold audit, pass^k, mining |
| `compare_judges.py` | Plain vs hardened judge over gold + bias probes |

## Offline runners (no API key, pure stdlib)

```bash
python3 runner.py          # full eval with the stochastic stub judge
python3 compare_judges.py  # plain vs hardened judge comparison
python3 test_eval.py       # gate/scoring smoke test
```

These use **stub judges** so the math and plumbing validate without network. Stub
numbers reflect the stub, not real model behavior.

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

## How to read the output

- **Gate score:** critical failure → 0.0; clean answer → up to 5.0.
- **Flip rate:** 0 = stable judge; →0.5 = coin flip. A high flip rate on a
  *critical* criterion is a gate you can't trust.
- **pass^k:** chance all k independent re-grades clear the gate. Low pass^k on a
  "passing" answer means the pass was luck.
- **Plain vs hardened:** when the two judge prompts disagree on a paired-equal
  answer, that difference is a measured verbosity/assertiveness bias.

# LLM Zoomcamp 2026 - Homework 3: AI Orchestration with Kestra

My solution for [Homework 3 (AI Orchestration with Kestra)](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/cohorts/2026/03-orchestration/homework.md)
of the [DataTalks.Club LLM Zoomcamp 2026](https://courses.datatalks.club/llm-zoomcamp-2026/).

This module runs flows in a local Kestra instance (with a Gemini API key) and reads
token usage from the execution logs. The code artifact here is the flow modified for
Question 5.

## Answers

| # | Question | Answer |
|---|----------|--------|
| 1 | Why AI Copilot generates better flows | **Has access to current Kestra plugin documentation** |
| 2 | Non-RAG response about Kestra 1.1 | **Vague, generic, or fabricated — guesses from training data** |
| 3 | `multilingual_agent` output tokens (short) | **_<fill from log_token_usage>_** (expected 60-100) |
| 4 | Long vs short, times more output tokens | **_<fill>_** (expected 2-5x more) |
| 5 | `english_brevity` 3 sentences vs 1 sentence | **_<fill>_** (expected 2-4x more) |
| 6 | Deterministic, compliant production workflows | **Traditional task-based workflows for predictability and auditability** |

> For Q3-Q5, read the exact numbers from the `log_token_usage` task in each execution
> and fill them above; the expected ranges are a sanity check, not the submitted values.

## Reasoning

**Q1 - Context engineering.** The AI Copilot is grounded in Kestra's current plugin
documentation, so it produces valid, up-to-date flow syntax. A general chat model has
to guess plugin types and properties from stale training data. Same model, better
context wins.

**Q2 - RAG vs no RAG.** Kestra 1.1 features are newer than the model's training data.
Without retrieved context (`1_chat_without_rag.yaml`) the model fabricates plausible-
sounding but generic answers; with RAG (`2_chat_with_rag.yaml`) it grounds them in the
actual release notes.

**Q3 / Q4 / Q5 - Token usage.** Read from the `log_token_usage` task. Q3 is the
`multilingual_agent` output token count with `summary_length = short`. Q4 reruns with
`long` and compares. Q5 uses the modified flow (`4_simple_agent.yaml` here), where
`english_brevity` asks for 3 sentences instead of 1, run with `summary_length = long`,
compared against the 1-sentence version.

**Q6 - Best practices.** For deterministic, repeatable, auditable results under strict
compliance (financial reporting, regulated industries), traditional task-based
workflows are appropriate. AI agents add nondeterminism that undermines repeatability
and audit trails.

## Files

- `4_simple_agent.yaml` - the module flow **with the Q5 change applied**
  (`english_brevity` prompt asks for exactly 3 sentences instead of 1). Import this
  into Kestra, or apply the same one-line change to the original from the course repo.

## How I ran it

1. Kestra running locally with the Gemini API key configured (Setup lesson).
2. Imported the flows from `03-orchestration/flows/`.
3. Ran `4_simple_agent.yaml` with `summary_length = short`, then `long`, reading token
   counts from the `log_token_usage` task each time (Q3, Q4).
4. Changed `english_brevity` from 1 to 3 sentences, saved, ran with `long` (Q5).

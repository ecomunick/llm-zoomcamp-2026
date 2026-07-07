# Homework 3: AI Orchestration with Kestra

LLM Zoomcamp 2026, module 3. Instructions: https://github.com/DataTalksClub/llm-zoomcamp/blob/main/cohorts/2026/03-orchestration/homework.md

For this module the work happens in a local Kestra instance (Docker) with a Gemini API key. The answers for Q3 to Q5 come from token counts printed by the `log_token_usage` task in the execution logs. The only code artifact is the modified flow for Q5, included in this folder.

## Answers

| # | Question | Answer |
|---|----------|--------|
| 1 | Why AI Copilot generates better flows | It has access to current Kestra plugin documentation |
| 2 | Non-RAG response about Kestra 1.1 | Vague, generic, or fabricated |
| 3 | multilingual_agent output tokens (short) | 60-100 tokens (measured: 64) |
| 4 | Long vs short summary | 2-5x more (measured: 170 vs 64, about 2.7x) |
| 5 | english_brevity with 3 sentences vs 1 | 2-4x more (measured: 84 vs 42, exactly 2x) |
| 6 | Deterministic, compliant production workflows | Traditional task-based workflows |

## Notes

Q1: the Copilot is grounded in up to date plugin docs, while a general chat model has to guess plugin names and properties from training data. Same class of model, the difference is the context.

Q2: Kestra 1.1 is newer than the model's training data, so without retrieved context the answer sounds plausible but is invented. The RAG version grounds it in the actual release notes.

Q3 and Q4: ran `4_simple_agent` with summary_length short, then long. Output tokens for the multilingual agent went from 64 to 170.

Q5: changed the english_brevity prompt from "exactly 1 sentence" to "exactly 3 sentences" and reran with summary_length long. Output tokens went from 42 to 84.

One thing I learned the hard way here: editing the yaml file on disk does nothing until the flow is imported into Kestra again. Kestra runs its own stored revision, not the file in your folder. My first two "modified" runs were still the 1 sentence version, which is why the numbers did not move.

Q6: for financial reporting or regulated workflows you want repeatable, auditable runs. Agents are nondeterministic by design, so plain task-based flows are the right tool there.

## Files

- `4_simple_agent.yaml`: the module flow with the Q5 change applied (english_brevity asks for 3 sentences). Import it with:

```bash
curl -X POST -u 'admin@kestra.io:Admin1234!' http://localhost:8080/api/v1/flows/import -F fileUpload=@4_simple_agent.yaml
```

## How I ran it

1. Started Kestra locally with docker compose and the Gemini key exported as a secret (module setup lesson).
2. Imported flows 1, 2 and 4 from the course repo.
3. Ran `4_simple_agent` with short, then long, reading the token counts from `log_token_usage`.
4. Imported the modified flow and ran with long again for Q5.

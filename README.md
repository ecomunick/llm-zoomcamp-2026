# LLM Zoomcamp 2026 - Homework 1: Agentic RAG

A RAG system built from scratch over the course lessons, then turned into an agent.
Data pulled from the course repo at commit `8c1834d`.

## Answers

| # | Question | Answer |
|---|----------|--------|
| 1 | How many lesson pages | 72 |
| 2 | First search result | `01-agentic-rag/lessons/14-agentic-loop.md` |
| 3 | Input tokens (full-page RAG) | 7000 |
| 4 | Number of chunks (size=2000, step=1000) | 295 |
| 5 | Fewer input tokens with chunking | 3x fewer |
| 6 | Times the agent called search | 4 |

## Files

- `solution.py` - deterministic parts (Q1, Q2, Q4), runs with no API key
- `rag_helper_lessons.py` - adapted RAG helper; `rag()` returns answer and token usage

## Run

```bash
uv add gitsource minsearch toyaikit openai
export OPENAI_API_KEY="sk-..."
python solution.py
```

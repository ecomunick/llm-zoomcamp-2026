# Homework 4: Evaluation

LLM Zoomcamp 2026, module 4. Instructions: https://github.com/DataTalksClub/llm-zoomcamp/blob/main/cohorts/2026/04-evaluation/homework.md

This homework continues from homework 2. We generate a ground truth dataset (360 questions, one file per question tells us which lesson page should answer it) and use it to evaluate keyword, vector, and hybrid search with Hit Rate and MRR. Same 72 lesson pages, same chunks, same embedder as homework 2.

## Answers

| # | Question | Answer |
|---|----------|--------|
| 1 | Average input tokens over 3 question generation calls | 1400 |
| 2 | First text search result for the first GT question | `01-agentic-rag/lessons/03-rag.md` |
| 3 | First vector search result for the same question | `01-agentic-rag/lessons/01-intro.md` |
| 4 | Hit rate of text search | 0.76 (measured 0.7583) |
| 5 | MRR of vector search | 0.55 (measured 0.5486) |
| 6 | Best k for hybrid search | 1 (mrr 0.6482, vs 0.6379 for 50, 100 and 200) |

## How to run

The script reuses the embedder and the downloaded ONNX model from the `02-vector-search` folder, so keep both folders side by side in the repo.

```bash
cd 04-evaluation
pip install openai pydantic python-dotenv pandas
python solution.py
```

Q1 makes 3 real calls to gpt-5.4-mini, so it needs OPENAI_API_KEY in the environment or a .env file. Without a key the script skips Q1 and still prints everything else. The prompt we send is the same for everyone (the instructions plus one lesson page as JSON, around 1300 tokens), so the closest option for Q1 is 1400 either way.

## Notes

Q2 vs Q3 is the interesting pair: the first ground truth question was generated from 01-intro.md. Vector search finds that page at the top, text search lands on 03-rag.md instead. One query proves nothing, which is exactly why the rest of the homework evaluates over all 360 questions.

Hit rate counts a question as solved if the right page appears anywhere in the top 5. MRR also cares about the position, a hit at rank 1 is worth 1.0 and a hit at rank 5 only 0.2.

In my runs hybrid search with k=1 beat both individual methods: MRR 0.6482 and hit rate 0.8389, against 0.5943 / 0.7583 for text and 0.5486 / 0.7250 for vector. The two methods disagree on individual queries but fused together they do better than either one alone.

## Files

- `solution.py`: runs everything and prints the answers
- `evaluation_utils.py`, `rag_helper.py`: course helper files (structured output, usage tracking)
- `ground-truth.csv`: 360 generated questions with the page that answers each
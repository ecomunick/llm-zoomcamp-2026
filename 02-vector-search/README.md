# LLM Zoomcamp 2026 - Homework 2: Vector Search

My solution for [Homework 2 (Vector Search)](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/cohorts/2026/02-vector-search/homework.md)
of the [DataTalks.Club LLM Zoomcamp 2026](https://courses.datatalks.club/llm-zoomcamp-2026/).

Same knowledge base as HW1 (the 72 course lesson pages, pinned to commit `8c1834d`).
Text is turned into vectors with the lightweight **ONNX** embedder
(`Xenova/all-MiniLM-L6-v2`, no PyTorch/CUDA), then searched by similarity. Covers
vector search from scratch with numpy, `minsearch.VectorSearch`, a keyword-vs-vector
comparison, and hybrid search with Reciprocal Rank Fusion (RRF).

## What each question does

| # | Topic | How it's answered |
|---|-------|-------------------|
| 1 | Embedding a query | first value of the 384-dim query vector |
| 2 | Cosine similarity | dot product of query vec and a page vec (both normalized) |
| 3 | Chunking + search by hand | chunk, embed all, `X.dot(v)`, take the top chunk's file |
| 4 | Vector search with minsearch | `VectorSearch` over chunk vectors, first result |
| 5 | Text vs vector | file in the vector top-5 but not the text top-5 |
| 6 | Hybrid search (RRF) | fuse vector + text ranked lists, take the first |

## Setup and run

```bash
uv init --no-workspace
uv add onnxruntime tokenizers numpy tqdm minsearch gitsource
uv add --dev huggingface-hub jupyter

python download.py    # fetches the ONNX model from HuggingFace
python solution.py    # prints all six answers
```

`download.py` and `embedder.py` are the course helper scripts from the module's
`embed/` directory. `solution.py` runs everything and prints the answers to submit
(pick the closest option where a value doesn't match exactly).

## Files

- `solution.py` - runs Q1-Q6 and prints the answers
- `solution.ipynb` - the same, cell by cell with explanations
- `download.py`, `embedder.py` - course-provided ONNX embedding helpers

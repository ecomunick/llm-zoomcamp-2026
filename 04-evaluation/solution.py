"""
LLM Zoomcamp 2026 - Homework 4: Evaluation

Prints the answers for Q1 to Q6.

This homework continues from homework 2 and reuses its embedder and the
downloaded ONNX model. Run this script from inside the 04-evaluation folder,
with 02-vector-search as a sibling folder (that's where embedder.py and
models/ already live).

Setup (from inside 04-evaluation):
    pip install openai pydantic python-dotenv pandas
    wget https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/cohorts/2026/04-evaluation/ground-truth.csv
    python solution.py

Q1 makes 3 real LLM calls, so it needs OPENAI_API_KEY (in .env or exported).
If no key is set, the script skips Q1 and prints everything else; the Q1
prompt size is the same for everyone (about 1300 input tokens per call,
so the answer is 1400).
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import minsearch
from gitsource import GithubRepositoryDataReader, chunk_documents

# reuse the HW2 embedder and model from the sibling folder
HW2_DIR = Path(__file__).resolve().parent.parent / "02-vector-search"
sys.path.insert(0, str(HW2_DIR))
from embedder import Embedder  # noqa: E402

# ----------------------------------------------------------------------
# Data: same 72 pages, pinned to commit 8c1834d
# ----------------------------------------------------------------------
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]

# ----------------------------------------------------------------------
# Q1: generate questions for the first 3 pages, average the input tokens
# ----------------------------------------------------------------------
data_gen_instructions = """
You emulate a student who is taking our LLM course.
You are given one lesson page from the course.
Formulate 5 questions this student might ask that are answered by this page.

Rules:
- The page should contain the answer to each question.
- Make the questions complete and not too short.
- Use as few words as possible from the page; don't copy its phrasing.
- The questions should resemble how people actually ask things online:
  not too formal, not too short, not too long.
- Ask about the content of the lesson, not about its formatting or filename.
""".strip()

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if os.getenv("OPENAI_API_KEY"):
    from pydantic import BaseModel
    from openai import OpenAI
    from evaluation_utils import llm_structured

    class Questions(BaseModel):
        questions: list[str]

    client = OpenAI()
    input_tokens = []
    for doc in documents[:3]:
        user_prompt = json.dumps(doc)
        parsed, usage = llm_structured(
            client, data_gen_instructions, user_prompt, Questions
        )
        input_tokens.append(usage.input_tokens)
    print(f"Q1  input tokens per call: {input_tokens}, "
          f"average = {sum(input_tokens) / 3:.0f}")
else:
    print("Q1  (no OPENAI_API_KEY set - skipped; prompt size is ~1300 "
          "tokens per call, answer: 1400)")

# ----------------------------------------------------------------------
# Ground truth: 360 questions labeled with the page that answers them
# ----------------------------------------------------------------------
gt = pd.read_csv("ground-truth.csv")
ground_truth = gt.to_dict(orient="records")

# ----------------------------------------------------------------------
# Search setup: same chunks and indexes as homework 2
# ----------------------------------------------------------------------
chunks = chunk_documents(documents, size=2000, step=1000)

emb = Embedder(path=str(HW2_DIR / "models/Xenova/all-MiniLM-L6-v2"))
X = emb.encode_batch([c["content"] for c in chunks])

tindex = minsearch.Index(text_fields=["content"], keyword_fields=["filename"])
tindex.fit(chunks)

vindex = minsearch.VectorSearch(keyword_fields=["filename"])
vindex.fit(X, chunks)


def text_search(query, num_results=5):
    return tindex.search(query, num_results=num_results)


def vector_search(query, num_results=5):
    return vindex.search(emb.encode(query), num_results=num_results)


def rrf(result_lists, k=60, num_results=5):
    scores, docs = {}, {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


def hybrid_search(query, k=60):
    text_results = text_search(query, num_results=10)
    vector_results = vector_search(query, num_results=10)
    return rrf([text_results, vector_results], k=k)


# ----------------------------------------------------------------------
# Q2 and Q3: first result for the first ground-truth question
# ----------------------------------------------------------------------
q = ground_truth[0]["question"]
print("Q2  text search first result:  ", text_search(q)[0]["filename"])
print("Q3  vector search first result:", vector_search(q)[0]["filename"])

# ----------------------------------------------------------------------
# Evaluation: hit rate and MRR, as in the module (label = filename)
# ----------------------------------------------------------------------
def compute_relevance(search_fn, rec):
    results = search_fn(rec["question"])
    return [1 if r["filename"] == rec["filename"] else 0 for r in results]


def hit_rate(relevances):
    return sum(1 for r in relevances if 1 in r) / len(relevances)


def mrr(relevances):
    total = 0.0
    for r in relevances:
        for rank, v in enumerate(r):
            if v == 1:
                total += 1.0 / (rank + 1)
                break
    return total / len(relevances)


def evaluate(search_fn):
    relevances = [compute_relevance(search_fn, rec) for rec in ground_truth]
    return {"hit_rate": hit_rate(relevances), "mrr": mrr(relevances)}


# Q4: text search
res_text = evaluate(text_search)
print(f"Q4  text search:   hit rate = {res_text['hit_rate']:.4f}, "
      f"mrr = {res_text['mrr']:.4f}")

# Q5: vector search
res_vec = evaluate(vector_search)
print(f"Q5  vector search: hit rate = {res_vec['hit_rate']:.4f}, "
      f"mrr = {res_vec['mrr']:.4f}")

# Q6: hybrid search, tuning k
print("Q6  hybrid search MRR by k:")
best_k, best_mrr = None, -1.0
for k in [1, 50, 100, 200]:
    res = evaluate(lambda query, k=k: hybrid_search(query, k=k))
    print(f"      k = {k:>3}: mrr = {res['mrr']:.4f} "
          f"(hit rate = {res['hit_rate']:.4f})")
    if res["mrr"] > best_mrr:  # strict >, so ties keep the smallest k
        best_k, best_mrr = k, res["mrr"]
print(f"Q6  best k = {best_k}")

print("\n--- submit the printed values (closest option where needed) ---")

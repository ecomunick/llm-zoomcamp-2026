"""
LLM Zoomcamp 2026 - Homework 2: Vector Search

Turnkey solution. Prints all six answers.

Run once, in order:
    uv add onnxruntime tokenizers numpy tqdm minsearch gitsource
    uv add --dev huggingface-hub jupyter
    python download.py          # fetches the ONNX model (Xenova/all-MiniLM-L6-v2)
    python solution.py

The embedder is the lightweight ONNX runtime from the module (no PyTorch/CUDA),
so this runs fast and anywhere.
"""

import numpy as np
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index, VectorSearch
from embedder import Embedder

# ----------------------------------------------------------------------
# Setup: embedder + data (pinned to commit 8c1834d, the same 72 pages)
# ----------------------------------------------------------------------
emb = Embedder()  # loads models/Xenova/all-MiniLM-L6-v2

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]

# ----------------------------------------------------------------------
# Q1. Embedding a query -> first value of the 384-dim vector
# ----------------------------------------------------------------------
q1_query = "How does approximate nearest neighbor search work?"
v = emb.encode(q1_query)
print("Q1  v[0] =", round(float(v[0]), 4))

# ----------------------------------------------------------------------
# Q2. Cosine similarity (normalized vectors -> dot product == cosine)
# ----------------------------------------------------------------------
target = next(d for d in documents
              if d["filename"] == "02-vector-search/lessons/07-sqlitesearch-vector.md")
page_vec = emb.encode(target["content"])
print("Q2  cosine similarity =", round(float(page_vec.dot(v)), 4))

# ----------------------------------------------------------------------
# Q3. Chunk, embed all chunks, score against the Q1 query -> best chunk's file
# ----------------------------------------------------------------------
chunks = chunk_documents(documents, size=2000, step=1000)
X = emb.encode_batch([c["content"] for c in chunks])   # (n_chunks, 384)
scores = X.dot(v)                                       # (n_chunks,)
best = chunks[int(np.argmax(scores))]
print("Q3  highest-scoring chunk file =", best["filename"])

# ----------------------------------------------------------------------
# Q4. Vector search with minsearch.VectorSearch -> first result's file
# ----------------------------------------------------------------------
vindex = VectorSearch(keyword_fields=["filename"])
vindex.fit(X, chunks)

q4_query = "What metric do we use to evaluate a search engine?"
q4_vec = emb.encode(q4_query)
q4_results = vindex.search(q4_vec, num_results=5)
print("Q4  first vector result =", q4_results[0]["filename"])

# ----------------------------------------------------------------------
# Q5. Text search vs vector search: file in vector top-5 but not text top-5
# ----------------------------------------------------------------------
tindex = Index(text_fields=["content"], keyword_fields=["filename"])
tindex.fit(chunks)

q5_query = "How do I store vectors in PostgreSQL?"
q5_vec = emb.encode(q5_query)
q5_vector = vindex.search(q5_vec, num_results=5)
q5_text = tindex.search(q5_query, num_results=5)

vector_files = {r["filename"] for r in q5_vector}
text_files = {r["filename"] for r in q5_text}
only_vector = vector_files - text_files
print("Q5  in vector but not text =", sorted(only_vector))

# ----------------------------------------------------------------------
# Q6. Hybrid search with Reciprocal Rank Fusion -> first fused result
# ----------------------------------------------------------------------
def rrf(result_lists, k=60, num_results=5):
    scores, docs = {}, {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]

q6_query = "How do I give the model access to tools?"
q6_vec = emb.encode(q6_query)
vector_results = vindex.search(q6_vec, num_results=10)
text_results = tindex.search(q6_query, num_results=10)
fused = rrf([vector_results, text_results])
print("Q6  first after RRF =", fused[0]["filename"])

print("\n--- submit the printed values (closest option where needed) ---")

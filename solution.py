"""LLM Zoomcamp 2026 - Homework 1: Agentic RAG. Deterministic parts (Q1, Q2, Q4)."""

from gitsource import GithubRepositoryDataReader, chunk_documents
import minsearch


def load_documents():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub", repo_name="llm-zoomcamp",
        commit_id="8c1834d", allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    return [f.parse() for f in reader.read()]


def main():
    documents = load_documents()
    print("Q1  lesson pages:", len(documents))  # 72

    index = minsearch.Index(text_fields=["content"], keyword_fields=["filename"])
    index.fit(documents)
    query = "How does the agentic loop keep calling the model until it stops?"
    results = index.search(query, num_results=5)
    print("Q2  first result:", results[0]["filename"])

    chunks = chunk_documents(documents, size=2000, step=1000)
    print("Q4  chunks:", len(chunks))  # 295


if __name__ == "__main__":
    main()

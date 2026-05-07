from __future__ import annotations

import argparse

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME, DEFAULT_TOP_K


def query_rag(question: str, top_k: int = DEFAULT_TOP_K, course_filter: str | None = None) -> None:
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    query_embedding = embedder.encode([question]).tolist()

    where_filter = None

    # Optional example:
    # If you store course in metadata later, this can become useful.
    # Right now this only works if your metadata contains "course".
    if course_filter:
        where_filter = {"course": course_filter}

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        print("No results found.")
        return

    print()
    print(f"Question: {question}")
    print("=" * 80)

    for rank, (doc, meta, distance) in enumerate(zip(documents, metadatas, distances), start=1):
        print(f"\nRank {rank}")
        print("-" * 80)
        print(f"Distance: {distance}")
        print(f"Source: {meta.get('source_file', '')}")
        print(f"Pages: {meta.get('pages', '')}")
        print(f"Section: {meta.get('section', '')}")
        print(f"Suggested tags: {meta.get('suggested_tags', '')}")
        print(f"Approved tags: {meta.get('approved_tags', '')}")
        print()
        print(doc[:1200])
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query local Chroma RAG DB.")
    parser.add_argument("question", help="Question to ask.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--course", type=str, default=None)
    args = parser.parse_args()

    query_rag(
        question=args.question,
        top_k=args.top_k,
        course_filter=args.course,
    )


if __name__ == "__main__":
    main()
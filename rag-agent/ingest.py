from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

from config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME
from tag_suggest import suggest_tags, tags_to_metadata_string


def make_hash(text: str) -> str:
    """
    Content hash for cache invalidation later.
    If chunk text changes, this hash changes.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_join(items: list[Any] | None) -> str:
    if not items:
        return ""
    return " > ".join(str(x) for x in items if x is not None)


def extract_docling_metadata(chunk: Any) -> dict[str, str]:
    """
    Docling chunk metadata may vary depending on document type.
    This function extracts useful fields defensively.

    Expected useful fields:
    - headings
    - captions
    - page numbers from provenance, if available
    """

    meta = getattr(chunk, "meta", None)

    headings = getattr(meta, "headings", None) if meta is not None else None
    captions = getattr(meta, "captions", None) if meta is not None else None

    pages: set[str] = set()

    # Try to extract page numbers from Docling provenance objects.
    doc_items = getattr(meta, "doc_items", None) if meta is not None else None
    if doc_items:
        for item in doc_items:
            prov_list = getattr(item, "prov", None)
            if not prov_list:
                continue

            for prov in prov_list:
                page_no = getattr(prov, "page_no", None)
                if page_no is not None:
                    pages.add(str(page_no))

    return {
        "section": safe_join(headings),
        "captions": safe_join(captions),
        "pages": ", ".join(sorted(pages)),
    }


def ingest_file(file_path: str | Path) -> None:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"[1/6] Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("[2/6] Connecting to Chroma")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print(f"[3/6] Parsing with Docling: {file_path}")
    converter = DocumentConverter()
    result = converter.convert(str(file_path))
    doc = result.document

    print("[4/6] Chunking with Docling HybridChunker")
    chunker = HybridChunker()
    chunks = list(chunker.chunk(dl_doc=doc))

    if not chunks:
        print("No chunks generated.")
        return

    print(f"Generated {len(chunks)} chunks.")

    documents: list[str] = []
    embedding_texts: list[str] = []
    ids: list[str] = []
    metadatas: list[dict[str, str | int | float | bool]] = []

    source_file = file_path.name
    document_id = file_path.stem

    for i, chunk in enumerate(chunks):
        raw_text = chunk.text.strip()

        if not raw_text:
            continue

        # Contextualized text is usually better for embeddings,
        # because it can include headings or surrounding structure.
        contextualized_text = chunker.contextualize(chunk).strip()

        docling_meta = extract_docling_metadata(chunk)

        suggested_tags = suggest_tags(
            text=contextualized_text,
            source_file=source_file,
        )

        chunk_id = f"{document_id}_chunk_{i:04d}"
        content_hash = make_hash(raw_text)

        documents.append(raw_text)
        embedding_texts.append(contextualized_text)
        ids.append(chunk_id)

        metadatas.append(
            {
                "document_id": document_id,
                "source_file": source_file,
                "chunk_index": i,
                "section": docling_meta["section"],
                "captions": docling_meta["captions"],
                "pages": docling_meta["pages"],
                "suggested_tags": tags_to_metadata_string(suggested_tags),
                "approved_tags": "",  # fill later after human review
                "content_hash": content_hash,
            }
        )

    if not documents:
        print("No non-empty chunks to store.")
        return

    print(f"[5/6] Embedding {len(documents)} chunks locally")
    embeddings = embedder.encode(embedding_texts).tolist()

    print("[6/6] Storing chunks in Chroma")
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Done. Stored {len(documents)} chunks in collection '{COLLECTION_NAME}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a file into local Chroma RAG DB.")
    parser.add_argument("file_path", help="Path to PDF/DOCX/PPTX/etc.")
    args = parser.parse_args()

    ingest_file(args.file_path)


if __name__ == "__main__":
    main()
"""
chunker.py

Splits loaded documents into overlapping text chunks using recursive
character splitting, producing chunk dicts ready for embedding and
indexing in the hybrid RAG pipeline.
"""

from __future__ import annotations

from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.loader import LoadedDocument


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_splitter(
    chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> RecursiveCharacterTextSplitter:
    """Build a RecursiveCharacterTextSplitter with the pipeline's default settings."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_document(
    document: LoadedDocument, splitter: RecursiveCharacterTextSplitter = None
) -> List[Dict]:
    """
    Split a single LoadedDocument into chunk dicts.

    Each chunk dict has the shape:
        {"text": str, "source": str, "chunk_id": str}

    Args:
        document: The LoadedDocument to split.
        splitter: Optional pre-built splitter to reuse across documents.

    Returns:
        A list of chunk dicts, in order, for this document.
    """
    splitter = splitter or get_splitter()
    pieces = splitter.split_text(document.text)

    chunks: List[Dict] = []
    for index, piece in enumerate(pieces):
        chunk_id = f"{document.source}::chunk-{index}"
        chunks.append(
            {
                "text": piece,
                "source": document.source,
                "chunk_id": chunk_id,
            }
        )
    return chunks


def chunk_documents(documents: List[LoadedDocument]) -> List[Dict]:
    """Split many LoadedDocuments into a single flat list of chunk dicts."""
    splitter = get_splitter()
    all_chunks: List[Dict] = []
    for document in documents:
        all_chunks.extend(chunk_document(document, splitter))
    return all_chunks


if __name__ == "__main__":
    import sys

    from ingestion.loader import load_corpus

    corpus = sys.argv[1] if len(sys.argv) > 1 else "corpus"
    docs = load_corpus(corpus)
    chunks = chunk_documents(docs)
    print(f"Produced {len(chunks)} chunks from {len(docs)} documents")
    for chunk in chunks[:5]:
        print(f"- {chunk['chunk_id']} ({len(chunk['text'])} chars)")
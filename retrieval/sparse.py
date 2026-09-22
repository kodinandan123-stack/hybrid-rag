"""Sparse retrieval using BM25 over the document corpus."""

import re
from typing import Any

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class SparseRetriever:
    """BM25-based lexical retriever over pre-chunked documents."""

    def __init__(self, chunks: list[dict[str, Any]] | None = None):
        self.chunks: list[dict[str, Any]] = []
        self.bm25: BM25Okapi = None
        if chunks:
            self.index(chunks)

    def index(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        tokenized_corpus = [_tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.bm25 is None:
            raise ValueError("SparseRetriever has not been indexed yet")
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(
            zip(scores, self.chunks), key=lambda pair: pair[0], reverse=True
        )
        return [{"score": float(score), **chunk} for score, chunk in ranked[:top_k]]

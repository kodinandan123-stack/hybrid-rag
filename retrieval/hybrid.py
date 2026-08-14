"""
hybrid.py

Hybrid retriever combining dense and sparse search results via Reciprocal
Rank Fusion (RRF). Accepts any combination of ranked hit lists and fuses
them into a single ranking without requiring score normalisation across
retrieval strategies.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever


class HybridRetriever:
    """Fuses dense and sparse retrieval rankings using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever, rrf_k: int = 60):
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k

    def _rrf_scores(self, ranked_lists: List[List[Dict[str, Any]]]) -> Dict[str, float]:
        scores: Dict[str, float] = defaultdict(float)
        for ranked in ranked_lists:
            for rank, item in enumerate(ranked):
                key = item.get("chunk_id") or item.get("text")
                scores[key] += 1.0 / (self.rrf_k + rank + 1)
        return scores

    def search(self, query: str, top_k: int = 5, candidate_k: int = 20) -> List[Dict[str, Any]]:
        """Return the top_k chunks most relevant to query, fused from dense and sparse hits."""
        dense_hits = self.dense.search(query, top_k=candidate_k)
        sparse_hits = self.sparse.search(query, top_k=candidate_k)
        fused_scores = self._rrf_scores([dense_hits, sparse_hits])

        by_key: Dict[str, Dict[str, Any]] = {}
        for item in dense_hits + sparse_hits:
            key = item.get("chunk_id") or item.get("text")
            by_key[key] = item

        ranked_keys = sorted(fused_scores, key=lambda k: fused_scores[k], reverse=True)
        results = []
        for key in ranked_keys[:top_k]:
            merged = dict(by_key[key])
            merged["hybrid_score"] = fused_scores[key]
            results.append(merged)
        return results

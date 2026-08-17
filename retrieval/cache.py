"""
cache.py

In-memory LRU cache for hybrid retrieval results. Wraps HybridRetriever and
returns cached hits when the same (query, top_k) pair is seen again, avoiding
redundant vector and BM25 lookups during interactive or batch sessions.
"""

from __future__ import annotations

from typing import Any, Dict, List


class RetrievalCache:
    """LRU cache wrapper around any retriever exposing a search method."""

    def __init__(self, retriever: Any, maxsize: int = 128) -> None:
        self._retriever = retriever
        self._maxsize = maxsize
        self._cache: Dict[tuple, List[Dict[str, Any]]] = {}
        self._order: List[tuple] = []

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return cached hits if available, otherwise delegate to the retriever."""
        key = (query.strip().lower(), top_k)
        if key in self._cache:
            return self._cache[key]
        results = self._retriever.search(query, top_k=top_k)
        self._store(key, results)
        return results

    def invalidate(self, query=None, top_k=None):
        """Remove a specific key from the cache, or clear the entire cache."""
        if query is None:
            self._cache.clear()
            self._order.clear()
            return
        key = (query.strip().lower(), top_k if top_k is not None else 5)
        if key in self._cache:
            del self._cache[key]
            self._order.remove(key)

    @property
    def size(self) -> int:
        return len(self._cache)

    def _store(self, key, results):
        if len(self._cache) >= self._maxsize:
            oldest = self._order.pop(0)
            del self._cache[oldest]
        self._cache[key] = results
        self._order.append(key)

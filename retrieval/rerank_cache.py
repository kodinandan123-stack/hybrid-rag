"""rerank_cache.py: LRU cache with TTL for cross-encoder reranker results."""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class CacheEntry:
      scores: List[float]
      created_at: float = field(default_factory=time.time)


class RerankCache:
      """LRU cache for reranker scores keyed on (query, doc_ids) hash."""

    def __init__(self, max_size: int = 512, ttl_seconds: float = 300.0) -> None:
              self.max_size = max_size
              self.ttl = ttl_seconds
              self._store: OrderedDict[str, CacheEntry] = OrderedDict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(query: str, doc_ids: List[str]) -> str:
              raw = query + "|".join(sorted(doc_ids))
              return hashlib.sha256(raw.encode()).hexdigest()

    def _is_expired(self, entry: CacheEntry) -> bool:
              return (time.time() - entry.created_at) > self.ttl

    def _evict_expired(self) -> None:
              expired = [k for k, v in self._store.items() if self._is_expired(v)]
              for k in expired:
                            del self._store[k]

          # ------------------------------------------------------------------
          # Public API
          # ------------------------------------------------------------------

    def get(self, query: str, doc_ids: List[str]) -> Optional[List[float]]:
              """Return cached scores or None on miss / expiry."""
              key = self._make_key(query, doc_ids)
              entry = self._store.get(key)
              if entry is None:
                            return None
                        if self._is_expired(entry):
                                      del self._store[key]
                                      return None
                                  self._store.move_to_end(key)
        return entry.scores

    def set(self, query: str, doc_ids: List[str], scores: List[float]) -> None:
              """Store scores for the given (query, doc_ids) pair."""
        self._evict_expired()
        key = self._make_key(query, doc_ids)
        self._store[key] = CacheEntry(scores=scores)
        self._store.move_to_end(key)
        if len(self._store) > self.max_size:
                      self._store.popitem(last=False)

    def invalidate(self, query: str, doc_ids: List[str]) -> bool:
              """Remove a specific entry; return True if it existed."""
        key = self._make_key(query, doc_ids)
        return self._store.pop(key, None) is not None

    def clear(self) -> None:
              """Remove all entries."""
        self._store.clear()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
              self._evict_expired()
        return {"size": len(self._store), "max_size": self.max_size, "ttl_seconds": self.ttl}

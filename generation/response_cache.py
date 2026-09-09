"""response_cache.py

Caches generated responses keyed by a hash of the query and compressed
context, avoiding redundant LLM API calls for identical or near-identical
retrieval results.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
      query: str
      answer: str
      context_hash: str
      created_at: float = field(default_factory=time.time)
      hits: int = 0


class ResponseCache:
      """LRU cache with TTL for generated responses.

          Args:
                  max_size: Maximum number of responses to hold in memory.
                          ttl: Time-to-live in seconds.  0 means entries never expire.
                              """

    def __init__(self, max_size: int = 256, ttl: float = 3600.0) -> None:
              self.max_size = max_size
              self.ttl = ttl
              self._cache: OrderedDict[str, CachedResponse] = OrderedDict()

    def get(self, query: str, context_chunks: list[str]) -> Optional[str]:
              """Return a cached answer, or None if not found / expired."""
              key = self._make_key(query, context_chunks)
              entry = self._cache.get(key)
              if entry is None:
                            logger.debug("Cache miss for key %s", key[:8])
                            return None
                        if self._is_expired(entry):
                                      logger.debug("Cache entry expired for key %s", key[:8])
                                      del self._cache[key]
                                      return None
                                  self._cache.move_to_end(key)
        entry.hits += 1
        logger.info("Cache hit (hits=%d) for key %s", entry.hits, key[:8])
        return entry.answer

    def set(self, query: str, context_chunks: list[str], answer: str) -> None:
              """Store a generated answer in the cache."""
        key = self._make_key(query, context_chunks)
        if key in self._cache:
                      self._cache.move_to_end(key)
                      self._cache[key].answer = answer
                      self._cache[key].created_at = time.time()
                      return
                  if len(self._cache) >= self.max_size:
                                evicted_key, _ = self._cache.popitem(last=False)
                                logger.debug("Evicted LRU entry %s", evicted_key[:8])
                            entry = CachedResponse(
                                          query=query,
                                          answer=answer,
                                          context_hash=self._hash_context(context_chunks),
                            )
        self._cache[key] = entry
        logger.info("Cached response for key %s (cache size=%d)", key[:8], len(self._cache))

    def invalidate(self, query: str, context_chunks: list[str]) -> bool:
              """Remove a specific entry.  Returns True if it existed."""
        key = self._make_key(query, context_chunks)
        if key in self._cache:
                      del self._cache[key]
                      return True
                  return False

    def clear(self) -> None:
              """Flush the entire cache."""
        self._cache.clear()
        logger.info("Response cache cleared.")

    def stats(self) -> dict:
              """Return current cache statistics."""
        total_hits = sum(e.hits for e in self._cache.values())
        return {
                      "size": len(self._cache),
                      "max_size": self.max_size,
                      "ttl": self.ttl,
                      "total_hits": total_hits,
        }

    @staticmethod
    def _hash_context(chunks: list[str]) -> str:
              combined = "\n".join(chunks)
        return hashlib.sha256(combined.encode()).hexdigest()

    def _make_key(self, query: str, context_chunks: list[str]) -> str:
              context_hash = self._hash_context(context_chunks)
        raw = f"{query.strip().lower()}|{context_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _is_expired(self, entry: CachedResponse) -> bool:
              if self.ttl <= 0:
                            return False
                        return (time.time() - entry.created_at) > self.ttl

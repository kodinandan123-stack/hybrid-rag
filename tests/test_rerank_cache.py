"""tests/test_rerank_cache.py: unit tests for RerankCache."""
import time
import pytest
from retrieval.rerank_cache import RerankCache


@pytest.fixture()
def cache() -> RerankCache:
      return RerankCache(max_size=4, ttl_seconds=2.0)


def test_miss_on_empty_cache(cache):
      assert cache.get("what is AI?", ["doc1", "doc2"]) is None


def test_set_and_get(cache):
      scores = [0.9, 0.7]
      cache.set("what is AI?", ["doc1", "doc2"], scores)
      assert cache.get("what is AI?", ["doc1", "doc2"]) == scores


def test_key_order_independent(cache):
      scores = [0.8, 0.6]
      cache.set("query", ["b", "a"], scores)
      assert cache.get("query", ["a", "b"]) == scores


def test_ttl_expiry(cache):
      cache.set("expire me", ["d1"], [0.5])
      time.sleep(2.1)
      assert cache.get("expire me", ["d1"]) is None


def test_lru_eviction(cache):
      for i in range(4):
                cache.set(f"q{i}", [f"d{i}"], [float(i)])
            cache.get("q0", ["d0"])
    cache.set("q4", ["d4"], [4.0])
    assert cache.get("q1", ["d1"]) is None
    assert cache.get("q0", ["d0"]) == [0.0]


def test_invalidate(cache):
      cache.set("q", ["d"], [1.0])
    assert cache.invalidate("q", ["d"]) is True
    assert cache.get("q", ["d"]) is None


def test_invalidate_missing(cache):
      assert cache.invalidate("nonexistent", ["d"]) is False


def test_clear(cache):
      cache.set("q1", ["d1"], [0.9])
    cache.set("q2", ["d2"], [0.8])
    cache.clear()
    assert cache.stats()["size"] == 0


def test_stats(cache):
      cache.set("q", ["d"], [0.5])
    s = cache.stats()
    assert s["size"] == 1
    assert s["max_size"] == 4
    assert s["ttl_seconds"] == 2.0

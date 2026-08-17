"""tests/test_cache.py - unit tests for RetrievalCache."""
import pytest
from retrieval.cache import RetrievalCache


class _FakeRetriever:
    def __init__(self):
        self.call_count = 0

    def search(self, query, top_k=5):
        self.call_count += 1
        return [{"text": f"result for {query}", "score": 1.0}] * top_k


def test_cache_miss_calls_retriever():
    fake = _FakeRetriever()
    cache = RetrievalCache(fake, maxsize=10)
    results = cache.search("hello", top_k=3)
    assert fake.call_count == 1
    assert len(results) == 3


def test_cache_hit_avoids_second_call():
    fake = _FakeRetriever()
    cache = RetrievalCache(fake, maxsize=10)
    cache.search("hello", top_k=3)
    cache.search("hello", top_k=3)
    assert fake.call_count == 1


def test_different_top_k_different_keys():
    fake = _FakeRetriever()
    cache = RetrievalCache(fake, maxsize=10)
    cache.search("hello", top_k=3)
    cache.search("hello", top_k=5)
    assert fake.call_count == 2


def test_case_insensitive_normalization():
    fake = _FakeRetriever()
    cache = RetrievalCache(fake, maxsize=10)
    cache.search("Hello World", top_k=5)
    cache.search("hello world", top_k=5)
    assert fake.call_count == 1


def test_size_property():
    fake = _FakeRetriever()
    cache = RetrievalCache(fake, maxsize=10)
    assert cache.size == 0
    cache.search("a", top_k=5)
    assert cache.size == 1
    cache.search("b", top_k=5)
    assert cache.size == 2


def test_lru_eviction():
    fake = _FakeRetriever()
    cache = RetrievalCache(fake, maxsize=2)
    cache.search("a", top_k=5)
    cache.search("b", top_k=5)
    cache.search("c", top_k=5)  # evicts "a"
    assert cache.size == 2
    cache.search("a", top_k=5)  # cache miss; must call retriever
    assert fake.call_count == 4


def test_invalidate_specific_entry():
    fake = _FakeRetriever()
    cache = RetrievalCache(fake, maxsize=10)
    cache.search("x", top_k=5)
    cache.invalidate("x", top_k=5)
    assert cache.size == 0
    cache.search("x", top_k=5)
    assert fake.call_count == 2


def test_invalidate_all():
    fake = _FakeRetriever()
    cache = RetrievalCache(fake, maxsize=10)
    cache.search("x", top_k=5)
    cache.search("y", top_k=5)
    cache.invalidate()
    assert cache.size == 0


def test_invalidate_nonexistent_key_is_safe():
    fake = _FakeRetriever()
    cache = RetrievalCache(fake, maxsize=10)
    cache.invalidate("does-not-exist", top_k=5)  # must not raise
    assert cache.size == 0

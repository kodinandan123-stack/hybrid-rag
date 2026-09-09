"""test_response_cache.py

Unit tests for generation.response_cache.ResponseCache.
"""

import time
import unittest

from generation.response_cache import ResponseCache


QUERY = "What is hybrid RAG?"
CHUNKS = ["Dense search retrieves semantically similar chunks.",
                    "BM25 retrieves lexically matching chunks."]
ANSWER = "Hybrid RAG combines dense and sparse retrieval."


class TestResponseCacheBasic(unittest.TestCase):
      def setUp(self):
                self.cache = ResponseCache(max_size=4, ttl=60.0)

      def test_miss_on_empty_cache(self):
                result = self.cache.get(QUERY, CHUNKS)
                self.assertIsNone(result)

      def test_set_and_get(self):
                self.cache.set(QUERY, CHUNKS, ANSWER)
                result = self.cache.get(QUERY, CHUNKS)
                self.assertEqual(result, ANSWER)

      def test_hit_increments_counter(self):
                self.cache.set(QUERY, CHUNKS, ANSWER)
                self.cache.get(QUERY, CHUNKS)
                self.cache.get(QUERY, CHUNKS)
                entry = list(self.cache._cache.values())[0]
                self.assertEqual(entry.hits, 2)

      def test_different_query_is_miss(self):
                self.cache.set(QUERY, CHUNKS, ANSWER)
                result = self.cache.get("Different question?", CHUNKS)
                self.assertIsNone(result)

      def test_different_chunks_is_miss(self):
                self.cache.set(QUERY, CHUNKS, ANSWER)
                result = self.cache.get(QUERY, ["Completely different chunk."])
                self.assertIsNone(result)

      def test_invalidate_removes_entry(self):
                self.cache.set(QUERY, CHUNKS, ANSWER)
                removed = self.cache.invalidate(QUERY, CHUNKS)
                self.assertTrue(removed)
                self.assertIsNone(self.cache.get(QUERY, CHUNKS))

      def test_invalidate_missing_key_returns_false(self):
                removed = self.cache.invalidate(QUERY, CHUNKS)
                self.assertFalse(removed)

      def test_clear_empties_cache(self):
                self.cache.set(QUERY, CHUNKS, ANSWER)
                self.cache.clear()
                self.assertEqual(len(self.cache._cache), 0)

      def test_stats_structure(self):
                self.cache.set(QUERY, CHUNKS, ANSWER)
                stats = self.cache.stats()
                self.assertIn("size", stats)
                self.assertIn("max_size", stats)
                self.assertIn("ttl", stats)
                self.assertIn("total_hits", stats)
                self.assertEqual(stats["size"], 1)


class TestResponseCacheLRUEviction(unittest.TestCase):
      def setUp(self):
                self.cache = ResponseCache(max_size=2, ttl=0)

      def test_lru_evicts_oldest_entry(self):
                self.cache.set("q1", ["c1"], "a1")
                self.cache.set("q2", ["c2"], "a2")
                self.cache.get("q1", ["c1"])
                self.cache.set("q3", ["c3"], "a3")
                self.assertIsNone(self.cache.get("q2", ["c2"]))
                self.assertEqual(self.cache.get("q1", ["c1"]), "a1")
                self.assertEqual(self.cache.get("q3", ["c3"]), "a3")

      def test_max_size_not_exceeded(self):
                for i in range(5):
                              self.cache.set(f"q{i}", [f"c{i}"], f"a{i}")
                          self.assertLessEqual(len(self.cache._cache), 2)


class TestResponseCacheTTL(unittest.TestCase):
      def test_expired_entry_returns_none(self):
                cache = ResponseCache(max_size=4, ttl=0.05)
                cache.set(QUERY, CHUNKS, ANSWER)
                time.sleep(0.1)
                result = cache.get(QUERY, CHUNKS)
                self.assertIsNone(result)

      def test_zero_ttl_never_expires(self):
                cache = ResponseCache(max_size=4, ttl=0)
                cache.set(QUERY, CHUNKS, ANSWER)
                result = cache.get(QUERY, CHUNKS)
                self.assertEqual(result, ANSWER)


if __name__ == "__main__":
      unittest.main()

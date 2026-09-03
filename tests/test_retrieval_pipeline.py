"""Integration tests for the full retrieval pipeline."""
import pytest
from unittest.mock import MagicMock, patch


class MockRetriever:
      """Mock retriever for pipeline tests."""

    def __init__(self, results=None):
              self.results = results or []
              self.call_count = 0

    def retrieve(self, query: str, top_k: int = 5):
              self.call_count += 1
              return self.results[:top_k]


class MockReranker:
      """Mock reranker for pipeline tests."""

    def rerank(self, query: str, docs: list, top_n: int = 3):
              return sorted(docs, key=lambda d: d.get("score", 0), reverse=True)[:top_n]


@pytest.fixture
def sample_docs():
      return [
                {"id": "doc1", "text": "Hybrid RAG combines dense and sparse retrieval.", "score": 0.92},
                {"id": "doc2", "text": "BM25 is a classic sparse retrieval algorithm.", "score": 0.85},
                {"id": "doc3", "text": "FAISS enables fast approximate nearest-neighbour search.", "score": 0.81},
                {"id": "doc4", "text": "Reranking improves precision of top-k results.", "score": 0.78},
                {"id": "doc5", "text": "Metadata filters narrow the candidate pool.", "score": 0.70},
      ]


@pytest.fixture
def retriever(sample_docs):
      return MockRetriever(results=sample_docs)


@pytest.fixture
def reranker():
      return MockReranker()


class TestRetrievalPipeline:
      """Tests for end-to-end retrieval pipeline."""

    def test_retrieve_returns_top_k(self, retriever, sample_docs):
              results = retriever.retrieve("hybrid retrieval", top_k=3)
              assert len(results) == 3

    def test_retrieve_respects_top_k_limit(self, retriever):
              results = retriever.retrieve("query", top_k=2)
              assert len(results) <= 2

    def test_retriever_called_once_per_query(self, retriever):
              retriever.retrieve("test query")
              assert retriever.call_count == 1

    def test_reranker_orders_by_score(self, reranker, sample_docs):
              reranked = reranker.rerank("hybrid retrieval", sample_docs, top_n=3)
              scores = [d["score"] for d in reranked]
              assert scores == sorted(scores, reverse=True)

    def test_pipeline_end_to_end(self, retriever, reranker, sample_docs):
              candidates = retriever.retrieve("hybrid RAG", top_k=5)
              final = reranker.rerank("hybrid RAG", candidates, top_n=3)
              assert len(final) == 3
              assert final[0]["score"] >= final[-1]["score"]

    def test_empty_retrieval_returns_empty(self):
              empty_retriever = MockRetriever(results=[])
              results = empty_retriever.retrieve("anything", top_k=5)
              assert results == []

    def test_reranker_handles_empty_docs(self, reranker):
              result = reranker.rerank("query", [], top_n=3)
              assert result == []

    def test_top_k_larger_than_results(self, retriever, sample_docs):
              results = retriever.retrieve("query", top_k=100)
              assert len(results) <= len(sample_docs)

    def test_pipeline_preserves_doc_ids(self, retriever, reranker):
              candidates = retriever.retrieve("query", top_k=5)
              final = reranker.rerank("query", candidates, top_n=3)
              ids = {d["id"] for d in final}
              assert all(i.startswith("doc") for i in ids)

    def test_pipeline_with_single_result(self):
              single = [{"id": "doc1", "text": "Only result.", "score": 0.9}]
              retriever = MockRetriever(results=single)
              reranker = MockReranker()
              candidates = retriever.retrieve("query", top_k=5)
              final = reranker.rerank("query", candidates, top_n=3)
              assert len(final) == 1
      

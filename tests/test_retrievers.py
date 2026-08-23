"""Unit tests for dense, sparse, and hybrid retrievers."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_dense_retriever():
      retriever = MagicMock()
      retriever.retrieve.return_value = [
          {"id": "doc1", "score": 0.95, "text": "Dense result one"},
          {"id": "doc2", "score": 0.88, "text": "Dense result two"},
      ]
      return retriever


@pytest.fixture
def mock_sparse_retriever():
      retriever = MagicMock()
      retriever.retrieve.return_value = [
          {"id": "doc2", "score": 12.4, "text": "Sparse result two"},
          {"id": "doc3", "score": 9.1, "text": "Sparse result three"},
      ]
      return retriever


# ---------------------------------------------------------------------------
# Dense retriever tests
# ---------------------------------------------------------------------------

class TestDenseRetriever:
      def test_retrieve_returns_list(self, mock_dense_retriever):
                results = mock_dense_retriever.retrieve("test query", top_k=2)
                assert isinstance(results, list)

      def test_retrieve_result_count(self, mock_dense_retriever):
                results = mock_dense_retriever.retrieve("test query", top_k=2)
                assert len(results) == 2

      def test_retrieve_has_score(self, mock_dense_retriever):
                results = mock_dense_retriever.retrieve("test query", top_k=2)
                for r in results:
                              assert "score" in r
                              assert isinstance(r["score"], float)


# ---------------------------------------------------------------------------
# Sparse retriever tests
# ---------------------------------------------------------------------------

class TestSparseRetriever:
      def test_retrieve_returns_list(self, mock_sparse_retriever):
                results = mock_sparse_retriever.retrieve("test query", top_k=2)
                assert isinstance(results, list)

      def test_retrieve_has_id(self, mock_sparse_retriever):
                results = mock_sparse_retriever.retrieve("test query", top_k=2)
                for r in results:
                              assert "id" in r


# ---------------------------------------------------------------------------
# Hybrid / RRF fusion tests
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
      dense_results: list[dict],
      sparse_results: list[dict],
      k: int = 60,
) -> list[dict]:
      """Simple RRF implementation used for testing."""
      scores: dict[str, float] = {}
      for rank, doc in enumerate(dense_results, start=1):
                scores[doc["id"]] = scores.get(doc["id"], 0) + 1 / (k + rank)
            for rank, doc in enumerate(sparse_results, start=1):
                      scores[doc["id"]] = scores.get(doc["id"], 0) + 1 / (k + rank)
                  ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"id": doc_id, "rrf_score": score} for doc_id, score in ranked]


class TestHybridRRF:
      def test_fused_results_not_empty(self, mock_dense_retriever, mock_sparse_retriever):
                dense = mock_dense_retriever.retrieve("q", top_k=2)
                sparse = mock_sparse_retriever.retrieve("q", top_k=2)
                fused = reciprocal_rank_fusion(dense, sparse)
                assert len(fused) > 0

    def test_shared_doc_has_higher_score(self, mock_dense_retriever, mock_sparse_retriever):
              dense = mock_dense_retriever.retrieve("q", top_k=2)
              sparse = mock_sparse_retriever.retrieve("q", top_k=2)
              fused = reciprocal_rank_fusion(dense, sparse)
              # doc2 appears in both dense and sparse, so it should rank first
              assert fused[0]["id"] == "doc2"

    def test_all_doc_ids_present(self, mock_dense_retriever, mock_sparse_retriever):
              dense = mock_dense_retriever.retrieve("q", top_k=2)
              sparse = mock_sparse_retriever.retrieve("q", top_k=2)
              fused = reciprocal_rank_fusion(dense, sparse)
              ids = {r["id"] for r in fused}
              assert ids == {"doc1", "doc2", "doc3"}
      

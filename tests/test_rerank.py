"""Unit tests for retrieval/rerank.py."""

from unittest.mock import patch, MagicMock

from retrieval.rerank import Reranker


def _candidate(chunk_id, text):
      return {"chunk_id": chunk_id, "text": text, "source": "doc.md"}


@patch("retrieval.rerank.CrossEncoder")
def test_rerank_orders_by_score_descending(mock_cross_encoder):
      mock_model = MagicMock()
      mock_model.predict.return_value = [0.2, 0.9, 0.5]
      mock_cross_encoder.return_value = mock_model

    reranker = Reranker()
    candidates = [_candidate("a", "alpha"), _candidate("b", "bravo"), _candidate("c", "charlie")]

    results = reranker.rerank("query", candidates, top_k=3)

    ids = [item["chunk_id"] for item in results]
    assert ids == ["b", "c", "a"]
    assert all("rerank_score" in item for item in results)


@patch("retrieval.rerank.CrossEncoder")
def test_rerank_respects_top_k(mock_cross_encoder):
      mock_model = MagicMock()
      mock_model.predict.return_value = [0.1, 0.4, 0.9]
      mock_cross_encoder.return_value = mock_model

    reranker = Reranker()
    candidates = [_candidate("a", "alpha"), _candidate("b", "bravo"), _candidate("c", "charlie")]

    results = reranker.rerank("query", candidates, top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c"


@patch("retrieval.rerank.CrossEncoder")
def test_rerank_handles_empty_candidates(mock_cross_encoder):
      mock_cross_encoder.return_value = MagicMock()
      reranker = Reranker()

    assert reranker.rerank("query", [], top_k=5) == []

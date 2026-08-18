"""Tests for retrieval/hybrid.py — HybridRetriever and RRF fusion."""

from unittest.mock import MagicMock

import pytest

from retrieval.hybrid import HybridRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hits(ids):
    """Return a ranked list of chunk dicts keyed by chunk_id."""
    return [{"chunk_id": cid, "text": f"text for {cid}"} for cid in ids]


def _mock_retriever(hits):
    """Return a mock retriever whose .search() returns *hits*."""
    m = MagicMock()
    m.search.return_value = hits
    return m


# ---------------------------------------------------------------------------
# HybridRetriever._rrf_scores
# ---------------------------------------------------------------------------

def test_rrf_scores_returns_dict():
    dense = _mock_retriever(_make_hits(["a", "b"]))
    sparse = _mock_retriever(_make_hits(["b", "c"]))
    hr = HybridRetriever(dense=dense, sparse=sparse)
    ranked_lists = [_make_hits(["a", "b"]), _make_hits(["b", "c"])]
    scores = hr._rrf_scores(ranked_lists)
    assert isinstance(scores, dict)


def test_rrf_scores_higher_for_top_ranked():
    dense = _mock_retriever([])
    sparse = _mock_retriever([])
    hr = HybridRetriever(dense=dense, sparse=sparse)
    ranked = [{"chunk_id": "first", "text": "x"}, {"chunk_id": "second", "text": "y"}]
    scores = hr._rrf_scores([ranked])
    assert scores["first"] > scores["second"]


def test_rrf_scores_accumulate_across_lists():
    dense = _mock_retriever([])
    sparse = _mock_retriever([])
    hr = HybridRetriever(dense=dense, sparse=sparse)
    list_a = [{"chunk_id": "shared"}, {"chunk_id": "only_a"}]
    list_b = [{"chunk_id": "shared"}, {"chunk_id": "only_b"}]
    scores = hr._rrf_scores([list_a, list_b])
    assert scores["shared"] > scores["only_a"]
    assert scores["shared"] > scores["only_b"]


def test_rrf_scores_uses_text_as_fallback_key():
    dense = _mock_retriever([])
    sparse = _mock_retriever([])
    hr = HybridRetriever(dense=dense, sparse=sparse)
    ranked = [{"text": "unique text here"}]
    scores = hr._rrf_scores([ranked])
    assert "unique text here" in scores


def test_rrf_scores_custom_rrf_k():
    dense = _mock_retriever([])
    sparse = _mock_retriever([])
    hr = HybridRetriever(dense=dense, sparse=sparse, rrf_k=1)
    ranked = [{"chunk_id": "x"}]
    scores = hr._rrf_scores([ranked])
    assert abs(scores["x"] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# HybridRetriever.search
# ---------------------------------------------------------------------------

def test_search_calls_both_retrievers():
    dense_hits = _make_hits(["d1", "d2"])
    sparse_hits = _make_hits(["s1", "s2"])
    dense = _mock_retriever(dense_hits)
    sparse = _mock_retriever(sparse_hits)
    hr = HybridRetriever(dense=dense, sparse=sparse)
    hr.search("query", top_k=5)
    dense.search.assert_called_once()
    sparse.search.assert_called_once()


def test_search_returns_list_of_dicts():
    dense = _mock_retriever(_make_hits(["a", "b"]))
    sparse = _mock_retriever(_make_hits(["b", "c"]))
    hr = HybridRetriever(dense=dense, sparse=sparse)
    results = hr.search("hello", top_k=3)
    assert isinstance(results, list)
    assert all(isinstance(r, dict) for r in results)


def test_search_respects_top_k():
    dense = _mock_retriever(_make_hits(["a", "b", "c"]))
    sparse = _mock_retriever(_make_hits(["d", "e", "f"]))
    hr = HybridRetriever(dense=dense, sparse=sparse)
    results = hr.search("query", top_k=2)
    assert len(results) <= 2


def test_search_includes_hybrid_score():
    dense = _mock_retriever(_make_hits(["a", "b"]))
    sparse = _mock_retriever(_make_hits(["a", "c"]))
    hr = HybridRetriever(dense=dense, sparse=sparse)
    results = hr.search("query", top_k=5)
    for r in results:
        assert "hybrid_score" in r


def test_search_ranks_shared_chunk_first():
    """A chunk that appears in both dense and sparse hits should rank highest."""
    dense = _mock_retriever(_make_hits(["shared", "only_dense"]))
    sparse = _mock_retriever(_make_hits(["shared", "only_sparse"]))
    hr = HybridRetriever(dense=dense, sparse=sparse)
    results = hr.search("query", top_k=3)
    assert results[0]["chunk_id"] == "shared"


def test_search_deduplicates_chunks():
    """The same chunk_id should appear at most once in results."""
    dense = _mock_retriever(_make_hits(["x", "y"]))
    sparse = _mock_retriever(_make_hits(["x", "z"]))
    hr = HybridRetriever(dense=dense, sparse=sparse)
    results = hr.search("query", top_k=5)
    ids = [r.get("chunk_id") for r in results]
    assert len(ids) == len(set(ids))


def test_search_empty_hits_returns_empty():
    dense = _mock_retriever([])
    sparse = _mock_retriever([])
    hr = HybridRetriever(dense=dense, sparse=sparse)
    results = hr.search("query", top_k=5)
    assert results == []

"""Tests for retrieval/sparse.py — SparseRetriever and _tokenize helper."""

import pytest

from retrieval.sparse import SparseRetriever, _tokenize


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

def test_tokenize_lowercases_and_splits():
    tokens = _tokenize("Hello World")
    assert tokens == ["hello", "world"]


def test_tokenize_strips_punctuation():
    tokens = _tokenize("foo, bar. baz!")
    assert "foo" in tokens
    assert "bar" in tokens
    assert "baz" in tokens


def test_tokenize_empty_string():
    assert _tokenize("") == []


# ---------------------------------------------------------------------------
# SparseRetriever.index
# ---------------------------------------------------------------------------

CHUNKS = [
    {"chunk_id": "c1", "text": "the cat sat on the mat"},
    {"chunk_id": "c2", "text": "dogs are loyal animals"},
    {"chunk_id": "c3", "text": "the cat chased the dog around the yard"},
]


def test_index_sets_bm25():
    sr = SparseRetriever()
    assert sr.bm25 is None
    sr.index(CHUNKS)
    assert sr.bm25 is not None


def test_index_stores_chunks():
    sr = SparseRetriever()
    sr.index(CHUNKS)
    assert sr.chunks == CHUNKS


def test_constructor_with_chunks_indexes_immediately():
    sr = SparseRetriever(chunks=CHUNKS)
    assert sr.bm25 is not None
    assert len(sr.chunks) == 3


# ---------------------------------------------------------------------------
# SparseRetriever.search
# ---------------------------------------------------------------------------

def test_search_returns_list_of_dicts():
    sr = SparseRetriever(chunks=CHUNKS)
    results = sr.search("cat", top_k=2)
    assert isinstance(results, list)
    assert all(isinstance(r, dict) for r in results)


def test_search_respects_top_k():
    sr = SparseRetriever(chunks=CHUNKS)
    results = sr.search("cat", top_k=1)
    assert len(results) <= 1


def test_search_includes_score_field():
    sr = SparseRetriever(chunks=CHUNKS)
    results = sr.search("cat", top_k=3)
    for r in results:
        assert "score" in r


def test_search_relevance_ordering():
    """The chunk most relevant to 'cat' should rank above the dog-only chunk."""
    sr = SparseRetriever(chunks=CHUNKS)
    results = sr.search("cat", top_k=3)
    chunk_ids = [r.get("chunk_id") for r in results]
    assert chunk_ids[0] != "c2"


def test_search_before_index_raises():
    sr = SparseRetriever()
    with pytest.raises(ValueError, match="not been indexed"):
        sr.search("anything")


def test_search_default_top_k_is_five():
    chunks = [{"chunk_id": f"c{i}", "text": f"word{i} content"} for i in range(10)]
    sr = SparseRetriever(chunks=chunks)
    results = sr.search("word")
    assert len(results) <= 5


def test_search_preserves_chunk_fields():
    sr = SparseRetriever(chunks=CHUNKS)
    results = sr.search("cat", top_k=3)
    for r in results:
        assert "chunk_id" in r
        assert "text" in r

"""Unit tests for retrieval/hybrid.py and retrieval/sparse.py."""

from retrieval.hybrid import HybridRetriever
from retrieval.sparse import SparseRetriever


class _FakeRetriever:
    """Stub retriever returning a fixed, pre-ranked list of hits."""

    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k=5):
        return self._hits[:top_k]


def _chunk(chunk_id, text):
    return {"chunk_id": chunk_id, "text": text, "source": "doc.md"}


def test_hybrid_retriever_fuses_dense_and_sparse_results():
    dense = _FakeRetriever([_chunk("a", "alpha"), _chunk("b", "bravo")])
    sparse = _FakeRetriever([_chunk("b", "bravo"), _chunk("c", "charlie")])
    retriever = HybridRetriever(dense=dense, sparse=sparse)

    results = retriever.search("query", top_k=3)

    ids = [item["chunk_id"] for item in results]
    assert "b" in ids
    assert results[0]["chunk_id"] == "b"
    assert all("hybrid_score" in item for item in results)


def test_hybrid_retriever_respects_top_k():
    dense = _FakeRetriever([_chunk("a", "alpha"), _chunk("b", "bravo"), _chunk("c", "charlie")])
    sparse = _FakeRetriever([])
    retriever = HybridRetriever(dense=dense, sparse=sparse)

    results = retriever.search("query", top_k=2)

    assert len(results) == 2


def test_sparse_retriever_ranks_by_lexical_overlap():
    chunks = [
        _chunk("a", "the quick brown fox"),
        _chunk("b", "a slow green turtle"),
    ]
    retriever = SparseRetriever(chunks)

    results = retriever.search("quick fox", top_k=1)

    assert results[0]["chunk_id"] == "a"


def test_sparse_retriever_raises_before_indexing():
    retriever = SparseRetriever()
    try:
        retriever.search("query")
        assert False, "expected ValueError"
    except ValueError:
        pass

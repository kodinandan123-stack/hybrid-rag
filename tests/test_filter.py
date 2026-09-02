"""Unit tests for retrieval.filter."""

import pytest

from retrieval.filter import MetadataFilter, apply_filters


CHUNKS = [
      {"text": "a", "metadata": {"source": "doc1.pdf", "page": 1}},
      {"text": "b", "metadata": {"source": "doc2.pdf", "page": 3}},
      {"text": "c", "metadata": {"source": "doc1.pdf", "page": 5}},
]


def test_eq_filter():
      flt = MetadataFilter(key="source", value="doc1.pdf")
      result = apply_filters(CHUNKS, [flt])
      assert len(result) == 2
      assert all(c["metadata"]["source"] == "doc1.pdf" for c in result)


def test_ne_filter():
      flt = MetadataFilter(key="source", value="doc1.pdf", operator="ne")
      result = apply_filters(CHUNKS, [flt])
      assert len(result) == 1
      assert result[0]["metadata"]["source"] == "doc2.pdf"


def test_in_filter():
      flt = MetadataFilter(key="page", value=[1, 5], operator="in")
      result = apply_filters(CHUNKS, [flt])
      assert len(result) == 2


def test_combined_filters():
      filters = [
                MetadataFilter(key="source", value="doc1.pdf"),
                MetadataFilter(key="page", value=1),
      ]
      result = apply_filters(CHUNKS, filters)
      assert len(result) == 1
      assert result[0]["text"] == "a"


def test_no_filters():
      result = apply_filters(CHUNKS, [])
      assert result == CHUNKS


def test_unknown_operator():
      flt = MetadataFilter(key="source", value="x", operator="gt")
      with pytest.raises(ValueError, match="Unknown operator"):
                apply_filters(CHUNKS, [flt])

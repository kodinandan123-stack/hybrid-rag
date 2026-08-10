"""Unit tests for retrieval/dense.py."""

from unittest.mock import MagicMock, patch

import numpy as np

from retrieval.dense import DenseRetriever


@patch("retrieval.dense.QdrantClient")
@patch("retrieval.dense.SentenceTransformer")
def test_index_creates_collection_when_missing(mock_transformer, mock_qdrant):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    mock_transformer.return_value = mock_model

    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])
    mock_qdrant.return_value = mock_client

    retriever = DenseRetriever()
    retriever.index([{"text": "alpha"}, {"text": "bravo"}])

    mock_client.create_collection.assert_called_once()
    mock_client.upsert.assert_called_once()
    upsert_points = mock_client.upsert.call_args.kwargs["points"]
    assert len(upsert_points) == 2


@patch("retrieval.dense.QdrantClient")
@patch("retrieval.dense.SentenceTransformer")
def test_index_skips_collection_creation_when_it_exists(mock_transformer, mock_qdrant):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_transformer.return_value = mock_model

    mock_client = MagicMock()
    existing = MagicMock()
    existing.name = "hybrid_rag_chunks"
    mock_client.get_collections.return_value = MagicMock(collections=[existing])
    mock_qdrant.return_value = mock_client

    retriever = DenseRetriever()
    retriever.index([{"text": "alpha"}])

    mock_client.create_collection.assert_not_called()


@patch("retrieval.dense.QdrantClient")
@patch("retrieval.dense.SentenceTransformer")
def test_search_returns_score_and_payload(mock_transformer, mock_qdrant):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_transformer.return_value = mock_model

    hit = MagicMock(score=0.87, payload={"text": "alpha", "source": "doc.md"})
    mock_client = MagicMock()
    mock_client.search.return_value = [hit]
    mock_qdrant.return_value = mock_client

    retriever = DenseRetriever()
    results = retriever.search("query", top_k=1)

    assert results == [{"score": 0.87, "text": "alpha", "source": "doc.md"}]

"""Unit tests for api/main.py."""

import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

with patch("retrieval.dense.SentenceTransformer"), patch("retrieval.dense.QdrantClient"):
    from api.main import app
    import api.main as main_module

from fastapi.testclient import TestClient

client = TestClient(app, raise_server_exceptions=False)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_indexes_chunks_and_returns_count():
    with patch.object(main_module, "_dense") as mock_dense:
        chunks = [{"text": "alpha"}, {"text": "bravo"}]
        response = client.post("/index", json=chunks)

    assert response.status_code == 200
    assert response.json() == {"indexed": 2}
    mock_dense.index.assert_called_once_with(chunks)


def test_query_before_index_returns_400():
    main_module._hybrid = None
    response = client.post("/query", json={"query": "what is RAG?"})
    assert response.status_code == 400
    assert "not indexed" in response.json()["detail"]


def test_index_with_empty_chunks_returns_400():
    response = client.post("/index", json=[])
    assert response.status_code == 400


def test_query_with_invalid_top_k_returns_422():
    response = client.post("/query", json={"query": "what is RAG?", "top_k": 0})
    assert response.status_code == 422


def test_query_returns_answer_and_sources():
    mock_hybrid = MagicMock()
    mock_hybrid.search.return_value = [{"text": "alpha", "source": "doc.md"}]
    main_module._hybrid = mock_hybrid

    with patch.object(main_module._generator, "generate", return_value="alpha is a chunk"):
        response = client.post("/query", json={"query": "what is alpha?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "alpha is a chunk"
    assert body["sources"] == [{"text": "alpha", "source": "doc.md"}]

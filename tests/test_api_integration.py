"""tests/test_api_integration.py - integration tests for FastAPI /index and /query endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    def test_liveness(self):
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_readiness_all_healthy(self):
        with patch("api.health.check_qdrant", return_value=True), \
             patch("api.health.check_redis", return_value=True):
            response = client.get("/health/ready")
            assert response.status_code == 200
            data = response.json()
            assert data["qdrant"] is True
            assert data["redis"] is True

    def test_readiness_degraded(self):
        with patch("api.health.check_qdrant", return_value=False), \
             patch("api.health.check_redis", return_value=True):
            response = client.get("/health/ready")
            assert response.status_code == 503


class TestIndexEndpoint:
    def test_index_success(self):
        with patch("api.main.ingest_documents", return_value={"chunks": 42}):
            response = client.post("/index", json={"corpus_dir": "data/corpus"})
            assert response.status_code == 200
            assert response.json()["chunks"] == 42

    def test_index_missing_body(self):
        response = client.post("/index", json={})
        assert response.status_code in (400, 422)

    def test_index_invalid_dir(self):
        with patch("api.main.ingest_documents", side_effect=FileNotFoundError("not found")):
            response = client.post("/index", json={"corpus_dir": "/no/such/path"})
            assert response.status_code == 400


class TestQueryEndpoint:
    def test_query_success(self):
        mock_result = {"answer": "Paris", "sources": ["doc1.pdf"], "score": 0.91}
        with patch("api.main.run_pipeline", return_value=mock_result):
            response = client.post("/query", json={"question": "What is the capital of France?"})
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "Paris"
            assert "sources" in data

    def test_query_missing_question(self):
        response = client.post("/query", json={})
        assert response.status_code in (400, 422)

    def test_query_empty_question(self):
        response = client.post("/query", json={"question": ""})
        assert response.status_code in (400, 422)

    def test_query_returns_sources(self):
        mock_result = {"answer": "42", "sources": ["a.pdf", "b.pdf"], "score": 0.85}
        with patch("api.main.run_pipeline", return_value=mock_result):
            response = client.post("/query", json={"question": "Answer?"})
            assert len(response.json()["sources"]) == 2

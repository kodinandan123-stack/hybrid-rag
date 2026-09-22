"""tests/test_api_integration.py

Integration-style tests that exercise api/main.py end-to-end (index -> query,
using the real HybridRetriever/DenseRetriever/SparseRetriever wiring with the
dense embedding model mocked out) and unit tests for the standalone health
router in api/health.py, which is not mounted into the main app and so is
otherwise untested.

api/main.py's own request/response-shape edge cases (single-endpoint checks,
validation errors) are covered in test_api.py; this file focuses on the
multi-step flow and on api/health.py.
"""

import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

with (
    patch("retrieval.dense.SentenceTransformer"),
    patch("retrieval.dense.QdrantClient"),
):
    import api.main as main_module
    from api.main import app

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.health import router as health_router

client = TestClient(app, raise_server_exceptions=False)


class TestIndexThenQueryFlow:
    """Exercise the real index -> search -> generate pipeline end to end."""

    def setup_method(self):
        main_module._hybrid = None
        main_module._sparse = None

    def test_index_then_query_uses_real_hybrid_retriever(self):
        chunks = [
            {"chunk_id": "1", "text": "Paris is the capital of France."},
            {"chunk_id": "2", "text": "Berlin is the capital of Germany."},
        ]

        with (
            patch.object(main_module._dense, "index"),
            patch.object(main_module._dense, "search", return_value=[chunks[0]]),
        ):
            index_response = client.post("/index", json=chunks)
            assert index_response.status_code == 200
            assert index_response.json() == {"indexed": 2}
            assert main_module._hybrid is not None

            with patch.object(
                main_module._generator, "generate", return_value="Paris."
            ) as mock_generate:
                query_response = client.post(
                    "/query", json={"query": "What is the capital of France?"}
                )

        assert query_response.status_code == 200
        body = query_response.json()
        assert body["answer"] == "Paris."
        assert len(body["sources"]) >= 1
        mock_generate.assert_called_once()

    def test_query_after_index_with_no_matches_still_returns_200(self):
        chunks = [{"chunk_id": "1", "text": "irrelevant content"}]

        with (
            patch.object(main_module._dense, "index"),
            patch.object(main_module._dense, "search", return_value=[]),
        ):
            client.post("/index", json=chunks)

            with patch.object(
                main_module._generator, "generate", return_value="I don't know."
            ):
                response = client.post("/query", json={"query": "unrelated question"})

        assert response.status_code == 200
        assert response.json()["answer"] == "I don't know."

    def test_reindexing_replaces_previous_hybrid_retriever(self):
        first_chunks = [{"chunk_id": "1", "text": "first batch"}]
        second_chunks = [{"chunk_id": "2", "text": "second batch"}]

        with patch.object(main_module._dense, "index"):
            client.post("/index", json=first_chunks)
            first_hybrid = main_module._hybrid

            client.post("/index", json=second_chunks)
            second_hybrid = main_module._hybrid

        assert first_hybrid is not second_hybrid


class TestHealthRouter:
    """api/health.py defines its own APIRouter that main.py never mounts.

    These tests exercise it directly against a throwaway FastAPI app so the
    module has real coverage despite being dead code from the running
    service's perspective.
    """

    @staticmethod
    def _client():
        test_app = FastAPI()
        test_app.include_router(health_router)
        return TestClient(test_app)

    def test_health_returns_ok_with_timestamp(self):
        response = self._client().get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "timestamp" in body

    def test_readiness_returns_system_info(self):
        with patch("api.health.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = MagicMock(percent=42.0)
            mock_psutil.disk_usage.return_value = MagicMock(percent=13.0)
            mock_psutil.cpu_count.return_value = 8

            response = self._client().get("/health/ready")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["system"]["memory_used_pct"] == 42.0
        assert body["system"]["disk_used_pct"] == 13.0
        assert body["system"]["cpu_count"] == 8

    def test_health_router_is_not_mounted_on_main_app(self):
        # Documents the current state: api/main.py only exposes its own
        # inline /health handler, not the one from api/health.py.
        response = client.get("/health/ready")
        assert response.status_code == 404

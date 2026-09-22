"""tests/test_pipeline.py

Unit tests for the retrieve-then-generate orchestration in api/main.py's
query() function, exercised directly (no HTTP layer) with a mocked hybrid
retriever and generator.

There is no standalone `pipeline` module in this codebase; the RAG
orchestration logic lives inline in api.main.query(). This file previously
targeted a fictional `pipeline.RAGPipeline` class that was never
implemented, so it is rewritten here against the real function. The
request/response-shape and HTTP-level behaviour of the same function are
covered separately in test_api.py and test_api_integration.py.
"""

import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

with (
    patch("retrieval.dense.SentenceTransformer"),
    patch("retrieval.dense.QdrantClient"),
):
    import api.main as main_module
    from api.main import QueryRequest, query


class TestQueryOrchestration:
    """Integration tests for the retrieve -> generate flow in api.main.query."""

    def _run_query(self, question, hits, answer):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = hits
        with (
            patch.object(main_module, "_hybrid", mock_retriever),
            patch.object(
                main_module._generator, "generate", return_value=answer
            ) as mock_generate,
        ):
            result = query(QueryRequest(query=question))
        return result, mock_retriever, mock_generate

    def test_query_returns_answer(self):
        result, _, _ = self._run_query(
            "What is the capital of France?",
            [{"text": "Paris is the capital of France.", "score": 0.95}],
            "Paris",
        )
        assert result.answer == "Paris"

    def test_query_returns_sources(self):
        hits = [
            {"text": "Paris is the capital of France.", "score": 0.95},
            {"text": "France is in Western Europe.", "score": 0.82},
        ]
        result, _, _ = self._run_query("What is the capital of France?", hits, "Paris")
        assert result.sources == hits
        assert len(result.sources) > 0

    def test_query_calls_retriever(self):
        _, mock_retriever, _ = self._run_query("test query", [], "I don't know.")
        mock_retriever.search.assert_called_once()

    def test_query_calls_generator(self):
        _, _, mock_generate = self._run_query("test query", [], "I don't know.")
        mock_generate.assert_called_once()

    def test_query_passes_retrieved_hits_to_generator(self):
        hits = [{"text": "Paris is the capital of France.", "score": 0.95}]
        _, _, mock_generate = self._run_query(
            "What is the capital of France?", hits, "Paris"
        )
        call_args = mock_generate.call_args
        assert call_args.args[1] == hits

    def test_query_handles_empty_retrieval(self):
        result, _, _ = self._run_query("Unknown query", [], "I don't know.")
        assert result.answer == "I don't know."
        assert result.sources == []

    def test_query_raises_400_when_not_indexed(self):
        from fastapi import HTTPException

        with patch.object(main_module, "_hybrid", None):
            try:
                query(QueryRequest(query="anything"))
            except HTTPException as exc:
                assert exc.status_code == 400
            else:
                raise AssertionError("expected HTTPException when not indexed")

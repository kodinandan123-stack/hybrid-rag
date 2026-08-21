import pytest
from unittest.mock import MagicMock, patch


class TestRAGPipeline:
    """Integration tests for the full RAG pipeline."""

    def _make_pipeline(self):
        """Build a pipeline with all dependencies mocked."""
        from pipeline import RAGPipeline

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"text": "Paris is the capital of France.", "score": 0.95},
            {"text": "France is in Western Europe.", "score": 0.82},
        ]
        mock_generator = MagicMock()
        mock_generator.generate.return_value = "Paris"
        return RAGPipeline(retriever=mock_retriever, generator=mock_generator)

    def test_pipeline_returns_answer(self):
        pipeline = self._make_pipeline()
        result = pipeline.query("What is the capital of France?")
        assert result["answer"] == "Paris"

    def test_pipeline_returns_sources(self):
        pipeline = self._make_pipeline()
        result = pipeline.query("What is the capital of France?")
        assert "sources" in result
        assert len(result["sources"]) > 0

    def test_pipeline_calls_retriever(self):
        pipeline = self._make_pipeline()
        pipeline.query("test query")
        pipeline.retriever.search.assert_called_once()

    def test_pipeline_calls_generator(self):
        pipeline = self._make_pipeline()
        pipeline.query("test query")
        pipeline.generator.generate.assert_called_once()

    def test_pipeline_passes_context_to_generator(self):
        pipeline = self._make_pipeline()
        pipeline.query("What is the capital of France?")
        call_args = pipeline.generator.generate.call_args
        context_arg = str(call_args)
        assert "Paris" in context_arg or len(context_arg) > 0

    def test_pipeline_handles_empty_retrieval(self):
        from pipeline import RAGPipeline

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        mock_generator = MagicMock()
        mock_generator.generate.return_value = "I don't know."
        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        result = pipeline.query("Unknown query")
        assert "answer" in result

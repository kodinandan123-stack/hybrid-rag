"""Unit tests for generation/generator.py."""

from unittest.mock import MagicMock, patch

import pytest

from generation.generator import Generator


@pytest.fixture
def sample_chunks():
    return [
        {
            "text": "Reciprocal Rank Fusion combines rankings from multiple retrievers.",
            "source": "docs/architecture.md",
        },
        {
            "text": "BM25 is a probabilistic sparse retrieval algorithm.",
            "source": "docs/retrieval.md",
        },
    ]


class TestGenerator:
    def test_generate_returns_string(self, sample_chunks):
        with patch("generation.generator.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value.content = [
                MagicMock(text="RRF combines rankings.")
            ]
            gen = Generator(api_key="test-key")
            answer = gen.generate(query="What is RRF?", chunks=sample_chunks)
            assert isinstance(answer, str)
            assert len(answer) > 0

    def test_generate_calls_api_once(self, sample_chunks):
        with patch("generation.generator.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value.content = [
                MagicMock(text="Answer.")
            ]
            gen = Generator(api_key="test-key")
            gen.generate(query="Test?", chunks=sample_chunks)
            mock_client.messages.create.assert_called_once()

    def test_generate_empty_chunks(self):
        with patch("generation.generator.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value.content = [
                MagicMock(text="I don't know.")
            ]
            gen = Generator(api_key="test-key")
            answer = gen.generate(query="Unknown?", chunks=[])
            assert isinstance(answer, str)

    def test_generate_uses_configured_model(self, sample_chunks):
        with patch("generation.generator.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value.content = [
                MagicMock(text="Answer.")
            ]
            gen = Generator(model="claude-3-opus-20240229", api_key="test-key")
            gen.generate(query="Test", chunks=sample_chunks)
            call_kwargs = mock_client.messages.create.call_args
            model = call_kwargs.kwargs.get("model") or call_kwargs[1].get("model")
            assert model == "claude-3-opus-20240229"

"""Unit tests for generation/generator.py."""

from unittest.mock import patch, MagicMock

from generation.generator import Generator


def _chunk(text, source="doc.md"):
      return {"text": text, "source": source}


@patch("generation.generator.anthropic.Anthropic")
def test_generate_returns_model_text(mock_anthropic_cls):
      mock_client = MagicMock()
      mock_block = MagicMock()
      mock_block.text = "The answer is 42."
      mock_response = MagicMock()
      mock_response.content = [mock_block]
      mock_client.messages.create.return_value = mock_response
      mock_anthropic_cls.return_value = mock_client

    generator = Generator(api_key="test-key")
    answer = generator.generate("What is the answer?", [_chunk("42 is the answer")])

    assert answer == "The answer is 42."
    mock_client.messages.create.assert_called_once()


@patch("generation.generator.anthropic.Anthropic")
def test_format_context_includes_source_and_index(mock_anthropic_cls):
      mock_anthropic_cls.return_value = MagicMock()
      generator = Generator(api_key="test-key")

    context = generator._format_context([_chunk("alpha", source="a.md"), _chunk("bravo", source="b.md")])

    assert "[1] (source: a.md)" in context
    assert "[2] (source: b.md)" in context
    assert "alpha" in context and "bravo" in context


@patch("generation.generator.anthropic.Anthropic")
def test_generate_passes_query_in_prompt(mock_anthropic_cls):
      mock_client = MagicMock()
      mock_block = MagicMock()
      mock_block.text = "ok"
      mock_response = MagicMock()
      mock_response.content = [mock_block]
      mock_client.messages.create.return_value = mock_response
      mock_anthropic_cls.return_value = mock_client

    generator = Generator(api_key="test-key")
    generator.generate("What is RAG?", [_chunk("context text")])

    _, kwargs = mock_client.messages.create.call_args
    user_message = kwargs["messages"][0]["content"]
    assert "What is RAG?" in user_message

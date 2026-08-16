"""Unit tests for generation/generator.py."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from generation.generator import Generator, SYSTEM_PROMPT


def _chunk(text: str, source: str = "doc.md", chunk_id: str = "c1") -> Dict[str, Any]:
    return {"text": text, "source": source, "chunk_id": chunk_id}


@pytest.fixture()
def mock_anthropic(monkeypatch):
    """Patch anthropic.Anthropic so no real API calls are made."""
    fake_client = MagicMock()
    fake_response = SimpleNamespace(
        content=[SimpleNamespace(text="Mocked answer from the context.")]
    )
    fake_client.messages.create.return_value = fake_response
    with patch("generation.generator.anthropic.Anthropic", return_value=fake_client):
        yield fake_client


class TestGeneratorFormatContext:
    def test_single_chunk(self):
        gen = Generator.__new__(Generator)  # bypass __init__
        ctx = gen._format_context([_chunk("Hello world", source="a.md")])
        assert "[1] (source: a.md)" in ctx
        assert "Hello world" in ctx

    def test_multiple_chunks_numbered_in_order(self):
        gen = Generator.__new__(Generator)
        chunks = [
            _chunk("First", source="a.md", chunk_id="c1"),
            _chunk("Second", source="b.md", chunk_id="c2"),
        ]
        ctx = gen._format_context(chunks)
        assert "[1] (source: a.md)" in ctx
        assert "[2] (source: b.md)" in ctx
        assert ctx.index("[1]") < ctx.index("[2]")

    def test_missing_source_defaults_to_unknown(self):
        gen = Generator.__new__(Generator)
        ctx = gen._format_context([{"text": "No source here", "chunk_id": "x"}])
        assert "(source: unknown)" in ctx

    def test_empty_chunks_returns_empty_string(self):
        gen = Generator.__new__(Generator)
        assert gen._format_context([]) == ""


class TestGeneratorGenerate:
    def test_generate_calls_anthropic_with_correct_model(self, mock_anthropic, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from config.settings import get_settings
        get_settings.cache_clear()

        gen = Generator(model="claude-test-model", api_key="test-key")
        answer = gen.generate("What is RRF?", [_chunk("RRF is a fusion method.")])

        assert answer == "Mocked answer from the context."
        call_kwargs = mock_anthropic.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-test-model"
        assert call_kwargs["system"] == SYSTEM_PROMPT

    def test_generate_includes_query_in_user_message(self, mock_anthropic, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from config.settings import get_settings
        get_settings.cache_clear()

        gen = Generator(model="claude-test-model", api_key="test-key")
        gen.generate("Explain dense retrieval", [_chunk("Dense retrieval uses embeddings.")])

        call_kwargs = mock_anthropic.messages.create.call_args[1]
        user_content = call_kwargs["messages"][0]["content"]
        assert "Explain dense retrieval" in user_content

    def test_generate_includes_chunk_text_in_context(self, mock_anthropic, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from config.settings import get_settings
        get_settings.cache_clear()

        gen = Generator(model="claude-test-model", api_key="test-key")
        gen.generate("Q?", [_chunk("Unique context text XYZ", source="ref.md")])

        call_kwargs = mock_anthropic.messages.create.call_args[1]
        user_content = call_kwargs["messages"][0]["content"]
        assert "Unique context text XYZ" in user_content
        assert "ref.md" in user_content

    def test_generate_respects_max_tokens(self, mock_anthropic, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from config.settings import get_settings
        get_settings.cache_clear()

        gen = Generator(model="claude-test-model", api_key="test-key")
        gen.generate("Q?", [_chunk("ctx")], max_tokens=256)

        call_kwargs = mock_anthropic.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 256

"""Unit tests for config/settings.py."""

from __future__ import annotations

import pytest

from config.settings import Settings, get_settings

_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "QDRANT_URL",
    "QDRANT_COLLECTION",
    "EMBEDDING_MODEL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
)


@pytest.fixture(autouse=True)
def _isolated_settings_environment(monkeypatch):
    """Clear the lru_cache and any settings env vars between tests."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_default_values():
    settings = Settings(_env_file=None)
    assert settings.anthropic_api_key == ""
    assert settings.anthropic_model == "claude-sonnet-4-5"
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "hybrid_rag_chunks"
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.top_k == 5


def test_settings_reads_from_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-5")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("TOP_K", "10")

    settings = Settings(_env_file=None)

    assert settings.anthropic_api_key == "test-key"
    assert settings.anthropic_model == "claude-opus-4-5"
    assert settings.qdrant_url == "http://qdrant:6333"
    assert settings.top_k == 10


def test_get_settings_returns_cached_instance():
    first = get_settings()
    second = get_settings()
    assert first is second


def test_get_settings_reflects_environment_at_first_call(monkeypatch):
    monkeypatch.setenv("QDRANT_COLLECTION", "custom_collection")
    settings = get_settings()
    assert settings.qdrant_collection == "custom_collection"

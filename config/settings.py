"""
settings.py

Centralized application settings loaded from environment variables, used
across the ingestion, retrieval, and generation stages of the hybrid RAG
pipeline.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven configuration for the hybrid RAG pipeline."""

    anthropic_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "hybrid_rag_chunks"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance built from the environment."""
    return Settings()


if __name__ == "__main__":
    print(get_settings().model_dump())

# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]
### Added
- Unit tests for the FastAPI service covering /health, /index, and /query.
- docker-compose.yml to run the API alongside a local Qdrant instance.
- Pre-commit hooks for ruff lint/format and basic file hygiene checks.

## [0.1.0] - 2026-08-10
### Added
- End-to-end hybrid RAG pipeline: corpus ingestion, chunking, dense (Qdrant)
  and sparse (BM25) retrieval fused with RRF, cross-encoder reranking, and
  grounded generation via the Anthropic API.
- FastAPI service (`/index`, `/query`, `/health`) with a minimal chat frontend.
- Test suite covering chunking, loading, retrieval, reranking, and generation.
- Docker, CI, Makefile, and project documentation (architecture, API, evaluation, contributing).

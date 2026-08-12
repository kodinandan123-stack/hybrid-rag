# Deployment

This document describes how to deploy the hybrid RAG API and its Qdrant dependency.

## Docker Compose (recommended for local and small deployments)

The simplest way to run the full stack (API + Qdrant) is via `docker-compose.yml`:

```bash
docker compose up --build
```

This starts the FastAPI service on port 8000 and a Qdrant instance on port 6333. Environment
variables are read from `.env`; copy `.env.example` and fill in your Anthropic API key and
Qdrant URL before starting the stack.

## Standalone Docker image

To build and run only the API container against an external Qdrant instance:

```bash
docker build -t hybrid-rag .
docker run -p 8000:8000 --env-file .env hybrid-rag
```

## Re-indexing a corpus after deployment

Once the API is running, load and index a corpus with the `scripts/reindex.py` CLI:

```bash
python scripts/reindex.py corpus/ --api-url http://localhost:8000
```

## Configuration

Runtime configuration is centralized in `config/settings.py` and driven by environment
variables (or a `.env` file). Key settings include the Anthropic API key, Qdrant URL and
collection name, embedding model, and chunking parameters.

## Scaling notes

The API is stateless aside from the in-memory sparse (BM25) index, which is rebuilt whenever
`/index` is called. For multi-instance deployments, route indexing requests to a single
instance, or persist the sparse index externally so all replicas stay in sync. Qdrant itself
can be scaled independently and shared across API replicas.

## Health checks

The `/health` endpoint provides a simple liveness probe suitable for container orchestrators
such as Docker Compose, Kubernetes, or a load balancer.

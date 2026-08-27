# Deployment Guide

This guide covers deploying the hybrid-RAG pipeline to production.

## Prerequisites

- Docker 24+ and Docker Compose v2
- Python 3.10+
- Qdrant 1.7+ (vector store)
- Redis 7+ (cache and BM25 index)
- Ollama or a compatible OpenAI-compatible endpoint

## Environment Variables

Copy `.env.example` to `.env`:

```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=hybrid_rag
REDIS_HOST=localhost
REDIS_PORT=6379
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
RAG_TOP_K=5
RAG_ALPHA=0.5
LOG_LEVEL=INFO
PROMETHEUS_PORT=8001
```

## Docker Compose

```bash
docker compose up -d
```

Services started: qdrant (6333), redis (6379), ollama (11434), api (8000), prometheus (9090), grafana (3000).

## Ingestion

```bash
python -m ingestion.pipeline --source ./data/docs --chunk-size 512
```

## Health Check

```bash
curl http://localhost:8000/health
```

## Scaling

- **API**: increase Uvicorn workers or add replicas in `docker-compose.yml`.
- **Qdrant**: enable distributed mode with shard replicas.
- **Redis**: use Sentinel or Cluster for fault tolerance.

## Monitoring

Prometheus scrapes `/metrics` on port 8001. Import `monitoring/grafana_dashboard.json` for the pre-built dashboard. Alert rules are in `monitoring/alerting_rules.yml`.

## Rollback

```bash
docker compose down
git checkout <previous-tag>
docker compose up -d
```

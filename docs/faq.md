# Frequently Asked Questions

## Setup

**Q: What Python version is required?**  
Python 3.10 or later is required. The project uses PEP 604 union types and `match` statements.

**Q: How do I set my Anthropic API key?**  
Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY=your-key-here`. The key is loaded by `config/settings.py` via `python-dotenv`.

**Q: Do I need a running Qdrant instance?**  
Yes. Start one locally with Docker:
```bash
docker run -p 6333:6333 qdrant/qdrant
```
Then set `QDRANT_URL=http://localhost:6333` in your `.env`.

**Q: Is Redis required?**  
Redis is optional. If `REDIS_URL` is not set, the system falls back to the in-process LRU cache defined in `config/cache_config.yaml`.

---

## Ingestion

**Q: What file formats can I ingest?**  
PDF and Markdown (`.pdf`, `.md`). Supported extensions are configured in `config/ingestion_config.yaml`.

**Q: How do I ingest a corpus?**  
```bash
python scripts/run_ingest.py --corpus-dir data/corpus
```
Use `--help` for all options.

**Q: Can I re-ingest updated documents?**  
Set `overwrite_existing: true` in `config/ingestion_config.yaml` before running ingestion again.

---

## Retrieval

**Q: How does hybrid retrieval work?**  
Dense (sentence-transformer embeddings in Qdrant) and sparse (BM25) results are fused with Reciprocal Rank Fusion (RRF), then reranked by a cross-encoder before generation. See `docs/retrieval_architecture.md` for details.

**Q: How do I change the number of retrieved chunks?**  
Pass `--top-k N` to `scripts/run_query.py`, or set `top_k` in the request body of `POST /query`.

**Q: The system is slow on the first query — why?**  
The embedding model and cross-encoder are loaded on first use. Subsequent queries reuse the cached models.

---

## Evaluation

**Q: How do I run the ragas evaluation?**  
```bash
python scripts/run_evaluation.py --testset eval/testset.jsonl --output eval/results/report.json
```
See `docs/evaluation_guide.md` for metric definitions.

**Q: What does a faithfulness score of 0.91 mean?**  
91% of the claims in generated answers are directly supported by the retrieved context, as judged by the ragas faithfulness metric.

---

## Deployment

**Q: How do I run the API in production?**  
See `docs/deployment_guide.md`. The recommended setup is Gunicorn + Uvicorn workers behind an Nginx reverse proxy, with Docker Compose for Qdrant and Redis.

**Q: How do I monitor the pipeline?**  
Prometheus metrics are exposed at `/metrics`. Import `monitoring/grafana_dashboard.json` into Grafana for pre-built panels. See `docs/monitoring_guide.md`.

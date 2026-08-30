# Troubleshooting Guide

Common issues and resolutions for the hybrid-rag pipeline.

## Qdrant Connection Errors

**Symptom:** `ConnectionRefusedError` or `qdrant_client.http.exceptions.UnexpectedResponse` on startup.

**Resolution:**
- Confirm Qdrant is running: `docker ps | grep qdrant`
- Check `QDRANT_URL` in `.env` matches the running instance (default: `http://localhost:6333`)
- If using Docker Compose, ensure the `qdrant` service started before the API: `docker compose up qdrant`

---

## Anthropic API Key Invalid

**Symptom:** `anthropic.AuthenticationError: 401` during generation.

**Resolution:**
- Verify `ANTHROPIC_API_KEY` is set in `.env` and has not expired
- Test the key independently: `python -c "import anthropic; print(anthropic.Anthropic().models.list())"`

---

## Empty Retrieval Results

**Symptom:** Queries return zero candidates despite documents being ingested.

**Resolution:**
- Confirm ingestion completed without errors: check logs for `Indexed N chunks`
- Verify the Qdrant collection exists: `GET http://localhost:6333/collections`
- Re-run ingestion: `python scripts/run_ingest.py --corpus-dir data/corpus`
- Increase `top_k` in `config/retrieval_config.yaml` to rule out threshold filtering

---

## BM25 Index Not Found

**Symptom:** `FileNotFoundError: bm25_index.pkl` on startup.

**Resolution:**
- The sparse index is built during ingestion. Run `python scripts/run_ingest.py` first
- Check that `BM25_INDEX_PATH` in `.env` points to the correct location

---

## Out-of-Memory During Embedding

**Symptom:** `torch.cuda.OutOfMemoryError` or process killed during ingestion.

**Resolution:**
- Reduce `EMBED_BATCH_SIZE` in `config/model_config.yaml` (try 16 or 8)
- Use CPU-only mode: set `DEVICE=cpu` in `.env`
- Split the corpus into smaller batches and ingest incrementally

---

## Slow Query Latency

**Symptom:** Queries take >5 seconds end-to-end.

**Resolution:**
- Profile with `python scripts/profile_retrieval.py` to identify the bottleneck
- Enable retrieval caching in `config/retrieval_config.yaml`: `cache_enabled: true`
- Reduce `rerank_top_k` to limit cross-encoder calls
- Run benchmark: `python scripts/run_benchmark.py`

---

## Docker Build Fails

**Symptom:** `pip install` error during `docker build`.

**Resolution:**
- Ensure `requirements.txt` is up to date: `pip freeze > requirements.txt`
- Clear Docker build cache: `docker build --no-cache -t hybrid-rag .`

---

## Logs Not Appearing

**Symptom:** No structured log output despite `LOG_LEVEL=DEBUG`.

**Resolution:**
- Confirm `config/logging_config.yaml` is loaded at startup (see `api/main.py`)
- Check that the log handler target directory exists and is writable

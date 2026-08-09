# API Reference

The FastAPI service in `api/main.py` exposes three endpoints for indexing a corpus and querying it through the hybrid RAG pipeline.

## POST /index

Indexes a batch of chunk dictionaries into both the dense (Qdrant) and sparse (BM25) retrievers, replacing any previously indexed corpus.

**Request body**: a JSON array of chunk objects, each containing at least a `text` field and an optional `source` field for citation metadata.

**Response**:

```json
{"indexed": 42}
```

## POST /query

Answers a natural language question by retrieving relevant chunks with hybrid search and generating a grounded answer via the Anthropic API.

**Request body**:

```json
{"query": "What does the ingestion pipeline do?", "top_k": 5}
```

`top_k` is optional and defaults to 5.

**Response**:

```json
{
  "answer": "...",
  "sources": [{"chunk_id": "...", "text": "...", "source": "..."}]
}
```

Calling `/query` before `/index` has been called at least once returns a runtime error, since there is no corpus to search yet.

## GET /health

A simple liveness probe used by orchestration and monitoring tools.

**Response**:

```json
{"status": "ok"}
```

# Query Guide

This guide covers how to query the hybrid RAG pipeline using the CLI script and the REST API.

## CLI Usage

The `scripts/run_query.py` script provides a command-line interface for querying the pipeline.

### Basic Query

```bash
python scripts/run_query.py "What is Reciprocal Rank Fusion?"
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--top-k` | 5 | Number of chunks to retrieve |
| `--output` | text | Output format: `text` or `json` |
| `--no-rerank` | false | Skip cross-encoder reranking |

### JSON Output

```bash
python scripts/run_query.py "How does BM25 work?" --output json --top-k 3
```

## REST API Usage

Start the server with `uvicorn api.main:app --reload`, then POST to `/query`:

```json
{"query": "What is the default chunk size?", "top_k": 5}
```

## Tips

- Increase `--top-k` for broader context at the cost of longer generation latency.
- Use `--output json` when integrating results into downstream scripts.
- Disable reranking with `--no-rerank` only for latency-sensitive use cases.
- Ensure `.env` has a valid `ANTHROPIC_API_KEY` and `QDRANT_URL` before querying.

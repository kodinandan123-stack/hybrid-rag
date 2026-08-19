# Architecture

This document describes the components of the hybrid RAG pipeline and how
data flows between them.

## Pipeline stages

1. **Ingestion** (`ingestion/`) — `loader.py` reads PDF and Markdown files
   from a corpus directory into `LoadedDocument` objects, tracking each
   file's path as its `source`. `chunker.py` splits each document into
   overlapping ~500-character chunks (50-character overlap) using a
   recursive character splitter, producing `{text, source, chunk_id}` dicts.

2. **Retrieval** (`retrieval/`) — chunks are indexed twice: `dense.py`
   embeds them with a sentence-transformers model and stores vectors in
   Qdrant, while `sparse.py` builds a BM25 lexical index over the same
   chunks. `hybrid.py` queries both retrievers and fuses their rankings
   with Reciprocal Rank Fusion (RRF) to produce a single ranked list.

3. **Generation** (`generation/`) — `generator.py` formats the top hybrid
   hits into a context block and calls the Anthropic API with a system
   prompt that restricts the model to grounded, citation-backed answers.

4. **API** (`api/main.py`) — a FastAPI service exposes `/index` to load a
   corpus's chunks into the retrievers and `/query` to run the full
   retrieve-then-generate flow, returning both the answer and its source
   chunks.

5. **Frontend** (`frontend/`) — a minimal static chat page posts questions
   to `/query` and renders the answer alongside its cited sources.

6. **Evaluation** (`eval/`) — `evaluate.py` replays a labeled test set
   through the retriever and generator, then scores the results with
   ragas metrics (faithfulness, answer relevancy, context precision and
   recall) to track quality over time.

## Data flow

```
corpus files -> loader -> chunker -> [dense index, sparse index]
                                            |
question -> hybrid retriever (RRF) -> generator -> answer + sources
```


## Configuration

Key parameters are centralised in `config/settings.py` and can be overridden via environment variables or the `.env` file:

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 500 | Token target per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between adjacent chunks |
| `TOP_K_DENSE` | 10 | Dense retriever candidate count |
| `TOP_K_SPARSE` | 10 | BM25 retriever candidate count |
| `RRF_K` | 60 | RRF constant (higher = less rank-sensitive) |
| `RERANK_TOP_N` | 5 | Chunks passed to the generator after reranking |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model for dense embeddings |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for reranking |

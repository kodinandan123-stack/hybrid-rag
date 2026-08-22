# Retrieval Architecture

This document describes the hybrid retrieval architecture used in this project.

## Overview

The system combines **dense** (vector) and **sparse** (BM25) retrieval using
Reciprocal Rank Fusion (RRF) to produce a single ranked list of candidate
documents before passing them to the language model.

```
Query
  ├── DenseRetriever   (FAISS / pgvector)  → top-k dense hits
  └── SparseRetriever  (BM25 / Elasticsearch) → top-k sparse hits
            │
            ▼
       RRF Fusion
            │
            ▼
    Re-ranked candidates
            │
            ▼
     LLM Generation
```

## Components

### DenseRetriever

- Embeds the query using a bi-encoder (e.g. `text-embedding-3-small`).
- Performs approximate nearest-neighbour search via FAISS or pgvector.
- Returns the top-`k` documents by cosine similarity.

### SparseRetriever

- Tokenises the query and indexes documents with BM25.
- Returns the top-`k` documents by BM25 score.
- Handles out-of-vocabulary and exact-match queries better than dense retrieval.

### HybridRetriever (RRF Fusion)

Given ranked lists from both retrievers, RRF assigns a score to each document:

```
score(d) = Σ_r  1 / (k + rank_r(d))
```

where `k = 60` (default) dampens the influence of very high ranks.
Documents are then sorted by descending RRF score.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dense_top_k` | 20 | Candidates fetched from the dense index |
| `sparse_top_k` | 20 | Candidates fetched from the sparse index |
| `rrf_k` | 60 | RRF damping constant |
| `final_top_k` | 10 | Documents passed to the LLM |

All values are configurable via `config/settings.py`.

## Benchmark Summary

See `eval/benchmark_results.json` for the latest numbers. Key takeaways:

- **Hybrid** achieves the highest recall@10 (0.91) at a modest latency cost vs dense-only.
- **Sparse** is fastest but lowest recall — useful as a fallback or filter.
- **Dense** offers the best accuracy-latency trade-off for single-mode retrieval.

## References

- Cormack et al., *Reciprocal Rank Fusion outperforms Condorcet and individual
  Rank Learning Methods* (SIGIR 2009).
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks* (NeurIPS 2020).

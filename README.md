# Hybrid RAG

A hybrid retrieval-augmented generation (RAG) system combining dense vector search with sparse BM25 retrieval, built for accurate, source-grounded question answering over a document corpus.

## Overview

TODO: Summarize the goal of this project, the problem it solves, and the high-level approach (hybrid dense + sparse retrieval, reranking, and grounded generation).

## Architecture

TODO: Describe the pipeline stages:
- Ingestion: loading PDFs/markdown from a corpus directory with source tracking
- Chunking: recursive character splitting with metadata (section, page)
- Retrieval: dense (sentence-transformers + Qdrant) and sparse (BM25) hybrid search
- Generation: answer synthesis grounded in retrieved context (Anthropic API)
- Evaluation: automated quality metrics (ragas)
- API/Frontend: serving and interacting with the system

## Setup

TODO: Document environment setup, dependencies, and configuration.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Results

TODO: Summarize evaluation results and benchmarks once available.

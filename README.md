# Hybrid RAG

A hybrid retrieval-augmented generation (RAG) system combining dense vector search with sparse BM25 retrieval, built for accurate, source-grounded question answering over a document corpus.

## Overview

This project implements an end-to-end RAG pipeline that answers questions grounded in a document corpus. It ingests PDFs and markdown files, splits them into overlapping chunks, and indexes those chunks with both a dense vector store (Qdrant) and a sparse BM25 index. At query time, results from both retrievers are fused with Reciprocal Rank Fusion and reranked before being passed to the generation step, which produces an answer grounded in the retrieved context via the Anthropic API.

## Architecture

The pipeline consists of the following stages:

- **Ingestion** (`ingestion/loader.py`): loads PDFs and markdown files from a corpus directory while tracking source metadata.
- - **Chunking** (`ingestion/chunker.py`): splits documents into overlapping chunks (size=500, overlap=50) using a recursive character splitter.
  - - **Retrieval** (`retrieval/`): combines dense search (sentence-transformers embeddings indexed in Qdrant) with sparse BM25 lexical search, fusing both via Reciprocal Rank Fusion in `retrieval/hybrid.py`.
    - - **Reranking** (`retrieval/rerank.py`): re-scores the fused candidates with a cross-encoder for improved precision before the top results reach generation.
      - - **Generation** (`generation/generator.py`): synthesizes a grounded answer from the retrieved chunks using the Anthropic API.
        - - **Evaluation** (`eval/evaluate.py`): scores retrieval and generation quality using ragas metrics.
          - - **API/Frontend** (`api/main.py`, `frontend/index.html`): a FastAPI service exposing `/index` and `/query` endpoints, with a minimal chat UI for interacting with the system.
           
            - See `docs/architecture.md` for more detail on the pipeline design.
           
            - ## Setup
           
            - ```bash
              python -m venv .venv
              source .venv/bin/activate
              pip install -r requirements.txt
              cp .env.example .env
              ```

              Fill in your Anthropic API key and Qdrant URL in `.env`, then run the API locally:

              ```bash
              uvicorn api.main:app --reload
              ```

              Or with Docker:

              ```bash
              docker build -t hybrid-rag .
              docker run -p 8000:8000 --env-file .env hybrid-rag
              ```

              ## Results

              TODO: Summarize evaluation results and benchmarks once eval/evaluate.py has been run against a representative corpus.
              

# Tracing and Observability

This document describes how to enable structured tracing for the hybrid RAG
pipeline so you can inspect latency, token counts, and retrieval quality
across every request.

## Overview

The pipeline exposes lightweight span-level timing via Python's standard
logging module (configured in config/logging.py). Each major stage --
ingestion, chunking, retrieval, reranking, and generation -- logs a structured
JSON line containing:

- stage: Pipeline stage name (retrieval, generation, ...)
- query: Incoming query text (truncated to 120 chars)
- latency_ms: Wall-clock duration for the stage in milliseconds
- top_k: Number of chunks retrieved or reranked
- model: Anthropic model used (generation stage only)
- input_tokens: Prompt token count (generation stage only)
- output_tokens: Completion token count (generation stage only)

## Enabling JSON Log Output

Set LOG_FORMAT=json in your .env file or environment:

    LOG_LEVEL=INFO
    LOG_FORMAT=json

## Instrumenting a Stage

Import the context manager from config.logging:

    from config.logging import trace_stage

    with trace_stage("retrieval", query=query, top_k=top_k) as span:
        hits = retriever.search(query, top_k=top_k)
        span["chunks"] = len(hits)

The context manager records latency_ms automatically on exit and emits a
single JSON log line at INFO level.

## Local Inspection

During development, pretty-print individual trace lines with jq:

    uvicorn api.main:app 2>&1 | grep stage | jq .

## Exporting to an Observability Backend

For production deployments, pipe the JSON log stream into your preferred
backend such as Datadog, Grafana Loki, or OpenTelemetry OTLP.

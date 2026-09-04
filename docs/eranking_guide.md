# Reranking Guide

This guide explains how the cross-encoder reranker works in the hybrid-RAG pipeline, how to configure it, and how to tune it for your use case.

## Overview

After the hybrid retriever (dense + BM25 + RRF) returns a candidate set, a cross-encoder reranker re-scores every candidate against the query. Cross-encoders attend jointly to the query and document text, producing more accurate relevance scores than the bi-encoder used for dense retrieval, at the cost of higher latency.

The reranker is implemented in `retrieval/rerank.py` and is wired into the full pipeline in `retrieval/hybrid.py`.

## Configuration

Reranker settings live in `config/model_config.yaml` under the `reranker` key.

Reranker result caching is configured in `config/rerank_cache_config.yaml` with `max_size`, `ttl_seconds`, and `enabled` fields.

## Choosing a Model

Smaller cross-encoders such as `cross-encoder/ms-marco-MiniLM-L-6-v2` are fast and suitable for production. Larger models such as `cross-encoder/ms-marco-electra-base` produce higher quality scores at the cost of latency. Start with the smallest model and move up only if evaluation metrics justify the trade-off.

## Tuning top_k

`top_k` controls how many reranked documents reach the generation step. Increasing it gives the generator more context but also increases prompt length and cost. Evaluate on your test set using the ragas metrics in `eval/evaluate.py` to find the sweet spot.

## Disabling the Reranker

Set `reranker.enabled: false` in `config/model_config.yaml` to skip reranking and pass raw RRF results directly to generation. This is useful for latency-sensitive deployments or ablation studies.

## Cache Behaviour

When `rerank_cache.enabled: true`, identical (query, candidate-set) pairs served within `ttl_seconds` return cached scores without a forward pass. The cache is an in-process LRU store; it does not persist across restarts. Monitor cache hit rate via `RerankCache.stats()`.

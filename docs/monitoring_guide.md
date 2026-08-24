# Monitoring Guide

This document describes how to monitor the hybrid-RAG pipeline using Prometheus and Grafana.

## Overview

The monitoring stack exposes Prometheus metrics via `monitoring/metrics.py` and defines
alert rules in `monitoring/alerting_rules.yml`. A pre-built Grafana dashboard is available
at `monitoring/grafana_dashboard.json`.

## Starting the Metrics Server

```python
from monitoring.metrics import start_metrics_server
start_metrics_server(port=8001)
```

Metrics are then available at `http://localhost:8001/metrics`.

## Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `rag_queries_total` | Counter | Total queries by status (success/error) |
| `rag_query_latency_seconds` | Histogram | End-to-end query latency |
| `rag_retriever_latency_seconds` | Histogram | Per-retriever latency (dense/sparse/hybrid) |
| `rag_docs_retrieved_count` | Histogram | Number of documents retrieved per query |
| `rag_generation_latency_seconds` | Histogram | LLM generation latency |
| `rag_tokens_generated_total` | Counter | Total tokens generated |
| `rag_cache_hits_total` | Counter | Cache hits |
| `rag_cache_misses_total` | Counter | Cache misses |
| `rag_index_document_count` | Gauge | Current number of indexed documents |

## Instrumentation Example

```python
from monitoring.metrics import QUERY_LATENCY, QUERY_COUNTER, MetricsTimer

def handle_query(query: str) -> str:
    with MetricsTimer(QUERY_LATENCY):
        try:
            result = pipeline.run(query)
            QUERY_COUNTER.labels(status="success").inc()
            return result
        except Exception as e:
            QUERY_COUNTER.labels(status="error").inc()
            raise
```

## Alerts

See `monitoring/alerting_rules.yml` for alert definitions. Critical alerts:

- **HighQueryLatency** – p95 latency > 3s for 2 minutes
- **HighErrorRate** – error rate > 5% for 2 minutes
- **LowCacheHitRate** – cache hit rate < 30% for 5 minutes
- **IndexSizeDropped** – fewer than 100 documents indexed

## Grafana Dashboard

Import `monitoring/grafana_dashboard.json` into Grafana (Dashboard → Import → Upload JSON).
The dashboard includes panels for query rate, latency percentiles, cache hit rate, and index size.

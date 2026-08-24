"""monitoring/metrics.py - Prometheus metrics for hybrid RAG pipeline."""
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

QUERY_COUNTER = Counter("rag_queries_total", "Total RAG queries", ["status"])
QUERY_LATENCY = Histogram("rag_query_latency_seconds", "Query latency", buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0])
RETRIEVER_LATENCY = Histogram("rag_retriever_latency_seconds", "Retriever latency", ["retriever_type"], buckets=[0.05, 0.1, 0.25, 0.5, 1.0])
DOCS_RETRIEVED = Histogram("rag_docs_retrieved_count", "Docs retrieved per query", ["retriever_type"], buckets=[1, 5, 10, 20, 50])
GENERATION_LATENCY = Histogram("rag_generation_latency_seconds", "LLM generation latency", buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 15.0])
TOKENS_GENERATED = Counter("rag_tokens_generated_total", "Total tokens generated")
CACHE_HITS = Counter("rag_cache_hits_total", "Cache hits")
CACHE_MISSES = Counter("rag_cache_misses_total", "Cache misses")
INDEX_SIZE = Gauge("rag_index_document_count", "Documents in index")


class MetricsTimer:
    """Context manager for recording histogram latency."""

    def __init__(self, histogram, labels=None):
        self._histogram = histogram
        self._labels = labels or {}
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._start
        if self._labels:
            self._histogram.labels(**self._labels).observe(elapsed)
        else:
            self._histogram.observe(elapsed)


def start_metrics_server(port: int = 8001) -> None:
    """Start the Prometheus metrics HTTP server."""
    start_http_server(port)
    print(f"Metrics server running on port {port}")

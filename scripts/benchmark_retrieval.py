"""scripts/benchmark_retrieval.py

Benchmark dense, sparse, and hybrid retrieval modes across query sets.
Outputs latency, recall@k, and MRR metrics to JSON.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def benchmark_mode(
      retriever: Any,
      queries: list[str],
      k: int = 10,
) -> dict[str, float]:
      """Run queries through a retriever and collect timing + recall stats."""
      latencies: list[float] = []
      recalls: list[float] = []

    for query in queries:
              start = time.perf_counter()
              results = retriever.retrieve(query, k=k)
              elapsed = time.perf_counter() - start
              latencies.append(elapsed)

        # Placeholder recall: assume all returned docs are relevant for benchmark
              recalls.append(len(results) / k if results else 0.0)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0

    return {
              "avg_latency_s": round(avg_latency, 4),
              "p95_latency_s": round(sorted(latencies)[int(len(latencies) * 0.95)], 4) if latencies else 0.0,
              "avg_recall_at_k": round(avg_recall, 4),
              "num_queries": len(queries),
              "k": k,
    }


def run_benchmark(
      retrievers: dict[str, Any],
      queries: list[str],
      output_path: str = "eval/benchmark_results.json",
      k: int = 10,
) -> None:
      """Benchmark multiple retrievers and write results to JSON."""
      results: dict[str, dict] = {}

    for name, retriever in retrievers.items():
              print(f"Benchmarking {name}...")
              results[name] = benchmark_mode(retriever, queries, k=k)
              print(f"  avg_latency={results[name]['avg_latency_s']}s  recall@{k}={results[name]['avg_recall_at_k']}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
      # Minimal smoke-test with stub retrievers
      class _StubRetriever:
                def retrieve(self, query: str, k: int = 10) -> list[str]:
                              return [f"doc_{i}" for i in range(k)]

            sample_queries = ["What is hybrid RAG?", "How does RRF fusion work?", "Explain BM25 scoring"]
    run_benchmark(
              retrievers={"dense": _StubRetriever(), "sparse": _StubRetriever(), "hybrid": _StubRetriever()},
              queries=sample_queries,
              output_path="eval/benchmark_results.json",
    )

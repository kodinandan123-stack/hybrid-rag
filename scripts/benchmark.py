"""
benchmark.py

Quick latency benchmark for the hybrid RAG pipeline. Fires a list of
queries against the running API and reports min/mean/max response times
so query-level latency regressions are visible without a full load test.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import List

import requests


def benchmark(queries: List[str], api_url: str, top_k: int = 5) -> None:
    """Run each query against the /query endpoint and print latency stats."""
    latencies: List[float] = []
    errors = 0

    for i, query in enumerate(queries, 1):
        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{api_url.rstrip('/')}/query",
                json={"query": query, "top_k": top_k},
                timeout=30,
            )
            resp.raise_for_status()
        except Exception as exc:
            print(f"  [{i}/{len(queries)}] ERROR: {exc}")
            errors += 1
            continue
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        print(f"  [{i}/{len(queries)}] {elapsed * 1000:.1f} ms  — {query[:60]}")

    if not latencies:
        print("No successful queries; nothing to report.")
        return

    print(
        f"\nResults over {len(latencies)} successful queries"
        f" ({errors} error(s)):\n"
        f"  min   : {min(latencies) * 1000:.1f} ms\n"
        f"  mean  : {statistics.mean(latencies) * 1000:.1f} ms\n"
        f"  median: {statistics.median(latencies) * 1000:.1f} ms\n"
        f"  max   : {max(latencies) * 1000:.1f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the hybrid RAG /query endpoint.")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Base URL of the running API")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve per query")
    parser.add_argument(
        "queries",
        nargs="*",
        default=["What is RAG?", "How does hybrid retrieval work?", "Explain reciprocal rank fusion."],
        help="One or more query strings to benchmark",
    )
    args = parser.parse_args()
    benchmark(args.queries, args.api_url, top_k=args.top_k)


if __name__ == "__main__":
    main()

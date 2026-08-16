"""
load_test.py

Concurrent load test for the hybrid RAG /query endpoint.  Spawns N worker
threads, each firing M queries in a tight loop, then reports per-thread and
aggregate throughput and latency statistics.  Unlike benchmark.py (which
fires queries sequentially to measure pipeline latency), this script
measures server behaviour under simultaneous user load.

Usage
-----
    python scripts/load_test.py                          # defaults
    python scripts/load_test.py --workers 4 --iters 20 --api-url http://host:8000
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from typing import List

import requests

DEFAULT_QUERIES: List[str] = [
    "What is Reciprocal Rank Fusion?",
    "How does the dense retriever work?",
    "What embedding model is used?",
    "Describe the ingestion pipeline.",
    "How are chunks sized?",
]


def _worker(
    worker_id: int,
    queries: List[str],
    api_url: str,
    iters: int,
    top_k: int,
    results: List[dict],
    barrier: threading.Barrier,
) -> None:
    """Send iters queries (cycling over queries list) and record per-request latency."""
    barrier.wait()  # all workers start simultaneously

    latencies: List[float] = []
    errors = 0

    for i in range(iters):
        query = queries[i % len(queries)]
        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{api_url.rstrip('/')}/query",
                json={"query": query, "top_k": top_k},
                timeout=60,
            )
            resp.raise_for_status()
            latencies.append(time.perf_counter() - start)
        except Exception:
            errors += 1

    results.append({"worker_id": worker_id, "latencies": latencies, "errors": errors})


def run_load_test(
    queries: List[str],
    api_url: str,
    workers: int,
    iters: int,
    top_k: int,
) -> None:
    """Spawn *workers* threads, each firing *iters* requests, then print stats."""
    results: List[dict] = []
    barrier = threading.Barrier(workers)

    threads = [
        threading.Thread(
            target=_worker,
            args=(i, queries, api_url, iters, top_k, results, barrier),
            daemon=True,
        )
        for i in range(workers)
    ]

    print(f"Starting load test: {workers} workers x {iters} requests = {workers * iters} total")
    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_elapsed = time.perf_counter() - wall_start

    all_latencies: List[float] = []
    total_errors = 0
    for r in results:
        all_latencies.extend(r["latencies"])
        total_errors += r["errors"]

    total_requests = workers * iters
    successful = len(all_latencies)

    print(f"\nResults ({successful}/{total_requests} succeeded, {total_errors} errors)")
    if successful:
        print(f"  min    : {min(all_latencies):.3f}s")
        print(f"  mean   : {statistics.mean(all_latencies):.3f}s")
        print(f"  median : {statistics.median(all_latencies):.3f}s")
        print(f"  p95    : {sorted(all_latencies)[int(len(all_latencies) * 0.95)]:.3f}s")
        print(f"  max    : {max(all_latencies):.3f}s")
    print(f"  wall   : {wall_elapsed:.3f}s")
    print(f"  rps    : {successful / wall_elapsed:.2f} req/s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Concurrent load test for the hybrid RAG API.")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Base URL of the running API")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent worker threads")
    parser.add_argument("--iters", type=int, default=10, help="Requests per worker")
    parser.add_argument("--top-k", type=int, default=5, help="top_k passed to /query")
    args = parser.parse_args()

    try:
        run_load_test(
            queries=DEFAULT_QUERIES,
            api_url=args.api_url,
            workers=args.workers,
            iters=args.iters,
            top_k=args.top_k,
        )
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

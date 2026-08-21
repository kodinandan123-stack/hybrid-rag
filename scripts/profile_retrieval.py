"""Profile retrieval latency and throughput for dense, sparse, and hybrid retrievers."""
import time
import statistics
import argparse
from typing import List, Dict


def profile_retriever(retriever, queries: List[str], top_k: int = 10) -> Dict:
    """Run queries through retriever and collect latency stats."""
    latencies = []
    for query in queries:
        start = time.perf_counter()
        retriever.search(query, top_k=top_k)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    return {
        "count": len(latencies),
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "throughput_qps": len(latencies) / (sum(latencies) / 1000),
    }


def print_report(name: str, stats: Dict) -> None:
    """Print a formatted profiling report."""
    print(f"\n=== {name} ===")
    print(f"  Queries run : {stats['count']}")
    print(f"  Mean latency: {stats['mean_ms']:.2f} ms")
    print(f"  Median      : {stats['median_ms']:.2f} ms")
    print(f"  P95         : {stats['p95_ms']:.2f} ms")
    print(f"  P99         : {stats['p99_ms']:.2f} ms")
    print(f"  Min / Max   : {stats['min_ms']:.2f} / {stats['max_ms']:.2f} ms")
    print(f"  Throughput  : {stats['throughput_qps']:.1f} QPS")


def load_queries(path: str) -> List[str]:
    """Load queries from a plain-text file, one per line."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile retrieval latency")
    parser.add_argument("--queries", required=True, help="Path to queries file")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retriever", choices=["dense", "sparse", "hybrid"], default="hybrid")
    args = parser.parse_args()

    queries = load_queries(args.queries)
    print(f"Loaded {len(queries)} queries from {args.queries}")

    if args.retriever == "dense":
        from retrieval.dense import DenseRetriever
        retriever = DenseRetriever()
    elif args.retriever == "sparse":
        from retrieval.sparse import SparseRetriever
        retriever = SparseRetriever()
    else:
        from retrieval.hybrid import HybridRetriever
        retriever = HybridRetriever()

    stats = profile_retriever(retriever, queries, top_k=args.top_k)
    print_report(args.retriever.capitalize() + " Retriever", stats)


if __name__ == "__main__":
    main()

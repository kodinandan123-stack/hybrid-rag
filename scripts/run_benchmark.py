"""Benchmarking script for hybrid-RAG retrieval latency and quality metrics."""

import time
import json
import argparse
import statistics
from typing import List, Dict, Any
from pathlib import Path


SAMPLE_QUERIES = [
      "What is retrieval-augmented generation?",
      "How does hybrid search combine dense and sparse retrieval?",
      "Explain reciprocal rank fusion for result merging.",
      "What are the trade-offs between BM25 and vector search?",
      "How to tune the alpha parameter in hybrid retrieval?",
]


def run_latency_benchmark(
      retriever,
      queries: List[str],
      top_k: int = 5,
      warmup_runs: int = 3,
) -> Dict[str, Any]:
      """Measure retrieval latency across queries."""
      for q in queries[:warmup_runs]:
                retriever.retrieve(q, top_k=top_k)

      latencies = []
      for query in queries:
                start = time.perf_counter()
                retriever.retrieve(query, top_k=top_k)
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

      return {
          "n_queries": len(queries),
          "top_k": top_k,
          "mean_ms": round(statistics.mean(latencies), 2),
          "median_ms": round(statistics.median(latencies), 2),
          "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
          "min_ms": round(min(latencies), 2),
          "max_ms": round(max(latencies), 2),
      }


def run_quality_benchmark(
      retriever,
      eval_set: List[Dict[str, Any]],
      top_k: int = 5,
) -> Dict[str, Any]:
      """Measure Hit@K and MRR on a labelled evaluation set."""
      hits = 0
      reciprocal_ranks = []

    for item in eval_set:
              query = item["query"]
              relevant_ids = set(item["relevant_doc_ids"])
              results = retriever.retrieve(query, top_k=top_k)
              retrieved_ids = [r["id"] for r in results]

        if any(rid in relevant_ids for rid in retrieved_ids):
                      hits += 1

        rr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
                      if rid in relevant_ids:
                                        rr = 1.0 / rank
                                        break
                                reciprocal_ranks.append(rr)

    return {
              "n_queries": len(eval_set),
              "top_k": top_k,
              "hit_at_k": round(hits / len(eval_set), 4),
              "mrr": round(statistics.mean(reciprocal_ranks), 4),
    }


def save_results(results: Dict[str, Any], output_path: Path) -> None:
      output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
              json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def main() -> None:
      parser = argparse.ArgumentParser(description="Benchmark hybrid-RAG retrieval")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("eval/benchmark_results.json"))
    parser.add_argument("--mode", choices=["latency", "quality", "both"], default="latency")
    args = parser.parse_args()
    print(f"Running {args.mode} benchmark with top_k={args.top_k}")
    results: Dict[str, Any] = {"mode": args.mode, "top_k": args.top_k}
    save_results(results, args.output)


if __name__ == "__main__":
      main()

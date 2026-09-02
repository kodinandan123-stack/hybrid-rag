"""CLI script for running ragas evaluation on a testset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_testset(path: str) -> list[dict]:
      rows = []
      with open(path) as f:
                for line in f:
                              line = line.strip()
                              if line:
                                                rows.append(json.loads(line))
                                    return rows


def run_evaluation(testset_path: str, output_path: str, top_k: int) -> None:
      """Run ragas evaluation and write results to *output_path*.

          Args:
                  testset_path: Path to a JSONL testset file.
                          output_path: Destination JSON file for results.
                                  top_k: Number of chunks to retrieve per query.
                                      """
      rows = load_testset(testset_path)
      print(f"Loaded {len(rows)} questions from {testset_path}")

    try:
              from eval.evaluate import evaluate_pipeline
except ImportError as exc:
          print(f"ERROR: {exc}", file=sys.stderr)
          sys.exit(1)

    results = evaluate_pipeline(rows, top_k=top_k)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
              json.dump(results, f, indent=2)
          print(f"Results written to {output_path}")


def main() -> None:
      parser = argparse.ArgumentParser(description="Run ragas evaluation pipeline.")
      parser.add_argument(
          "--testset",
          default="eval/testset.jsonl",
          help="Path to JSONL testset (default: eval/testset.jsonl)",
      )
      parser.add_argument(
          "--output",
          default="eval/results/latest.json",
          help="Output path for results JSON (default: eval/results/latest.json)",
      )
      parser.add_argument(
          "--top-k",
          type=int,
          default=5,
          help="Number of chunks to retrieve per query (default: 5)",
      )
      args = parser.parse_args()
      run_evaluation(args.testset, args.output, args.top_k)


if __name__ == "__main__":
      main()

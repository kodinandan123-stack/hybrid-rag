"""scripts/run_evaluation.py - CLI for running ragas evaluation over a testset."""
import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ragas evaluation on a hybrid-RAG testset."
    )
    parser.add_argument(
        "--testset",
        type=str,
        default="eval/testset.jsonl",
        help="Path to the JSONL testset file (default: eval/testset.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/results/evaluation_report.json",
        help="Path to write the JSON evaluation report.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["context_recall", "context_precision", "faithfulness", "answer_relevancy"],
        help="Ragas metrics to evaluate (space-separated).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks passed to generation (default: 5).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-question scores to stdout.",
    )
    return parser.parse_args()


def load_testset(path: str) -> list[dict]:
    testset_path = Path(path)
    if not testset_path.exists():
        print(f"[error] testset not found: {path}", file=sys.stderr)
        sys.exit(1)
    records = []
    with open(testset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        print("[error] testset is empty.", file=sys.stderr)
        sys.exit(1)
    print(f"[info] loaded {len(records)} questions from {path}")
    return records


def run_evaluation(records: list[dict], metrics: list[str], top_k: int, verbose: bool) -> dict:
    """Run the RAG pipeline on each record and score with ragas."""
    try:
        from eval.evaluate import evaluate_pipeline
    except ImportError as exc:
        print(f"[error] could not import eval.evaluate: {exc}", file=sys.stderr)
        sys.exit(1)

    results = evaluate_pipeline(records, metrics=metrics, top_k=top_k)

    if verbose:
        for item in results.get("per_question", []):
            print(json.dumps(item, indent=2))

    return results


def write_report(results: dict, output: str) -> None:
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[info] evaluation report written to {output}")


def main() -> None:
    args = parse_args()
    records = load_testset(args.testset)
    results = run_evaluation(records, metrics=args.metrics, top_k=args.top_k, verbose=args.verbose)
    write_report(results, args.output)

    print("\n=== Aggregate Scores ===")
    for metric, score in results.get("aggregate", {}).items():
        print(f"  {metric}: {score:.4f}")


if __name__ == "__main__":
    main()

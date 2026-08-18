"""
evaluate_ragas.py

Run RAGAS evaluation metrics (faithfulness, answer_relevancy, context_recall)
against the labeled QA pairs in eval/testset.jsonl, using the live hybrid RAG
pipeline to generate answers and retrieve context.

Usage
-----
    python scripts/evaluate_ragas.py [--testset eval/testset.jsonl]
                                     [--corpus  corpus/]
                                     [--top-k   5]
                                     [--output  eval/ragas_results.json]

Environment variables
---------------------
    ANTHROPIC_API_KEY  – required for generation
    QDRANT_URL         – Qdrant instance used by DenseRetriever (default: http://localhost:6333)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from config.logging import configure_logging
from ingestion.loader import Loader
from ingestion.chunker import Chunker
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.hybrid import HybridRetriever
from generation.generator import Generator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline bootstrap
# ---------------------------------------------------------------------------

def _build_pipeline(corpus_dir: str, top_k: int):
    """Index *corpus_dir* and return a ready (retriever, generator) pair."""
    logger.info("Loading corpus from %s", corpus_dir)
    loader = Loader()
    docs = loader.load(corpus_dir)

    chunker = Chunker()
    chunks = chunker.chunk(docs)
    logger.info("Chunked corpus into %d chunks", len(chunks))

    dense = DenseRetriever()
    dense.index(chunks)

    sparse = SparseRetriever(chunks=chunks)
    hybrid = HybridRetriever(dense=dense, sparse=sparse)
    generator = Generator()

    return hybrid, generator, top_k


# ---------------------------------------------------------------------------
# RAGAS dataset construction
# ---------------------------------------------------------------------------

def _build_ragas_dataset(
    testset_path: str,
    hybrid: HybridRetriever,
    generator: Generator,
    top_k: int,
) -> Dict[str, List[Any]]:
    """
    For each QA pair in the testset, retrieve context, generate an answer,
    and collect the four columns RAGAS expects:
        question, answer, contexts, ground_truth
    """
    questions: List[str] = []
    answers: List[str] = []
    contexts: List[List[str]] = []
    ground_truths: List[str] = []

    testset = Path(testset_path)
    if not testset.exists():
        raise FileNotFoundError(f"Testset not found: {testset_path}")

    rows = [json.loads(line) for line in testset.read_text().splitlines() if line.strip()]
    logger.info("Evaluating %d QA pairs from %s", len(rows), testset_path)

    for i, row in enumerate(rows, 1):
        question = row["question"]
        ground_truth = row.get("answer", row.get("ground_truth", ""))

        hits = hybrid.search(question, top_k=top_k)
        context_texts = [h.get("text", "") for h in hits]

        answer = generator.generate(question, hits)

        questions.append(question)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(ground_truth)

        logger.info("[%d/%d] Q: %s", i, len(rows), question[:80])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _run_ragas(dataset_dict: Dict[str, List[Any]]) -> Dict[str, float]:
    """Evaluate with RAGAS and return a dict of metric_name -> score."""
    try:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            faithfulness,
            answer_relevancy,
            context_recall,
        )
    except ImportError as exc:
        logger.error(
            "RAGAS or HuggingFace datasets not installed. "
            "Run: pip install ragas datasets"
        )
        raise SystemExit(1) from exc

    ds = Dataset.from_dict(dataset_dict)
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_recall])

    scores: Dict[str, float] = {}
    for metric in [faithfulness, answer_relevancy, context_recall]:
        key = metric.name
        scores[key] = float(result[key])

    return scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid RAG pipeline with RAGAS metrics."
    )
    parser.add_argument(
        "--testset",
        default="eval/testset.jsonl",
        help="Path to the labeled QA testset (JSONL). Default: eval/testset.jsonl",
    )
    parser.add_argument(
        "--corpus",
        default="corpus/",
        help="Path to the document corpus directory. Default: corpus/",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query. Default: 5",
    )
    parser.add_argument(
        "--output",
        default="eval/ragas_results.json",
        help="Where to write the JSON results. Default: eval/ragas_results.json",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = _parse_args()

    hybrid, generator, top_k = _build_pipeline(args.corpus, args.top_k)
    dataset_dict = _build_ragas_dataset(args.testset, hybrid, generator, top_k)
    scores = _run_ragas(dataset_dict)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scores, indent=2))

    logger.info("RAGAS scores written to %s", output_path)
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")


if __name__ == "__main__":
    main()

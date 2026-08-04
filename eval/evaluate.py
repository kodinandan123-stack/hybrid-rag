"""Evaluation harness for the hybrid RAG pipeline using ragas metrics.

Given a small labeled test set of (question, ground_truth) pairs, this
script runs each question through the hybrid retriever and generator,
then scores the results with ragas so retrieval and generation quality
can be tracked as the pipeline evolves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from generation.generator import Generator
from retrieval.hybrid import HybridRetriever


def load_testset(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL test set of {"question": ..., "ground_truth": ...} records."""
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_pipeline(
    testset: List[Dict[str, Any]], retriever: HybridRetriever, generator: Generator, top_k: int = 5
) -> Dataset:
    """Run each test question through retrieval + generation, collecting a ragas-ready dataset."""
    questions, answers, contexts, ground_truths = [], [], [], []
    for record in testset:
        question = record["question"]
        hits = retriever.search(question, top_k=top_k)
        answer = generator.generate(question, hits)

        questions.append(question)
        answers.append(answer)
        contexts.append([hit["text"] for hit in hits])
        ground_truths.append(record["ground_truth"])

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def evaluate_pipeline(
    testset_path: str, retriever: HybridRetriever, generator: Generator, top_k: int = 5
) -> Dict[str, float]:
    """Evaluate the pipeline on a test set and return ragas metric scores."""
    testset = load_testset(testset_path)
    dataset = run_pipeline(testset, retriever, generator, top_k=top_k)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return result


if __name__ == "__main__":
    import sys

    from retrieval.dense import DenseRetriever
    from retrieval.sparse import SparseRetriever

    testset_path = sys.argv[1] if len(sys.argv) > 1 else "eval/testset.jsonl"
    dense = DenseRetriever()
    sparse = SparseRetriever()
    hybrid = HybridRetriever(dense=dense, sparse=sparse)
    scores = evaluate_pipeline(testset_path, hybrid, Generator())
    print(scores)

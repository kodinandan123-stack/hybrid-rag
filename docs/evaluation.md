# Evaluation

This document describes how retrieval and generation quality are measured for the hybrid RAG pipeline, using the harness in `eval/evaluate.py`.

## Test set format

Evaluation expects a JSONL file (default `eval/testset.jsonl`) where each line is a JSON object containing a `question` and its `ground_truth` answer:

```json
{"question": "What vector store does the pipeline use?", "ground_truth": "Qdrant"}
```

## Running an evaluation

```bash
python eval/evaluate.py eval/testset.jsonl
```

For every question in the test set, the script retrieves context with `HybridRetriever`, generates an answer with `Generator`, and scores the resulting question/answer/context/ground-truth tuples with [ragas](https://github.com/explodinggradients/ragas).

## Metrics

- Faithfulness: whether the generated answer is grounded in the retrieved context, without unsupported claims.
- Answer relevancy: how directly the answer addresses the question that was asked.
- Context precision: whether the retrieved chunks that are actually relevant are ranked near the top.
- Context recall: whether the retrieved context contains enough information to support the ground-truth answer.

## Interpreting results

`evaluate_pipeline` returns a ragas result with one score per metric, averaged across the test set. Track these scores as the corpus, chunking parameters, or retrieval and reranking settings change, and record notable findings in the README's Results section.

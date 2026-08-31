# Evaluation Guide

This guide explains how to measure retrieval and generation quality in the
hybrid RAG pipeline using the metrics implemented in `retrieval/metrics.py`.

## Retrieval Metrics

### Precision@K

Fraction of the top-K results that are relevant.

Precision@K = (relevant in top K) / K

### Recall@K

Fraction of all relevant documents that appear in the top-K results.

Recall@K = (relevant in top K) / (total relevant in corpus)

### Mean Average Precision (MAP)

Average of per-query average precisions across a test set.
AP rewards placing relevant results higher in the ranked list.
MAP = mean of AP(q) over all queries.

### Mean Reciprocal Rank (MRR)

Reciprocal of the rank of the first relevant result, averaged over queries.
MRR = mean of 1/rank_of_first_relevant over all queries.

## Running an Evaluation

from retrieval.metrics import mean_average_precision, mean_reciprocal_rank
from retrieval.hybrid import HybridRetriever
from config.settings import Settings

settings = Settings()
retriever = HybridRetriever(settings=settings)
test_set = load_test_set("data/eval_queries.json")

all_relevant = []
for query, relevant_ids in test_set:
    results = retriever.search(query, top_k=10)
        relevance = [r.doc_id in relevant_ids for r in results]
            all_relevant.append(relevance)

            print("MAP:", mean_average_precision(all_relevant))
            print("MRR:", mean_reciprocal_rank(all_relevant))

            ## Recommended Thresholds

            Precision@5: acceptable > 0.40, good > 0.70
            Recall@10:   acceptable > 0.50, good > 0.80
            MAP:         acceptable > 0.35, good > 0.60
            MRR:         acceptable > 0.50, good > 0.75

            ## Generation Metrics

            Pair retrieval metrics with generation quality checks:
            - Faithfulness: does the answer only use facts from retrieved chunks?
            - Answer relevance: does the answer address the question asked?
            - Context precision: what fraction of retrieved chunks are actually used?

            Tools such as RAGAS automate these checks and integrate with the Anthropic
            API used by this pipeline.
            

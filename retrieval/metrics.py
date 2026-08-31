"""Retrieval evaluation metrics for the hybrid RAG pipeline."""
from __future__ import annotations

from typing import List, Sequence


def precision_at_k(relevant: Sequence[bool], k: int) -> float:
      """Fraction of top-k results that are relevant.

          Args:
                  relevant: Boolean sequence where True means the result is relevant.
                          k: Cut-off rank.

                              Returns:
                                      Precision@k in [0, 1].
                                          """
      if k <= 0:
                raise ValueError("k must be positive")
            top_k = list(relevant)[:k]
    return sum(top_k) / k


def recall_at_k(relevant: Sequence[bool], total_relevant: int, k: int) -> float:
      """Fraction of all relevant documents retrieved in top-k.

          Args:
                  relevant: Boolean sequence where True means the result is relevant.
                          total_relevant: Total number of relevant documents in the corpus.
                                  k: Cut-off rank.

                                      Returns:
                                              Recall@k in [0, 1].
                                                  """
    if total_relevant <= 0:
              return 0.0
          top_k = list(relevant)[:k]
    return sum(top_k) / total_relevant


def average_precision(relevant: Sequence[bool]) -> float:
      """Mean of precision values at each relevant rank position.

          Args:
                  relevant: Boolean sequence ordered by retrieval rank.

                      Returns:
                              Average precision in [0, 1].
                                  """
    hits = 0
    score = 0.0
    for rank, rel in enumerate(relevant, start=1):
              if rel:
                            hits += 1
                            score += hits / rank
                    return score / hits if hits else 0.0


def mean_average_precision(queries_relevant: List[Sequence[bool]]) -> float:
      """MAP over a list of queries.

          Args:
                  queries_relevant: Each element is the relevance sequence for one query.

                      Returns:
                              Mean average precision in [0, 1].
                                  """
    if not queries_relevant:
              return 0.0
    return sum(average_precision(r) for r in queries_relevant) / len(queries_relevant)


def reciprocal_rank(relevant: Sequence[bool]) -> float:
      """Reciprocal of the rank of the first relevant result.

          Args:
                  relevant: Boolean sequence ordered by retrieval rank.

                      Returns:
                              Reciprocal rank in (0, 1], or 0 if no relevant result found.
                                  """
    for rank, rel in enumerate(relevant, start=1):
              if rel:
                            return 1.0 / rank
                    return 0.0


def mean_reciprocal_rank(queries_relevant: List[Sequence[bool]]) -> float:
      """MRR over a list of queries.

          Args:
                  queries_relevant: Each element is the relevance sequence for one query.

                      Returns:
                              Mean reciprocal rank in [0, 1].
                                  """
    if not queries_relevant:
              return 0.0
    return sum(reciprocal_rank(r) for r in queries_relevant) / len(queries_relevant)

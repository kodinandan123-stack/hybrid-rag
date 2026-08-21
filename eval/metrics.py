"""Retrieval evaluation metrics: precision, recall, F1, MRR, and NDCG."""
from typing import List, Set


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of relevant docs found in top-k results."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(relevant)


def f1_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Harmonic mean of precision and recall at k."""
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """Reciprocal rank of the first relevant document."""
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(results: List[List[str]], relevant_sets: List[Set[str]]) -> float:
    """MRR averaged over multiple queries."""
    if not results:
        return 0.0
    rr_scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(results, relevant_sets)
    ]
    return sum(rr_scores) / len(rr_scores)


def dcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    import math
    score = 0.0
    for i, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            score += 1.0 / math.log2(i + 1)
    return score


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Normalised DCG at k."""
    ideal = dcg_at_k(list(relevant), relevant, k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / ideal

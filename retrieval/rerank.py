"""Cross-encoder reranking for refining hybrid retrieval results."""

from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

class Reranker:
    """Reranks candidate chunks using a cross-encoder relevance model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = [(query, candidate["text"]) for candidate in candidates]
        scores = self.model.predict(pairs)

        reranked = [
            {**candidate, "rerank_score": float(score)}
            for candidate, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked[:top_k]

"""hybrid_scorer.py – weighted combination of dense and sparse retrieval scores."""
  from __future__ import annotations

from dataclasses import dataclass, field
  from typing import Dict, List, Optional


  @dataclass
class ScoredDoc:
    """A document with an associated hybrid score."""

      doc_id: str
    text: str
      dense_score: float = 0.0
      sparse_score: float = 0.0
      hybrid_score: float = field(init=False, default=0.0)
      metadata: Dict = field(default_factory=dict)


  class HybridScorer:
    """Combine dense (embedding) and sparse (BM25) scores into a single hybrid score.

          Scores from each retriever are first normalised to [0, 1] using min-max
      normalisation within the result set, then blended via a configurable
    alpha weight::

        hybrid = alpha * dense_norm + (1 - alpha) * sparse_norm

    Args:
        alpha: Weight for the dense score.  ``1.0`` means dense-only;
               ``0.0`` means sparse-only.  Defaults to ``0.5``.
                         min_score_threshold: Docs whose hybrid score falls below this
                             threshold are excluded from the output.  Defaults to ``0.0``
            (keep all docs).
                     """

                     def __init__(
        self,
        alpha: float = 0.5,
        min_score_threshold: float = 0.0,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
                         self.min_score_threshold = min_score_threshold

                     # ------------------------------------------------------------------
                     # Public API
                     # ------------------------------------------------------------------

                     def score(
        self,
        docs: List[ScoredDoc],
        *,
        top_k: Optional[int] = None,
                     ) -> List[ScoredDoc]:
                         """Compute hybrid scores for *docs* and return them sorted descending.

                         Args:
            docs: Documents with ``dense_score`` and ``sparse_score`` populated.
                             top_k: If given, return only the top-*k* results after scoring.

                         Returns:
                           List of :class:`ScoredDoc` objects with ``hybrid_score`` set,
                             sorted by ``hybrid_score`` descending.
                         """
                         if not docs:
                             return []

                         dense_vals = [d.dense_score for d in docs]
        sparse_vals = [d.sparse_score for d in docs]

        dense_norm = self._minmax_normalise(dense_vals)
        sparse_norm = self._minmax_normalise(sparse_vals)

                         results: List[ScoredDoc] = []
                         for doc, dn, sn in zip(docs, dense_norm, sparse_norm):
            doc.hybrid_score = self.alpha * dn + (1.0 - self.alpha) * sn
                             if doc.hybrid_score >= self.min_score_threshold:
                results.append(doc)

                         results.sort(key=lambda d: d.hybrid_score, reverse=True)

        if top_k is not None:
            results = results[:top_k]

                         return results

                     def update_alpha(self, alpha: float) -> None:
        """Update the dense weight at runtime."""
                         if not 0.0 <= alpha <= 1.0:
                             raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha

    # ------------------------------------------------------------------
                     # Internal helpers
                     # ------------------------------------------------------------------

                     @staticmethod
                     def _minmax_normalise(values: List[float]) -> List[float]:
        """Return min-max normalised copy of *values* to the range [0, 1]."""
                         if not values:
                             return []
                         lo, hi = min(values), max(values)
                         if hi == lo:
            return [1.0] * len(values)
                         span = hi - lo
                         return [(v - lo) / span for v in values]
                 

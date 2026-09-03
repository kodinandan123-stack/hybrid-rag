"""Tests for retrieval/hybrid_scorer.py."""
import pytest

from retrieval.hybrid_scorer import HybridScorer, ScoredDoc


def make_doc(doc_id, dense, sparse):
      return ScoredDoc(doc_id=doc_id, text=f"text-{doc_id}", dense_score=dense, sparse_score=sparse)


@pytest.fixture
def scorer():
      return HybridScorer(alpha=0.5)


@pytest.fixture
def docs():
      return [
                make_doc("a", dense=0.9, sparse=0.1),
                make_doc("b", dense=0.5, sparse=0.5),
                make_doc("c", dense=0.1, sparse=0.9),
      ]


class TestHybridScorer:
      def test_returns_all_docs_by_default(self, scorer, docs):
                result = scorer.score(docs)
                assert len(result) == 3

      def test_dense_only_alpha_one(self, docs):
                s = HybridScorer(alpha=1.0)
                result = s.score(docs)
                assert result[0].doc_id == "a"

      def test_sparse_only_alpha_zero(self, docs):
                s = HybridScorer(alpha=0.0)
                result = s.score(docs)
                assert result[0].doc_id == "c"

      def test_sorted_descending(self, scorer, docs):
                result = scorer.score(docs)
                scores = [d.hybrid_score for d in result]
                assert scores == sorted(scores, reverse=True)

      def test_top_k_limits_results(self, scorer, docs):
                result = scorer.score(docs, top_k=2)
                assert len(result) == 2

      def test_top_k_larger_than_docs(self, scorer, docs):
                result = scorer.score(docs, top_k=10)
                assert len(result) == 3

      def test_empty_input(self, scorer):
                assert scorer.score([]) == []

      def test_single_doc_score_is_one(self, scorer):
                doc = make_doc("x", dense=0.7, sparse=0.3)
                result = scorer.score([doc])
                assert len(result) == 1
                assert result[0].hybrid_score == 1.0

      def test_min_score_threshold_filters(self, docs):
                s = HybridScorer(alpha=0.5, min_score_threshold=0.6)
                result = s.score(docs)
                assert all(d.hybrid_score >= 0.6 for d in result)

      def test_invalid_alpha_raises(self):
                with pytest.raises(ValueError):
                              HybridScorer(alpha=1.5)

            def test_update_alpha(self, scorer):
                      scorer.update_alpha(0.8)
                      assert scorer.alpha == 0.8

    def test_update_alpha_invalid(self, scorer):
              with pytest.raises(ValueError):
                            scorer.update_alpha(-0.1)

          def test_hybrid_score_set_on_docs(self, scorer, docs):
                    scorer.score(docs)
                    for d in docs:
                                  assert d.hybrid_score >= 0.0

                def test_doc_ids_preserved(self, scorer, docs):
                          result = scorer.score(docs)
                          ids = {d.doc_id for d in result}
                          assert ids == {"a", "b", "c"}
                  

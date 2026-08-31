"""Unit tests for retrieval/metrics.py."""
import pytest
from retrieval.metrics import (
    average_precision,
    mean_average_precision,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestPrecisionAtK:
      def test_all_relevant(self):
                assert precision_at_k([True, True, True], k=3) == pytest.approx(1.0)

      def test_none_relevant(self):
                assert precision_at_k([False, False, False], k=3) == pytest.approx(0.0)

      def test_half_relevant(self):
                assert precision_at_k([True, False, True, False], k=4) == pytest.approx(0.5)

      def test_k_smaller_than_list(self):
                assert precision_at_k([True, False, False], k=1) == pytest.approx(1.0)

      def test_invalid_k(self):
                with pytest.raises(ValueError):
                              precision_at_k([True], k=0)


class TestRecallAtK:
      def test_all_retrieved(self):
                assert recall_at_k([True, True], total_relevant=2, k=2) == pytest.approx(1.0)

      def test_partial_retrieval(self):
                assert recall_at_k([True, False, False], total_relevant=3, k=3) == pytest.approx(1 / 3)

      def test_zero_total_relevant(self):
                assert recall_at_k([True], total_relevant=0, k=1) == pytest.approx(0.0)


class TestAveragePrecision:
      def test_perfect(self):
                assert average_precision([True, True, True]) == pytest.approx(1.0)

      def test_no_relevant(self):
                assert average_precision([False, False]) == pytest.approx(0.0)

      def test_single_hit_at_rank2(self):
                ap = average_precision([False, True])
                assert ap == pytest.approx(0.5)

      def test_mixed(self):
                # Hits at ranks 1 and 3 -> (1/1 + 2/3) / 2
                ap = average_precision([True, False, True])
                assert ap == pytest.approx((1.0 + 2 / 3) / 2)


class TestMeanAveragePrecision:
      def test_empty(self):
                assert mean_average_precision([]) == pytest.approx(0.0)

      def test_single_query(self):
                assert mean_average_precision([[True, False, True]]) == pytest.approx(
                              average_precision([True, False, True])
                )

      def test_multiple_queries(self):
                aps = [average_precision(r) for r in [[True], [False, True]]]
                assert mean_average_precision([[True], [False, True]]) == pytest.approx(
                    sum(aps) / len(aps)
                )


class TestReciprocalRank:
      def test_first_hit_at_rank1(self):
                assert reciprocal_rank([True, False]) == pytest.approx(1.0)

      def test_first_hit_at_rank3(self):
                assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)

      def test_no_hit(self):
                assert reciprocal_rank([False, False]) == pytest.approx(0.0)


class TestMeanReciprocalRank:
      def test_empty(self):
                assert mean_reciprocal_rank([]) == pytest.approx(0.0)

      def test_single(self):
                assert mean_reciprocal_rank([[True]]) == pytest.approx(1.0)

      def test_multiple(self):
                mrr = mean_reciprocal_rank([[True], [False, True]])
                assert mrr == pytest.approx((1.0 + 0.5) / 2)
        

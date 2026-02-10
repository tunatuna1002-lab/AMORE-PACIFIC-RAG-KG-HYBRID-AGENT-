"""Tests for metric aggregator."""

import pytest

from eval.metrics.aggregator import (
    DEFAULT_THRESHOLDS,
    DEFAULT_WEIGHTS,
    MetricAggregator,
    check_gating,
    compute_overall_score,
    get_fail_reason_description,
)
from eval.schemas import ItemMetadata, L1Metrics, L2Metrics, L3Metrics, L4Metrics, L5Metrics


class TestMetricAggregator:
    """Tests for MetricAggregator."""

    @pytest.fixture
    def aggregator(self):
        """Create default aggregator."""
        return MetricAggregator()

    @pytest.fixture
    def perfect_metrics(self):
        """Create perfect metrics for all layers."""
        return {
            "l1": L1Metrics(
                entity_link_f1=1.0,
                concept_map_f1=1.0,
                constraint_extraction_f1=1.0,
            ),
            "l2": L2Metrics(
                context_recall_at_k=1.0,
                context_precision_at_k=1.0,
                mrr=1.0,
            ),
            "l3": L3Metrics(
                hits_at_k=1.0,
                kg_edge_f1=1.0,
            ),
            "l4": L4Metrics(
                constraint_violation_rate=0.0,
                type_consistency_rate=1.0,
            ),
            "l5": L5Metrics(
                answer_exact_match=1.0,
                answer_f1=1.0,
                groundedness_score=1.0,
                answer_relevance_score=1.0,
            ),
        }

    @pytest.fixture
    def failing_metrics(self):
        """Create failing metrics for all layers."""
        return {
            "l1": L1Metrics(
                entity_link_f1=0.3,
                concept_map_f1=0.3,
                constraint_extraction_f1=0.3,
            ),
            "l2": L2Metrics(
                context_recall_at_k=0.3,
                context_precision_at_k=0.3,
                mrr=0.3,
            ),
            "l3": L3Metrics(
                hits_at_k=0.3,
                kg_edge_f1=0.3,
            ),
            "l4": L4Metrics(
                constraint_violation_rate=0.5,
                type_consistency_rate=0.5,
            ),
            "l5": L5Metrics(
                answer_exact_match=0.0,
                answer_f1=0.3,
                groundedness_score=0.3,
                answer_relevance_score=0.3,
            ),
        }

    def test_default_weights(self):
        """Test default weights are set."""
        assert "l5" in DEFAULT_WEIGHTS
        assert "l2_l3" in DEFAULT_WEIGHTS
        assert "l1" in DEFAULT_WEIGHTS
        assert "l4" in DEFAULT_WEIGHTS
        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)

    def test_default_thresholds(self):
        """Test default thresholds are set."""
        assert "groundedness_min" in DEFAULT_THRESHOLDS
        assert "constraint_violation_max" in DEFAULT_THRESHOLDS
        assert "context_recall_min" in DEFAULT_THRESHOLDS
        assert "hits_at_k_min" in DEFAULT_THRESHOLDS

    def test_compute_overall_score_perfect(self, aggregator, perfect_metrics):
        """Test overall score with perfect metrics."""
        score = aggregator.compute_overall_score(
            perfect_metrics["l1"],
            perfect_metrics["l2"],
            perfect_metrics["l3"],
            perfect_metrics["l4"],
            perfect_metrics["l5"],
        )
        assert score == pytest.approx(1.0)

    def test_compute_overall_score_failing(self, aggregator, failing_metrics):
        """Test overall score with failing metrics."""
        score = aggregator.compute_overall_score(
            failing_metrics["l1"],
            failing_metrics["l2"],
            failing_metrics["l3"],
            failing_metrics["l4"],
            failing_metrics["l5"],
        )
        assert score < 0.5

    def test_compute_overall_score_requires_kg_true(self, aggregator, perfect_metrics):
        """Test overall score with requires_kg=True."""
        metadata = ItemMetadata(requires_kg=True)
        score = aggregator.compute_overall_score(
            perfect_metrics["l1"],
            perfect_metrics["l2"],
            perfect_metrics["l3"],
            perfect_metrics["l4"],
            perfect_metrics["l5"],
            metadata=metadata,
        )
        assert score == pytest.approx(1.0)

    def test_compute_overall_score_requires_kg_false(self, aggregator, perfect_metrics):
        """Test overall score with requires_kg=False."""
        metadata = ItemMetadata(requires_kg=False)
        score = aggregator.compute_overall_score(
            perfect_metrics["l1"],
            perfect_metrics["l2"],
            perfect_metrics["l3"],
            perfect_metrics["l4"],
            perfect_metrics["l5"],
            metadata=metadata,
        )
        assert score == pytest.approx(1.0)

    def test_check_gating_pass(self, aggregator, perfect_metrics):
        """Test gating check with passing metrics."""
        passed, fail_reasons = aggregator.check_gating(
            perfect_metrics["l1"],
            perfect_metrics["l2"],
            perfect_metrics["l3"],
            perfect_metrics["l4"],
            perfect_metrics["l5"],
        )
        assert passed
        assert len(fail_reasons) == 0

    def test_check_gating_l1_fail(self, aggregator, perfect_metrics):
        """Test gating check with L1 failure."""
        l1 = L1Metrics(
            entity_link_f1=0.3,  # Below threshold
            concept_map_f1=1.0,
            constraint_extraction_f1=1.0,
        )
        passed, fail_reasons = aggregator.check_gating(
            l1,
            perfect_metrics["l2"],
            perfect_metrics["l3"],
            perfect_metrics["l4"],
            perfect_metrics["l5"],
        )
        assert not passed
        assert "L1_mapping_fail" in fail_reasons

    def test_check_gating_l2_fail(self, aggregator, perfect_metrics):
        """Test gating check with L2 failure (requires_kg=False)."""
        l2 = L2Metrics(
            context_recall_at_k=0.5,  # Below threshold
            context_precision_at_k=1.0,
            mrr=1.0,
        )
        metadata = ItemMetadata(requires_kg=False)
        passed, fail_reasons = aggregator.check_gating(
            perfect_metrics["l1"],
            l2,
            perfect_metrics["l3"],
            perfect_metrics["l4"],
            perfect_metrics["l5"],
            metadata=metadata,
        )
        assert not passed
        assert "L2_doc_retrieval_fail" in fail_reasons

    def test_check_gating_l3_fail(self, aggregator, perfect_metrics):
        """Test gating check with L3 failure (requires_kg=True)."""
        l3 = L3Metrics(
            hits_at_k=0.5,  # Below threshold
            kg_edge_f1=1.0,
        )
        metadata = ItemMetadata(requires_kg=True)
        passed, fail_reasons = aggregator.check_gating(
            perfect_metrics["l1"],
            perfect_metrics["l2"],
            l3,
            perfect_metrics["l4"],
            perfect_metrics["l5"],
            metadata=metadata,
        )
        assert not passed
        assert "L3_kg_fail" in fail_reasons

    def test_check_gating_l4_violation(self, aggregator, perfect_metrics):
        """Test gating check with L4 constraint violation."""
        l4 = L4Metrics(
            constraint_violation_rate=0.2,  # Above threshold
            type_consistency_rate=1.0,
        )
        passed, fail_reasons = aggregator.check_gating(
            perfect_metrics["l1"],
            perfect_metrics["l2"],
            perfect_metrics["l3"],
            l4,
            perfect_metrics["l5"],
        )
        assert not passed
        assert "L4_constraint_violation" in fail_reasons

    def test_check_gating_l4_inconsistency(self, aggregator, perfect_metrics):
        """Test gating check with L4 type inconsistency."""
        l4 = L4Metrics(
            constraint_violation_rate=0.0,
            type_consistency_rate=0.5,  # Below threshold
        )
        passed, fail_reasons = aggregator.check_gating(
            perfect_metrics["l1"],
            perfect_metrics["l2"],
            perfect_metrics["l3"],
            l4,
            perfect_metrics["l5"],
        )
        assert not passed
        assert "L4_type_inconsistency" in fail_reasons

    def test_check_gating_l5_wrong_answer(self, aggregator, perfect_metrics):
        """Test gating check with L5 wrong answer."""
        l5 = L5Metrics(
            answer_exact_match=0.0,
            answer_f1=0.3,  # Below threshold
            groundedness_score=None,
            answer_relevance_score=None,
        )
        passed, fail_reasons = aggregator.check_gating(
            perfect_metrics["l1"],
            perfect_metrics["l2"],
            perfect_metrics["l3"],
            perfect_metrics["l4"],
            l5,
        )
        assert not passed
        assert "L5_wrong_answer" in fail_reasons

    def test_check_gating_l5_grounding_fail(self, aggregator, perfect_metrics):
        """Test gating check with L5 grounding failure."""
        l5 = L5Metrics(
            answer_exact_match=1.0,
            answer_f1=1.0,
            groundedness_score=0.5,  # Below threshold
            answer_relevance_score=1.0,
        )
        passed, fail_reasons = aggregator.check_gating(
            perfect_metrics["l1"],
            perfect_metrics["l2"],
            perfect_metrics["l3"],
            perfect_metrics["l4"],
            l5,
        )
        assert not passed
        assert "L5_grounding_fail" in fail_reasons

    def test_check_gating_multiple_failures(self, aggregator, failing_metrics):
        """Test gating check with multiple failures."""
        passed, fail_reasons = aggregator.check_gating(
            failing_metrics["l1"],
            failing_metrics["l2"],
            failing_metrics["l3"],
            failing_metrics["l4"],
            failing_metrics["l5"],
        )
        assert not passed
        assert len(fail_reasons) > 1

    def test_custom_weights(self, perfect_metrics):
        """Test aggregator with custom weights."""
        custom_weights = {"l5": 1.0, "l2_l3": 0.0, "l1": 0.0, "l4": 0.0}
        aggregator = MetricAggregator(weights=custom_weights)

        score = aggregator.compute_overall_score(
            perfect_metrics["l1"],
            perfect_metrics["l2"],
            perfect_metrics["l3"],
            perfect_metrics["l4"],
            perfect_metrics["l5"],
        )
        assert score == pytest.approx(1.0)

    def test_custom_thresholds(self, perfect_metrics):
        """Test aggregator with custom thresholds."""
        custom_thresholds = {"entity_link_f1_min": 0.99}
        aggregator = MetricAggregator(thresholds=custom_thresholds)

        # With 1.0 F1, should still pass
        l1 = L1Metrics(
            entity_link_f1=0.98,  # Below custom threshold
            concept_map_f1=1.0,
            constraint_extraction_f1=1.0,
        )
        passed, fail_reasons = aggregator.check_gating(
            l1,
            perfect_metrics["l2"],
            perfect_metrics["l3"],
            perfect_metrics["l4"],
            perfect_metrics["l5"],
        )
        assert not passed
        assert "L1_mapping_fail" in fail_reasons


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compute_overall_score_function(self):
        """Test compute_overall_score convenience function."""
        l1 = L1Metrics(entity_link_f1=1.0, concept_map_f1=1.0, constraint_extraction_f1=1.0)
        l2 = L2Metrics(context_recall_at_k=1.0, context_precision_at_k=1.0, mrr=1.0)
        l3 = L3Metrics(hits_at_k=1.0, kg_edge_f1=1.0)
        l4 = L4Metrics(constraint_violation_rate=0.0, type_consistency_rate=1.0)
        l5 = L5Metrics(
            answer_exact_match=1.0,
            answer_f1=1.0,
            groundedness_score=None,
            answer_relevance_score=None,
        )

        score = compute_overall_score(l1, l2, l3, l4, l5)
        assert score == pytest.approx(1.0)

    def test_check_gating_function(self):
        """Test check_gating convenience function."""
        l1 = L1Metrics(entity_link_f1=1.0, concept_map_f1=1.0, constraint_extraction_f1=1.0)
        l2 = L2Metrics(context_recall_at_k=1.0, context_precision_at_k=1.0, mrr=1.0)
        l3 = L3Metrics(hits_at_k=1.0, kg_edge_f1=1.0)
        l4 = L4Metrics(constraint_violation_rate=0.0, type_consistency_rate=1.0)
        l5 = L5Metrics(
            answer_exact_match=1.0,
            answer_f1=1.0,
            groundedness_score=None,
            answer_relevance_score=None,
        )

        passed, fail_reasons = check_gating(l1, l2, l3, l4, l5)
        assert passed
        assert len(fail_reasons) == 0

    def test_get_fail_reason_description(self):
        """Test get_fail_reason_description function."""
        desc = get_fail_reason_description("L1_mapping_fail")
        assert "entity" in desc.lower() or "link" in desc.lower()

        desc = get_fail_reason_description("unknown_tag")
        assert "unknown" in desc.lower()

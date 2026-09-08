"""Tests for L5 answer metrics."""

import pytest

from eval.metrics.l5_answer import (
    L5AnswerMetrics,
    answer_exact_match,
    answer_token_f1,
    compute_answer_quality,
)
from eval.schemas import AnswerTrace, GoldEvidence


class TestL5AnswerMetrics:
    """Tests for L5 answer metrics."""

    @pytest.fixture
    def calculator(self):
        """Create L5 metrics calculator."""
        return L5AnswerMetrics()

    def test_exact_match_identical(self, calculator):
        """Test exact match with identical answers."""
        trace = AnswerTrace(
            final_answer="LANEIGE is the top brand",
            citations=[],
            confidence=None,
        )
        gold = GoldEvidence(answer="LANEIGE is the top brand")

        score = calculator._compute_exact_match(trace, gold)
        assert score == 1.0

    def test_exact_match_case_insensitive(self, calculator):
        """Test exact match is case insensitive."""
        trace = AnswerTrace(
            final_answer="LANEIGE IS THE TOP BRAND",
            citations=[],
            confidence=None,
        )
        gold = GoldEvidence(answer="laneige is the top brand")

        score = calculator._compute_exact_match(trace, gold)
        assert score == 1.0

    def test_exact_match_ignores_punctuation(self, calculator):
        """Test exact match ignores punctuation."""
        trace = AnswerTrace(
            final_answer="LANEIGE is the top brand!",
            citations=[],
            confidence=None,
        )
        gold = GoldEvidence(answer="LANEIGE is the top brand")

        score = calculator._compute_exact_match(trace, gold)
        assert score == 1.0

    def test_exact_match_different(self, calculator):
        """Test exact match with different answers."""
        trace = AnswerTrace(
            final_answer="LANEIGE is the top brand",
            citations=[],
            confidence=None,
        )
        gold = GoldEvidence(answer="COSRX is the top brand")

        score = calculator._compute_exact_match(trace, gold)
        assert score == 0.0

    def test_exact_match_no_gold(self, calculator):
        """Test exact match with no gold answer."""
        trace = AnswerTrace(
            final_answer="LANEIGE is the top brand",
            citations=[],
            confidence=None,
        )
        gold = GoldEvidence(answer=None)

        score = calculator._compute_exact_match(trace, gold)
        assert score == 1.0  # No gold = automatic pass

    def test_token_f1_identical(self, calculator):
        """Test token F1 with identical answers."""
        trace = AnswerTrace(
            final_answer="LANEIGE is the top brand",
            citations=[],
            confidence=None,
        )
        gold = GoldEvidence(answer="LANEIGE is the top brand")

        score = calculator._compute_token_f1(trace, gold)
        assert score == 1.0

    def test_token_f1_partial_overlap(self, calculator):
        """Test token F1 with partial overlap."""
        trace = AnswerTrace(
            final_answer="LANEIGE is the best brand in skin care",
            citations=[],
            confidence=None,
        )
        gold = GoldEvidence(answer="LANEIGE is the top brand")

        score = calculator._compute_token_f1(trace, gold)
        # Some overlap: "laneige", "is", "the", "brand"
        assert 0.5 < score < 1.0

    def test_token_f1_no_overlap(self, calculator):
        """Test token F1 with no overlap."""
        trace = AnswerTrace(
            final_answer="completely different answer",
            citations=[],
            confidence=None,
        )
        gold = GoldEvidence(answer="LANEIGE is the top brand")

        score = calculator._compute_token_f1(trace, gold)
        assert score == 0.0

    def test_token_f1_no_gold(self, calculator):
        """Test token F1 with no gold answer."""
        trace = AnswerTrace(
            final_answer="LANEIGE is the top brand",
            citations=[],
            confidence=None,
        )
        gold = GoldEvidence(answer=None)

        score = calculator._compute_token_f1(trace, gold)
        assert score == 1.0  # No gold = automatic pass

    def test_normalize_answer(self, calculator):
        """Test answer normalization."""
        normalized = L5AnswerMetrics._normalize_answer("  LANEIGE  is   the TOP brand!  ")
        assert normalized == "laneige is the top brand"

    def test_tokenize(self, calculator):
        """Test tokenization."""
        tokens = L5AnswerMetrics._tokenize("LANEIGE is the top brand!")
        assert tokens == ["laneige", "is", "the", "top", "brand"]

    def test_tokenize_empty(self, calculator):
        """Test tokenization of empty string."""
        tokens = L5AnswerMetrics._tokenize("")
        assert tokens == []

    def test_compute_sync(self, calculator):
        """Test synchronous compute without judge."""
        trace = AnswerTrace(
            final_answer="LANEIGE is the top brand",
            citations=[],
            confidence=0.9,
        )
        gold = GoldEvidence(answer="LANEIGE is the top brand")

        metrics = calculator.compute_sync(trace, gold)

        assert metrics.answer_exact_match == 1.0
        assert metrics.answer_f1 == 1.0
        assert metrics.groundedness_score is None
        assert metrics.answer_relevance_score is None

    @pytest.mark.asyncio
    async def test_compute_async_without_judge(self, calculator):
        """Test async compute without judge."""
        trace = AnswerTrace(
            final_answer="LANEIGE is the top brand",
            citations=[],
            confidence=0.9,
        )
        gold = GoldEvidence(answer="LANEIGE is the top brand")

        metrics = await calculator.compute(
            trace, gold, question="What is the top brand?", context="test context", use_judge=False
        )

        assert metrics.answer_exact_match == 1.0
        assert metrics.groundedness_score is None

    @pytest.mark.asyncio
    async def test_compute_async_with_judge(self, calculator):
        """Test async compute with judge (stub)."""
        trace = AnswerTrace(
            final_answer="LANEIGE is the top brand",
            citations=[],
            confidence=0.9,
        )
        gold = GoldEvidence(answer="LANEIGE is the top brand")

        metrics = await calculator.compute(
            trace,
            gold,
            question="What is the top brand?",
            context="LANEIGE is the top brand in the market.",
            use_judge=True,
        )

        assert metrics.answer_exact_match == 1.0
        assert metrics.groundedness_score is not None
        assert metrics.answer_relevance_score is not None


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_answer_exact_match_function(self):
        """Test answer_exact_match convenience function."""
        trace = AnswerTrace(
            final_answer="test answer",
            citations=[],
            confidence=None,
        )
        gold = GoldEvidence(answer="test answer")

        score = answer_exact_match(trace, gold)
        assert score == 1.0

    def test_answer_token_f1_function(self):
        """Test answer_token_f1 convenience function."""
        trace = AnswerTrace(
            final_answer="test answer",
            citations=[],
            confidence=None,
        )
        gold = GoldEvidence(answer="test answer")

        score = answer_token_f1(trace, gold)
        assert score == 1.0

    def test_compute_answer_quality_function(self):
        """Test compute_answer_quality convenience function."""
        metrics = compute_answer_quality(
            answer="LANEIGE is the top brand",
            gold_answer="LANEIGE is the top brand",
            context="LANEIGE is the top brand in the market.",
        )

        assert metrics["exact_match"] == 1.0
        assert metrics["token_f1"] == 1.0
        assert "context_overlap" in metrics

    def test_compute_answer_quality_no_gold(self):
        """Test compute_answer_quality with no gold answer."""
        metrics = compute_answer_quality(
            answer="LANEIGE is the top brand",
            gold_answer=None,
        )

        assert metrics["exact_match"] == 1.0
        assert metrics["token_f1"] == 1.0


class TestNumericAccuracy:
    """수치 정확도 (2026-09-06, 5단계).

    골든셋 40문항에 있던 expected_values를 어떤 채점기도 읽지 않아, 데이터형
    문항에서 정답 일치 계열 지표가 문체 유사도만 재고 있었다.
    (docs/eval/rag-eval-review-2026-09-06.md §2-c 2번)
    """

    def test_none_when_no_expected_values(self):
        from eval.metrics.l5_answer import numeric_accuracy

        assert numeric_accuracy("LANEIGE SoS는 2.0%입니다.", {}) is None

    def test_exact_value_matches(self):
        from eval.metrics.l5_answer import numeric_accuracy

        assert numeric_accuracy("LANEIGE Lip Care SoS는 2.0%입니다.", {"sos": 2.0}) == 1.0

    def test_within_ten_percent_relative_error_matches(self):
        from eval.metrics.l5_answer import numeric_accuracy

        # 2.0 대비 2.15는 7.5% 오차 → 정답
        assert numeric_accuracy("SoS는 2.15%", {"sos": 2.0}) == 1.0
        # 2.0 대비 2.5는 25% 오차 → 오답
        assert numeric_accuracy("SoS는 2.5%", {"sos": 2.0}) == 0.0

    def test_thousands_separator_is_handled(self):
        from eval.metrics.l5_answer import numeric_accuracy

        assert numeric_accuracy("리뷰 37,356건", {"reviews": 37356}) == 1.0

    def test_range_keys_use_interval_containment(self):
        from eval.metrics.l5_answer import numeric_accuracy

        values = {"hhi_low": 0.06, "hhi_high": 0.08}
        assert numeric_accuracy("HHI는 0.0681입니다", values) == 1.0
        assert numeric_accuracy("HHI는 0.15입니다", values) == 0.0

    def test_partial_credit_across_keys(self):
        from eval.metrics.l5_answer import numeric_accuracy

        score = numeric_accuracy("SoS 2.0%, HHI 0.15", {"sos": 2.0, "hhi": 0.0681})
        assert score == 0.5

    def test_zero_expectation_requires_zero(self):
        from eval.metrics.l5_answer import numeric_accuracy

        assert numeric_accuracy("해당 카테고리 SoS는 0%입니다.", {"sos": 0.0}) == 1.0
        assert numeric_accuracy("SoS는 3.0%입니다.", {"sos": 0.0}) == 0.0


class TestNumericAccuracyGating:
    """게이트는 골드를 원자료로 검증할 수 있는 문항에만 적용한다."""

    @staticmethod
    def _metrics(numeric):
        from eval.schemas import L1Metrics, L2Metrics, L3Metrics, L4Metrics, L5Metrics

        return (
            L1Metrics(entity_link_f1=1.0, concept_map_f1=1.0, constraint_extraction_f1=1.0),
            L2Metrics(
                context_recall_at_k=1.0,
                context_precision_at_k=1.0,
                mrr=1.0,
                context_recall_at_k_concept=1.0,
            ),
            L3Metrics(hits_at_k=1.0, kg_edge_f1=1.0, kg_edge_recall=1.0),
            L4Metrics(constraint_violation_rate=0.0, type_consistency_rate=1.0),
            L5Metrics(
                answer_exact_match=1.0,
                answer_f1=1.0,
                groundedness_score=1.0,
                answer_relevance_score=1.0,
                numeric_accuracy=numeric,
            ),
        )

    def _gate(self, numeric, gold_source):
        from eval.metrics.aggregator import MetricAggregator
        from eval.schemas import ItemMetadata

        l1, l2, l3, l4, l5 = self._metrics(numeric)
        return MetricAggregator().check_gating(
            l1, l2, l3, l4, l5, ItemMetadata(gold_source=gold_source)
        )

    def test_snapshot_item_fails_on_numeric_mismatch(self):
        passed, reasons = self._gate(0.0, "snapshot")

        assert passed is False
        assert "L5_numeric_mismatch" in reasons

    def test_document_item_is_only_reported(self):
        passed, reasons = self._gate(0.0, "document")

        assert "L5_numeric_mismatch" not in reasons
        assert passed is True

    def test_domain_expectation_item_is_only_reported(self):
        passed, reasons = self._gate(0.0, "domain_expectation")

        assert "L5_numeric_mismatch" not in reasons
        assert passed is True

    def test_domain_expectation_skips_wrong_answer_gate(self):
        """원자료로 검증할 수 없는 골드에 정답 일치를 요구하지 않는다."""
        from eval.metrics.aggregator import MetricAggregator
        from eval.schemas import ItemMetadata, L5Metrics

        l1, l2, l3, l4, _ = self._metrics(None)
        weak_answer = L5Metrics(
            answer_exact_match=0.0,
            answer_f1=0.0,
            semantic_similarity=0.1,
            groundedness_score=1.0,
            answer_relevance_score=1.0,
        )

        _, domain_reasons = MetricAggregator().check_gating(
            l1, l2, l3, l4, weak_answer, ItemMetadata(gold_source="domain_expectation")
        )
        _, snapshot_reasons = MetricAggregator().check_gating(
            l1, l2, l3, l4, weak_answer, ItemMetadata(gold_source="snapshot")
        )

        assert "L5_wrong_answer" not in domain_reasons
        assert "L5_wrong_answer" in snapshot_reasons

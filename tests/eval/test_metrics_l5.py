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

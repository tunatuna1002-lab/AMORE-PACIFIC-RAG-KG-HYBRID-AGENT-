"""Tests for semantic similarity metrics."""

import pytest

from eval.metrics.semantic import (
    SemanticSimilarity,
    answer_semantic_similarity,
    cosine_similarity,
    semantic_similarity,
)


class TestCosineSimilarity:
    """Tests for cosine similarity function."""

    def test_identical_vectors(self):
        """Test identical vectors return 1.0."""
        import numpy as np

        vec = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Test orthogonal vectors return 0.0."""
        import numpy as np

        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Test opposite vectors return -1.0."""
        import numpy as np

        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

    def test_zero_vector(self):
        """Test zero vector returns 0.0."""
        import numpy as np

        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])
        assert cosine_similarity(vec1, vec2) == 0.0


class TestSemanticSimilarity:
    """Tests for SemanticSimilarity class."""

    @pytest.fixture
    def calculator(self):
        """Create a semantic similarity calculator."""
        return SemanticSimilarity()

    def test_empty_text(self, calculator):
        """Test empty text returns 0.0."""
        assert calculator.compute("", "some text") == 0.0
        assert calculator.compute("some text", "") == 0.0
        assert calculator.compute("", "") == 0.0

    def test_identical_text(self, calculator):
        """Test identical text returns high similarity."""
        text = "LANEIGE Lip Sleeping Mask is a popular product"
        similarity = calculator.compute(text, text)
        # Should be very high (close to 1.0)
        assert similarity > 0.95

    def test_similar_text(self, calculator):
        """Test similar texts return positive similarity."""
        text1 = "LANEIGE의 Lip Care 점유율은 5%입니다"
        text2 = "라네즈 립케어 시장점유율은 약 5퍼센트"
        similarity = calculator.compute(text1, text2)
        # Should have some positive similarity (Korean-English cross-lingual)
        # Note: actual similarity varies by model capabilities
        assert similarity > 0.1

    def test_different_text(self, calculator):
        """Test very different texts return low similarity."""
        text1 = "LANEIGE Lip Care market share"
        text2 = "Weather forecast for tomorrow"
        similarity = calculator.compute(text1, text2)
        # Should have low similarity
        assert similarity < 0.5

    def test_korean_english_similarity(self, calculator):
        """Test Korean-English text similarity."""
        korean = "라네즈 립슬리핑마스크"
        english = "LANEIGE Lip Sleeping Mask"
        similarity = calculator.compute(korean, english)
        # Should have some similarity (multilingual model)
        # Note: This depends on model capabilities
        assert 0 <= similarity <= 1

    def test_fallback_token_overlap(self, calculator):
        """Test token overlap fallback method."""
        text1 = "LANEIGE is a Korean beauty brand"
        text2 = "LANEIGE brand from Korea beauty"

        similarity = calculator._token_overlap_similarity(text1, text2)
        # Both share: laneige, brand, beauty (korea/korean are different tokens)
        # Jaccard: intersection/union = 3/8 = 0.375
        assert similarity > 0.3

    def test_fallback_empty_text(self, calculator):
        """Test fallback with empty text."""
        assert calculator._token_overlap_similarity("", "text") == 0.0
        assert calculator._token_overlap_similarity("text", "") == 0.0
        assert calculator._token_overlap_similarity("", "") == 0.0

    def test_is_available_property(self, calculator):
        """Test is_available property."""
        # Should be True if sentence-transformers is installed
        # or False if not (will use fallback)
        assert isinstance(calculator.is_available, bool)


class TestSemanticSimilarityBatch:
    """Tests for batch semantic similarity."""

    @pytest.fixture
    def calculator(self):
        """Create a semantic similarity calculator."""
        return SemanticSimilarity()

    def test_batch_single_pair(self, calculator):
        """Test batch with single pair."""
        texts1 = ["LANEIGE Lip Care"]
        texts2 = ["라네즈 립케어"]
        results = calculator.compute_batch(texts1, texts2)

        assert len(results) == 1
        assert 0 <= results[0] <= 1

    def test_batch_multiple_pairs(self, calculator):
        """Test batch with multiple pairs."""
        texts1 = [
            "LANEIGE Lip Care",
            "COSRX skincare",
            "Market share analysis",
        ]
        texts2 = [
            "라네즈 립케어",
            "코스알엑스 스킨케어",
            "시장 점유율 분석",
        ]
        results = calculator.compute_batch(texts1, texts2)

        assert len(results) == 3
        for score in results:
            assert 0 <= score <= 1

    def test_batch_mismatched_length(self, calculator):
        """Test batch with mismatched lengths raises error."""
        texts1 = ["text1", "text2"]
        texts2 = ["text1"]

        with pytest.raises(ValueError):
            calculator.compute_batch(texts1, texts2)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_semantic_similarity_function(self):
        """Test semantic_similarity convenience function."""
        text1 = "LANEIGE products"
        text2 = "LANEIGE beauty items"
        similarity = semantic_similarity(text1, text2)

        assert 0 <= similarity <= 1

    def test_answer_semantic_similarity_with_gold(self):
        """Test answer_semantic_similarity with gold answer."""
        answer = "LANEIGE의 Lip Care 점유율은 5.2%입니다."
        gold = "LANEIGE Lip Care 카테고리 Share of Shelf는 약 5.2%입니다."

        similarity = answer_semantic_similarity(answer, gold)

        assert similarity is not None
        assert 0 <= similarity <= 1

    def test_answer_semantic_similarity_no_gold(self):
        """Test answer_semantic_similarity without gold answer."""
        answer = "Some answer"
        similarity = answer_semantic_similarity(answer, None)

        assert similarity is None

    def test_answer_semantic_similarity_empty_gold(self):
        """Test answer_semantic_similarity with empty gold answer."""
        answer = "Some answer"
        similarity = answer_semantic_similarity(answer, "")

        assert similarity is None


class TestL5Integration:
    """Tests for L5 metrics integration with semantic similarity."""

    def test_l5_metrics_with_semantic(self):
        """Test L5AnswerMetrics with semantic similarity enabled."""
        from eval.metrics.l5_answer import L5AnswerMetrics
        from eval.schemas import AnswerTrace, GoldEvidence

        calc = L5AnswerMetrics(use_semantic_similarity=True)

        trace = AnswerTrace(final_answer="LANEIGE의 Lip Care 점유율은 약 5%입니다.")
        gold = GoldEvidence(answer="LANEIGE Lip Care 카테고리 SoS는 5.2%입니다.")

        metrics = calc.compute_sync(trace, gold)

        assert metrics.semantic_similarity is not None
        assert 0 <= metrics.semantic_similarity <= 1

    def test_l5_metrics_without_semantic(self):
        """Test L5AnswerMetrics without semantic similarity."""
        from eval.metrics.l5_answer import L5AnswerMetrics
        from eval.schemas import AnswerTrace, GoldEvidence

        calc = L5AnswerMetrics(use_semantic_similarity=False)

        trace = AnswerTrace(final_answer="Some answer")
        gold = GoldEvidence(answer="Gold answer")

        metrics = calc.compute_sync(trace, gold)

        assert metrics.semantic_similarity is None

    def test_l5_metrics_no_gold_answer(self):
        """Test L5AnswerMetrics with no gold answer."""
        from eval.metrics.l5_answer import L5AnswerMetrics
        from eval.schemas import AnswerTrace, GoldEvidence

        calc = L5AnswerMetrics(use_semantic_similarity=True)

        trace = AnswerTrace(final_answer="Some answer")
        gold = GoldEvidence(answer=None)

        metrics = calc.compute_sync(trace, gold)

        # Should be None when no gold answer
        assert metrics.semantic_similarity is None


class TestComputeAnswerQuality:
    """Tests for compute_answer_quality function."""

    def test_with_semantic_similarity(self):
        """Test compute_answer_quality with semantic similarity."""
        from eval.metrics.l5_answer import compute_answer_quality

        metrics = compute_answer_quality(
            answer="LANEIGE Lip Care 점유율 5%",
            gold_answer="라네즈 립케어 시장점유율 5퍼센트",
            use_semantic_similarity=True,
        )

        assert "exact_match" in metrics
        assert "token_f1" in metrics
        assert "semantic_similarity" in metrics
        assert 0 <= metrics["semantic_similarity"] <= 1

    def test_without_semantic_similarity(self):
        """Test compute_answer_quality without semantic similarity."""
        from eval.metrics.l5_answer import compute_answer_quality

        metrics = compute_answer_quality(
            answer="Some answer",
            gold_answer="Gold answer",
            use_semantic_similarity=False,
        )

        assert "exact_match" in metrics
        assert "token_f1" in metrics
        assert "semantic_similarity" not in metrics

    def test_with_context_overlap(self):
        """Test compute_answer_quality with context."""
        from eval.metrics.l5_answer import compute_answer_quality

        metrics = compute_answer_quality(
            answer="LANEIGE Lip Care is popular",
            gold_answer="LANEIGE Lip Care 인기 제품",
            context="LANEIGE is a Korean brand known for Lip Care products.",
            use_semantic_similarity=True,
        )

        assert "context_overlap" in metrics
        assert 0 <= metrics["context_overlap"] <= 1

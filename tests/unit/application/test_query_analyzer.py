"""
Tests for QueryAnalyzer Service
================================
Tests query complexity analysis and intent detection.
"""

import pytest

from src.application.services.query_analyzer import (
    ComplexityLevel,
    QueryAnalyzer,
    QueryIntent,
)


class TestQueryAnalyzer:
    """Test suite for QueryAnalyzer"""

    @pytest.fixture
    def analyzer(self):
        """Create QueryAnalyzer instance"""
        return QueryAnalyzer()

    # Complexity Analysis Tests
    def test_simple_query_complexity(self, analyzer):
        """Test simple query detection"""
        query = "LANEIGE 순위 알려줘"
        complexity = analyzer.analyze_complexity(query)
        assert complexity == ComplexityLevel.SIMPLE

    def test_moderate_query_complexity(self, analyzer):
        """Test moderate query detection"""
        query = "LANEIGE와 Burt's Bees 비교해줘"
        complexity = analyzer.analyze_complexity(query)
        assert complexity == ComplexityLevel.MODERATE

    def test_complex_query_with_analysis_keyword(self, analyzer):
        """Test complex query detection with analysis keywords"""
        query = "LANEIGE의 경쟁력을 분석해주고 전략을 제안해줘"
        complexity = analyzer.analyze_complexity(query)
        assert complexity == ComplexityLevel.COMPLEX

    def test_complex_query_with_multiple_questions(self, analyzer):
        """Test complex query with multiple questions"""
        query = "LANEIGE 순위가 어떻게 되고, SoS는 어떻게 변했고, 경쟁사 대비 어떤 위치인가?"
        complexity = analyzer.analyze_complexity(query)
        assert complexity == ComplexityLevel.COMPLEX

    def test_complex_query_with_multistep_reasoning(self, analyzer):
        """Test complex query requiring multi-step reasoning"""
        query = "경쟁사 분석 후 LANEIGE 포지셔닝 전략 수립"
        complexity = analyzer.analyze_complexity(query)
        assert complexity == ComplexityLevel.COMPLEX

    # Intent Detection Tests
    def test_rank_query_intent(self, analyzer):
        """Test rank query intent detection"""
        query = "LANEIGE 순위 알려줘"
        intent = analyzer.detect_intent(query)
        assert intent == QueryIntent.RANK_QUERY

    def test_metric_query_intent(self, analyzer):
        """Test metric query intent detection"""
        query = "LANEIGE SoS 얼마야?"
        intent = analyzer.detect_intent(query)
        assert intent == QueryIntent.METRIC_QUERY

    def test_comparison_intent(self, analyzer):
        """Test comparison intent detection"""
        query = "LANEIGE와 Burt's Bees 비교"
        intent = analyzer.detect_intent(query)
        assert intent == QueryIntent.COMPARISON

    def test_trend_analysis_intent(self, analyzer):
        """Test trend analysis intent detection"""
        query = "LANEIGE 추세 분석해줘"
        intent = analyzer.detect_intent(query)
        assert intent == QueryIntent.TREND_ANALYSIS

    def test_recommendation_intent(self, analyzer):
        """Test recommendation intent detection"""
        query = "LANEIGE 전략 제안해줘"
        intent = analyzer.detect_intent(query)
        assert intent == QueryIntent.RECOMMENDATION

    def test_product_detail_intent(self, analyzer):
        """Test product detail intent detection"""
        query = "Lip Sleeping Mask 제품 정보"
        intent = analyzer.detect_intent(query)
        assert intent == QueryIntent.PRODUCT_DETAIL

    def test_general_question_intent(self, analyzer):
        """Test general question intent detection"""
        query = "안녕하세요"
        intent = analyzer.detect_intent(query)
        assert intent == QueryIntent.GENERAL

    # Edge Cases
    def test_empty_query(self, analyzer):
        """Test empty query handling"""
        complexity = analyzer.analyze_complexity("")
        intent = analyzer.detect_intent("")
        assert complexity == ComplexityLevel.SIMPLE
        assert intent == QueryIntent.GENERAL

    def test_very_long_query(self, analyzer):
        """Test very long query"""
        query = "LANEIGE " * 50  # 50 repetitions
        complexity = analyzer.analyze_complexity(query)
        assert complexity in [ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE]

    def test_mixed_language_query(self, analyzer):
        """Test query with mixed Korean/English"""
        query = "LANEIGE rank는 어떻게 되나요?"
        complexity = analyzer.analyze_complexity(query)
        intent = analyzer.detect_intent(query)
        assert complexity in [ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE]
        assert intent == QueryIntent.RANK_QUERY

    # Full Analysis Test
    def test_full_analysis(self, analyzer):
        """Test full analysis method"""
        query = "LANEIGE 경쟁력 분석해줘"
        result = analyzer.analyze(query)

        assert "complexity" in result
        assert "intent" in result
        assert "keywords" in result
        assert result["complexity"] in [
            ComplexityLevel.MODERATE,
            ComplexityLevel.COMPLEX,
        ]
        assert isinstance(result["keywords"], list)

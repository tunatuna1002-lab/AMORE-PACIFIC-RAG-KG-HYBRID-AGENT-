"""
Unified Intent Classification Tests
====================================
Tests for src/core/intent.py - the single source of truth for query intent.
"""

import pytest

from src.core.intent import (
    INTENT_DOC_TYPE_PRIORITY,
    UnifiedIntent,
    classify_intent,
    get_doc_type_filter,
    to_query_category,
    to_query_intent,
    to_query_type,
)

# ---------------------------------------------------------------------------
# UnifiedIntent enum
# ---------------------------------------------------------------------------


class TestUnifiedIntentEnum:
    """UnifiedIntent enum completeness tests."""

    def test_all_expected_values_exist(self):
        expected = {
            "diagnosis",
            "trend",
            "crisis",
            "metric",
            "general",
            "competitive",
            "definition",
            "interpretation",
            "combination",
            "insight_rule",
            "data_query",
            "analysis",
        }
        actual = {i.value for i in UnifiedIntent}
        assert actual == expected

    def test_enum_count(self):
        assert len(UnifiedIntent) == 12


# ---------------------------------------------------------------------------
# classify_intent - keyword groups
# ---------------------------------------------------------------------------


class TestClassifyIntentTrend:
    """TREND intent detection."""

    @pytest.mark.parametrize(
        "query",
        [
            "최근 트렌드가 뭐야",
            "요즘 인기 있는 제품",
            "바이럴된 키워드 알려줘",
            "펩타이드 트렌드 분석",
            "성분 분석",
            "pdrn 트렌드",
            "글래스스킨 유행",
            "추이 분석해줘",
            "성장 트렌드",
        ],
    )
    def test_trend_keywords(self, query):
        assert classify_intent(query) == UnifiedIntent.TREND


class TestClassifyIntentCrisis:
    """CRISIS intent detection."""

    @pytest.mark.parametrize(
        "query",
        [
            "부정 리뷰가 많은데 어떻게 해",
            "이슈 발생 시 대응",
            "위기 상황 대처법",
            "불만 사항 처리",
            "인플루언서 마케팅 필요해",
            "메시지 전략",
        ],
    )
    def test_crisis_keywords(self, query):
        assert classify_intent(query) == UnifiedIntent.CRISIS


class TestClassifyIntentDiagnosis:
    """DIAGNOSIS intent detection."""

    @pytest.mark.parametrize(
        "query",
        [
            "왜 순위가 떨어졌어",
            "원인 분석해줘",
            "갑자기 변동이 생긴 이유",
            "급변한 원인 확인",
            "진단 결과",
            "급등한 이유",
            "급락 원인",
        ],
    )
    def test_diagnosis_keywords(self, query):
        assert classify_intent(query) == UnifiedIntent.DIAGNOSIS


class TestClassifyIntentCompetitive:
    """COMPETITIVE intent detection."""

    @pytest.mark.parametrize(
        "query",
        [
            "경쟁사 분석",
            "COSRX vs LANEIGE 비교",
            "경쟁 브랜드 대비",
            "상대적 위치",
            "competitor analysis",
        ],
    )
    def test_competitive_keywords(self, query):
        assert classify_intent(query) == UnifiedIntent.COMPETITIVE


class TestClassifyIntentMetric:
    """METRIC intent detection."""

    @pytest.mark.parametrize(
        "query",
        [
            "SoS가 뭐야",
            "HHI 지표 해석",
            "CPI 의미 알려줘",
            "점유율 정의",
            "순위 알려줘",
            "share 수치",
        ],
    )
    def test_metric_keywords(self, query):
        assert classify_intent(query) == UnifiedIntent.METRIC


class TestClassifyIntentDefinition:
    """DEFINITION intent detection."""

    @pytest.mark.parametrize(
        "query",
        [
            "립케어가 뭐야",
            "산출식 알려줘",
        ],
    )
    def test_definition_keywords(self, query):
        assert classify_intent(query) == UnifiedIntent.DEFINITION

    def test_calculation_formula_maps_to_metric(self):
        """'계산식' contains '계산' which is METRIC (higher priority)."""
        assert classify_intent("계산식 설명") == UnifiedIntent.METRIC


class TestClassifyIntentGeneral:
    """GENERAL fallback intent."""

    @pytest.mark.parametrize(
        "query",
        [
            "LANEIGE 제품 보여줘",
            "브랜드 정보",
            "카테고리 현황",
            "안녕하세요",
            "hello",
        ],
    )
    def test_general_fallback(self, query):
        assert classify_intent(query) == UnifiedIntent.GENERAL


class TestClassifyIntentEdgeCases:
    """Edge cases and empty inputs."""

    def test_empty_query(self):
        assert classify_intent("") == UnifiedIntent.GENERAL

    def test_whitespace_only(self):
        assert classify_intent("   ") == UnifiedIntent.GENERAL


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


class TestIntentPriority:
    """Priority ordering: TREND > CRISIS > DIAGNOSIS > COMPETITIVE > METRIC."""

    def test_trend_over_diagnosis(self):
        """'분석' is a DIAGNOSIS keyword, but '트렌드' has higher priority."""
        assert classify_intent("최근 트렌드 분석해줘") == UnifiedIntent.TREND

    def test_trend_over_metric(self):
        """TREND keywords override METRIC keywords."""
        assert classify_intent("요즘 지표 트렌드") == UnifiedIntent.TREND

    def test_crisis_over_diagnosis(self):
        """CRISIS keywords override DIAGNOSIS keywords."""
        assert classify_intent("이슈 원인 파악") == UnifiedIntent.CRISIS

    def test_diagnosis_over_competitive(self):
        """DIAGNOSIS keywords override COMPETITIVE keywords."""
        assert classify_intent("왜 경쟁사 대비 떨어졌나") == UnifiedIntent.DIAGNOSIS

    def test_diagnosis_over_metric(self):
        """DIAGNOSIS keywords override METRIC keywords."""
        assert classify_intent("왜 SoS가 변동이 있나") == UnifiedIntent.DIAGNOSIS

    def test_competitive_over_metric(self):
        """COMPETITIVE keywords override METRIC keywords."""
        assert classify_intent("경쟁사 점유율 비교") == UnifiedIntent.COMPETITIVE


# ---------------------------------------------------------------------------
# Backward compatibility mappings
# ---------------------------------------------------------------------------


class TestToQueryIntent:
    """to_query_intent backward compat mapping."""

    def test_diagnosis_maps(self):
        assert to_query_intent(UnifiedIntent.DIAGNOSIS) == "diagnosis"

    def test_trend_maps(self):
        assert to_query_intent(UnifiedIntent.TREND) == "trend"

    def test_crisis_maps(self):
        assert to_query_intent(UnifiedIntent.CRISIS) == "crisis"

    def test_metric_maps(self):
        assert to_query_intent(UnifiedIntent.METRIC) == "metric"

    def test_general_maps(self):
        assert to_query_intent(UnifiedIntent.GENERAL) == "general"

    def test_competitive_maps_to_general(self):
        """COMPETITIVE has no old QueryIntent equivalent -> general."""
        assert to_query_intent(UnifiedIntent.COMPETITIVE) == "general"

    def test_definition_maps_to_metric(self):
        assert to_query_intent(UnifiedIntent.DEFINITION) == "metric"

    def test_all_intents_covered(self):
        """Every UnifiedIntent has a mapping."""
        for intent in UnifiedIntent:
            result = to_query_intent(intent)
            assert isinstance(result, str)
            assert len(result) > 0


class TestToQueryCategory:
    """to_query_category backward compat mapping."""

    def test_metric_maps(self):
        assert to_query_category(UnifiedIntent.METRIC) == "metric"

    def test_trend_maps(self):
        assert to_query_category(UnifiedIntent.TREND) == "trend"

    def test_competitive_maps(self):
        assert to_query_category(UnifiedIntent.COMPETITIVE) == "competitive"

    def test_diagnosis_maps_to_diagnostic(self):
        assert to_query_category(UnifiedIntent.DIAGNOSIS) == "diagnostic"

    def test_general_maps(self):
        assert to_query_category(UnifiedIntent.GENERAL) == "general"

    def test_all_intents_covered(self):
        for intent in UnifiedIntent:
            result = to_query_category(intent)
            assert isinstance(result, str)


class TestToQueryType:
    """to_query_type backward compat mapping."""

    def test_definition_maps(self):
        assert to_query_type(UnifiedIntent.DEFINITION) == "definition"

    def test_interpretation_maps(self):
        assert to_query_type(UnifiedIntent.INTERPRETATION) == "interpretation"

    def test_analysis_maps(self):
        assert to_query_type(UnifiedIntent.ANALYSIS) == "analysis"

    def test_diagnosis_maps_to_unknown(self):
        """DIAGNOSIS has no old QueryType equivalent -> unknown."""
        assert to_query_type(UnifiedIntent.DIAGNOSIS) == "unknown"

    def test_all_intents_covered(self):
        for intent in UnifiedIntent:
            result = to_query_type(intent)
            assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Doc type priority
# ---------------------------------------------------------------------------


class TestDocTypePriority:
    """INTENT_DOC_TYPE_PRIORITY mapping tests."""

    def test_all_intents_have_mapping(self):
        for intent in UnifiedIntent:
            assert intent in INTENT_DOC_TYPE_PRIORITY

    def test_diagnosis_priority(self):
        doc_types = INTENT_DOC_TYPE_PRIORITY[UnifiedIntent.DIAGNOSIS]
        assert "playbook" in doc_types
        assert "metric_guide" in doc_types

    def test_trend_priority(self):
        doc_types = INTENT_DOC_TYPE_PRIORITY[UnifiedIntent.TREND]
        assert "intelligence" in doc_types
        assert "knowledge_base" in doc_types

    def test_crisis_priority(self):
        doc_types = INTENT_DOC_TYPE_PRIORITY[UnifiedIntent.CRISIS]
        assert "response_guide" in doc_types

    def test_general_returns_none(self):
        assert INTENT_DOC_TYPE_PRIORITY[UnifiedIntent.GENERAL] is None

    def test_data_query_returns_none(self):
        assert INTENT_DOC_TYPE_PRIORITY[UnifiedIntent.DATA_QUERY] is None


class TestGetDocTypeFilter:
    """get_doc_type_filter function tests."""

    def test_diagnosis_filter(self):
        result = get_doc_type_filter(UnifiedIntent.DIAGNOSIS)
        assert isinstance(result, list)
        assert "playbook" in result

    def test_general_filter_is_none(self):
        assert get_doc_type_filter(UnifiedIntent.GENERAL) is None

    def test_trend_filter(self):
        result = get_doc_type_filter(UnifiedIntent.TREND)
        assert isinstance(result, list)
        assert "intelligence" in result

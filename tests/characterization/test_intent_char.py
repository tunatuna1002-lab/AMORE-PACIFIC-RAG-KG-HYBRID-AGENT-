"""
Characterization: intent classification entry points.

Pins the enum values currently returned by
- src.core.intent.classify_intent (+ legacy mapping helpers)
- src.rag.router.RAGRouter.route
- src.rag.retrieval_strategy.get_intent_retrieval_config
for a fixed set of Korean/English queries. No collaborators, pure functions.
"""

from __future__ import annotations

import pytest

from src.core.intent import (
    UnifiedIntent,
    classify_intent,
    get_doc_type_filter,
    to_query_category,
    to_query_intent,
    to_query_type,
)
from src.rag.retrieval_strategy import get_intent_retrieval_config
from src.rag.router import QueryType, RAGRouter

# ---------------------------------------------------------------------------
# classify_intent: first keyword group (by priority) that matches wins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("립케어 순위", UnifiedIntent.METRIC),
        ("LANEIGE SoS 왜 떨어졌어", UnifiedIntent.DIAGNOSIS),
        # "sos" (METRIC group) outranks "뭐야" (DEFINITION group) - so a
        # definition question is classified as METRIC.
        ("SoS가 뭐야", UnifiedIntent.METRIC),
        ("경쟁사 비교해줘", UnifiedIntent.COMPETITIVE),
        ("트렌드 분석", UnifiedIntent.TREND),
        ("안녕", UnifiedIntent.GENERAL),
        # "경쟁" substring of "경쟁력" -> COMPETITIVE (no price intent exists)
        ("가격 경쟁력", UnifiedIntent.COMPETITIVE),
        # "hhi" (METRIC) outranks "높으면" (INTERPRETATION)
        ("HHI가 높으면 어떤 의미야", UnifiedIntent.METRIC),
        ("라네즈 현재 순위", UnifiedIntent.METRIC),
        ("인사이트 요약 문구 생성해줘", UnifiedIntent.INSIGHT_RULE),
        ("CPI 높고 평점 낮으면?", UnifiedIntent.METRIC),
        ("LANEIGE 전략 제언 보고서", UnifiedIntent.ANALYSIS),
        ("리뷰 부정 이슈 대응", UnifiedIntent.CRISIS),
        ("hello", UnifiedIntent.GENERAL),
        # IR vocabulary (매출/실적/분기) is unknown to the unified classifier
        ("3분기 매출 실적", UnifiedIntent.GENERAL),
        ("Lip Care 카테고리 Top 브랜드", UnifiedIntent.GENERAL),
        ("COSRX vs LANEIGE", UnifiedIntent.COMPETITIVE),
        ("펩타이드 성분 인기", UnifiedIntent.TREND),
    ],
)
def test_classify_intent_pins_enum(query: str, expected: UnifiedIntent) -> None:
    assert classify_intent(query) is expected


def test_classify_intent_is_case_insensitive() -> None:
    assert classify_intent("SOS") is UnifiedIntent.METRIC
    assert classify_intent("sos") is UnifiedIntent.METRIC


def test_classify_intent_empty_query_is_general() -> None:
    assert classify_intent("") is UnifiedIntent.GENERAL


# ---------------------------------------------------------------------------
# Legacy mapping helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("intent", "query_intent", "query_category", "query_type"),
    [
        (UnifiedIntent.METRIC, "metric", "metric", "unknown"),
        (UnifiedIntent.TREND, "trend", "trend", "unknown"),
        (UnifiedIntent.DIAGNOSIS, "diagnosis", "diagnostic", "unknown"),
        (UnifiedIntent.COMPETITIVE, "general", "competitive", "unknown"),
        (UnifiedIntent.CRISIS, "crisis", "general", "unknown"),
        (UnifiedIntent.DEFINITION, "metric", "metric", "definition"),
        (UnifiedIntent.INTERPRETATION, "metric", "metric", "interpretation"),
        (UnifiedIntent.DATA_QUERY, "general", "metric", "data_query"),
        (UnifiedIntent.ANALYSIS, "diagnosis", "diagnostic", "analysis"),
        (UnifiedIntent.INSIGHT_RULE, "general", "general", "insight_rule"),
        (UnifiedIntent.GENERAL, "general", "general", "unknown"),
    ],
)
def test_legacy_mappings(
    intent: UnifiedIntent, query_intent: str, query_category: str, query_type: str
) -> None:
    assert to_query_intent(intent) == query_intent
    assert to_query_category(intent) == query_category
    assert to_query_type(intent) == query_type


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        (UnifiedIntent.METRIC, ["metric_guide", "playbook"]),
        (UnifiedIntent.TREND, ["intelligence", "knowledge_base", "response_guide"]),
        (UnifiedIntent.DIAGNOSIS, ["playbook", "metric_guide", "intelligence"]),
        (UnifiedIntent.CRISIS, ["response_guide", "intelligence", "playbook"]),
        (UnifiedIntent.GENERAL, None),
        (UnifiedIntent.DATA_QUERY, None),
    ],
)
def test_get_doc_type_filter(intent: UnifiedIntent, expected: list[str] | None) -> None:
    assert get_doc_type_filter(intent) == expected


# ---------------------------------------------------------------------------
# RAGRouter.route (sync) - independent scoring scheme from classify_intent
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def router() -> RAGRouter:
    return RAGRouter()


@pytest.mark.parametrize(
    ("query", "query_type", "confidence", "max_score", "has_fallback"),
    [
        ("립케어 순위", QueryType.DATA_QUERY, 1.0, 2.0, False),
        # Only the "sos" indicator matches (1.5 == threshold) and it is shared by
        # DEFINITION/INTERPRETATION; dict order makes DEFINITION win with 1/3 confidence.
        ("LANEIGE SoS 왜 떨어졌어", QueryType.DEFINITION, 1 / 3, 1.5, False),
        ("SoS가 뭐야", QueryType.DEFINITION, 0.7, 3.5, False),
        ("경쟁사 비교해줘", QueryType.ANALYSIS, 1.0, 3.0, False),
        ("트렌드 분석", QueryType.ANALYSIS, 1.0, 3.0, False),
        ("안녕", QueryType.UNKNOWN, 0.0, 0.0, True),
        ("가격 경쟁력", QueryType.UNKNOWN, 0.0, 0.0, True),
        ("HHI가 높으면 어떤 의미야", QueryType.INTERPRETATION, 5.5 / 7.0, 5.5, False),
        ("라네즈 현재 순위", QueryType.DATA_QUERY, 1.0, 5.5, False),
        ("인사이트 요약 문구 생성해줘", QueryType.INSIGHT_RULE, 1.0, 8.0, False),
        ("CPI 높고 평점 낮으면?", QueryType.INTERPRETATION, 3.5 / 6.0, 3.5, False),
        ("LANEIGE 전략 제언 보고서", QueryType.ANALYSIS, 0.8, 6.0, False),
        # CRISIS vocabulary is unknown to the router -> fallback message
        ("리뷰 부정 이슈 대응", QueryType.UNKNOWN, 0.0, 0.0, True),
        ("hello", QueryType.UNKNOWN, 0.0, 0.0, True),
        ("3분기 매출 실적", QueryType.DATA_QUERY, 1.0, 3.0, False),
        ("Lip Care 카테고리 Top 브랜드", QueryType.DATA_QUERY, 1.0, 3.0, False),
        # "laneige" entity (1.5) beats the "vs" pattern (1.0) -> DATA_QUERY, not ANALYSIS
        ("COSRX vs LANEIGE", QueryType.DATA_QUERY, 0.6, 1.5, False),
        ("펩타이드 성분 인기", QueryType.UNKNOWN, 0.0, 0.0, True),
    ],
)
def test_rag_router_route_pins_query_type(
    router: RAGRouter,
    query: str,
    query_type: QueryType,
    confidence: float,
    max_score: float,
    has_fallback: bool,
) -> None:
    result = router.route(query)
    assert result["query_type"] is query_type
    assert result["confidence"] == pytest.approx(confidence)
    assert result["max_score"] == pytest.approx(max_score)
    assert (result["fallback_message"] is not None) is has_fallback
    if has_fallback:
        assert result["fallback_message"].startswith("질문의 의도를 정확히 파악하지 못했습니다.")


def test_rag_router_route_result_shape(router: RAGRouter) -> None:
    result = router.route("SoS가 뭐야")
    assert set(result) == {
        "query_type",
        "confidence",
        "max_score",
        "matched_keywords",
        "target_doc",
        "requires_data",
        "requires_rag",
        "fallback_message",
    }
    assert result["target_doc"] == "strategic_indicators"
    assert result["requires_data"] is False
    assert result["requires_rag"] is True
    assert set(result["matched_keywords"]) == {"뭐야", "sos"}


def test_rag_router_data_query_requires_data_not_rag(router: RAGRouter) -> None:
    result = router.route("라네즈 현재 순위")
    assert result["requires_data"] is True
    assert result["requires_rag"] is False
    assert result["target_doc"] is None


# ---------------------------------------------------------------------------
# get_intent_retrieval_config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("intent", "weights", "top_k", "doc_type_filter", "description", "fusion"),
    [
        (
            UnifiedIntent.METRIC,
            {"kg": 0.4, "rag": 0.4, "inference": 0.2},
            5,
            ["metric_guide", "playbook"],
            "balanced/metric",
            "weighted_sum",
        ),
        (
            UnifiedIntent.TREND,
            {"kg": 0.35, "rag": 0.35, "inference": 0.3},
            7,
            ["intelligence", "knowledge_base", "response_guide"],
            "hybrid/trend",
            "harmonic_mean",
        ),
        (
            UnifiedIntent.DIAGNOSIS,
            {"kg": 0.5, "rag": 0.3, "inference": 0.2},
            5,
            ["playbook", "metric_guide", "intelligence"],
            "graph-heavy/diagnosis",
            "weighted_sum",
        ),
    ],
)
def test_get_intent_retrieval_config(
    intent: UnifiedIntent,
    weights: dict[str, float],
    top_k: int,
    doc_type_filter: list[str],
    description: str,
    fusion: str,
) -> None:
    cfg = get_intent_retrieval_config(intent)
    assert cfg.weights == weights
    assert cfg.top_k == top_k
    assert cfg.doc_type_filter == doc_type_filter
    assert cfg.description == description
    assert cfg.fusion_strategy == fusion
    # Strategy doc_type_filter mirrors the unified doc-type priority table.
    assert cfg.doc_type_filter == get_doc_type_filter(intent)


def test_retrieval_config_is_frozen() -> None:
    cfg = get_intent_retrieval_config(UnifiedIntent.METRIC)
    with pytest.raises(AttributeError):
        cfg.top_k = 99  # type: ignore[misc]

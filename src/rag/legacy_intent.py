"""Legacy 5-value query intent (backward-compatibility shim)
=========================================================

The single definition of query intent is ``src.core.intent.UnifiedIntent``.
This module is the pre-unification 5-value projection of it: the enum, its
doc-type table and the two helpers.

Nothing under ``src/`` reads these names — ``HybridRetriever`` runs entirely on
``UnifiedIntent`` and only records ``QueryIntent.value`` in its metadata, and
the retrieval doc-type filter comes from ``retrieval_strategy``. They survive
because ``tests/integration/test_rag_integration.py`` still imports them from
``src.rag.hybrid_retriever``, which re-exports them.

Delete this module once that import is updated to ``src.core.intent``.

Moved verbatim out of ``hybrid_retriever.py`` (F3 split).
"""

from __future__ import annotations

from enum import Enum

from src.core.intent import classify_intent as _unified_classify
from src.core.intent import to_query_intent as _to_query_intent


class QueryIntent(Enum):
    """쿼리 의도 분류 (backward compat - delegates to UnifiedIntent)"""

    DIAGNOSIS = "diagnosis"  # 원인 분석 → Type A (플레이북) 우선
    TREND = "trend"  # 트렌드 → Type B (인텔리전스) 우선
    CRISIS = "crisis"  # 위기 대응 → Type C (대응 가이드) 우선
    METRIC = "metric"  # 지표 해석 → Type D (기존 가이드) 우선
    GENERAL = "general"  # 일반 → 모든 문서


# 의도별 우선 검색 문서 유형 매핑 — src.core.intent.INTENT_DOC_TYPE_PRIORITY의
# 5값 투영. 실제 검색 필터는 retrieval_strategy의 인텐트 설정에서 나온다.
INTENT_DOC_TYPE_PRIORITY = {
    QueryIntent.DIAGNOSIS: ["playbook", "metric_guide", "intelligence"],
    QueryIntent.TREND: ["intelligence", "knowledge_base", "response_guide"],
    QueryIntent.CRISIS: ["response_guide", "intelligence", "playbook"],
    QueryIntent.METRIC: ["metric_guide", "playbook"],
    QueryIntent.GENERAL: None,  # 모든 문서 검색
}


def classify_intent(query: str) -> QueryIntent:
    """
    쿼리 의도 분류 - delegates to unified classifier.

    Args:
        query: 사용자 쿼리

    Returns:
        QueryIntent enum 값

    Note:
        키워드 우선순위: TREND > CRISIS > DIAGNOSIS > METRIC > GENERAL
        트렌드/위기 키워드가 있으면 분석 키워드보다 우선
    """
    value = _to_query_intent(_unified_classify(query))
    try:
        return QueryIntent(value)
    except ValueError:
        return QueryIntent.GENERAL


def get_doc_type_filter(intent: QueryIntent) -> list[str] | None:
    """
    의도에 따른 문서 유형 필터 반환

    Args:
        intent: 쿼리 의도

    Returns:
        우선 검색할 문서 유형 리스트 (None이면 모든 문서)
    """
    return INTENT_DOC_TYPE_PRIORITY.get(intent)

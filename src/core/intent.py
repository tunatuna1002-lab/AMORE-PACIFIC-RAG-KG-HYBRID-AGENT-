"""
Unified Intent Classification
==============================
Single source of truth for query intent classification.
Merges keyword-based classification from:
  - src/rag/hybrid_retriever.py (QueryIntent)
  - src/core/query_router.py (QueryCategory)
  - src/rag/router.py (QueryType - document-level intents)

Decision_maker.py and hybrid_chatbot_agent.py are excluded:
decision_maker uses LLM-first classification, chatbot delegates to RAGRouter.
"""

from __future__ import annotations

from enum import Enum


class UnifiedIntent(Enum):
    """Unified query intent - superset of all previous keyword-based enums."""

    # From QueryIntent (hybrid_retriever) - coarse intents
    DIAGNOSIS = "diagnosis"
    TREND = "trend"
    CRISIS = "crisis"
    METRIC = "metric"
    GENERAL = "general"
    # From QueryCategory (query_router) - additional coarse intents
    COMPETITIVE = "competitive"
    # From QueryType (router) - document-level intents
    DEFINITION = "definition"
    INTERPRETATION = "interpretation"
    COMBINATION = "combination"
    INSIGHT_RULE = "insight_rule"
    DATA_QUERY = "data_query"
    ANALYSIS = "analysis"


# ---------------------------------------------------------------------------
# Merged keyword maps
# ---------------------------------------------------------------------------
# Priority ordering (highest first):
#   TREND > CRISIS > DIAGNOSIS > COMPETITIVE > METRIC >
#   DEFINITION > INTERPRETATION > COMBINATION > INSIGHT_RULE >
#   DATA_QUERY > ANALYSIS > GENERAL

_INTENT_KEYWORDS: list[tuple[UnifiedIntent, list[str]]] = [
    # 1. TREND (from hybrid_retriever trend_keywords + query_router TREND)
    (
        UnifiedIntent.TREND,
        [
            "트렌드",
            "요즘",
            "최근",
            "인기",
            "바이럴",
            "키워드",
            "성분",
            "펩타이드",
            "pdrn",
            "글래스스킨",
            "모닝쉐드",
            # from query_router TREND
            "추이",
            "변화",
            "추세",
            "증가",
            "감소",
            "상승",
            "하락",
            "성장",
        ],
    ),
    # 2. CRISIS (from hybrid_retriever crisis_keywords)
    (
        UnifiedIntent.CRISIS,
        [
            "부정",
            "문제",
            "이슈",
            "대응",
            "어떻게 해",
            "위기",
            "리뷰",
            "불만",
            "인플루언서",
            "마케팅",
            "메시지",
        ],
    ),
    # 3. DIAGNOSIS (from hybrid_retriever diagnosis_keywords + query_router DIAGNOSTIC)
    (
        UnifiedIntent.DIAGNOSIS,
        [
            "왜",
            "원인",
            "갑자기",
            "급변",
            "떨어",
            "올라",
            "변동",
            "이유",
            "진단",
            "체크",
            "확인",
            # from query_router DIAGNOSTIC extras
            "어떻게",
            "급등",
            "급락",
        ],
    ),
    # 4. COMPETITIVE (from query_router only)
    (
        UnifiedIntent.COMPETITIVE,
        [
            "경쟁",
            "비교",
            "대비",
            "차이",
            "경쟁사",
            "competitor",
            "versus",
            "vs",
            "상대적",
        ],
    ),
    # 5. METRIC (from hybrid_retriever metric_keywords + query_router METRIC)
    (
        UnifiedIntent.METRIC,
        [
            "sos",
            "hhi",
            "cpi",
            "지표",
            "점유율",
            "해석",
            "의미",
            "정의",
            "공식",
            "계산",
            # from query_router METRIC extras
            "순위",
            "rank",
            "share",
            "수치",
            "퍼센트",
            "%",
            "몇",
            "얼마",
        ],
    ),
    # 6. DEFINITION (from router.py DEFINITION)
    (
        UnifiedIntent.DEFINITION,
        [
            "뭐야",
            "무엇",
            "산출식",
            "계산식",
            "어떻게 계산",
        ],
    ),
    # 7. INTERPRETATION (from router.py INTERPRETATION)
    (
        UnifiedIntent.INTERPRETATION,
        [
            "높으면",
            "낮으면",
            "어떻게 봐야",
            "이해",
        ],
    ),
    # 8. COMBINATION (from router.py COMBINATION)
    (
        UnifiedIntent.COMBINATION,
        [
            "조합",
            "같이",
            "동시에",
            "결합",
            "복합",
            "시나리오",
        ],
    ),
    # 9. INSIGHT_RULE (from router.py INSIGHT_RULE)
    (
        UnifiedIntent.INSIGHT_RULE,
        [
            "인사이트",
            "요약",
            "문구",
            "생성",
            "템플릿",
            "톤",
            "표현",
            "어떻게 쓰",
            "어떻게 표현",
            "문장",
        ],
    ),
    # 10. DATA_QUERY (from router.py DATA_QUERY)
    (
        UnifiedIntent.DATA_QUERY,
        [
            "현재",
            "오늘",
            "어제",
            "지금",
            "랭킹",
        ],
    ),
    # 11. ANALYSIS (from router.py ANALYSIS)
    (
        UnifiedIntent.ANALYSIS,
        [
            "전략",
            "제언",
            "보고서",
            "리포트",
            "평가",
        ],
    ),
]

# NOTE: Some keywords overlap across old systems (e.g. "분석" in diagnosis_keywords,
# DIAGNOSTIC, and ANALYSIS). Here we resolve by priority: the first matching group wins.
# Keywords that appeared in multiple old systems are placed in the highest-priority
# group where they originally appeared. "분석" → DIAGNOSIS (from hybrid_retriever),
# "문제" → CRISIS (from hybrid_retriever), "리뷰" → CRISIS,
# "최근" → TREND (from hybrid_retriever; also DATA_QUERY in router.py).
# "변화" → TREND (from query_router TREND; also overlaps DIAGNOSIS "변동").
# "증가"/"감소"/"상승"/"하락" → TREND (from query_router; also COMBINATION patterns).


def classify_intent(query: str) -> UnifiedIntent:
    """
    Single entry point for intent classification.

    Merges keyword maps from all 3 keyword-based systems.
    Priority: TREND > CRISIS > DIAGNOSIS > COMPETITIVE > METRIC >
              DEFINITION > INTERPRETATION > COMBINATION > INSIGHT_RULE >
              DATA_QUERY > ANALYSIS > GENERAL

    Args:
        query: user query string

    Returns:
        UnifiedIntent enum value
    """
    query_lower = query.lower()

    for intent, keywords in _INTENT_KEYWORDS:
        if any(kw in query_lower for kw in keywords):
            return intent

    return UnifiedIntent.GENERAL


# ---------------------------------------------------------------------------
# Backward-compatibility mappings
# ---------------------------------------------------------------------------

# UnifiedIntent -> legacy QueryIntent value string
_INTENT_TO_QUERY_INTENT: dict[UnifiedIntent, str] = {
    UnifiedIntent.DIAGNOSIS: "diagnosis",
    UnifiedIntent.TREND: "trend",
    UnifiedIntent.CRISIS: "crisis",
    UnifiedIntent.METRIC: "metric",
    UnifiedIntent.GENERAL: "general",
    # intents that don't exist in old QueryIntent → fallback to "general"
    UnifiedIntent.COMPETITIVE: "general",
    UnifiedIntent.DEFINITION: "metric",
    UnifiedIntent.INTERPRETATION: "metric",
    UnifiedIntent.COMBINATION: "general",
    UnifiedIntent.INSIGHT_RULE: "general",
    UnifiedIntent.DATA_QUERY: "general",
    UnifiedIntent.ANALYSIS: "diagnosis",
}

# UnifiedIntent -> legacy QueryCategory value string
_INTENT_TO_QUERY_CATEGORY: dict[UnifiedIntent, str] = {
    UnifiedIntent.METRIC: "metric",
    UnifiedIntent.TREND: "trend",
    UnifiedIntent.COMPETITIVE: "competitive",
    UnifiedIntent.DIAGNOSIS: "diagnostic",
    UnifiedIntent.GENERAL: "general",
    # intents that don't exist in old QueryCategory → best fit
    UnifiedIntent.CRISIS: "general",
    UnifiedIntent.DEFINITION: "metric",
    UnifiedIntent.INTERPRETATION: "metric",
    UnifiedIntent.COMBINATION: "general",
    UnifiedIntent.INSIGHT_RULE: "general",
    UnifiedIntent.DATA_QUERY: "metric",
    UnifiedIntent.ANALYSIS: "diagnostic",
}

# UnifiedIntent -> legacy QueryType value string
_INTENT_TO_QUERY_TYPE: dict[UnifiedIntent, str] = {
    UnifiedIntent.DEFINITION: "definition",
    UnifiedIntent.INTERPRETATION: "interpretation",
    UnifiedIntent.COMBINATION: "combination",
    UnifiedIntent.INSIGHT_RULE: "insight_rule",
    UnifiedIntent.DATA_QUERY: "data_query",
    UnifiedIntent.ANALYSIS: "analysis",
    # intents that don't exist in old QueryType → "unknown"
    UnifiedIntent.DIAGNOSIS: "unknown",
    UnifiedIntent.TREND: "unknown",
    UnifiedIntent.CRISIS: "unknown",
    UnifiedIntent.METRIC: "unknown",
    UnifiedIntent.GENERAL: "unknown",
    UnifiedIntent.COMPETITIVE: "unknown",
}


def to_query_intent(intent: UnifiedIntent) -> str:
    """Map UnifiedIntent to legacy QueryIntent value for backward compat."""
    return _INTENT_TO_QUERY_INTENT.get(intent, "general")


def to_query_category(intent: UnifiedIntent) -> str:
    """Map UnifiedIntent to legacy QueryCategory value for backward compat."""
    return _INTENT_TO_QUERY_CATEGORY.get(intent, "general")


def to_query_type(intent: UnifiedIntent) -> str:
    """Map UnifiedIntent to legacy QueryType value for backward compat."""
    return _INTENT_TO_QUERY_TYPE.get(intent, "unknown")


# ---------------------------------------------------------------------------
# Doc type priority (moved from hybrid_retriever.py)
# ---------------------------------------------------------------------------

INTENT_DOC_TYPE_PRIORITY: dict[UnifiedIntent, list[str] | None] = {
    UnifiedIntent.DIAGNOSIS: ["playbook", "metric_guide", "intelligence"],
    UnifiedIntent.TREND: ["intelligence", "knowledge_base", "response_guide"],
    UnifiedIntent.CRISIS: ["response_guide", "intelligence", "playbook"],
    UnifiedIntent.METRIC: ["metric_guide", "playbook"],
    UnifiedIntent.GENERAL: None,
    # Additional intents - sensible defaults
    UnifiedIntent.COMPETITIVE: ["intelligence", "playbook"],
    UnifiedIntent.DEFINITION: ["metric_guide", "playbook"],
    UnifiedIntent.INTERPRETATION: ["metric_guide", "playbook"],
    UnifiedIntent.COMBINATION: ["playbook", "metric_guide"],
    UnifiedIntent.INSIGHT_RULE: ["intelligence", "knowledge_base"],
    UnifiedIntent.DATA_QUERY: None,
    UnifiedIntent.ANALYSIS: ["intelligence", "playbook", "metric_guide"],
}


def get_doc_type_filter(intent: UnifiedIntent) -> list[str] | None:
    """
    Return document-type filter list for intent-based retrieval.

    Args:
        intent: unified intent

    Returns:
        Priority document type list, or None for all documents.
    """
    return INTENT_DOC_TYPE_PRIORITY.get(intent)

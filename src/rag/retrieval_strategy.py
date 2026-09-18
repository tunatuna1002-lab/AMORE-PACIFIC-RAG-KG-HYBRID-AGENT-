"""
Intent-Based Retrieval Configuration
====================================
검색 경로는 `HybridRetriever` 하나뿐이다. 이 모듈은 인텐트별 검색 설정만 제공한다.

Intent-Based Configuration:
    UnifiedIntent (from src.core.intent) is mapped to IntentRetrievalConfig,
    which tunes retrieval weights (kg / rag / inference), top_k, and
    document-type filters per intent category.

2026-09: OWL 검색 전략(`OWLRetrievalStrategy`·`create_owl_strategy`·`RetrievalStrategy`
프로토콜)과 플래그 `retriever.use_owl_strategy`를 삭제했다. 온톨로지 신호(엔티티·카테고리
일치)는 legacy 경로의 재순위 보너스로 표현한다.

[2026-09 사후 정정, 트랙 O6] 예전 문구 "OWL은 어휘(카테고리 계층·일관성 검사)로만 쓴다"는
사실과 달랐다. 카테고리 계층은 `config/category_hierarchy.json`에서 온다. 온톨로지 원본은
JSON(`config/ontology/`) + 로더 `src/ontology/ontology.py`(런타임은 Python 폐포, Pellet 교차
검증은 개발 전용 `scripts/check_ontology_owl.py`)이고, `owl_reasoner.py`는 삭제됐다(서비스
호출처 0건). 질의 경로 사용은 `src/rag/ontology_context.py`, 플래그 `ontology.use_class_reasoning`
(기본 OFF, 효과는 O7 측정 전).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.intent import UnifiedIntent

# ============================================================================
# Intent-Based Retrieval Configuration
# ============================================================================


@dataclass(frozen=True)
class IntentRetrievalConfig:
    """Per-intent retrieval tuning parameters.

    Attributes:
        weights: blend ratios for kg / rag / inference (sum to 1.0).
        top_k: maximum number of RAG chunks to retrieve.
        doc_type_filter: document types to prioritize (None = all).
        description: human-readable strategy label (for logging/debug).
        fusion_strategy: ConfidenceFusion strategy name for overall scoring.
    """

    weights: dict[str, float] = field(
        default_factory=lambda: {"kg": 0.4, "rag": 0.4, "inference": 0.2}
    )
    top_k: int = 5
    doc_type_filter: list[str] | None = None
    description: str = "default"
    fusion_strategy: str = "weighted_sum"


# 검색 단계가 공급해야 할 RAG 청크 수. `config/retrieval_weights.json`의
# `max_context_items.rag_chunks`(=8)가 선언한 **컨텍스트 예산**과 같은 값이어야 한다.
#
# 사이클 2에서 컨텍스트 상한을 3→8로 올렸지만 인텐트별 top_k는 5(일부 7)로 남아
# 검색이 애초에 예산보다 적게 공급했고, 상한 8은 한 번도 걸리지 않았다.
# 그 결과 골든셋 개념 recall@8이 구조적으로 눌려 있었다 — 2026-08-30 사이클 8 실측
# (검색 단계만 교체): 전체 0.576 → 0.671, 기존 160문항 0.575 → 0.671, IR 0.583 → 0.667.
# 인텐트별 차등은 `weights`(kg/rag/inference 배분)로 표현하고, 검색 공급량을
# 줄여서 표현하지 않는다. 드리프트 방지 테스트: tests/unit/rag/test_retrieval_strategy.py
_CONTEXT_BUDGET_TOP_K = 8

# Mapping from UnifiedIntent → retrieval configuration.
# Graph-heavy: high kg weight (DIAGNOSIS, COMPETITIVE)
# Vector-heavy: high rag weight (GENERAL, DEFINITION, DATA_QUERY)
# Inference-heavy: high inference weight (ANALYSIS, INSIGHT_RULE)
# Balanced/Hybrid: mixed weights (TREND, CRISIS, METRIC)

_INTENT_STRATEGY_MAP: dict[UnifiedIntent, IntentRetrievalConfig] = {
    # --- Graph-heavy strategies (KG relationships dominate) ---
    UnifiedIntent.DIAGNOSIS: IntentRetrievalConfig(
        weights={"kg": 0.5, "rag": 0.3, "inference": 0.2},
        top_k=_CONTEXT_BUDGET_TOP_K,
        doc_type_filter=["playbook", "metric_guide", "intelligence"],
        description="graph-heavy/diagnosis",
        fusion_strategy="weighted_sum",
    ),
    UnifiedIntent.COMPETITIVE: IntentRetrievalConfig(
        weights={"kg": 0.5, "rag": 0.25, "inference": 0.25},
        top_k=_CONTEXT_BUDGET_TOP_K,
        doc_type_filter=["intelligence", "playbook"],
        description="graph-heavy/competitive",
        fusion_strategy="weighted_sum",
    ),
    # --- Vector-heavy strategies (RAG document search dominates) ---
    UnifiedIntent.GENERAL: IntentRetrievalConfig(
        weights={"kg": 0.3, "rag": 0.5, "inference": 0.2},
        top_k=_CONTEXT_BUDGET_TOP_K,
        doc_type_filter=None,
        description="vector-heavy/general",
        fusion_strategy="weighted_sum",
    ),
    UnifiedIntent.DEFINITION: IntentRetrievalConfig(
        weights={"kg": 0.2, "rag": 0.6, "inference": 0.2},
        top_k=_CONTEXT_BUDGET_TOP_K,
        doc_type_filter=["metric_guide", "playbook"],
        description="vector-heavy/definition",
        fusion_strategy="weighted_sum",
    ),
    UnifiedIntent.DATA_QUERY: IntentRetrievalConfig(
        weights={"kg": 0.4, "rag": 0.4, "inference": 0.2},
        top_k=_CONTEXT_BUDGET_TOP_K,
        doc_type_filter=None,
        description="balanced/data_query",
        fusion_strategy="weighted_sum",
    ),
    UnifiedIntent.INTERPRETATION: IntentRetrievalConfig(
        weights={"kg": 0.3, "rag": 0.5, "inference": 0.2},
        top_k=_CONTEXT_BUDGET_TOP_K,
        doc_type_filter=["metric_guide", "playbook"],
        description="vector-heavy/interpretation",
        fusion_strategy="weighted_sum",
    ),
    # --- Inference-heavy strategies (ontology reasoning dominates) ---
    UnifiedIntent.ANALYSIS: IntentRetrievalConfig(
        weights={"kg": 0.3, "rag": 0.3, "inference": 0.4},
        top_k=_CONTEXT_BUDGET_TOP_K,
        doc_type_filter=["intelligence", "playbook", "metric_guide"],
        description="inference-heavy/analysis",
        fusion_strategy="geometric_mean",
    ),
    UnifiedIntent.INSIGHT_RULE: IntentRetrievalConfig(
        weights={"kg": 0.25, "rag": 0.35, "inference": 0.4},
        top_k=_CONTEXT_BUDGET_TOP_K,
        doc_type_filter=["intelligence", "knowledge_base"],
        description="inference-heavy/insight_rule",
        fusion_strategy="geometric_mean",
    ),
    # --- Hybrid strategies (balanced blend) ---
    UnifiedIntent.TREND: IntentRetrievalConfig(
        weights={"kg": 0.35, "rag": 0.35, "inference": 0.3},
        top_k=_CONTEXT_BUDGET_TOP_K,
        doc_type_filter=["intelligence", "knowledge_base", "response_guide"],
        description="hybrid/trend",
        fusion_strategy="harmonic_mean",
    ),
    UnifiedIntent.CRISIS: IntentRetrievalConfig(
        weights={"kg": 0.35, "rag": 0.4, "inference": 0.25},
        top_k=_CONTEXT_BUDGET_TOP_K,
        doc_type_filter=["response_guide", "intelligence", "playbook"],
        description="hybrid/crisis",
        fusion_strategy="harmonic_mean",
    ),
    UnifiedIntent.METRIC: IntentRetrievalConfig(
        weights={"kg": 0.4, "rag": 0.4, "inference": 0.2},
        top_k=_CONTEXT_BUDGET_TOP_K,
        doc_type_filter=["metric_guide", "playbook"],
        description="balanced/metric",
        fusion_strategy="weighted_sum",
    ),
    UnifiedIntent.COMBINATION: IntentRetrievalConfig(
        weights={"kg": 0.35, "rag": 0.35, "inference": 0.3},
        top_k=_CONTEXT_BUDGET_TOP_K,
        doc_type_filter=["playbook", "metric_guide"],
        description="hybrid/combination",
        fusion_strategy="harmonic_mean",
    ),
}


def get_intent_retrieval_config(intent: UnifiedIntent) -> IntentRetrievalConfig:
    """Return the retrieval configuration for a given intent.

    Falls back to the GENERAL config if the intent is not mapped.

    Args:
        intent: unified query intent

    Returns:
        IntentRetrievalConfig with weights, top_k, doc_type_filter
    """
    return _INTENT_STRATEGY_MAP.get(
        intent,
        _INTENT_STRATEGY_MAP[UnifiedIntent.GENERAL],
    )

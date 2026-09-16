"""Weighted merge — the single per-item scoring pass
==================================================

Scores the three retrieval sources of a ``HybridContext`` onto one scale,
sorts each source by that score, caps it, and records the per-item scores plus
an aggregate ``ConfidenceFusion`` verdict in the context metadata.

Weight precedence (unchanged by the F3 split):
    1. ``intent_weights`` — from ``_INTENT_STRATEGY_MAP`` via the intent config.
    2. ``config/retrieval_weights.json``.
    3. the in-code defaults below.

Only ``freshness`` and ``max_context_items`` of the JSON are actually read on
the production path: ``HybridRetriever.retrieve`` always passes
``intent_config.weights``, so the JSON's ``weights`` block never wins. See the
F3 report.

Moved out of ``hybrid_retriever.py`` (F3 split). The context is duck-typed so
this module does not import back into ``hybrid_retriever``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_RETRIEVAL_WEIGHTS: dict[str, Any] = {
    "weights": {"kg": 0.4, "rag": 0.4, "inference": 0.2},
    "freshness": {"weekly": 1.0, "quarterly": 0.9, "static": 0.8},
    "max_context_items": {"ontology_facts": 5, "inferences": 5, "rag_chunks": 3},
}

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "retrieval_weights.json"

# 사실 유형별 기본 점수 (kg 가중치와 곱해진다)
_FACT_BASE_SCORES: dict[str, float] = {
    "brand_info": 1.0,
    "competitors": 1.0,
    "competitor_network": 1.0,
    "category_brands": 0.8,
    "category_hierarchy": 0.8,
}
_FACT_BASE_SCORE_DEFAULT = 0.6

# 문서 유형별 freshness 등급
_DOC_TYPE_FRESHNESS: dict[str, str] = {
    "intelligence": "weekly",
    "response_guide": "weekly",
    "playbook": "quarterly",
    "knowledge_base": "quarterly",
}
_DOC_TYPE_FRESHNESS_DEFAULT = "static"


def load_retrieval_weights() -> dict:
    """config/retrieval_weights.json에서 가중치 로드 (없으면 기본값)."""
    defaults = {key: dict(value) for key, value in DEFAULT_RETRIEVAL_WEIGHTS.items()}

    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                loaded = json.load(f)
                # Merge with defaults (loaded overrides)
                for key in defaults:
                    if key in loaded:
                        defaults[key] = loaded[key]
            logger.info(f"Retrieval weights loaded from {CONFIG_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load retrieval weights: {e}, using defaults")

    return defaults


def _score_ontology_facts(facts: list[dict], kg_weight: float, limit: int) -> list[dict]:
    for fact in facts:
        base_score = _FACT_BASE_SCORES.get(fact.get("type", ""), _FACT_BASE_SCORE_DEFAULT)
        fact["_weighted_score"] = kg_weight * base_score
    facts.sort(key=lambda x: x.get("_weighted_score", 0), reverse=True)
    return facts[:limit]


def _score_rag_chunks(
    chunks: list[dict], rag_weight: float, freshness: dict[str, float], limit: int
) -> list[dict]:
    for chunk in chunks:
        similarity_score = chunk.get("score", 0.5)
        doc_type = chunk.get("metadata", {}).get("doc_type", "")
        grade = _DOC_TYPE_FRESHNESS.get(doc_type, _DOC_TYPE_FRESHNESS_DEFAULT)
        chunk["_weighted_score"] = rag_weight * similarity_score * freshness[grade]
    chunks.sort(key=lambda x: x.get("_weighted_score", 0), reverse=True)
    return chunks[:limit]


def _score_inferences(inferences: list, inference_weight: float, limit: int) -> list:
    for inference in inferences:
        confidence = getattr(inference, "confidence", 0.5)
        # Store score as attribute (not in dict)
        inference._weighted_score = inference_weight * confidence
    inferences.sort(key=lambda x: getattr(x, "_weighted_score", 0), reverse=True)
    return inferences[:limit]


def weighted_merge(
    context: Any,
    retrieval_weights: dict,
    intent_weights: dict[str, float] | None = None,
) -> Any:
    """가중치 기반 컨텍스트 병합.

    KG facts, RAG chunks, Ontology inferences에 가중치를 부여하고
    최종 점수로 정렬하여 상위 항목만 유지합니다.

    Args:
        context: 병합 전 HybridContext (duck-typed)
        retrieval_weights: :func:`load_retrieval_weights` 결과
        intent_weights: 인텐트 기반 가중치 (optional override)

    Returns:
        가중치 적용된 context (같은 객체)
    """
    weights = intent_weights if intent_weights is not None else retrieval_weights["weights"]
    freshness = retrieval_weights["freshness"]
    max_items = retrieval_weights["max_context_items"]

    weighted_scores: dict[str, list[float]] = {}

    if context.ontology_facts:
        context.ontology_facts = _score_ontology_facts(
            list(context.ontology_facts), weights["kg"], max_items["ontology_facts"]
        )
        weighted_scores["ontology_facts"] = [
            f.get("_weighted_score", 0) for f in context.ontology_facts
        ]

    if context.rag_chunks:
        context.rag_chunks = _score_rag_chunks(
            list(context.rag_chunks), weights["rag"], freshness, max_items["rag_chunks"]
        )
        weighted_scores["rag_chunks"] = [c.get("_weighted_score", 0) for c in context.rag_chunks]

    if context.inferences:
        context.inferences = _score_inferences(
            list(context.inferences), weights["inference"], max_items["inferences"]
        )
        weighted_scores["inferences"] = [
            getattr(i, "_weighted_score", 0) for i in context.inferences
        ]

    # 메타데이터에 점수 저장
    if not context.metadata:
        context.metadata = {}
    context.metadata["weighted_scores"] = weighted_scores

    # ConfidenceFusion: 전체 신뢰도 계산 + 충돌 감지
    fusion_meta = compute_fusion_confidence(context, intent_weights)
    context.metadata["fusion"] = fusion_meta

    logger.info(
        f"Weighted merge applied: {len(context.ontology_facts)} facts, "
        f"{len(context.rag_chunks)} chunks, {len(context.inferences)} inferences"
        f" | fusion_confidence={fusion_meta.get('confidence', 0):.3f}"
        f" strategy={fusion_meta.get('strategy', 'n/a')}"
    )

    if fusion_meta.get("warnings"):
        for w in fusion_meta["warnings"]:
            logger.warning(f"Fusion conflict: {w}")

    return context


def _resolve_fusion_strategy_name(query: str) -> str:
    """인텐트 설정에서 fusion_strategy 이름을 읽는다 (실패 시 weighted_sum)."""
    try:
        from src.core.intent import classify_intent
        from src.rag.retrieval_strategy import get_intent_retrieval_config

        return get_intent_retrieval_config(classify_intent(query)).fusion_strategy
    except Exception:
        return "weighted_sum"


def compute_fusion_confidence(
    context: Any,
    intent_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """ConfidenceFusion으로 전체 신뢰도를 계산하고 소스 간 충돌을 감지한다.

    :func:`weighted_merge` 의 per-item 점수 산정은 그대로 두고, 이 함수는
    3개 소스의 aggregate 신뢰도 + 충돌 경고를 추가한다.

    Args:
        context: 가중 병합 완료된 HybridContext (duck-typed)
        intent_weights: 인텐트별 가중치 (kg/rag/inference)

    Returns:
        confidence, strategy, warnings, source_scores, explanation
    """
    from src.infrastructure.feature_flags import FeatureFlags

    if not FeatureFlags.get_instance().use_confidence_fusion():
        logger.info("Confidence fusion disabled by feature flag")
        return {"confidence": 0.0, "strategy": "disabled", "warnings": []}

    try:
        from src.rag.confidence_fusion import (
            ConfidenceFusion,
            FusedEntity,
            FusionInferenceResult,
            FusionStrategy,
            ScoreNormalizationMethod,
            SearchResult,
        )
    except ImportError:
        logger.debug("confidence_fusion module not available, skipping fusion scoring")
        return {"confidence": 0.0, "strategy": "unavailable", "warnings": []}

    # 인텐트 가중치 → ConfidenceFusion 가중치 매핑
    # ConfidenceFusion uses: vector(=rag), ontology(=inference), entity(=kg)
    w = intent_weights or {"kg": 0.4, "rag": 0.4, "inference": 0.2}
    fusion_weights = {
        "vector": w.get("rag", 0.4),
        "ontology": w.get("inference", 0.2),
        "entity": w.get("kg", 0.4),
    }

    strategy_map = {
        "weighted_sum": FusionStrategy.WEIGHTED_SUM,
        "harmonic_mean": FusionStrategy.HARMONIC_MEAN,
        "geometric_mean": FusionStrategy.GEOMETRIC_MEAN,
        "max_score": FusionStrategy.MAX_SCORE,
        "rrf": FusionStrategy.RRF,
    }
    strategy = strategy_map.get(
        _resolve_fusion_strategy_name(context.query), FusionStrategy.WEIGHTED_SUM
    )

    # harmonic/geometric mean은 0 점수에 취약 → 정규화 생략 (원점수가 이미 0-1)
    if strategy in (FusionStrategy.HARMONIC_MEAN, FusionStrategy.GEOMETRIC_MEAN):
        normalization = ScoreNormalizationMethod.NONE
    else:
        normalization = ScoreNormalizationMethod.MIN_MAX

    fusion = ConfidenceFusion(
        weights=fusion_weights,
        normalization=normalization,
        strategy=strategy,
        min_sources=1,
        conflict_threshold=0.3,
    )

    # HybridContext → ConfidenceFusion 입력 변환
    vector_results = [
        SearchResult(
            content=chunk.get("content", chunk.get("text", "")),
            score=chunk.get("_weighted_score", chunk.get("score", 0.5)),
            metadata=chunk.get("metadata", {}),
            source="vector",
        )
        for chunk in context.rag_chunks or []
    ]

    ontology_results = [
        FusionInferenceResult(
            insight=getattr(inf, "conclusion", str(inf)),
            confidence=getattr(inf, "confidence", 0.5),
            evidence=getattr(inf, "evidence", {}),
            rule_name=getattr(inf, "rule_name", None),
        )
        for inf in context.inferences or []
    ]

    entity_links = [
        FusedEntity(
            entity_id=fact.get("type", "unknown"),
            entity_name=fact.get("subject", fact.get("type", "")),
            entity_type=fact.get("type", "KG_Fact"),
            link_confidence=fact.get("_weighted_score", 0.6),
            context=str(fact.get("data", "")),
        )
        for fact in context.ontology_facts or []
    ]

    result = fusion.fuse(
        vector_results=vector_results or None,
        ontology_results=ontology_results or None,
        entity_links=entity_links or None,
        query=context.query,
    )

    return {
        "confidence": round(result.confidence, 4),
        "strategy": result.fusion_strategy,
        "warnings": result.warnings,
        "explanation": result.explanation,
        "source_scores": [
            {
                "source": s.source_name,
                "raw": round(s.raw_score, 3),
                "normalized": round(s.normalized_score, 3),
                "weight": round(s.weight, 3),
                "contribution": round(s.contribution, 3),
                "level": s.confidence_level,
            }
            for s in result.source_scores
        ],
    }

"""
Ontology Module
온톨로지 스키마(도메인 엔티티), 관계, 지식 그래프, 추론 엔진

- 엔티티·관계 타입의 정본은 src.domain.entities 이다 (여기서는 편의 재수출).
- knowledge_graph: JSON 트리플 저장소
- reasoner + rules: 규칙 기반 추론 엔진
- owl_reasoner: OWL(owlready2) 추론 (배치에서 실행, 챗 경로는 물질화 결과만 읽음)
"""

import importlib

_LAZY: dict[str, str] = {
    "Brand": "src.domain.entities.brand",
    "BrandMetrics": "src.domain.entities.brand",
    "Category": "src.domain.entities.market",
    "MarketMetrics": "src.domain.entities.market",
    "ProductMetrics": "src.domain.entities.market",
    "Snapshot": "src.domain.entities.market",
    "Product": "src.domain.entities.product",
    "RankRecord": "src.domain.entities.product",
    "BadgeType": "src.domain.entities.product",
    "RelationType": "src.domain.entities.relations",
    "InsightType": "src.domain.entities.relations",
    "MarketPosition": "src.domain.entities.relations",
    "Relation": "src.domain.entities.relations",
    "InferenceResult": "src.domain.entities.relations",
    "create_brand_product_relation": "src.domain.entities.relations",
    "create_product_category_relation": "src.domain.entities.relations",
    "create_competition_relation": "src.domain.entities.relations",
    "KnowledgeGraph": ".knowledge_graph",
    "OntologyReasoner": ".reasoner",
    "InferenceRule": ".reasoner",
    "RuleCondition": ".reasoner",
    "StandardConditions": ".reasoner",
    "ALL_BUSINESS_RULES": ".rules",
    "register_all_rules": ".rules",
    "get_rules_by_category": ".rules",
    "get_high_priority_rules": ".rules",
    "Thresholds": ".thresholds",
    "get_thresholds": ".thresholds",
    "load_thresholds": ".thresholds",
    "set_thresholds": ".thresholds",
    "build_inference_context": ".inference_context",
    "normalize_sentiment_clusters": ".inference_context",
    "OntologyBuilder": ".builder",
    "BuildResult": ".builder",
    "canonical_brand": ".builder",
    "materialize": ".materializer",
    "list_inferred": ".materializer",
    "inferred_facts": ".materializer",
    "define_tbox": ".tbox",
}
_OPTIONAL = frozenset(())

__all__ = list(_LAZY)


def __getattr__(name: str):
    """지연 로딩: 하위 모듈은 실제로 접근할 때만 import한다 (무거운 의존성 로딩 방지)."""
    if name in _LAZY:
        try:
            module = importlib.import_module(_LAZY[name], __name__)
        except ImportError:
            if name in _OPTIONAL:
                globals()[name] = None
                return None
            raise
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))

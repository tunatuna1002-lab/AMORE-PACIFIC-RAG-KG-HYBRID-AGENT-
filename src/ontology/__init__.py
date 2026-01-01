"""
Ontology Module
온톨로지 스키마, 관계, 지식 그래프, 추론 엔진

Components:
- schema: 데이터 모델 정의 (Pydantic)
- relations: 관계 타입 및 트리플 구조
- knowledge_graph: 지식 그래프 구현
- reasoner: 추론 규칙 엔진
- business_rules: 비즈니스 추론 규칙
"""

# Schema (기존)
from .schema import (
    Brand,
    Product,
    Category,
    Snapshot,
    RankRecord,
    ProductMetrics,
    BrandMetrics,
    MarketMetrics
)

# Relations (신규)
from .relations import (
    RelationType,
    InsightType,
    MarketPosition,
    Relation,
    InferenceResult,
    create_brand_product_relation,
    create_product_category_relation,
    create_competition_relation
)

# Knowledge Graph (신규)
from .knowledge_graph import KnowledgeGraph

# Reasoner (신규)
from .reasoner import (
    OntologyReasoner,
    InferenceRule,
    RuleCondition,
    StandardConditions
)

# Business Rules (신규)
from .business_rules import (
    ALL_BUSINESS_RULES,
    register_all_rules,
    get_rules_by_category,
    get_high_priority_rules
)

__all__ = [
    # Schema
    "Brand",
    "Product",
    "Category",
    "Snapshot",
    "RankRecord",
    "ProductMetrics",
    "BrandMetrics",
    "MarketMetrics",

    # Relations
    "RelationType",
    "InsightType",
    "MarketPosition",
    "Relation",
    "InferenceResult",
    "create_brand_product_relation",
    "create_product_category_relation",
    "create_competition_relation",

    # Knowledge Graph
    "KnowledgeGraph",

    # Reasoner
    "OntologyReasoner",
    "InferenceRule",
    "RuleCondition",
    "StandardConditions",

    # Business Rules
    "ALL_BUSINESS_RULES",
    "register_all_rules",
    "get_rules_by_category",
    "get_high_priority_rules",
]

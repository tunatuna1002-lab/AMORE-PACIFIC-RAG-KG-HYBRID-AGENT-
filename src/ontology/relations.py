"""
Ontology Relations
==================
엔티티 간 관계 타입 정의 및 트리플(Triple) 구조

DEPRECATED: 이 모듈은 기존 import 경로 호환성을 위해 유지됩니다.
새 코드는 src.domain.entities.relations에서 직접 import하세요:

    from src.domain.entities.relations import (
        RelationType, InsightType, MarketPosition,
        Relation, InferenceResult
    )
"""

import warnings
from typing import TYPE_CHECKING

# Re-export from new domain layer for backward compatibility
from src.domain.entities.relations import (
    # Enums
    RelationType,
    InsightType,
    MarketPosition,
    # Dataclasses
    Relation,
    InferenceResult,
    # Helper functions
    create_brand_product_relation,
    create_product_category_relation,
    create_competition_relation,
    create_metric_insight_relation,
    create_ai_summary_relation,
    create_sentiment_relation,
    create_brand_sentiment_profile,
    create_sentiment_cluster_relation,
    # Constants
    SENTIMENT_CLUSTERS,
    get_cluster_for_sentiment,
)

# All exports for star import
__all__ = [
    # Enums
    "RelationType",
    "InsightType",
    "MarketPosition",
    # Dataclasses
    "Relation",
    "InferenceResult",
    # Helper functions
    "create_brand_product_relation",
    "create_product_category_relation",
    "create_competition_relation",
    "create_metric_insight_relation",
    "create_ai_summary_relation",
    "create_sentiment_relation",
    "create_brand_sentiment_profile",
    "create_sentiment_cluster_relation",
    # Constants
    "SENTIMENT_CLUSTERS",
    "get_cluster_for_sentiment",
]


def __getattr__(name: str):
    """Emit deprecation warning for direct access."""
    if name in __all__:
        warnings.warn(
            f"Importing {name} from src.ontology.relations is deprecated. "
            f"Use 'from src.domain.entities.relations import {name}' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return globals().get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

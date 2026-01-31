"""
Domain Entities
===============
핵심 비즈니스 엔티티 정의

이 패키지의 모든 엔티티는:
- Pydantic BaseModel 또는 dataclass 기반
- 외부 의존성 없음
- 불변성 권장
"""

from src.domain.entities.brain_models import (
    ConfidenceLevel,
    ContextBase,
    ContextProtocol,
    KGFact,
    ResponseBase,
    ResponseProtocol,
    SystemState,
    ToolResultBase,
    ToolResultProtocol,
)
from src.domain.entities.brand import (
    Brand,
    BrandMetrics,
)
from src.domain.entities.market import (
    Category,
    MarketMetrics,
    ProductMetrics,
    Snapshot,
)
from src.domain.entities.product import (
    BadgeType,
    Product,
    RankRecord,
)
from src.domain.entities.relations import (
    InferenceResult,
    InsightType,
    MarketPosition,
    Relation,
    RelationType,
    create_brand_product_relation,
    create_competition_relation,
    create_metric_insight_relation,
    create_product_category_relation,
)

__all__ = [
    # Brain Models
    "ConfidenceLevel",
    "KGFact",
    "SystemState",
    "ContextProtocol",
    "ToolResultProtocol",
    "ResponseProtocol",
    "ContextBase",
    "ToolResultBase",
    "ResponseBase",
    # Product
    "BadgeType",
    "Product",
    "RankRecord",
    # Brand
    "Brand",
    "BrandMetrics",
    # Market
    "Category",
    "Snapshot",
    "MarketMetrics",
    "ProductMetrics",
    # Relations
    "RelationType",
    "InsightType",
    "MarketPosition",
    "Relation",
    "InferenceResult",
    # Helpers
    "create_brand_product_relation",
    "create_product_category_relation",
    "create_competition_relation",
    "create_metric_insight_relation",
]

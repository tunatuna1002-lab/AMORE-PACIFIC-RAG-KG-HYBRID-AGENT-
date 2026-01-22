"""
Ontology Schema Definitions
===========================
5개 핵심 엔티티: Brand, Product, Category, Snapshot, RankRecord

DEPRECATED: 이 모듈은 기존 import 경로 호환성을 위해 유지됩니다.
새 코드는 src.domain.entities에서 직접 import하세요:

    from src.domain.entities import RankRecord, Product, Brand, Category, Snapshot
    from src.domain.entities import BrandMetrics, MarketMetrics, ProductMetrics, BadgeType
"""

import warnings
from typing import TYPE_CHECKING

# Re-export from new domain layer for backward compatibility
from src.domain.entities.product import (
    BadgeType,
    Product,
    RankRecord,
)
from src.domain.entities.brand import (
    Brand,
    BrandMetrics,
)
from src.domain.entities.market import (
    Category,
    Snapshot,
    ProductMetrics,
    MarketMetrics,
)

# All exports for star import
__all__ = [
    # Product entities
    "BadgeType",
    "Product",
    "RankRecord",
    # Brand entities
    "Brand",
    "BrandMetrics",
    # Market entities
    "Category",
    "Snapshot",
    "ProductMetrics",
    "MarketMetrics",
]


def __getattr__(name: str):
    """Emit deprecation warning for direct access."""
    if name in __all__:
        warnings.warn(
            f"Importing {name} from src.ontology.schema is deprecated. "
            f"Use 'from src.domain.entities import {name}' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Return the already imported symbol
        return globals().get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

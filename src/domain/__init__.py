"""
Domain Layer
============
Clean Architecture의 Entities Layer (Enterprise Business Rules)

이 패키지는 프레임워크나 외부 의존성 없이 순수 도메인 모델만 포함합니다.

구조:
- entities/: 핵심 엔티티 (Product, Brand, Category, Metrics 등)
- value_objects/: 값 객체 (계산된 메트릭 등)
- interfaces/: 의존성 역전을 위한 Protocol (추상 인터페이스)

원칙:
- 외부 의존성 없음 (FastAPI, LiteLLM, Playwright 등 금지)
- 순수 Python 타입과 표준 라이브러리만 사용
- 비즈니스 로직의 핵심만 포함
"""

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
)
from src.domain.exceptions import (
    AmoreAgentError,
    ConfigurationError,
    DataValidationError,
    KnowledgeGraphError,
    LLMAPIError,
    NetworkError,
    ReasonerError,
    RetrieverError,
    ScraperError,
)

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
    "MarketMetrics",
    "ProductMetrics",
    # Relations
    "RelationType",
    "InsightType",
    "MarketPosition",
    "Relation",
    "InferenceResult",
    # Exceptions
    "AmoreAgentError",
    "NetworkError",
    "LLMAPIError",
    "DataValidationError",
    "ScraperError",
    "KnowledgeGraphError",
    "ReasonerError",
    "RetrieverError",
    "ConfigurationError",
]

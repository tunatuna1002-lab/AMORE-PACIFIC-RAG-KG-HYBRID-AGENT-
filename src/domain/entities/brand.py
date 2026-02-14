"""
Brand Domain Entities
=====================
브랜드 관련 핵심 엔티티: Brand, BrandMetrics
"""

from datetime import datetime

from pydantic import BaseModel, Field


class Brand(BaseModel):
    """
    브랜드 엔티티

    Amazon에서 추적하는 브랜드 정보

    Attributes:
        name: 브랜드명 (예: LANEIGE)
        is_target: 추적 대상 브랜드 여부 (LANEIGE = True)
    """

    name: str = Field(..., description="브랜드명 (예: LANEIGE)")
    is_target: bool = Field(default=False, description="추적 대상 브랜드 여부")

    model_config = {"json_schema_extra": {"example": {"name": "LANEIGE", "is_target": True}}}


class BrandMetrics(BaseModel):
    """
    브랜드별 계산된 지표

    Level 1 (Market & Brand) 및 Level 2 (Category & Price) 지표 포함

    Attributes:
        brand: 브랜드명
        category_id: 카테고리 ID

        # Level 1 지표
        sos: Share of Shelf (%)
        brand_avg_rank: 브랜드 평균 순위
        product_count: Top 100 내 제품 수

        # Level 2 지표
        cpi: Category Price Index
        avg_rating_gap: 평균 평점 격차

        calculated_at: 계산 시점
    """

    brand: str = Field(..., description="브랜드명")
    category_id: str = Field(..., description="카테고리 ID")

    # Level 1: Market & Brand 지표
    sos: float = Field(..., description="Share of Shelf (%)")
    brand_avg_rank: float | None = Field(default=None, description="브랜드 평균 순위")
    product_count: int = Field(default=0, description="Top 100 내 제품 수")

    # Level 2: Category & Price 지표
    cpi: float | None = Field(default=None, description="Category Price Index")
    avg_rating_gap: float | None = Field(default=None, description="평균 평점 격차")

    calculated_at: datetime = Field(default_factory=datetime.now, description="계산 시점")

    model_config = {
        "json_schema_extra": {
            "example": {
                "brand": "LANEIGE",
                "category_id": "lip_care",
                "sos": 15.5,
                "brand_avg_rank": 5.2,
                "product_count": 3,
                "cpi": 105.2,
                "avg_rating_gap": 0.3,
            }
        }
    }

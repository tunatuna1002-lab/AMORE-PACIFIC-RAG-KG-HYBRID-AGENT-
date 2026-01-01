"""
Ontology Schema Definitions
5개 핵심 엔티티: Brand, Product, Category, Snapshot, RankRecord
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date
from enum import Enum


class BadgeType(str, Enum):
    """Amazon 뱃지 유형"""
    BEST_SELLER = "Best Seller"
    AMAZONS_CHOICE = "Amazon's Choice"
    CLIMATE_PLEDGE = "Climate Pledge Friendly"
    NONE = ""


class Brand(BaseModel):
    """브랜드 엔티티"""
    name: str = Field(..., description="브랜드명 (예: LANEIGE)")
    is_target: bool = Field(default=False, description="추적 대상 브랜드 여부")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "LANEIGE",
                "is_target": True
            }
        }


class Category(BaseModel):
    """카테고리 엔티티"""
    id: str = Field(..., description="카테고리 ID (예: lip_care)")
    name: str = Field(..., description="카테고리명 (예: Lip Care)")
    url: str = Field(..., description="Amazon 베스트셀러 URL")
    parent_id: Optional[str] = Field(default=None, description="상위 카테고리 ID")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "lip_care",
                "name": "Lip Care",
                "url": "https://www.amazon.com/Best-Sellers-Beauty-Personal-Care-Lip-Care-Products/zgbs/beauty/3761351",
                "parent_id": "beauty"
            }
        }


class Product(BaseModel):
    """제품 엔티티"""
    asin: str = Field(..., description="Amazon Standard Identification Number")
    product_name: str = Field(..., description="제품명")
    brand: str = Field(..., description="브랜드명")
    product_url: str = Field(..., description="제품 상세 페이지 URL")
    first_seen_date: Optional[date] = Field(default=None, description="시스템에서 최초 발견된 날짜")
    launch_date: Optional[date] = Field(default=None, description="실제 출시일 (Amazon 상세페이지에서 수집)")

    class Config:
        json_schema_extra = {
            "example": {
                "asin": "B08XYZ1234",
                "product_name": "LANEIGE Lip Sleeping Mask - Berry",
                "brand": "LANEIGE",
                "product_url": "https://www.amazon.com/dp/B08XYZ1234",
                "first_seen_date": "2025-01-01",
                "launch_date": "2024-11-15"
            }
        }


class Snapshot(BaseModel):
    """스냅샷 엔티티 - 특정 시점의 데이터 수집 기록"""
    snapshot_date: date = Field(..., description="스냅샷 날짜")
    category_id: str = Field(..., description="카테고리 ID")
    collected_at: datetime = Field(default_factory=datetime.now, description="실제 수집 시간")
    total_products: int = Field(default=100, description="수집된 제품 수")
    success: bool = Field(default=True, description="수집 성공 여부")
    error_message: Optional[str] = Field(default=None, description="에러 메시지")

    class Config:
        json_schema_extra = {
            "example": {
                "snapshot_date": "2025-01-15",
                "category_id": "lip_care",
                "collected_at": "2025-01-15T09:30:00",
                "total_products": 100,
                "success": True
            }
        }


class RankRecord(BaseModel):
    """순위 기록 엔티티 - 특정 시점의 제품 순위 정보"""
    snapshot_date: date = Field(..., description="스냅샷 날짜")
    category_id: str = Field(..., description="카테고리 ID")
    asin: str = Field(..., description="제품 ASIN")
    product_name: str = Field(..., description="제품명")
    brand: str = Field(..., description="브랜드명")
    rank: int = Field(..., ge=1, le=100, description="순위 (1-100)")
    price: Optional[float] = Field(default=None, description="가격 (USD)")
    rating: Optional[float] = Field(default=None, ge=0, le=5, description="평점 (0-5)")
    reviews_count: Optional[int] = Field(default=None, ge=0, description="리뷰 수")
    badge: str = Field(default="", description="뱃지 (Best Seller, Amazon's Choice 등)")
    product_url: str = Field(..., description="제품 URL")

    class Config:
        json_schema_extra = {
            "example": {
                "snapshot_date": "2025-01-15",
                "category_id": "lip_care",
                "asin": "B08XYZ1234",
                "product_name": "LANEIGE Lip Sleeping Mask - Berry",
                "brand": "LANEIGE",
                "rank": 1,
                "price": 24.00,
                "rating": 4.7,
                "reviews_count": 89234,
                "badge": "Best Seller",
                "product_url": "https://www.amazon.com/dp/B08XYZ1234"
            }
        }


class ProductMetrics(BaseModel):
    """제품별 계산된 지표"""
    asin: str
    category_id: str

    # Level 3: Product & Risk 지표
    rank_volatility: Optional[float] = Field(default=None, description="순위 변동성 (7일 표준편차)")
    rank_shock: bool = Field(default=False, description="순위 급변 발생 여부")
    rank_change: Optional[int] = Field(default=None, description="전일 대비 순위 변화")
    streak_days: int = Field(default=0, description="Top N 연속 체류일")
    rating_trend: Optional[float] = Field(default=None, description="평점 추세 (기울기)")
    best_rank: Optional[int] = Field(default=None, description="최고 순위")
    days_in_top_n: dict = Field(default_factory=dict, description="Top N별 체류일 수")

    calculated_at: datetime = Field(default_factory=datetime.now)


class BrandMetrics(BaseModel):
    """브랜드별 계산된 지표"""
    brand: str
    category_id: str

    # Level 1: Market & Brand 지표
    sos: float = Field(..., description="Share of Shelf (%)")
    brand_avg_rank: Optional[float] = Field(default=None, description="브랜드 평균 순위")
    product_count: int = Field(default=0, description="Top 100 내 제품 수")

    # Level 2: Category & Price 지표
    cpi: Optional[float] = Field(default=None, description="Category Price Index")
    avg_rating_gap: Optional[float] = Field(default=None, description="평균 평점 격차")

    calculated_at: datetime = Field(default_factory=datetime.now)


class MarketMetrics(BaseModel):
    """시장(카테고리)별 계산된 지표"""
    category_id: str
    snapshot_date: date

    # Level 1: Market 지표
    hhi: float = Field(..., description="Herfindahl Index (시장 집중도)")

    # Level 2: Market 지표
    churn_rate: Optional[float] = Field(default=None, description="순위 교체율")
    category_avg_price: Optional[float] = Field(default=None, description="카테고리 평균 가격")
    category_avg_rating: Optional[float] = Field(default=None, description="카테고리 평균 평점")

    calculated_at: datetime = Field(default_factory=datetime.now)

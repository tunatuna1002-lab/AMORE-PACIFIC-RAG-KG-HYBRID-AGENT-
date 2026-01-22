"""
Product Domain Entities
=======================
제품 관련 핵심 엔티티: Product, RankRecord, ProductMetrics, BadgeType
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime, date
from enum import Enum


class BadgeType(str, Enum):
    """Amazon 뱃지 유형"""
    BEST_SELLER = "Best Seller"
    AMAZONS_CHOICE = "Amazon's Choice"
    CLIMATE_PLEDGE = "Climate Pledge Friendly"
    NONE = ""


class Product(BaseModel):
    """
    제품 엔티티

    Amazon 제품의 기본 정보를 나타내는 핵심 엔티티입니다.

    Attributes:
        asin: Amazon Standard Identification Number (고유 식별자)
        product_name: 제품명
        brand: 브랜드명
        product_url: 제품 상세 페이지 URL
        first_seen_date: 시스템에서 최초 발견된 날짜
        launch_date: 실제 출시일 (Amazon 상세페이지에서 수집)
    """
    asin: str = Field(..., description="Amazon Standard Identification Number")
    product_name: str = Field(..., description="제품명")
    brand: str = Field(..., description="브랜드명")
    product_url: str = Field(..., description="제품 상세 페이지 URL")
    first_seen_date: Optional[date] = Field(default=None, description="시스템에서 최초 발견된 날짜")
    launch_date: Optional[date] = Field(default=None, description="실제 출시일")

    model_config = {
        "json_schema_extra": {
            "example": {
                "asin": "B08XYZ1234",
                "product_name": "LANEIGE Lip Sleeping Mask - Berry",
                "brand": "LANEIGE",
                "product_url": "https://www.amazon.com/dp/B08XYZ1234",
                "first_seen_date": "2025-01-01",
                "launch_date": "2024-11-15"
            }
        }
    }


class RankRecord(BaseModel):
    """
    순위 기록 엔티티

    특정 시점의 제품 순위 정보 (프로모션 정보 포함)

    Attributes:
        snapshot_date: 스냅샷 날짜
        category_id: 카테고리 ID
        asin: 제품 ASIN
        product_name: 제품명
        brand: 브랜드명
        rank: 순위 (1-100)
        price: 현재 판매가 (USD)
        list_price: 정가/원가 (USD)
        discount_percent: 할인율 (%)
        rating: 평점 (0-5)
        reviews_count: 리뷰 수
        badge: 뱃지 (Best Seller, Amazon's Choice 등)
        coupon_text: 쿠폰 정보
        is_subscribe_save: Subscribe & Save 여부
        promo_badges: 프로모션 배지
        product_url: 제품 URL
        collected_at: 정확한 수집 시간
        discount_trend: 할인율 추세
        previous_price: 이전 가격
        previous_discount: 이전 할인율
    """
    snapshot_date: date = Field(..., description="스냅샷 날짜")
    category_id: str = Field(..., description="카테고리 ID")
    asin: str = Field(..., description="제품 ASIN")
    product_name: str = Field(..., description="제품명")
    brand: str = Field(..., description="브랜드명")
    rank: int = Field(..., ge=1, le=100, description="순위 (1-100)")
    price: Optional[float] = Field(default=None, description="현재 판매가 (USD)")
    list_price: Optional[float] = Field(default=None, description="정가/원가 (USD)")
    discount_percent: Optional[float] = Field(default=None, description="할인율 (%)")
    rating: Optional[float] = Field(default=None, ge=0, le=5, description="평점 (0-5)")
    reviews_count: Optional[int] = Field(default=None, ge=0, description="리뷰 수")
    badge: str = Field(default="", description="뱃지")
    coupon_text: str = Field(default="", description="쿠폰 정보")
    is_subscribe_save: bool = Field(default=False, description="Subscribe & Save 여부")
    promo_badges: str = Field(default="", description="프로모션 배지")
    product_url: str = Field(..., description="제품 URL")

    # 신규 필드 (할인 추적 및 정확한 수집 시간)
    collected_at: Optional[datetime] = Field(default=None, description="정확한 수집 시간")
    discount_trend: Optional[str] = Field(default=None, description="할인율 추세 (up, down, stable)")
    previous_price: Optional[float] = Field(default=None, description="이전 가격 (USD)")
    previous_discount: Optional[float] = Field(default=None, description="이전 할인율 (%)")

    @field_validator('rank')
    @classmethod
    def validate_rank(cls, v: int) -> int:
        """rank는 1-100 범위여야 함"""
        if v < 1 or v > 100:
            raise ValueError(f"rank must be between 1 and 100, got {v}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "snapshot_date": "2025-01-15",
                "category_id": "lip_care",
                "asin": "B08XYZ1234",
                "product_name": "LANEIGE Lip Sleeping Mask - Berry",
                "brand": "LANEIGE",
                "rank": 1,
                "price": 24.00,
                "list_price": 30.00,
                "discount_percent": 20.0,
                "rating": 4.7,
                "reviews_count": 89234,
                "badge": "Best Seller",
                "coupon_text": "Save 5% with coupon",
                "is_subscribe_save": True,
                "promo_badges": "Limited Time Deal",
                "product_url": "https://www.amazon.com/dp/B08XYZ1234"
            }
        }
    }


class ProductMetrics(BaseModel):
    """
    제품별 계산된 지표

    제품의 성과와 위험을 추적하는 Level 3 지표 모음입니다.

    Attributes:
        asin: 제품 ASIN
        category_id: 카테고리 ID
        rank_volatility: 순위 변동성 (7일 표준편차)
        rank_shock: 순위 급변 발생 여부
        rank_change: 전일 대비 순위 변화
        streak_days: Top N 연속 체류일
        rating_trend: 평점 추세 (기울기)
        best_rank: 최고 순위
        days_in_top_n: Top N별 체류일 수
        calculated_at: 지표 계산 시간
    """
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

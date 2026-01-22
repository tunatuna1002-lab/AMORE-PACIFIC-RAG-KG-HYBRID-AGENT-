"""
Market Domain Entities
======================
시장/카테고리 관련 핵심 엔티티: Category, Snapshot, MarketMetrics, ProductMetrics
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date


class Category(BaseModel):
    """
    카테고리 엔티티

    Amazon 베스트셀러 카테고리 정보

    Attributes:
        id: 카테고리 ID (예: lip_care)
        name: 카테고리명 (예: Lip Care)
        url: Amazon 베스트셀러 URL
        parent_id: 상위 카테고리 ID
        amazon_node_id: Amazon 노드 ID
        level: 계층 레벨 (0=root, 1=중분류, 2=소분류)
        path: 전체 경로
        children: 자식 카테고리 ID 목록
    """
    id: str = Field(..., description="카테고리 ID")
    name: str = Field(..., description="카테고리명")
    url: str = Field(..., description="Amazon 베스트셀러 URL")
    parent_id: Optional[str] = Field(default=None, description="상위 카테고리 ID")
    amazon_node_id: str = Field(default="", description="Amazon 노드 ID")
    level: int = Field(default=0, description="계층 레벨")
    path: List[str] = Field(default_factory=list, description="전체 경로")
    children: List[str] = Field(default_factory=list, description="자식 카테고리 ID 목록")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "lip_care",
                "name": "Lip Care",
                "url": "https://www.amazon.com/Best-Sellers-Beauty/zgbs/beauty/3761351",
                "parent_id": "beauty",
                "amazon_node_id": "3761351",
                "level": 2,
                "path": ["beauty", "skin_care", "lip_care"],
                "children": []
            }
        }
    }


class Snapshot(BaseModel):
    """
    스냅샷 엔티티

    특정 시점의 데이터 수집 기록

    Attributes:
        snapshot_date: 스냅샷 날짜
        category_id: 카테고리 ID
        collected_at: 실제 수집 시간
        total_products: 수집된 제품 수
        success: 수집 성공 여부
        error_message: 에러 메시지 (실패 시)
    """
    snapshot_date: date = Field(..., description="스냅샷 날짜")
    category_id: str = Field(..., description="카테고리 ID")
    collected_at: datetime = Field(default_factory=datetime.now, description="실제 수집 시간")
    total_products: int = Field(default=100, description="수집된 제품 수")
    success: bool = Field(default=True, description="수집 성공 여부")
    error_message: Optional[str] = Field(default=None, description="에러 메시지")

    model_config = {
        "json_schema_extra": {
            "example": {
                "snapshot_date": "2025-01-15",
                "category_id": "lip_care",
                "collected_at": "2025-01-15T09:30:00",
                "total_products": 100,
                "success": True
            }
        }
    }


class ProductMetrics(BaseModel):
    """
    제품별 계산된 지표

    Level 3 (Product & Risk) 지표

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
        calculated_at: 계산 시점
    """
    asin: str = Field(..., description="제품 ASIN")
    category_id: str = Field(..., description="카테고리 ID")

    # Level 3: Product & Risk 지표
    rank_volatility: Optional[float] = Field(default=None, description="순위 변동성")
    rank_shock: bool = Field(default=False, description="순위 급변 발생 여부")
    rank_change: Optional[int] = Field(default=None, description="전일 대비 순위 변화")
    streak_days: int = Field(default=0, description="Top N 연속 체류일")
    rating_trend: Optional[float] = Field(default=None, description="평점 추세")
    best_rank: Optional[int] = Field(default=None, description="최고 순위")
    days_in_top_n: dict = Field(default_factory=dict, description="Top N별 체류일 수")

    calculated_at: datetime = Field(default_factory=datetime.now, description="계산 시점")

    model_config = {
        "json_schema_extra": {
            "example": {
                "asin": "B08XYZ1234",
                "category_id": "lip_care",
                "rank_volatility": 2.5,
                "rank_shock": False,
                "rank_change": -2,
                "streak_days": 30,
                "rating_trend": 0.01,
                "best_rank": 1,
                "days_in_top_n": {"top_10": 25, "top_20": 30}
            }
        }
    }


class MarketMetrics(BaseModel):
    """
    시장(카테고리)별 계산된 지표

    Level 1 및 Level 2 시장 지표

    Attributes:
        category_id: 카테고리 ID
        snapshot_date: 스냅샷 날짜
        hhi: Herfindahl Index (시장 집중도)
        churn_rate: 순위 교체율
        category_avg_price: 카테고리 평균 가격
        category_avg_rating: 카테고리 평균 평점
        calculated_at: 계산 시점
    """
    category_id: str = Field(..., description="카테고리 ID")
    snapshot_date: date = Field(..., description="스냅샷 날짜")

    # Level 1: Market 지표
    hhi: float = Field(..., description="Herfindahl Index (시장 집중도)")

    # Level 2: Market 지표
    churn_rate: Optional[float] = Field(default=None, description="순위 교체율")
    category_avg_price: Optional[float] = Field(default=None, description="카테고리 평균 가격")
    category_avg_rating: Optional[float] = Field(default=None, description="카테고리 평균 평점")

    calculated_at: datetime = Field(default_factory=datetime.now, description="계산 시점")

    model_config = {
        "json_schema_extra": {
            "example": {
                "category_id": "lip_care",
                "snapshot_date": "2025-01-15",
                "hhi": 0.08,
                "churn_rate": 12.5,
                "category_avg_price": 25.99,
                "category_avg_rating": 4.3
            }
        }
    }

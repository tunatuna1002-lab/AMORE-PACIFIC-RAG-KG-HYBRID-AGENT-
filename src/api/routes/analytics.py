"""
Analytics Routes (KPI / SoS)
=============================
카테고리별 KPI, SoS(Share of Shelf), 브랜드 비교 엔드포인트.

라우트는 쿼리 검증과 서비스 호출만 담당한다 (F6). 집계 로직은
``src/application/services/analytics_service.py``에 있다.
"""

import json
import logging

from fastapi import APIRouter, Request

from src.api.dependencies import get_data_service, limiter
from src.application.services.analytics_service import AnalyticsService
from src.application.services.sos_trend_service import SosTrendService
from src.tools.storage.sqlite_storage import get_sqlite_storage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Analytics"])


# ============= Helper =============


def _load_crawl_data_for_sos():
    """JSON 파일에서 크롤링 데이터 로드 (SQLite fallback)"""
    crawl_path = get_data_service().latest_crawl_json_path
    if crawl_path.exists():
        with open(crawl_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _service() -> AnalyticsService:
    """모듈 전역을 호출 시점에 읽어 주입한다 (테스트가 교체할 수 있도록)."""
    return AnalyticsService(
        sqlite_factory=get_sqlite_storage,
        crawl_loader=_load_crawl_data_for_sos,
    )


def _trend_service() -> SosTrendService:
    return SosTrendService(sqlite_factory=get_sqlite_storage)


# ============= Category KPI =============


@router.get("/api/category/kpi")
@limiter.limit("10/minute")
async def get_category_kpi(
    request: Request,
    category_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    brand: str = "LANEIGE",
):
    """
    카테고리별 KPI 데이터 조회 (기간 필터링 지원)

    Args:
        category_id: 카테고리 ID (beauty, skin_care, lip_care, lip_makeup, face_powder)
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        brand: 타겟 브랜드 (기본값: LANEIGE)

    Returns:
        KPI 데이터: sos, best_rank, cpi, new_competitors
    """
    return await _service().category_kpi(
        category_id=category_id,
        start_date=start_date,
        end_date=end_date,
        brand=brand,
    )


# ============= SoS (Share of Shelf) =============


@router.get("/api/sos/category")
@limiter.limit("10/minute")
async def get_sos_by_category(
    request: Request,
    start_date: str | None = None,
    end_date: str | None = None,
    compare_brands: str | None = None,
):
    """
    카테고리별 SoS (Share of Shelf) 데이터 조회

    SoS = (해당 브랜드 제품 수 / Top 100) * 100

    Args:
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        compare_brands: 비교할 브랜드 (콤마로 구분)

    Returns:
        카테고리별 SoS 데이터
    """
    return await _service().sos_by_category(
        start_date=start_date,
        end_date=end_date,
        compare_brands=compare_brands,
    )


@router.get("/api/sos/brands")
@limiter.limit("10/minute")
async def get_available_brands(
    request: Request, category_id: str | None = None, min_count: int = 1
):
    """
    비교 가능한 브랜드 목록 조회 (Top 100에 포함된 브랜드들)

    Args:
        category_id: 특정 카테고리만 조회 (선택)
        min_count: 최소 제품 수 (기본: 1)

    Returns:
        브랜드 목록 (제품 수 기준 정렬)
    """
    return await _service().available_brands(category_id=category_id, min_count=min_count)


@router.get("/api/sos/trend")
@limiter.limit("10/minute")
async def get_sos_trend(
    request: Request,
    brand: str = "LANEIGE",
    category_id: str | None = None,
    days: int = 7,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """
    브랜드의 SoS 추세 데이터 (일별)

    Args:
        brand: 브랜드명 (기본: LANEIGE)
        category_id: 카테고리 (선택, 없으면 전체)
        days: 조회 기간 (기본: 7일, start_date/end_date가 없을 때만 사용)
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (YYYY-MM-DD)

    Returns:
        일별 SoS 추세 데이터
    """
    return await _trend_service().sos_trend(
        brand=brand,
        category_id=category_id,
        days=days,
        start_date=start_date,
        end_date=end_date,
    )


@router.get("/api/sos/trend/competitors-avg")
@limiter.limit("10/minute")
async def get_competitors_avg_sos_trend(
    request: Request,
    category_id: str | None = None,
    days: int = 7,
    start_date: str | None = None,
    end_date: str | None = None,
    top_n: int = 10,
    exclude_brand: str = "LANEIGE",
):
    """
    경쟁 브랜드 평균 SoS 추세 데이터 (일별)
    Top N 브랜드(LANEIGE 제외)의 평균 시장점유율 추이

    Args:
        category_id: 카테고리 (선택, 없으면 전체)
        days: 조회 기간 (기본: 7일)
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (YYYY-MM-DD)
        top_n: 상위 몇 개 브랜드 (기본: 10)
        exclude_brand: 제외할 브랜드 (기본: LANEIGE)

    Returns:
        경쟁 브랜드 평균 SoS 추세 데이터
    """
    return await _trend_service().competitors_avg_sos_trend(
        category_id=category_id,
        days=days,
        start_date=start_date,
        end_date=end_date,
        top_n=top_n,
        exclude_brand=exclude_brand,
    )

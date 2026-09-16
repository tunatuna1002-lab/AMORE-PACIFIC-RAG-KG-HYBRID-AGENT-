"""
Data Routes
===========
Dashboard data and historical data endpoints (SQLite-first, Sheets/local fallback).

라우트는 검증과 서비스 호출만 담당한다 (F6):
- 대시보드 데이터: ``application/services/dashboard_data_service.py``
- 히스토리컬 집계: ``application/services/historical_service.py``
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.dependencies import (
    get_data_service,
    get_sheets_writer,
    limiter,
    load_dashboard_data,
    verify_api_key,
)
from src.application.services.historical_service import HistoricalService
from src.tools.storage.sqlite_storage import get_sqlite_storage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["data"])


def _historical_service() -> HistoricalService:
    """모듈 전역을 호출 시점에 읽어 주입한다 (테스트가 교체할 수 있도록)."""
    return HistoricalService(
        sqlite_factory=get_sqlite_storage,
        sheets_factory=get_sheets_writer,
        dashboard_loader=load_dashboard_data,
        data_service=get_data_service(),
    )


@router.get("/api/data")
@limiter.limit("30/minute")
async def get_data(request: Request):
    """
    대시보드 데이터 조회 (JSON 캐시 우선, SQLite 폴백, 없으면 빈 스켈레톤).

    F6: 경로 해석·폴백·staleness 메타는 DashboardDataService 한 곳에만 있다
    (라우트가 들고 있던 두 번째 SQLite 폴백 구현 제거).
    """
    return await get_data_service().get_dashboard_data()


@router.post("/api/data/refresh", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
async def refresh_data(request: Request):
    """dashboard_data.json 재생성 (SQLite 기반)"""
    try:
        from src.tools.exporters.dashboard_exporter import DashboardExporter

        exporter = DashboardExporter(enable_ontology=False)
        await exporter.initialize()
        output_path = str(get_data_service().dashboard_json_path)
        result = await exporter.export_dashboard_data(output_path)

        if isinstance(result, dict) and "error" in result:
            return {"success": False, "error": result["error"]}

        return {
            "success": True,
            "message": "Dashboard data refreshed from SQLite",
        }

    except Exception as e:
        logger.error(f"Data refresh failed: {e}")
        raise HTTPException(status_code=500, detail=f"Refresh failed: {e}") from e


@router.get("/api/historical")
@limiter.limit("30/minute")
async def get_historical_data(
    request: Request,
    start_date: str,
    end_date: str,
    category_id: str | None = None,
    brand: str | None = "LANEIGE",
):
    """
    히스토리컬 데이터 조회 (SQLite 우선, Google Sheets fallback)

    Args:
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (YYYY-MM-DD)
        category_id: 카테고리 필터 (선택)
        brand: 브랜드 필터 (기본값: LANEIGE)

    Returns:
        - data: 날짜별 지표 데이터
        - sos_history: SoS 추이 데이터
        - raw_data: 순위 추이 데이터
    """
    return await _historical_service().historical(
        start_date=start_date,
        end_date=end_date,
        category_id=category_id,
        brand=brand,
    )


async def _get_historical_from_local(
    start_date: str, end_date: str, brand: str = "LANEIGE"
) -> dict[str, Any]:
    """로컬 JSON 폴백 (HistoricalService.from_local 위임)"""
    return await _historical_service().from_local(start_date, end_date, brand)

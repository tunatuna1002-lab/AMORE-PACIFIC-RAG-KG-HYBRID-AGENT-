"""
Deals Routes - Amazon Deals monitoring endpoints
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.api.dependencies import verify_api_key
from src.tools.scrapers.deals_scraper import get_deals_scraper
from src.tools.storage.sqlite_storage import get_sqlite_storage

router = APIRouter(prefix="/api/deals", tags=["deals"])


class DealsRequest(BaseModel):
    """Deals 크롤링 요청"""

    max_items: int = 50
    beauty_only: bool = True


class DealsResponse(BaseModel):
    """Deals 응답"""

    success: bool
    count: int
    lightning_count: int
    competitor_count: int
    snapshot_datetime: str
    deals: list[dict[str, Any]]
    competitor_deals: list[dict[str, Any]]
    error: str | None = None


@router.get("/")
async def get_deals_data(brand: str | None = None, hours: int = 24, limit: int = 100):
    """
    저장된 Deals 데이터 조회

    Args:
        brand: 브랜드 필터 (선택)
        hours: 최근 N시간 데이터 (기본: 24시간)
        limit: 최대 개수

    Returns:
        - deals: 딜 데이터 리스트
        - summary: 요약 통계
    """
    try:
        storage = get_sqlite_storage()
        await storage.initialize()

        # 경쟁사 딜 조회
        deals = await storage.get_competitor_deals(brand=brand, hours=hours)

        # 최대 개수 제한
        deals = deals[:limit] if len(deals) > limit else deals

        # 요약 통계
        summary = await storage.get_deals_summary(days=7)

        return {
            "success": True,
            "deals": deals,
            "count": len(deals),
            "summary": summary,
            "filters": {"brand": brand, "hours": hours},
        }

    except Exception as e:
        logging.error(f"Deals data error: {e}")
        return {"success": False, "deals": [], "count": 0, "error": str(e)}


@router.get("/summary")
async def get_deals_summary(days: int = 7):
    """
    Deals 요약 통계

    Args:
        days: 분석 기간 (일)

    Returns:
        - by_brand: 브랜드별 딜 현황
        - by_date: 일별 추이
    """
    try:
        storage = get_sqlite_storage()
        await storage.initialize()

        summary = await storage.get_deals_summary(days=days)

        return {"success": True, **summary}

    except Exception as e:
        logging.error(f"Deals summary error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/scrape", dependencies=[Depends(verify_api_key)])
async def scrape_deals(request: DealsRequest):
    """
    Amazon Deals 페이지 크롤링 (API Key 필요)

    경쟁사 할인 정보를 수집하고 저장합니다.

    Args:
        max_items: 최대 수집 개수
        beauty_only: 뷰티 카테고리만 필터링

    Returns:
        - deals: 수집된 딜 데이터
        - competitor_deals: 경쟁사 딜
        - lightning_count: Lightning Deal 수
    """
    try:
        scraper = await get_deals_scraper()

        # 크롤링 실행
        result = await scraper.scrape_deals(
            max_items=request.max_items, beauty_only=request.beauty_only
        )

        if result["success"]:
            # SQLite에 저장
            storage = get_sqlite_storage()
            await storage.initialize()

            # 모든 딜 저장
            for deal in result["deals"]:
                await storage.save_deal(deal)

            # 경쟁사 딜에 대한 알림 생성
            if result["competitor_deals"]:
                try:
                    from src.agents.alert_agent import AlertAgent

                    alert_agent = AlertAgent()

                    for deal in result["competitor_deals"]:
                        await alert_agent.process_deal_alert(deal)

                except Exception as alert_err:
                    logging.error(f"Alert processing error: {alert_err}")
                    # 알림 실패해도 크롤링 결과는 반환

            logging.info(
                f"Deals scraped: {result['count']} total, {len(result['competitor_deals'])} competitors"
            )

        return DealsResponse(
            success=result["success"],
            count=result["count"],
            lightning_count=result["lightning_count"],
            competitor_count=len(result["competitor_deals"]),
            snapshot_datetime=result["snapshot_datetime"],
            deals=result["deals"],
            competitor_deals=result["competitor_deals"],
            error=result.get("error"),
        )

    except Exception as e:
        logging.error(f"Deals scrape error: {e}")
        return DealsResponse(
            success=False,
            count=0,
            lightning_count=0,
            competitor_count=0,
            snapshot_datetime=datetime.now().isoformat(),
            deals=[],
            competitor_deals=[],
            error=str(e),
        )


@router.get("/alerts")
async def get_deals_alerts(limit: int = 50, unsent_only: bool = False):
    """
    Deals 알림 목록 조회

    Args:
        limit: 최대 개수
        unsent_only: 미발송 알림만 조회

    Returns:
        - alerts: 알림 목록
        - count: 총 개수
    """
    try:
        storage = get_sqlite_storage()
        await storage.initialize()

        if unsent_only:
            alerts = await storage.get_unsent_alerts(limit=limit)
        else:
            # 모든 알림 조회 (최근 7일)
            with storage.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT * FROM deals_alerts
                    ORDER BY alert_datetime DESC
                    LIMIT ?
                """,
                    (limit,),
                )
                alerts = [dict(row) for row in cursor.fetchall()]

        return {"success": True, "alerts": alerts, "count": len(alerts)}

    except Exception as e:
        logging.error(f"Deals alerts error: {e}")
        return {"success": False, "alerts": [], "count": 0, "error": str(e)}


@router.post("/export")
async def export_deals_report(days: int = 7, format: str = "excel"):
    """
    Deals 리포트 내보내기

    Args:
        days: 분석 기간 (일)
        format: 출력 형식 (excel, json)

    Returns:
        - 엑셀: 파일 다운로드
        - JSON: 데이터 반환
    """
    try:
        storage = get_sqlite_storage()
        await storage.initialize()

        if format == "json":
            # JSON 형식 반환
            summary = await storage.get_deals_summary(days=days)

            # 전체 딜 데이터
            with storage.get_connection() as conn:
                cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                cursor = conn.execute(
                    """
                    SELECT * FROM deals
                    WHERE DATE(snapshot_datetime) >= ?
                    ORDER BY snapshot_datetime DESC
                """,
                    (cutoff_date,),
                )
                deals = [dict(row) for row in cursor.fetchall()]

            return {
                "success": True,
                "summary": summary,
                "deals": deals,
                "count": len(deals),
                "period": {
                    "days": days,
                    "start_date": cutoff_date,
                    "end_date": datetime.now().strftime("%Y-%m-%d"),
                },
            }

        elif format == "excel":
            # 엑셀 파일 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"./data/exports/Deals_Report_{timestamp}.xlsx"

            result = storage.export_deals_to_excel(output_path=output_path, days=days)

            if not result.get("success"):
                raise HTTPException(status_code=500, detail=result.get("error", "Export failed"))

            file_path = Path(result["file_path"])
            if not file_path.exists():
                raise HTTPException(status_code=500, detail="Generated file not found")

            return FileResponse(
                path=str(file_path),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename={file_path.name}"},
            )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Deals export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

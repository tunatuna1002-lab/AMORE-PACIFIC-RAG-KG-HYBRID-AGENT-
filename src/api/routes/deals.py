"""
Deals Routes - Amazon Deals monitoring endpoints
=================================================
dashboard_api.py의 deals 엔드포인트를 추출한 모듈입니다.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from src.api.dependencies import limiter, verify_api_key
from src.api.models import DealsRequest, DealsResponse
from src.tools.storage.sqlite_storage import get_sqlite_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/deals", tags=["deals"])


@router.get("/")
@limiter.limit("10/minute")
async def get_deals_data(
    request: Request, brand: str | None = None, hours: int = 24, limit: int = 100
):
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
@limiter.limit("10/minute")
async def get_deals_summary(request: Request, days: int = 7):
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
@limiter.limit("10/minute")
async def scrape_deals(request: Request, body: DealsRequest):
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
        from src.tools.scrapers.deals_scraper import get_deals_scraper

        scraper = await get_deals_scraper()

        # 크롤링 실행
        result = await scraper.scrape_deals(max_items=body.max_items, beauty_only=body.beauty_only)

        if result["success"]:
            # SQLite에 저장
            storage = get_sqlite_storage()
            await storage.initialize()

            # 모든 딜 저장
            if result["deals"]:
                await storage.save_deals(result["deals"], is_competitor=False)

            # 경쟁사 딜은 is_competitor=True로 별도 저장
            if result["competitor_deals"]:
                await storage.save_deals(result["competitor_deals"], is_competitor=True)

                # 알림 서비스로 알림 처리
                try:
                    from src.infrastructure.container import Container

                    alert_service = Container.get_alert_service()
                    alerts = await alert_service.process_deals_for_alerts(
                        result["competitor_deals"]
                    )

                    # DB에 알림 저장
                    for alert in alerts:
                        await storage.save_deal_alert(alert)

                    logging.info(
                        f"Processed {len(alerts)} alerts from {len(result['competitor_deals'])} competitor deals"
                    )
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
@limiter.limit("10/minute")
async def get_deals_alerts(request: Request, limit: int = 50, unsent_only: bool = False):
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
@limiter.limit("10/minute")
async def export_deals_report(request: Request, days: int = 7, format: str = "excel"):
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
                all_deals = [dict(row) for row in cursor.fetchall()]

            return {
                "success": True,
                "summary": summary,
                "deals": all_deals,
                "export_date": datetime.now().isoformat(),
                "period_days": days,
            }

        else:  # Excel
            # 엑셀 파일 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"./data/exports/Deals_Report_{timestamp}.xlsx"

            result = storage.export_deals_report(output_path=output_path, days=days)

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
        logger.error(f"Deals export error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Deals 내보내기 중 오류가 발생했습니다") from e

"""
External Signal Routes - 외부 트렌드 신호 수집 API
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.api.dependencies import limiter
from src.tools.collectors.external_signal_collector import (
    ExternalSignalCollector,
)

router = APIRouter(prefix="/api/signals", tags=["signals"])
logger = logging.getLogger(__name__)

# 싱글톤 인스턴스
_collector: ExternalSignalCollector | None = None


async def get_collector() -> ExternalSignalCollector:
    """ExternalSignalCollector 싱글톤 반환"""
    global _collector
    if _collector is None:
        _collector = ExternalSignalCollector()
        await _collector.initialize()
    return _collector


class ManualSignalInput(BaseModel):
    """수동 신호 입력"""

    source: str  # allure, tiktok, reddit, etc.
    date: str  # YYYY-MM-DD
    title: str
    url: str | None = ""
    quotes: list[str] | None = []
    keywords: list[str] | None = []
    views: int | None = None
    upvotes: int | None = None


class TrendRadarItem(BaseModel):
    """주간 트렌드 레이더 항목"""

    rank: int
    keyword: str
    source: str
    date: str
    consumer_need: str
    laneige_connection: str
    evidence: str


@router.get("/")
@limiter.limit("10/minute")
async def get_signals(
    request: Request, days: int = 7, tier: str | None = None, source: str | None = None
):
    """
    수집된 신호 조회

    Args:
        days: 최근 N일 이내 신호
        tier: 필터링할 tier (tier1_viral, tier2_validation, tier3_authority, tier4_pr)
        source: 필터링할 소스 (allure, reddit, tiktok, etc.)
    """
    collector = await get_collector()

    signals = collector.signals

    # 날짜 필터링
    cutoff_date = (datetime.now() - __import__("datetime").timedelta(days=days)).strftime(
        "%Y-%m-%d"
    )
    signals = [s for s in signals if s.published_at >= cutoff_date]

    # Tier 필터링
    if tier:
        signals = [s for s in signals if s.tier == tier]

    # Source 필터링
    if source:
        signals = [s for s in signals if s.source == source]

    return {
        "count": len(signals),
        "signals": [s.to_dict() for s in signals],
        "filters": {"days": days, "tier": tier, "source": source},
    }


@router.get("/stats")
@limiter.limit("10/minute")
async def get_signal_stats(request: Request):
    """신호 통계 조회"""
    collector = await get_collector()
    return collector.get_stats()


@router.get("/report")
@limiter.limit("10/minute")
async def get_signal_report(request: Request, days: int = 7):
    """보고서용 신호 섹션 생성"""
    collector = await get_collector()
    report = collector.generate_report_section(days=days)
    return {"days": days, "report_section": report}


@router.post("/fetch/rss")
@limiter.limit("10/minute")
async def fetch_rss_signals(
    request: Request, keywords: list[str] | None = None, max_articles: int = 10
):
    """
    RSS 피드에서 기사 수집 (Tier 3: 전문 매체)

    Sources: Allure, Byrdie, Refinery29
    """
    collector = await get_collector()

    try:
        signals = await collector.fetch_all_rss_feeds(keywords)
        return {
            "status": "success",
            "fetched_count": len(signals),
            "signals": [s.to_dict() for s in signals],
        }
    except Exception as e:
        logger.error(f"RSS fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/fetch/reddit")
@limiter.limit("10/minute")
async def fetch_reddit_signals(
    request: Request,
    subreddits: list[str] | None = None,
    keywords: list[str] | None = None,
    max_posts: int = 10,
):
    """
    Reddit에서 트렌드 수집 (Tier 2: 검증)

    Default subreddits: SkincareAddiction, AsianBeauty, MakeupAddiction
    """
    collector = await get_collector()

    try:
        signals = await collector.fetch_reddit_trends(
            subreddits=subreddits, keywords=keywords, max_posts=max_posts
        )
        return {
            "status": "success",
            "fetched_count": len(signals),
            "signals": [s.to_dict() for s in signals],
        }
    except Exception as e:
        logger.error(f"Reddit fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/manual")
@limiter.limit("10/minute")
async def add_manual_signal(request: Request, input: ManualSignalInput):
    """
    수동 신호 입력

    TikTok Creative Center, Instagram, Twitter 등에서
    수동으로 확인한 데이터를 입력합니다.
    """
    collector = await get_collector()

    try:
        signal = collector.add_manual_media_input(
            {
                "source": input.source,
                "date": input.date,
                "title": input.title,
                "url": input.url,
                "quotes": input.quotes,
                "keywords": input.keywords,
                "views": input.views,
                "upvotes": input.upvotes,
            }
        )
        return {"status": "success", "signal": signal.to_dict()}
    except Exception as e:
        logger.error(f"Manual signal input failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/trend-radar")
@limiter.limit("10/minute")
async def add_trend_radar(request: Request, items: list[TrendRadarItem]):
    """
    주간 트렌드 레이더 일괄 입력

    미국 뷰티 트렌드 레이더.md 형식의 데이터를 입력합니다.
    """
    collector = await get_collector()

    try:
        radar_data = [item.dict() for item in items]
        signals = collector.add_weekly_trend_radar(radar_data)
        return {
            "status": "success",
            "added_count": len(signals),
            "signals": [s.to_dict() for s in signals],
        }
    except Exception as e:
        logger.error(f"Trend radar input failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/clear")
@limiter.limit("10/minute")
async def clear_signals(request: Request):
    """모든 신호 삭제 (개발용)"""
    collector = await get_collector()
    collector.signals = []
    collector._save_signals()
    return {"status": "success", "message": "All signals cleared"}

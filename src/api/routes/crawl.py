"""
Crawl Routes
============
Crawling status and control endpoints
"""

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import limiter, verify_api_key
from src.core.crawl_manager import get_crawl_manager

# Router
router = APIRouter()


@router.get("/status")
@limiter.limit("2/minute")
async def get_crawl_status(request: Request):
    """
    Get crawling status

    Returns:
        - status: idle/running/completed/failed
        - date: Target crawl date
        - progress: Progress percentage (0-100)
        - data_date: Current data date
        - needs_crawl: Whether crawling is needed
    """
    crawl_manager = await get_crawl_manager()
    return {
        **crawl_manager.state.to_dict(),
        "data_date": crawl_manager.get_data_date(),
        "needs_crawl": crawl_manager.needs_crawl(),
        "is_today_available": crawl_manager.is_today_data_available(),
        "status_message": crawl_manager.get_status_message(),
    }


@router.post("/start", dependencies=[Depends(verify_api_key)])
@limiter.limit("2/minute")
async def start_crawl(request: Request):
    """
    Manually start crawling (requires API Key)

    Returns:
        - started: Whether crawling started
        - message: Status message
    """
    crawl_manager = await get_crawl_manager()

    if crawl_manager.is_crawling():
        return {
            "started": False,
            "message": "크롤링이 이미 진행 중입니다.",
            "status": crawl_manager.state.to_dict(),
        }

    if crawl_manager.is_today_data_available():
        return {
            "started": False,
            "message": "오늘 데이터가 이미 존재합니다.",
            "status": crawl_manager.state.to_dict(),
        }

    started = await crawl_manager.start_crawl()
    return {
        "started": started,
        "message": "크롤링을 시작했습니다." if started else "크롤링 시작 실패",
        "status": crawl_manager.state.to_dict(),
    }

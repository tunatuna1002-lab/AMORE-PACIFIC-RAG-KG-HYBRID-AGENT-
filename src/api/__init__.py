"""
API Routes Package
==================
FastAPI 라우터 모듈 (dashboard_api.py에서 분리)

구조:
- routes/chat.py: 챗봇 API (/api/v4/chat)
- routes/data.py: 데이터 API (/api/data, /api/historical)
- routes/crawl.py: 크롤링 API (/api/crawl/*)
- routes/brain.py: Brain API (/api/v4/brain/*)
- routes/export.py: 내보내기 API (/api/export/*)
- routes/deals.py: Deals API (/api/deals/*)
- routes/alerts.py: 알림 API (/api/alerts/*, /api/v3/alert-settings/*)
- dependencies.py: 공통 의존성 (인증, 레이트리밋 등)
"""

from fastapi import APIRouter

# 메인 라우터 (각 서브 라우터를 통합)
api_router = APIRouter()


def include_routers():
    """모든 서브 라우터 포함"""
    from src.api.routes import alerts, brain, chat, crawl, data, deals, export

    api_router.include_router(chat.router, tags=["Chat"])
    api_router.include_router(data.router, tags=["Data"])
    api_router.include_router(crawl.router, prefix="/crawl", tags=["Crawl"])
    api_router.include_router(brain.router, prefix="/v4/brain", tags=["Brain"])
    api_router.include_router(export.router, prefix="/export", tags=["Export"])
    api_router.include_router(deals.router, prefix="/deals", tags=["Deals"])
    api_router.include_router(alerts.router, tags=["Alerts"])

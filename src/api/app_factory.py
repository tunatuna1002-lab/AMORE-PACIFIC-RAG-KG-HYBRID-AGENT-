"""
App Factory
===========
FastAPI 앱 생성 팩토리 (미들웨어, 라우터, 정적 파일 등록)

dashboard_api.py의 앱 초기화 코드를 추출한 모듈입니다.
"""

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.api.dependencies import limiter
from src.api.middleware import SecurityHeadersMiddleware
from src.api.middleware.csrf import CSRFMiddleware

logger = logging.getLogger(__name__)


def create_app(
    lifespan: Callable[..., Any] | None = None,
) -> FastAPI:
    """
    FastAPI 앱 생성 및 설정

    Args:
        lifespan: Optional lifespan async context manager for startup/shutdown.

    Returns:
        설정 완료된 FastAPI 인스턴스
    """
    kwargs: dict[str, Any] = {
        "title": "AMORE Dashboard API",
        "description": "LANEIGE Amazon 대시보드 백엔드 API (RAG + Ontology 통합)",
        "version": "2.0.0",
    }
    if lifespan is not None:
        kwargs["lifespan"] = lifespan

    app = FastAPI(**kwargs)

    # Rate Limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Trusted Host (anti-host-header injection)
    allowed_hosts_str = os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1,.railway.app")
    allowed_hosts = [h.strip() for h in allowed_hosts_str.split(",") if h.strip()]
    if allowed_hosts and "*" not in allowed_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    # CORS
    allowed_origins = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:8001,http://127.0.0.1:8001"
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    )

    # CSRF Protection (exempt all /api/ paths — those use API key auth)
    app.add_middleware(CSRFMiddleware, exempt_paths=["/api/"])

    # Security Headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Register Routers
    _register_routers(app)

    # Static Files
    _mount_static_files(app)

    return app


def _register_routers(app: FastAPI) -> None:
    """라우터 등록"""
    # Already-extracted route modules
    from src.api.routes.alerts import router as alerts_router
    from src.api.routes.analytics import router as analytics_router
    from src.api.routes.brain import router as brain_router
    from src.api.routes.chat import router as chat_router
    from src.api.routes.competitors import router as competitors_router
    from src.api.routes.crawl import router as crawl_router
    from src.api.routes.data import router as data_router
    from src.api.routes.deals import router as deals_router
    from src.api.routes.export import router as export_router
    from src.api.routes.health import router as health_router
    from src.api.routes.market_intelligence import router as mi_router
    from src.api.routes.signals import router as signals_router
    from src.api.routes.sync import router as sync_router

    app.include_router(health_router)
    app.include_router(crawl_router, prefix="/api/crawl", tags=["crawl"])
    app.include_router(data_router)
    app.include_router(deals_router)
    app.include_router(alerts_router)
    app.include_router(chat_router)
    app.include_router(brain_router)
    app.include_router(competitors_router)
    app.include_router(analytics_router)
    app.include_router(sync_router)
    app.include_router(mi_router)
    app.include_router(signals_router)
    app.include_router(export_router)

    # Telegram Admin Bot Router (optional)
    try:
        from src.tools.notifications.telegram_bot import telegram_router

        app.include_router(telegram_router)
        logger.info("Telegram Admin Bot router enabled")
    except ImportError as e:
        logger.warning(f"Telegram Bot not available: {e}")


def _mount_static_files(app: FastAPI) -> None:
    """정적 파일 마운트"""
    static_dir = Path(__file__).parent.parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        fonts_dir = static_dir / "fonts"
        if fonts_dir.exists():
            app.mount("/fonts", StaticFiles(directory=str(fonts_dir)), name="fonts")

    dashboard_dir = Path(__file__).parent.parent.parent / "dashboard"
    if dashboard_dir.exists():
        app.mount(
            "/dashboard",
            StaticFiles(directory=str(dashboard_dir), html=True),
            name="dashboard",
        )

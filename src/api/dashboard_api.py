"""
Dashboard API Server
====================
대시보드용 FastAPI 백엔드 서버 (메인 엔트리포인트)

## 핵심 기능
- 챗봇 API (ChatGPT + RAG + Ontology 연동)
- DOCX 인사이트 리포트 생성
- 대화 메모리 지원 (세션별 TTL 기반)
- Audit Trail 로깅

## 아키텍처 흐름
```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Server                                │
│   dashboard_api.py (PORT 8001)                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  /api/chat ─────────────► HybridChatbotAgent ─────────► LLM (GPT-4.1)  │
│                                   │                                     │
│                         ┌─────────┴─────────┐                          │
│                         ▼                   ▼                          │
│                  KnowledgeGraph      DocumentRetriever                  │
│                  (온톨로지)          (RAG 가이드라인)                   │
│                                                                         │
│  /api/crawl/start ────► UnifiedBrain ────► AmazonScraper               │
│                              │              (Playwright)                │
│                              ▼                                          │
│                        MetricCalculator                                 │
│                              │                                          │
│                              ▼                                          │
│                       SheetsWriter / SQLite                             │
│                                                                         │
│  /api/data ───────────► dashboard_data.json (캐시된 데이터)            │
│                                                                         │
│  /dashboard ──────────► amore_unified_dashboard_v4.html                │
└─────────────────────────────────────────────────────────────────────────┘
```

## 주요 엔드포인트
- GET  /           : 헬스체크
- GET  /api/data   : 대시보드 데이터 JSON
- POST /api/chat   : 챗봇 v1 (RAG)
- POST /api/v2/chat: 챗봇 v2 (Unified Brain)
- POST /api/v3/chat: 챗봇 v3 (Simple Chat)
- POST /api/crawl/start: 크롤링 시작 (API Key 필요)
- GET  /dashboard  : 대시보드 UI

## 환경 변수
- OPENAI_API_KEY: OpenAI API 키 (필수)
- API_KEY: 보호된 엔드포인트용 인증키
- AUTO_START_SCHEDULER: 서버 시작 시 스케줄러 자동 시작 (default: true)
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# App Factory (미들웨어, 라우터, 정적 파일 등록 포함)
from src.api.app_factory import create_app

# Core 모듈 (startup 이벤트에서 사용)
from src.core.brain import get_initialized_brain
from src.core.crawl_manager import get_crawl_manager

load_dotenv()

logger = logging.getLogger(__name__)


# ============= Lifespan (startup/shutdown) =============

# Railway 배포 시 healthcheck 타임아웃 방지: 기본값 false
# 로컬 개발 시 AUTO_START_SCHEDULER=true 로 설정하면 스케줄러 자동 시작
AUTO_START_SCHEDULER = os.getenv("AUTO_START_SCHEDULER", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """서버 시작/종료 시 설정 검증, 자동 스케줄러 시작 및 즉시 크롤링 체크

    ⚠️ 중요: 크롤링은 백그라운드에서 실행하여 healthcheck 타임아웃 방지
    """
    # === STARTUP ===

    # 0. 설정 검증 (필수 설정 누락 시 경고, 서버는 계속 시작)
    try:
        from src.infrastructure.config.config_manager import AppConfig

        config = AppConfig.from_env_validated(fail_fast=False)
        logging.info(
            f"설정 검증 완료 (port={config.port}, scheduler={config.auto_start_scheduler})"
        )
    except Exception as e:
        logging.error(f"설정 검증 중 오류: {e}")

    # 1. 크롤링 필요 여부 체크 후 백그라운드 실행 (비블로킹)
    try:
        crawl_manager = await get_crawl_manager()
        if crawl_manager.needs_crawl():
            logging.info(
                f"서버 시작: 오늘({crawl_manager.get_kst_today()}) 데이터 없음"
                " → 크롤링 백그라운드 시작"
            )
            # ⚠️ await 대신 create_task로 백그라운드 실행 (healthcheck 블로킹 방지)
            asyncio.create_task(crawl_manager.start_crawl())
        else:
            logging.info(
                f"서버 시작: 오늘 데이터 있음 또는 크롤링 중"
                f" (data_date={crawl_manager.get_data_date()})"
            )
    except Exception as e:
        logging.error(f"서버 시작 크롤링 체크 실패: {e}")

    # 2. 자율 스케줄러 시작 (매일 06:00 정기 크롤링용)
    if AUTO_START_SCHEDULER:
        try:
            brain = await get_initialized_brain()
            await brain.start_scheduler()
            logging.info("자율 스케줄러 자동 시작 완료 (매일 한국시간 06:00 크롤링)")
        except Exception as e:
            logging.error(f"자율 스케줄러 자동 시작 실패: {e}")

    # 3. Export Job Queue Worker 시작 (비동기 내보내기용)
    try:
        from src.tools.exporters.export_handlers import register_all_handlers
        from src.tools.utilities.job_queue import get_job_queue

        queue = get_job_queue()
        await queue.initialize()
        register_all_handlers(queue)
        await queue.start_worker()
        logging.info("Export Job Queue Worker 시작 완료")
    except Exception as e:
        logging.error(f"Export Job Queue Worker 시작 실패: {e}")

    # 4. Telegram Admin Bot 알림 (서버 시작)
    try:
        from src.tools.notifications.telegram_bot import get_bot

        bot = get_bot()
        if bot.is_enabled():
            await bot.send_alert("🚀 서버 시작됨", level="info")
            logging.info("Telegram Admin Bot 활성화됨")
    except Exception as e:
        logging.debug(f"Telegram Bot 알림 실패 (무시): {e}")

    yield

    # === SHUTDOWN ===
    # (현재는 별도 종료 로직 없음)


# App 생성 (app_factory에서 미들웨어, 라우터, 정적 파일 등록 완료)
app = create_app(lifespan=lifespan)


# 글로벌 예외 핸들러 - 에러 발생 시 Telegram 알림
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """모든 예외를 잡아서 Telegram 알림 전송"""

    error_detail = f"{type(exc).__name__}: {str(exc)[:200]}"
    endpoint = f"{request.method} {request.url.path}"

    # 로깅
    logger.error(f"Unhandled exception at {endpoint}: {error_detail}")

    # Telegram 알림 (비동기, 실패해도 무시)
    try:
        from src.tools.notifications.telegram_bot import notify_error

        asyncio.create_task(notify_error(exc, context=f"API: {endpoint}"))
    except Exception:
        pass  # Telegram 알림 실패는 무시

    # 클라이언트에게는 일반 에러 응답
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": error_detail},
    )


# ============= 서버 실행 =============

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)

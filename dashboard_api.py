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
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from litellm import acompletion
from pydantic import BaseModel, Field

# Rate Limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

# Export Routes (DOCX, Excel 비동기 내보내기)
from src.api.routes.export import router as export_router

# External Signal Routes
from src.api.routes.signals import router as signals_router

# Ontology 스키마
# 통합 오케스트레이터 (deprecated - use UnifiedBrain instead)
# Level 4 Brain (LLM-First Autonomous Agent)
from src.core.brain import BrainMode, get_brain, get_initialized_brain
from src.core.crawl_manager import get_crawl_manager
from src.rag.retriever import DocumentRetriever

# RAG 시스템 연동
from src.rag.router import QueryType, RAGRouter

logger = logging.getLogger(__name__)

# Market Intelligence Engine
from src.tools.market_intelligence import MarketIntelligenceEngine

# SQLite Storage
from src.tools.sqlite_storage import get_sqlite_storage

# 환경 변수 로드
load_dotenv()

# Rate Limiter 설정 (IP 기반)
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="AMORE Dashboard API",
    description="LANEIGE Amazon 대시보드 백엔드 API (RAG + Ontology 통합)",
    version="2.0.0",
)

# Rate Limit 초과 시 에러 핸들러
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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
        from src.tools.telegram_bot import notify_error

        asyncio.create_task(notify_error(exc, context=f"API: {endpoint}"))
    except Exception:
        pass  # Telegram 알림 실패는 무시

    # 클라이언트에게는 일반 에러 응답
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": error_detail},
    )


# ============================================================================
# URL 헬퍼 함수 (Railway 환경 자동 감지)
# ============================================================================
def get_base_url() -> str:
    """
    배포 환경에 맞는 Base URL 반환

    우선순위:
    1. DASHBOARD_URL 환경변수 (명시적 설정)
    2. RAILWAY_PUBLIC_DOMAIN (Railway 자동 제공)
    3. localhost:8001 (로컬 개발)
    """
    # 1. 명시적 설정 우선
    if dashboard_url := os.getenv("DASHBOARD_URL"):
        return dashboard_url.rstrip("/")

    # 2. Railway 환경 자동 감지
    if railway_domain := os.getenv("RAILWAY_PUBLIC_DOMAIN"):
        return f"https://{railway_domain}"

    # 3. 로컬 개발 환경
    port = os.getenv("PORT", "8001")
    return f"http://localhost:{port}"


# CORS 설정 (환경변수로 허용 도메인 설정 가능)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8001,http://127.0.0.1:8001").split(
    ","
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)


# Security Headers Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """보안 헤더 추가 미들웨어"""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"  # iframe 임베딩은 같은 도메인만 허용
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response


app.add_middleware(SecurityHeadersMiddleware)

# External Signals Router 등록
app.include_router(signals_router)

# Export Router 등록 (비동기 DOCX/Excel 내보내기)
app.include_router(export_router)

# Telegram Admin Bot Router (관리자 전용)
try:
    from src.tools.telegram_bot import telegram_router

    app.include_router(telegram_router)
    logger.info("Telegram Admin Bot router enabled")
except ImportError as e:
    logger.warning(f"Telegram Bot not available: {e}")

# Static Files 서빙 (폰트, 이미지 등)
# Arita 폰트 파일: /fonts/AritaDotumKR-Medium.ttf 등으로 접근 가능
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    # 편의를 위해 /fonts 경로도 별도 마운트
    FONTS_DIR = STATIC_DIR / "fonts"
    if FONTS_DIR.exists():
        app.mount("/fonts", StaticFiles(directory=str(FONTS_DIR)), name="fonts")

# Dashboard 폴더 서빙
DASHBOARD_DIR = Path(__file__).parent / "dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")

# ============= 서버 시작 시 자동 스케줄러 =============

# Railway 배포 시 healthcheck 타임아웃 방지: 기본값 false
# 로컬 개발 시 AUTO_START_SCHEDULER=true 로 설정하면 스케줄러 자동 시작
AUTO_START_SCHEDULER = os.getenv("AUTO_START_SCHEDULER", "false").lower() == "true"


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 자동 스케줄러 시작 및 즉시 크롤링 체크

    ⚠️ 중요: 크롤링은 백그라운드에서 실행하여 healthcheck 타임아웃 방지
    """
    # 1. 크롤링 필요 여부 체크 후 백그라운드 실행 (비블로킹)
    try:
        crawl_manager = await get_crawl_manager()
        if crawl_manager.needs_crawl():
            logging.info(
                f"서버 시작: 오늘({crawl_manager.get_kst_today()}) 데이터 없음 → 크롤링 백그라운드 시작"
            )
            # ⚠️ await 대신 create_task로 백그라운드 실행 (healthcheck 블로킹 방지)
            asyncio.create_task(crawl_manager.start_crawl())
        else:
            logging.info(
                f"서버 시작: 오늘 데이터 있음 또는 크롤링 중 (data_date={crawl_manager.get_data_date()})"
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
        from src.tools.export_handlers import register_all_handlers
        from src.tools.job_queue import get_job_queue

        queue = get_job_queue()
        await queue.initialize()
        register_all_handlers(queue)
        await queue.start_worker()
        logging.info("Export Job Queue Worker 시작 완료")
    except Exception as e:
        logging.error(f"Export Job Queue Worker 시작 실패: {e}")

    # 4. Telegram Admin Bot 알림 (서버 시작)
    try:
        from src.tools.telegram_bot import get_bot

        bot = get_bot()
        if bot.is_enabled():
            await bot.send_alert("🚀 서버 시작됨", level="info")
            logging.info("Telegram Admin Bot 활성화됨")
    except Exception as e:
        logging.debug(f"Telegram Bot 알림 실패 (무시): {e}")


# 데이터 경로
DATA_PATH = "./data/dashboard_data.json"
DOCS_PATH = "./"  # MD 파일들이 루트에 있음
AUDIT_LOG_DIR = "./logs"

# ============= API Key 인증 설정 =============

# API_KEY: 환경변수 필수 - 기본값 없음 (보안상 하드코딩 금지)
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logging.warning(
        "⚠️ API_KEY 환경변수가 설정되지 않았습니다. 보호된 엔드포인트에 접근할 수 없습니다."
    )
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    API Key 검증 (민감한 엔드포인트용)

    사용법: 엔드포인트에 dependencies=[Depends(verify_api_key)] 추가
    """
    import hmac

    if api_key is None:
        raise HTTPException(
            status_code=401, detail="API Key가 필요합니다. 헤더에 X-API-Key를 추가하세요."
        )
    # 타이밍 공격 방어: hmac.compare_digest 사용
    if not API_KEY or not hmac.compare_digest(api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=403, detail="유효하지 않은 API Key입니다.")
    return api_key


# ============= Audit Trail Logger 설정 =============


def setup_audit_logger():
    """Audit Trail 로거 설정"""
    # 로그 디렉토리 생성
    Path(AUDIT_LOG_DIR).mkdir(parents=True, exist_ok=True)

    # 오늘 날짜 기반 로그 파일
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = Path(AUDIT_LOG_DIR) / f"chatbot_audit_{today}.log"

    # 로거 생성
    audit_logger = logging.getLogger("audit_trail")
    audit_logger.setLevel(logging.INFO)

    # 기존 핸들러 제거 (중복 방지)
    audit_logger.handlers.clear()

    # 파일 핸들러 추가
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # 포맷 설정
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    audit_logger.addHandler(file_handler)

    return audit_logger


audit_logger = setup_audit_logger()


def log_chat_interaction(
    session_id: str,
    user_query: str,
    ai_response: str,
    query_type: str,
    confidence: float,
    entities: dict,
    sources: list[str],
    response_time_ms: float,
):
    """챗봇 대화 Audit Trail 기록"""
    audit_entry = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "user_query": user_query,
        "ai_response": ai_response[:500] + "..." if len(ai_response) > 500 else ai_response,
        "query_type": query_type,
        "confidence": round(confidence, 4),
        "entities": entities,
        "sources": sources,
        "response_time_ms": round(response_time_ms, 2),
    }

    # JSON 형식으로 로그 기록
    audit_logger.info(json.dumps(audit_entry, ensure_ascii=False))


# ============= Global Instances =============

# RAG 시스템
rag_router = RAGRouter()
doc_retriever = DocumentRetriever(DOCS_PATH)

# 세션별 대화 메모리 (TTL 기반 자동 정리)
conversation_memory: dict[str, list[dict[str, str]]] = defaultdict(list)
session_last_activity: dict[str, datetime] = {}  # 세션별 마지막 활동 시간
MAX_MEMORY_TURNS = 10
SESSION_TTL_HOURS = 1  # 세션 만료 시간 (1시간)
MAX_SESSIONS = 1000  # 최대 세션 수


def cleanup_expired_sessions() -> int:
    """만료된 세션 정리 (TTL 기반)"""
    now = datetime.now()
    expired = [
        sid
        for sid, last_time in session_last_activity.items()
        if (now - last_time).total_seconds() > SESSION_TTL_HOURS * 3600
    ]
    for sid in expired:
        if sid in conversation_memory:
            del conversation_memory[sid]
        if sid in session_last_activity:
            del session_last_activity[sid]
    return len(expired)


# 통합 시스템은 UnifiedBrain (get_brain())으로 관리됨

# Market Intelligence Engine 싱글톤
_market_intelligence_engine: MarketIntelligenceEngine | None = None


async def get_market_intelligence() -> MarketIntelligenceEngine:
    """Market Intelligence Engine 싱글톤 반환"""
    global _market_intelligence_engine
    if _market_intelligence_engine is None:
        _market_intelligence_engine = MarketIntelligenceEngine()
        await _market_intelligence_engine.initialize()
    return _market_intelligence_engine


# ============= Pydantic Models =============


class ChatRequest(BaseModel):
    """챗봇 요청"""

    message: str = Field(..., max_length=10000, description="최대 10,000자")
    session_id: str | None = Field(default="default", max_length=100)
    context: dict | None = None


class ChatResponse(BaseModel):
    """챗봇 응답"""

    response: str
    query_type: str
    confidence: float
    sources: list[str]
    suggestions: list[str]
    entities: dict[str, Any]


class ExportRequest(BaseModel):
    """내보내기 요청"""

    start_date: str | None = None
    end_date: str | None = None
    include_strategy: bool = True


class MarketIntelligenceStatusResponse(BaseModel):
    """Market Intelligence 상태 응답"""

    initialized: bool
    layers_collected: list[int]
    last_collection: str | None = None
    stats: dict[str, Any]


class LayerDataResponse(BaseModel):
    """레이어 데이터 응답"""

    layer: int
    layer_name: str
    collected_at: str
    data: dict[str, Any]
    sources: list[dict[str, Any]]


# ============= Helper Functions =============


def load_dashboard_data() -> dict[str, Any]:
    """대시보드 데이터 로드"""
    try:
        with open(DATA_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def get_conversation_history(session_id: str, limit: int = 5) -> str:
    """대화 기록 조회 (문자열 형태)"""
    history = conversation_memory.get(session_id, [])[-limit:]
    if not history:
        return ""

    lines = []
    for turn in history:
        role = "사용자" if turn["role"] == "user" else "AI"
        content = turn["content"][:150] + "..." if len(turn["content"]) > 150 else turn["content"]
        lines.append(f"[{role}]: {content}")

    return "\n".join(lines)


def add_to_memory(session_id: str, role: str, content: str) -> None:
    """대화 메모리에 추가"""
    now = datetime.now()

    # 주기적 정리 (매 100번째 호출마다 또는 세션 수 초과 시)
    if len(session_last_activity) > MAX_SESSIONS or len(session_last_activity) % 100 == 0:
        cleanup_expired_sessions()

    # 마지막 활동 시간 업데이트
    session_last_activity[session_id] = now

    conversation_memory[session_id].append(
        {"role": role, "content": content, "timestamp": now.isoformat()}
    )
    # 최대 개수 유지
    if len(conversation_memory[session_id]) > MAX_MEMORY_TURNS * 2:
        conversation_memory[session_id] = conversation_memory[session_id][-MAX_MEMORY_TURNS * 2 :]


def build_data_context(data: dict, query_type: QueryType, entities: dict) -> str:
    """
    데이터 컨텍스트 구성 (Ontology 기반)

    질문 유형과 추출된 엔티티에 따라 필요한 데이터만 선택
    """
    if not data:
        return "현재 데이터가 없습니다."

    context_parts = []

    # 메타데이터 (항상 포함)
    metadata = data.get("metadata", {})
    context_parts.append(f"""[데이터 현황]
- 기준일: {metadata.get('data_date', 'N/A')}
- 총 제품 수: {metadata.get('total_products', 0)}개
- LANEIGE 제품 수: {metadata.get('laneige_products', 0)}개""")

    # 질문 유형별 데이터 선택
    brand_kpis = data.get("brand", {}).get("kpis", {})

    # 시장/브랜드 지표 (DEFINITION, INTERPRETATION, ANALYSIS)
    if query_type in [
        QueryType.DEFINITION,
        QueryType.INTERPRETATION,
        QueryType.ANALYSIS,
        QueryType.COMBINATION,
    ]:
        if brand_kpis:
            context_parts.append(f"""
[LANEIGE 브랜드 KPI] (Ontology: BrandMetrics)
- SoS (Share of Shelf): {brand_kpis.get('sos', 0)}% {brand_kpis.get('sos_delta', '')}
- Top 10 제품 수: {brand_kpis.get('top10_count', 0)}개
- 평균 순위: {brand_kpis.get('avg_rank', 0)}위
- HHI (시장 집중도): {brand_kpis.get('hhi', 0)}""")

    # 경쟁사 정보 (ANALYSIS, DATA_QUERY에서 경쟁사 언급 시)
    competitors = data.get("brand", {}).get("competitors", [])
    brands_mentioned = entities.get("brands", [])

    if query_type == QueryType.ANALYSIS or any(
        b for b in brands_mentioned if b.lower() != "laneige"
    ):
        if competitors:
            top_comps = competitors[:5]
            comp_lines = [
                f"  - {c['brand']}: SoS {c['sos']}%, 평균 순위 {c['avg_rank']}위, 제품 {c['product_count']}개"
                for c in top_comps
            ]
            context_parts.append("[경쟁사 현황]\n" + "\n".join(comp_lines))

    # 제품 정보 (DATA_QUERY, 특정 제품 언급 시)
    products = data.get("products", {})
    products_mentioned = entities.get("products", [])

    if query_type == QueryType.DATA_QUERY or products_mentioned:
        if products:
            prod_lines = []
            for _asin, p in list(products.items())[:5]:
                prod_lines.append(f"""  - {p['name'][:40]}
    순위: #{p['rank']} ({p['rank_delta']}), 평점: {p['rating']}, 변동성: {p.get('volatility_status', 'N/A')}""")
            context_parts.append(
                "[LANEIGE 제품 현황] (Ontology: ProductMetrics)\n" + "\n".join(prod_lines)
            )

    # 카테고리 정보
    categories = data.get("categories", {})
    categories_mentioned = entities.get("categories", [])

    if categories_mentioned or query_type in [QueryType.ANALYSIS, QueryType.INTERPRETATION]:
        if categories:
            cat_lines = []
            for _cat_id, cat in categories.items():
                cat_lines.append(
                    f"  - {cat['name']}: SoS {cat['sos']}%, 최고 순위 #{cat['best_rank']}, CPI {cat.get('cpi', 100)}"
                )
            context_parts.append(
                "[카테고리 현황] (Ontology: MarketMetrics)\n" + "\n".join(cat_lines)
            )

    # 액션 아이템 (전략 질문)
    if query_type == QueryType.ANALYSIS:
        action_items = data.get("home", {}).get("action_items", [])
        if action_items:
            action_lines = [
                f"  - [{a['priority']}] {a['product_name']}: {a['signal']} → {a['action_tag']}"
                for a in action_items[:4]
            ]
            context_parts.append("[현재 액션 아이템]\n" + "\n".join(action_lines))

    return "\n\n".join(context_parts)


async def get_rag_context(query: str, query_type: QueryType) -> tuple[str, list[str]]:
    """
    RAG 컨텍스트 검색

    Returns:
        (컨텍스트 문자열, 출처 목록)
    """
    # DocumentRetriever 초기화 (처음 호출 시)
    if not doc_retriever._initialized:
        await doc_retriever.initialize()

    # 질문 유형에 맞는 문서 검색
    target_doc = rag_router.get_target_document(query_type)

    # 검색 실행
    results = await doc_retriever.search(query, top_k=3, doc_filter=target_doc)

    if not results:
        return "", []

    # 컨텍스트 구성
    context_parts = []
    sources = []

    for result in results:
        metadata = result.get("metadata", {})
        content = result.get("content", "")
        title = metadata.get("title", "Unknown")
        doc_id = metadata.get("doc_id", "")

        context_parts.append(f"[{title}]\n{content}")

        # 출처 추가
        doc_name_map = {
            "strategic_indicators": "Strategic Indicators Definition",
            "metric_interpretation": "Metric Interpretation Guide",
            "indicator_combination": "Indicator Combination Playbook",
            "home_insight_rules": "Home Page Insight Rules",
        }
        if doc_id in doc_name_map and doc_name_map[doc_id] not in sources:
            sources.append(doc_name_map[doc_id])

    return "\n\n---\n\n".join(context_parts), sources


def generate_dynamic_suggestions(
    query_type: QueryType, entities: dict, response: str, user_query: str = ""
) -> list[str]:
    """
    동적 후속 질문 제안 (v2 - 개선 버전)

    응답 내용, 엔티티, 쿼리 유형을 종합하여 맞춤형 제안 생성

    우선순위:
    1. 응답 키워드 기반 (response 분석)
    2. 엔티티 기반 (브랜드, 카테고리, 지표 활용)
    3. 쿼리 유형 기반 (폴백)

    Args:
        query_type: 질문 유형
        entities: 추출된 엔티티
        response: AI 응답 내용
        user_query: 원본 사용자 질문

    Returns:
        3개의 후속 질문 리스트
    """
    suggestions = []

    # 엔티티 추출
    brands = entities.get("brands", [])
    indicators = entities.get("indicators", [])
    categories = entities.get("categories", [])

    # 1순위: 응답 키워드 기반 제안
    if response:
        keyword_suggestions = _extract_response_keywords(response)
        suggestions.extend(keyword_suggestions)

    # 2순위: 엔티티 기반 제안
    if len(suggestions) < 3:
        entity_suggestions = _generate_entity_suggestions(brands, categories, indicators)
        suggestions.extend(entity_suggestions)

    # 3순위: 쿼리 유형 기반 제안 (폴백)
    if len(suggestions) < 3:
        type_suggestions = _generate_type_suggestions(query_type, brands, indicators)
        suggestions.extend(type_suggestions)

    # 중복 제거 및 상위 3개
    unique = list(dict.fromkeys(suggestions))
    return unique[:3]


def _extract_response_keywords(response: str) -> list[str]:
    """응답에서 후속 질문 관련 키워드 추출"""
    import re

    keywords = []

    # 패턴 매칭 - 응답 내용에 따라 관련 후속 질문 생성
    patterns = {
        r"순위.{0,10}(하락|급락|떨어)": "순위 하락 원인을 분석해주세요",
        r"순위.{0,10}(상승|급등|올라)": "상승 요인을 상세 분석해주세요",
        r"경쟁사|경쟁 브랜드|competitor": "경쟁사 상세 비교를 해주세요",
        r"가격.{0,10}(인상|인하|변동)": "가격 전략을 분석해주세요",
        r"리뷰|평점|rating": "소비자 피드백을 상세 분석해주세요",
        r"트렌드|유행|trend": "트렌드 상세 분석을 해주세요",
        r"성장.{0,5}(기회|가능|potential)": "성장 전략을 제안해주세요",
        r"위험|리스크|위협|risk": "리스크 대응 전략은?",
        r"SoS|점유율|share": "점유율 개선 전략은?",
        r"Top.{0,3}(10|5)|상위": "Top 10 진입 전략은?",
    }

    for pattern, suggestion in patterns.items():
        if re.search(pattern, response, re.IGNORECASE):
            keywords.append(suggestion)
            if len(keywords) >= 2:  # 최대 2개
                break

    return keywords


def _generate_entity_suggestions(
    brands: list[str], categories: list[str], indicators: list[str]
) -> list[str]:
    """엔티티 기반 동적 제안 생성"""
    suggestions = []

    # 브랜드 기반
    if brands:
        brand = brands[0]
        suggestions.append(f"{brand} 경쟁사 비교 분석")
        if len(brands) > 1:
            suggestions.append(f"{brands[0]} vs {brands[1]} 비교")

    # 카테고리 기반
    if categories:
        cat = categories[0]
        suggestions.append(f"{cat} 시장 트렌드 분석")

    # 지표 기반
    if indicators:
        ind = indicators[0].upper()
        suggestions.append(f"{ind} 개선 전략")

    return suggestions


def _generate_type_suggestions(
    query_type: QueryType, brands: list[str], indicators: list[str]
) -> list[str]:
    """쿼리 유형 기반 폴백 제안"""
    suggestions = []

    if query_type == QueryType.DEFINITION:
        if indicators:
            ind = indicators[0].upper()
            suggestions.append(f"{ind}가 높으면 어떤 의미인가요?")
        suggestions.extend(["관련된 다른 지표는?", "실제 데이터에 적용해주세요"])

    elif query_type == QueryType.INTERPRETATION:
        suggestions.extend(
            ["현재 LANEIGE 수치 분석", "경쟁사와 비교해주세요", "개선 액션 아이템은?"]
        )

    elif query_type == QueryType.DATA_QUERY:
        suggestions.extend(["이 수치가 좋은 건가요?", "최근 7일 추이 분석", "경쟁사 대비 현황"])

    elif query_type == QueryType.ANALYSIS:
        suggestions.extend(["가장 시급한 액션은?", "Top 10 진입 전략", "리스크 요인 분석"])

    elif query_type == QueryType.COMBINATION:
        suggestions.extend(["다른 시나리오 분석", "현재 해당 상황 존재 여부"])

    else:
        # 기본 제안
        suggestions = ["SoS(점유율) 설명해주세요", "LANEIGE 현재 순위는?", "전략적 권고사항"]

    return suggestions


# ============= API Endpoints =============


@app.get("/")
async def root():
    """헬스 체크"""
    return {
        "status": "ok",
        "message": "AMORE Dashboard API v2.0 (RAG + Ontology)",
        "features": ["chatbot", "rag", "ontology", "memory", "docx_export"],
    }


@app.get("/api/data")
async def get_data():
    """대시보드 데이터 조회"""
    data = load_dashboard_data()
    if not data:
        raise HTTPException(status_code=404, detail="Dashboard data not found")
    return data


@app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")  # 분당 10회 제한 (보안 강화)
async def chat(request: Request, body: ChatRequest):
    """
    ChatGPT + RAG + Ontology 통합 챗봇 API

    1. 질문 분석 (RAGRouter)
    2. 엔티티 추출 (Ontology 기반)
    3. 관련 문서 검색 (RAG)
    4. 데이터 컨텍스트 구성
    5. 대화 기록 참조
    6. LLM 응답 생성
    7. Audit Trail 로깅
    """
    import time

    start_time = time.time()

    message = body.message.strip()
    session_id = body.session_id or "default"

    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    # 1. 질문 분류 (RAGRouter 사용)
    route_result = rag_router.route(message)
    query_type = route_result["query_type"]
    confidence = route_result["confidence"]

    # 2. 엔티티 추출 (Ontology 기반)
    entities = rag_router.extract_entities(message)

    # 3. 명확화 필요 여부 확인
    clarification = rag_router.needs_clarification(route_result, entities)
    if clarification and confidence < 0.5:
        # 명확화 요청
        add_to_memory(session_id, "user", message)
        add_to_memory(session_id, "assistant", clarification)

        return ChatResponse(
            response=clarification,
            query_type=query_type.value if hasattr(query_type, "value") else str(query_type),
            confidence=confidence,
            sources=[],
            suggestions=[
                "예, 전체 브랜드 분석해주세요",
                "LANEIGE만 분석해주세요",
                "Lip Care 카테고리만",
            ],
            entities=entities,
        )

    # 4. RAG 컨텍스트 검색
    rag_context, sources = await get_rag_context(message, query_type)

    # 5. 데이터 로드 및 컨텍스트 구성
    data = load_dashboard_data()
    data_context = build_data_context(data, query_type, entities)

    # 6. 대화 기록 조회
    conversation_history = get_conversation_history(session_id)

    # 7. 시스템 프롬프트 구성
    system_prompt = """당신은 AMORE Pacific의 LANEIGE 브랜드 Amazon 분석 전문가입니다.

역할:
- Amazon US 베스트셀러 데이터를 분석하여 인사이트 제공
- LANEIGE 브랜드의 시장 포지션 분석
- 경쟁사 대비 전략적 권고 제공
- 지표 정의 및 해석 가이드 제공

Ontology 엔티티 이해:
- Brand: 브랜드 정보 (LANEIGE, 경쟁사 등)
- Product: 제품 정보 (ASIN, 순위, 평점, 가격 등)
- Category: 카테고리 (Lip Care, Skin Care 등)
- BrandMetrics: SoS, 평균순위, 제품수 등
- ProductMetrics: 순위변동성, 연속체류일, 평점추세 등
- MarketMetrics: HHI(시장집중도), 교체율 등

응답 가이드라인:
1. 데이터에 기반한 구체적인 수치 인용
2. RAG 문서의 정의/해석 기준 활용
3. 이전 대화 맥락 고려
4. 간결하고 액션 가능한 인사이트 제공
5. 불확실한 경우 명확히 밝힐 것
6. 단정적 표현 피하기
7. 한국어로 응답

질문 유형별 응답 스타일:
- 정의(DEFINITION): 지표의 정의, 산출식, 의미를 설명
- 해석(INTERPRETATION): 수치의 의미, 좋고 나쁨의 기준 설명
- 조합(COMBINATION): 여러 지표를 함께 해석, 시나리오별 액션 제안
- 데이터조회(DATA_QUERY): 현재 수치와 변동 현황 안내
- 분석(ANALYSIS): 종합 분석과 전략적 권고 제공
"""

    # 8. 사용자 프롬프트 구성
    user_prompt = f"""## 사용자 질문
{message}

## 질문 유형
{query_type.value if hasattr(query_type, 'value') else str(query_type)} (신뢰도: {confidence:.1%})

## 추출된 엔티티
- 브랜드: {', '.join(entities.get('brands', [])) or '없음'}
- 카테고리: {', '.join(entities.get('categories', [])) or '없음'}
- 지표: {', '.join(entities.get('indicators', [])) or '없음'}
- 기간: {entities.get('time_range') or '없음'}

## RAG 참조 문서
{rag_context if rag_context else '관련 문서 없음'}

## 현재 데이터
{data_context}

## 이전 대화
{conversation_history if conversation_history else '이전 대화 없음'}

위 정보를 바탕으로 질문에 답변해주세요.
- 질문 유형에 맞는 응답 스타일을 사용하세요.
- RAG 문서에 관련 정의/해석이 있으면 인용하세요.
- 이전 대화 맥락이 있으면 고려하세요.
"""

    try:
        response = await acompletion(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content

        # 9. 대화 메모리에 저장
        add_to_memory(session_id, "user", message)
        add_to_memory(session_id, "assistant", answer)

        # 10. 동적 후속 질문 제안 (v2 - 개선 버전)
        suggestions = generate_dynamic_suggestions(query_type, entities, answer, message)

        # 11. Audit Trail 로깅
        response_time_ms = (time.time() - start_time) * 1000
        log_chat_interaction(
            session_id=session_id,
            user_query=message,
            ai_response=answer,
            query_type=query_type.value if hasattr(query_type, "value") else str(query_type),
            confidence=confidence,
            entities=entities,
            sources=sources,
            response_time_ms=response_time_ms,
        )

        return ChatResponse(
            response=answer,
            query_type=query_type.value if hasattr(query_type, "value") else str(query_type),
            confidence=confidence,
            sources=sources,
            suggestions=suggestions,
            entities=entities,
        )

    except Exception as e:
        logger.error(f"LLM Error: {e}")

        # Fallback 응답
        fallback = route_result.get("fallback_message") or rag_router.get_fallback_response(
            "unknown"
        )

        # 데이터 기반 기본 응답 추가
        if data and query_type == QueryType.DATA_QUERY:
            brand_kpis = data.get("brand", {}).get("kpis", {})
            fallback = f"""현재 LANEIGE 현황:
- SoS: {brand_kpis.get('sos', 0)}%
- Top 10 제품: {brand_kpis.get('top10_count', 0)}개
- 평균 순위: {brand_kpis.get('avg_rank', 0)}위

(상세 분석을 위해 잠시 후 다시 시도해주세요)"""

        # Fallback 응답도 Audit Trail 기록
        response_time_ms = (time.time() - start_time) * 1000
        log_chat_interaction(
            session_id=session_id,
            user_query=message,
            ai_response=f"[ERROR] {str(e)[:100]} | Fallback: {fallback[:200]}",
            query_type=query_type.value if hasattr(query_type, "value") else str(query_type),
            confidence=0.0,
            entities=entities,
            sources=["fallback"],
            response_time_ms=response_time_ms,
        )

        return ChatResponse(
            response=fallback,
            query_type=query_type.value if hasattr(query_type, "value") else str(query_type),
            confidence=0.0,
            sources=[],
            suggestions=["다시 질문해주세요", "SoS가 뭔가요?", "현재 순위 알려주세요"],
            entities=entities,
        )


@app.delete("/api/chat/memory/{session_id}")
async def clear_memory(session_id: str):
    """세션 대화 기록 초기화"""
    if session_id in conversation_memory:
        del conversation_memory[session_id]
    return {"status": "ok", "message": f"Session {session_id} memory cleared"}


# ============= Simple Chat API (v3 - 단순화) =============

from src.core.simple_chat import get_chat_service


class SimpleChatRequest(BaseModel):
    """Simple Chat 요청"""

    message: str
    session_id: str | None = "default"


class SimpleChatResponse(BaseModel):
    """Simple Chat 응답"""

    text: str
    suggestions: list[str]
    tools_used: list[str]
    sources: list[dict[str, Any]] = []  # AI 출처 정보 추가
    data_date: str
    processing_time_ms: float


@app.post("/api/v3/chat", response_model=SimpleChatResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")  # 분당 10회 제한 (보안 강화)
async def chat_v3(request: Request, body: SimpleChatRequest):
    """
    Simple LLM Chat API (v3)

    단순화된 구조:
    - LLM이 모든 판단 담당
    - Function Calling으로 도구 사용
    - 불필요한 레이어 제거
    """
    message = body.message.strip()
    session_id = body.session_id or "default"

    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    # 크롤링 상태 체크
    crawl_manager = await get_crawl_manager()
    crawl_notification = None
    crawl_started = False

    if crawl_manager.needs_crawl():
        crawl_started = await crawl_manager.start_crawl()

    if crawl_manager.should_notify(session_id):
        crawl_notification = crawl_manager.get_notification_message()
        crawl_manager.mark_notified(session_id)

    # Simple Chat Service로 처리
    chat_service = get_chat_service()
    result = await chat_service.chat(message, session_id)

    # 크롤링 알림 추가
    response_text = result["text"]
    if crawl_notification:
        response_text = f"{crawl_notification}\n\n---\n\n{response_text}"
    elif crawl_started:
        data_date = crawl_manager.get_data_date() or "없음"
        response_text = (
            f"📡 **백그라운드에서 오늘 데이터 수집을 시작합니다.**\n"
            f"현재 데이터: {data_date}\n"
            f"수집이 완료되면 알려드리겠습니다.\n\n---\n\n{response_text}"
        )

    return SimpleChatResponse(
        text=response_text,
        suggestions=result.get("suggestions", []),
        tools_used=result.get("tools_used", []),
        sources=result.get("sources", []),  # AI 출처 정보 전달
        data_date=result.get("data_date", "N/A"),
        processing_time_ms=result.get("processing_time_ms", 0),
    )


@app.post("/api/v3/chat/stream", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")  # 분당 10회 제한 (보안 강화)
async def chat_v3_stream(request: Request, body: SimpleChatRequest):
    """
    Simple LLM Chat API with SSE Streaming (v3)

    SSE 형식으로 실시간 스트리밍 응답을 반환합니다.

    이벤트 타입:
    - text: 응답 텍스트 청크
    - tool_call: 도구 호출 정보
    - done: 완료 (후속 질문 등 메타데이터 포함)
    - error: 오류 발생
    """
    message = body.message.strip()
    session_id = body.session_id or "default"

    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    chat_service = get_chat_service()

    async def generate():
        """SSE 이벤트 생성기"""
        try:
            async for chunk in chat_service.chat_stream(message, session_id):
                event_type = chunk.get("type", "text")
                content = chunk.get("content", "")

                # SSE 형식으로 변환
                if event_type == "text":
                    data = json.dumps({"type": "text", "content": content}, ensure_ascii=False)
                elif event_type == "tool_call":
                    data = json.dumps({"type": "tool_call", "content": content}, ensure_ascii=False)
                elif event_type == "done":
                    data = json.dumps({"type": "done", "content": content}, ensure_ascii=False)
                elif event_type == "error":
                    data = json.dumps({"type": "error", "content": content}, ensure_ascii=False)
                else:
                    data = json.dumps(chunk, ensure_ascii=False)

                yield f"data: {data}\n\n"

        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            error_data = json.dumps({"type": "error", "content": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx 버퍼링 비활성화
        },
    )


# ============= LLM Orchestrator API (v2 - 기존, deprecated) =============


class OrchestratorChatRequest(BaseModel):
    """LLM Orchestrator 챗봇 요청"""

    message: str
    session_id: str | None = "default"
    skip_cache: bool = False


class OrchestratorChatResponse(BaseModel):
    """LLM Orchestrator 챗봇 응답"""

    text: str
    query_type: str
    confidence_level: str
    confidence_score: float
    sources: list[str]
    entities: dict[str, Any]
    tools_called: list[str]
    suggestions: list[str]
    is_fallback: bool
    is_clarification: bool
    processing_time_ms: float


@app.post(
    "/api/v2/chat", response_model=OrchestratorChatResponse, dependencies=[Depends(verify_api_key)]
)
@limiter.limit("10/minute")  # 분당 10회 제한 (보안 강화)
async def chat_v2(request: Request, body: OrchestratorChatRequest):
    """
    통합 오케스트레이터 기반 챗봇 API (v2)

    동작 흐름:
    1. 질문 수신
    2. 시스템 상태 점검 (데이터 신선도, 사용 가능 에이전트)
    3. LLM이 상황 판단 → 에이전트 선택
    4. 에이전트 실행 (에러 시 전략에 따라 처리)
    5. 응답 생성

    에러 전략:
    - RETRY: 재시도 (최대 2회)
    - FALLBACK: 캐시 데이터 사용
    - SKIP: 건너뛰고 계속
    - ABORT: 중단 + 사용자 알림
    """
    import time

    start_time = time.time()

    message = body.message.strip()
    session_id = body.session_id or "default"

    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    # === 크롤링 상태 체크 ===
    crawl_manager = await get_crawl_manager()
    crawl_notification = None
    crawl_started = False

    # 오늘 데이터가 없고, 크롤링 중이 아니면 시작
    if crawl_manager.needs_crawl():
        crawl_started = await crawl_manager.start_crawl()
        if crawl_started:
            logging.info("Background crawl started for today's data")

    # 크롤링 완료 알림 체크 (이 세션에서 아직 안 알렸으면)
    if crawl_manager.should_notify(session_id):
        crawl_notification = crawl_manager.get_notification_message()
        crawl_manager.mark_notified(session_id)

    try:
        # UnifiedBrain으로 처리
        brain = get_brain()

        # 현재 메트릭 데이터 로드
        data = load_dashboard_data()
        current_metrics = data if data else None

        # 처리
        response = await brain.process_query(
            query=message,
            session_id=session_id,
            current_metrics=current_metrics,
            skip_cache=body.skip_cache,
        )

        # 응답 변환 (UnifiedBrain response 처리)
        response_dict = response.to_dict() if hasattr(response, "to_dict") else response

        # 응답 텍스트 구성
        response_text = response_dict.get("text", response_dict.get("content", ""))

        # 크롤링 알림 추가
        if crawl_notification:
            response_text = f"{crawl_notification}\n\n---\n\n{response_text}"
        elif crawl_started:
            # 크롤링 시작 알림
            data_date = crawl_manager.get_data_date() or "없음"
            response_text = (
                f"📡 **백그라운드에서 오늘 데이터 수집을 시작합니다.**\n"
                f"현재 데이터: {data_date}\n"
                f"수집이 완료되면 알려드리겠습니다.\n\n---\n\n{response_text}"
            )

        # 응답 변환
        return OrchestratorChatResponse(
            text=response_text,
            query_type=response_dict.get("query_type", "unknown"),
            confidence_level=response_dict.get("confidence_level", "medium"),
            confidence_score=response_dict.get(
                "confidence_score", response_dict.get("confidence", 0.5)
            ),
            sources=response_dict.get("sources", []),
            entities=response_dict.get("entities", {}),
            tools_called=response_dict.get("tools_called", response_dict.get("tools_used", [])),
            suggestions=response_dict.get("suggestions", []),
            is_fallback=response_dict.get("is_fallback", False),
            is_clarification=response_dict.get("is_clarification", False),
            processing_time_ms=response_dict.get("processing_time_ms", 0),
        )

    except Exception as e:
        logging.error(f"Orchestrator error: {e}")
        return OrchestratorChatResponse(
            text=f"처리 중 오류가 발생했습니다: {str(e)}",
            query_type="error",
            confidence_level="unknown",
            confidence_score=0.0,
            sources=[],
            entities={},
            tools_called=[],
            suggestions=["다시 질문해주세요"],
            is_fallback=True,
            is_clarification=False,
            processing_time_ms=(time.time() - start_time) * 1000,
        )


@app.get("/api/v2/stats")
async def get_orchestrator_stats():
    """UnifiedBrain 통계 조회"""
    brain = get_brain()
    return brain.get_stats() if hasattr(brain, "get_stats") else {"status": "ok"}


@app.get("/api/v2/state")
async def get_orchestrator_state():
    """UnifiedBrain 상태 조회"""
    brain = get_brain()
    return {
        "summary": brain.get_state_summary() if hasattr(brain, "get_state_summary") else {},
        "state": brain.state.to_dict()
        if hasattr(brain, "state") and hasattr(brain.state, "to_dict")
        else {},
    }


@app.get("/api/v2/errors")
async def get_orchestrator_errors():
    """UnifiedBrain 최근 에러 조회"""
    brain = get_brain()
    return {
        "recent_errors": brain.get_recent_errors(limit=20)
        if hasattr(brain, "get_recent_errors")
        else [],
        "stats": brain.get_stats() if hasattr(brain, "get_stats") else {},
    }


@app.post("/api/v2/reset-errors")
async def reset_orchestrator_errors():
    """실패한 에이전트 목록 초기화"""
    brain = get_brain()
    if hasattr(brain, "reset_failed_agents"):
        brain.reset_failed_agents()
    return {"status": "ok", "message": "Failed agents list cleared"}


@app.get("/api/crawl/status")
async def get_crawl_status():
    """
    크롤링 상태 조회

    Returns:
        - status: idle/running/completed/failed
        - date: 크롤링 대상 날짜
        - progress: 진행률 (0-100)
        - data_date: 현재 데이터 날짜
        - needs_crawl: 크롤링 필요 여부
    """
    crawl_manager = await get_crawl_manager()
    return {
        **crawl_manager.state.to_dict(),
        "data_date": crawl_manager.get_data_date(),
        "needs_crawl": crawl_manager.needs_crawl(),
        "is_today_available": crawl_manager.is_today_data_available(),
        "status_message": crawl_manager.get_status_message(),
    }


@app.post("/api/crawl/start", dependencies=[Depends(verify_api_key)])
async def start_crawl():
    """
    수동으로 크롤링 시작 (API Key 필요)

    Returns:
        - started: 크롤링 시작 여부
        - message: 상태 메시지
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


# ============= Historical Data API =============

from datetime import UTC, timedelta

from src.tools.sheets_writer import SheetsWriter

# SheetsWriter 싱글톤 인스턴스
_sheets_writer: SheetsWriter | None = None


def get_sheets_writer() -> SheetsWriter:
    """SheetsWriter 싱글톤 인스턴스 반환"""
    global _sheets_writer
    if _sheets_writer is None:
        _sheets_writer = SheetsWriter()
    return _sheets_writer


@app.get("/api/historical")
async def get_historical_data(
    start_date: str, end_date: str, category_id: str | None = None, brand: str | None = "LANEIGE"
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
    try:
        records = []
        data_source = None

        # 1차: SQLite에서 조회 (빠름)
        try:
            sqlite = get_sqlite_storage()
            await sqlite.initialize()
            records = await sqlite.get_raw_data(
                start_date=start_date,
                end_date=end_date,
                category_id=category_id,
                limit=50000,  # 충분히 큰 limit
            )
            if records:
                data_source = "sqlite"
                logging.info(
                    f"Historical: loaded {len(records)} records from SQLite ({start_date} ~ {end_date})"
                )
        except Exception as sqlite_err:
            logging.warning(f"Historical: SQLite 조회 실패: {sqlite_err}")

        # 2차: SQLite 실패/빈 결과 시 Google Sheets fallback
        if not records:
            try:
                sheets_writer = get_sheets_writer()
                if not sheets_writer._initialized:
                    await sheets_writer.initialize()
                records = await sheets_writer.get_raw_data(
                    start_date=start_date, end_date=end_date, category_id=category_id
                )
                if records:
                    data_source = "sheets"
                    logging.info(
                        f"Historical: loaded {len(records)} records from Sheets ({start_date} ~ {end_date})"
                    )
            except Exception as sheets_err:
                logging.warning(f"Historical: Google Sheets 조회 실패: {sheets_err}")

        if not records:
            # 모든 소스에서 데이터 없음 - 로컬 JSON 파일에서 시도
            return await _get_historical_from_local(start_date, end_date, brand)

        # 날짜 범위 계산
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end_dt - start_dt).days + 1

        # 날짜별 데이터 집계 (특정 브랜드 필터링)
        daily_data = {}
        brand_lower = brand.lower() if brand else ""
        for record in records:
            snapshot_date = record.get("snapshot_date", "")
            if not snapshot_date or snapshot_date < start_date or snapshot_date > end_date:
                continue

            # 특정 브랜드 필터링 (SoS 추이 계산용)
            record_brand = record.get("brand", "")
            if brand_lower and record_brand.lower() != brand_lower:
                continue

            if snapshot_date not in daily_data:
                daily_data[snapshot_date] = {
                    "date": snapshot_date,
                    "products": [],
                    "total_count": 0,
                    "top10_count": 0,
                }

            rank = int(record.get("rank", 0)) if record.get("rank") else 0
            daily_data[snapshot_date]["products"].append(
                {
                    "asin": record.get("asin", ""),
                    "product_name": record.get("product_name", ""),
                    "brand": record_brand,
                    "rank": rank,
                    "price": record.get("price", ""),
                    "rating": record.get("rating", ""),
                }
            )
            daily_data[snapshot_date]["total_count"] += 1
            if rank <= 10:
                daily_data[snapshot_date]["top10_count"] += 1

        # SoS 추이 계산 (Top 100 기준, 해당 브랜드 기준)
        sos_history = []
        raw_data = []
        for date_str in sorted(daily_data.keys()):
            day_data = daily_data[date_str]
            products = day_data["products"]

            # SoS = (브랜드 제품 수 / 100) * 100
            sos = round(len(products) / 100 * 100, 1) if products else 0
            sos_history.append(
                {
                    "date": date_str,
                    "sos": sos,
                    "product_count": len(products),
                    "top10_count": day_data["top10_count"],
                }
            )

            # 평균 순위 (있는 경우)
            if products:
                avg_rank = round(sum(p["rank"] for p in products) / len(products), 1)
                raw_data.append(
                    {
                        "date": date_str,
                        "rank": avg_rank,
                        "best_rank": min(p["rank"] for p in products),
                        "worst_rank": max(p["rank"] for p in products),
                    }
                )

        # available_dates 계산
        available_dates = sorted(daily_data.keys())

        # brand_metrics 계산 (전체 기간 통합 - 모든 브랜드 포함)
        brand_metrics = await _calculate_brand_metrics_for_period(records, daily_data, brand)

        # rank_history 생성 (Product View 차트용)
        # 형식: { "2026-01-14": { "products": [{ "name": "...", "rank": 5, "price": 21.5 }, ...] } }
        rank_history = {}
        for record in records:
            snapshot_date = record.get("snapshot_date", "")
            if not snapshot_date or snapshot_date < start_date or snapshot_date > end_date:
                continue

            if snapshot_date not in rank_history:
                rank_history[snapshot_date] = {"products": []}

            rank = int(record.get("rank", 0)) if record.get("rank") else 0
            price_val = record.get("price", 0)
            try:
                price = float(str(price_val).replace("$", "").replace(",", "")) if price_val else 0
            except (ValueError, TypeError):
                price = 0

            rank_history[snapshot_date]["products"].append(
                {
                    "name": record.get("product_name", ""),
                    "product_name": record.get("product_name", ""),
                    "brand": record.get("brand", ""),
                    "asin": record.get("asin", ""),
                    "rank": rank,
                    "price": price,
                    "rating": record.get("rating", ""),
                    "discount_percent": record.get("discount_percent", 0),
                }
            )

        # 전체 데이터의 사용 가능한 날짜 범위 조회 (SQLite에서)
        available_date_range = {"min": None, "max": None}
        try:
            sqlite = get_sqlite_storage()
            stats = sqlite.get_stats()
            if "date_range" in stats:
                available_date_range = stats["date_range"]
        except Exception:
            pass

        return {
            "success": True,
            "available_dates": available_dates,
            "available_date_range": available_date_range,
            "data_source": data_source,
            "brand_metrics": brand_metrics,
            "rank_history": rank_history,
            "data": {
                "sos_history": sos_history,
                "raw_data": raw_data,
                "daily_data": list(daily_data.values()),
                "period": {"start": start_date, "end": end_date, "days": days},
                "brand": brand,
            },
        }

    except Exception as e:
        logging.error(f"Historical data error: {e}")
        # 폴백: 로컬 데이터에서 시도
        return await _get_historical_from_local(start_date, end_date, brand)


async def _calculate_brand_metrics_for_period(
    records: list[dict], daily_data: dict, target_brand: str
) -> list[dict]:
    """
    기간 내 모든 브랜드의 메트릭 계산 (SoS × Avg Rank 차트용)

    Note:
        기간 조회 시 동일 ASIN이 여러 날짜에 중복 등장하므로,
        ASIN 기준 유니크 카운트를 적용하여 정확한 제품 수 계산

    Returns:
        브랜드별 SoS, 평균 순위, 제품 수 등
    """
    # 전체 제품 데이터 집계 (모든 브랜드)
    brand_data = {}
    brand_unique_asins: dict[str, set] = {}  # ASIN 중복 제거용

    for record in records:
        brand_name = record.get("brand", "Unknown")
        asin = record.get("asin", "")
        rank = int(record.get("rank", 0)) if record.get("rank") else 0

        # Unknown 브랜드 및 빈 브랜드 제외 (대시보드에서 의미 없음)
        if not brand_name or brand_name.lower() == "unknown" or rank == 0:
            continue

        if brand_name not in brand_data:
            brand_data[brand_name] = {"brand": brand_name, "ranks": [], "product_count": 0}
            brand_unique_asins[brand_name] = set()

        # 순위는 모든 레코드에서 수집 (평균 계산용)
        brand_data[brand_name]["ranks"].append(rank)

        # 제품 수는 ASIN 기준 유니크 카운트 (중복 제거)
        if asin and asin not in brand_unique_asins[brand_name]:
            brand_unique_asins[brand_name].add(asin)
            brand_data[brand_name]["product_count"] += 1
        elif not asin:
            # ASIN이 없는 경우 기존 방식으로 카운트 (폴백)
            brand_data[brand_name]["product_count"] += 1

    # 총 유니크 제품 수 (모든 브랜드 - Unknown 제외 후)
    total_products = sum(b["product_count"] for b in brand_data.values())

    # 메트릭 계산
    brand_metrics = []
    for brand_name, data in brand_data.items():
        if not data["ranks"]:
            continue

        sos = round(data["product_count"] / max(total_products, 100) * 100, 2)
        avg_rank = round(sum(data["ranks"]) / len(data["ranks"]), 1)

        # 버블 크기: 제품 수 기반 (최소 5, 최대 25)
        bubble_size = max(5, min(25, data["product_count"] * 2))

        is_laneige = target_brand.upper() in brand_name.upper()

        brand_metrics.append(
            {
                "brand": brand_name,
                "sos": sos,
                "avg_rank": avg_rank,
                "product_count": data["product_count"],
                "bubble_size": bubble_size,
                "is_laneige": is_laneige,
            }
        )

    # SoS 기준 내림차순 정렬
    brand_metrics.sort(key=lambda x: x["sos"], reverse=True)

    # 상위 10개 추출
    top_10 = brand_metrics[:10]

    # LANEIGE가 top_10에 포함되어 있는지 확인 (brand_data에 있는지가 아니라 top_10에 있는지!)
    laneige_in_top10 = any(b.get("is_laneige") for b in top_10)

    # LANEIGE가 top_10에 없으면 추가 (데이터가 존재할 경우)
    if not laneige_in_top10 and target_brand:
        # brand_data에서 LANEIGE 찾기 (대소문자 변형 모두 시도)
        laneige_data = None
        for key in [
            target_brand,
            target_brand.upper(),
            target_brand.lower(),
            target_brand.capitalize(),
        ]:
            if key in brand_data:
                laneige_data = brand_data[key]
                break

        if laneige_data and laneige_data["ranks"]:
            sos = round(laneige_data["product_count"] / max(total_products, 100) * 100, 2)
            avg_rank = round(sum(laneige_data["ranks"]) / len(laneige_data["ranks"]), 1)
            bubble_size = max(5, min(25, laneige_data["product_count"] * 2))
            top_10.append(
                {
                    "brand": target_brand,
                    "sos": sos,
                    "avg_rank": avg_rank,
                    "product_count": laneige_data["product_count"],
                    "bubble_size": bubble_size,
                    "is_laneige": True,
                }
            )
            # 다시 정렬 후 상위 11개 유지 (LANEIGE 포함 보장)
            top_10.sort(key=lambda x: x["sos"], reverse=True)

    # Summer Fridays 특별 처리 (고객 요청 tracked competitor)
    # top_10에 없으면 강제 추가
    TRACKED_COMPETITORS = ["Summer Fridays"]
    for tracked_brand in TRACKED_COMPETITORS:
        tracked_in_top = any(b.get("brand") == tracked_brand for b in top_10)
        if not tracked_in_top and tracked_brand in brand_data:
            tracked_data = brand_data[tracked_brand]
            if tracked_data["ranks"]:
                sos = round(tracked_data["product_count"] / max(total_products, 100) * 100, 2)
                avg_rank = round(sum(tracked_data["ranks"]) / len(tracked_data["ranks"]), 1)
                bubble_size = max(5, min(25, tracked_data["product_count"] * 2))
                top_10.append(
                    {
                        "brand": tracked_brand,
                        "sos": sos,
                        "avg_rank": avg_rank,
                        "product_count": tracked_data["product_count"],
                        "bubble_size": bubble_size,
                        "is_laneige": False,
                        "is_tracked": True,  # tracked competitor 표시
                    }
                )
        # 데이터가 없어도 placeholder 추가 (UI에서 "-" 대신 "데이터 없음" 표시 가능)
        elif not tracked_in_top:
            top_10.append(
                {
                    "brand": tracked_brand,
                    "sos": 0,
                    "avg_rank": None,
                    "product_count": 0,
                    "bubble_size": 5,
                    "is_laneige": False,
                    "is_tracked": True,
                    "no_data": True,  # 해당 기간 데이터 없음 표시
                }
            )

    # 최종 정렬 (SoS 내림차순, tracked는 하단에 유지)
    top_10.sort(key=lambda x: (not x.get("is_tracked", False), x["sos"]), reverse=True)

    return top_10


def _get_brand_metrics_from_dashboard(dashboard_data: dict | None, target_brand: str) -> list[dict]:
    """
    대시보드 데이터에서 브랜드 메트릭 추출 (로컬 폴백용)
    """
    if not dashboard_data:
        return []

    # 대시보드의 brand_matrix 데이터 사용
    brand_matrix = dashboard_data.get("charts", {}).get("brand_matrix", [])
    if brand_matrix:
        return brand_matrix

    # 경쟁사 데이터에서 생성
    competitors = dashboard_data.get("brand", {}).get("competitors", [])
    if not competitors:
        return []

    brand_metrics = []
    for comp in competitors:
        brand_metrics.append(
            {
                "brand": comp.get("brand", "Unknown"),
                "sos": comp.get("sos", 0),
                "avg_rank": comp.get("avg_rank", 50),
                "product_count": comp.get("product_count", 0),
                "bubble_size": max(5, min(25, comp.get("product_count", 0) * 2)),
                "is_laneige": target_brand.upper() in comp.get("brand", "").upper(),
            }
        )

    return brand_metrics


async def _get_historical_from_local(
    start_date: str, end_date: str, brand: str = "LANEIGE"
) -> dict[str, Any]:
    """
    로컬 JSON 파일에서 히스토리컬 데이터 조회 (폴백)

    data/ 폴더의 날짜별 JSON 파일이나 dashboard_data.json의 히스토리 데이터 활용
    """
    try:
        # 메인 대시보드 데이터 로드
        data = load_dashboard_data()
        sos_history = []
        raw_data = []

        # 1. 대시보드 데이터에서 현재 SoS/순위 정보 추출
        if data:
            brand_kpis = data.get("brand", {}).get("kpis", {})
            current_sos = brand_kpis.get("sos", 0)
            data_date = data.get("metadata", {}).get(
                "data_date", datetime.now().strftime("%Y-%m-%d")
            )

            # 현재 날짜가 요청 범위에 포함되면 추가
            if start_date <= data_date <= end_date:
                sos_history.append(
                    {
                        "date": data_date,
                        "sos": current_sos,
                        "product_count": brand_kpis.get("product_count", 0),
                        "top10_count": brand_kpis.get("top10_count", 0),
                    }
                )

                avg_rank = brand_kpis.get("avg_rank", 0)
                if avg_rank:
                    raw_data.append(
                        {
                            "date": data_date,
                            "rank": avg_rank,
                            "best_rank": brand_kpis.get("best_rank", avg_rank),
                            "worst_rank": brand_kpis.get("worst_rank", avg_rank),
                        }
                    )

        # 2. latest_crawl_result.json에서 데이터 추출
        latest_crawl_path = Path("./data/latest_crawl_result.json")
        if latest_crawl_path.exists():
            try:
                with open(latest_crawl_path, encoding="utf-8") as f:
                    crawl_data = json.load(f)

                # 모든 카테고리에서 브랜드 제품 찾기
                brand_products = []
                crawl_date = None

                for _cat_id, cat_data in crawl_data.get("categories", {}).items():
                    for product in cat_data.get("products", []):
                        product_brand = product.get("brand", "")
                        product_name = product.get("product_name", "")

                        # 브랜드 매칭 (대소문자 무시, 부분 매칭)
                        if (
                            brand.upper() in product_brand.upper()
                            or brand.upper() in product_name.upper()
                        ):
                            brand_products.append(product)
                            if not crawl_date:
                                crawl_date = product.get("snapshot_date")

                if brand_products and crawl_date and start_date <= crawl_date <= end_date:
                    # 중복 제거 확인
                    if not any(h["date"] == crawl_date for h in sos_history):
                        # 카테고리별 총 제품 수 (Top 100 기준)
                        total_products = sum(
                            len(cat.get("products", []))
                            for cat in crawl_data.get("categories", {}).values()
                        )

                        sos = round(len(brand_products) / max(total_products, 100) * 100, 2)
                        avg_rank = round(
                            sum(p.get("rank", 0) for p in brand_products) / len(brand_products), 1
                        )

                        sos_history.append(
                            {
                                "date": crawl_date,
                                "sos": sos,
                                "product_count": len(brand_products),
                                "top10_count": sum(
                                    1 for p in brand_products if p.get("rank", 100) <= 10
                                ),
                            }
                        )
                        raw_data.append(
                            {
                                "date": crawl_date,
                                "rank": avg_rank,
                                "best_rank": min(p.get("rank", 100) for p in brand_products),
                                "worst_rank": max(p.get("rank", 100) for p in brand_products),
                            }
                        )

            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"Failed to parse latest_crawl_result.json: {e}")

        # 3. raw_products 폴더에서 날짜별 데이터 검색 (기존 로직)
        raw_data_dir = Path("./data/raw_products")
        if raw_data_dir.exists():
            for json_file in raw_data_dir.glob("*.json"):
                try:
                    file_date = json_file.stem  # 파일명이 YYYY-MM-DD 형식이라고 가정
                    if start_date <= file_date <= end_date:
                        with open(json_file, encoding="utf-8") as f:
                            daily_raw = json.load(f)

                        # 브랜드 제품만 필터링
                        brand_products = [
                            p
                            for p in daily_raw
                            if brand.upper() in p.get("brand", "").upper()
                            or brand.upper() in p.get("product_name", "").upper()
                        ]

                        if brand_products:
                            sos = round(len(brand_products) / 100 * 100, 1)
                            avg_rank = round(
                                sum(p.get("rank", 0) for p in brand_products) / len(brand_products),
                                1,
                            )

                            # 중복 제거
                            if not any(h["date"] == file_date for h in sos_history):
                                sos_history.append(
                                    {
                                        "date": file_date,
                                        "sos": sos,
                                        "product_count": len(brand_products),
                                        "top10_count": sum(
                                            1 for p in brand_products if p.get("rank", 100) <= 10
                                        ),
                                    }
                                )
                                raw_data.append(
                                    {
                                        "date": file_date,
                                        "rank": avg_rank,
                                        "best_rank": min(
                                            p.get("rank", 100) for p in brand_products
                                        ),
                                        "worst_rank": max(
                                            p.get("rank", 100) for p in brand_products
                                        ),
                                    }
                                )
                except (json.JSONDecodeError, ValueError):
                    continue

        # 날짜순 정렬
        sos_history.sort(key=lambda x: x["date"])
        raw_data.sort(key=lambda x: x["date"])

        # available_dates 계산
        available_dates = [h["date"] for h in sos_history]

        # brand_metrics 계산 (현재 대시보드 데이터에서)
        brand_metrics = _get_brand_metrics_from_dashboard(data, brand)

        if not sos_history:
            return {
                "success": False,
                "error": "No historical data found for the specified period",
                "available_dates": [],
                "brand_metrics": [],
                "data": None,
            }

        return {
            "success": True,
            "available_dates": available_dates,
            "brand_metrics": brand_metrics,
            "data": {
                "sos_history": sos_history,
                "raw_data": raw_data,
                "period": {"start": start_date, "end": end_date},
                "brand": brand,
                "source": "local",
            },
        }

    except Exception as e:
        logging.error(f"Local historical data error: {e}")
        return {
            "success": False,
            "error": str(e),
            "available_dates": [],
            "brand_metrics": [],
            "data": None,
        }


@app.post("/api/export/docx")
async def export_docx(request: ExportRequest):
    """
    인사이트 리포트 DOCX 생성 및 다운로드
    """
    data = load_dashboard_data()
    if not data:
        raise HTTPException(status_code=404, detail="Dashboard data not found")

    # DOCX 문서 생성
    doc = Document()

    # 스타일 설정
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Arial"
    font.size = Pt(11)

    # ===== 표지 =====
    title = doc.add_heading("AMORE INSIGHT Report", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph("LANEIGE Amazon US 분석 리포트")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 날짜
    metadata = data.get("metadata", {})
    date_para = doc.add_paragraph()
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_para.add_run(
        f"분석 기준일: {metadata.get('data_date', datetime.now().strftime('%Y-%m-%d'))}"
    )
    date_para.add_run(f"\n생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    doc.add_page_break()

    # ===== 1. Executive Summary =====
    doc.add_heading("1. Executive Summary", level=1)

    brand_kpis = data.get("brand", {}).get("kpis", {})
    home_status = data.get("home", {}).get("status", {})

    summary_text = f"""
LANEIGE 브랜드는 Amazon US 시장에서 {home_status.get('exposure', 'N/A')} 상태입니다.

• Share of Shelf (SoS): {brand_kpis.get('sos', 0)}%
• Top 10 진입 제품: {brand_kpis.get('top10_count', 0)}개
• 평균 순위: {brand_kpis.get('avg_rank', 0)}위
• 시장 집중도 (HHI): {brand_kpis.get('hhi', 0)}

현재 시장 포지션: {home_status.get('position', 'N/A')}
주의 필요 제품: {home_status.get('warning_count', 0)}개
"""
    doc.add_paragraph(summary_text)

    # ===== 2. 제품별 현황 =====
    doc.add_heading("2. LANEIGE 제품 현황", level=1)

    products = data.get("products", {})
    if products:
        # 테이블 생성
        table = doc.add_table(rows=1, cols=5)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER

        # 헤더
        header_cells = table.rows[0].cells
        headers = ["제품명", "순위", "변동", "평점", "변동성"]
        for i, header in enumerate(headers):
            header_cells[i].text = header
            header_cells[i].paragraphs[0].runs[0].bold = True

        # 데이터 행
        for _asin, product in products.items():
            row = table.add_row().cells
            row[0].text = product.get("name", "")[:40]
            row[1].text = f"#{product.get('rank', 'N/A')}"
            row[2].text = product.get("rank_delta", "-")
            row[3].text = str(product.get("rating", "-"))
            row[4].text = product.get("volatility_status", "-")

    doc.add_paragraph()

    # ===== 3. 경쟁사 분석 =====
    doc.add_heading("3. 경쟁사 분석", level=1)

    competitors = data.get("brand", {}).get("competitors", [])
    if competitors:
        table = doc.add_table(rows=1, cols=4)
        table.style = "Table Grid"

        header_cells = table.rows[0].cells
        headers = ["브랜드", "SoS (%)", "평균 순위", "제품 수"]
        for i, header in enumerate(headers):
            header_cells[i].text = header
            header_cells[i].paragraphs[0].runs[0].bold = True

        for comp in competitors[:10]:
            row = table.add_row().cells
            row[0].text = comp.get("brand", "")
            row[1].text = str(comp.get("sos", 0))
            row[2].text = str(comp.get("avg_rank", "-"))
            row[3].text = str(comp.get("product_count", 0))

    doc.add_paragraph()

    # ===== 4. 액션 아이템 =====
    doc.add_heading("4. 액션 아이템", level=1)

    action_items = data.get("home", {}).get("action_items", [])
    if action_items:
        for item in action_items:
            priority_marker = "🔴" if item.get("priority") == "P1" else "🟠"
            para = doc.add_paragraph()
            para.add_run(f"{priority_marker} [{item.get('priority')}] ").bold = True
            para.add_run(f"{item.get('product_name', '')}\n")
            para.add_run(f"   신호: {item.get('signal', '')}\n")
            para.add_run(f"   권장 액션: {item.get('action_tag', '')}")
    else:
        doc.add_paragraph("현재 특별한 액션 아이템이 없습니다.")

    # ===== 5. 전략적 권고사항 =====
    if request.include_strategy:
        doc.add_heading("5. 전략적 권고사항", level=1)

        # ChatGPT로 전략 생성 (RAG 컨텍스트 활용)
        try:
            # RAG에서 전략 관련 컨텍스트 검색
            strategy_context, _ = await get_rag_context("전략 액션 권고", QueryType.ANALYSIS)

            strategy_prompt = f"""다음 데이터와 가이드라인을 바탕으로 LANEIGE 브랜드의 전략적 권고사항 3가지를 작성해주세요.

데이터:
- SoS: {brand_kpis.get('sos', 0)}%
- Top 10 제품: {brand_kpis.get('top10_count', 0)}개
- 평균 순위: {brand_kpis.get('avg_rank', 0)}위
- 주요 경쟁사: {', '.join([c['brand'] for c in competitors[:3]])}

참고 가이드라인:
{strategy_context if strategy_context else '기본 전략 기준 적용'}

각 권고사항은 1-2문장으로 간결하게 작성하세요.
"""
            response = await acompletion(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 뷰티 이커머스 전문 컨설턴트입니다. 간결하고 실행 가능한 전략을 제안합니다.",
                    },
                    {"role": "user", "content": strategy_prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            strategy_text = response.choices[0].message.content
            doc.add_paragraph(strategy_text)

        except Exception:
            # 폴백 전략
            doc.add_paragraph("""
1. Top 10 유지 전략: 현재 상위권 제품의 리뷰 관리 및 재고 확보를 통한 포지션 유지

2. 경쟁사 모니터링: e.l.f., Maybelline 등 주요 경쟁사의 가격 및 프로모션 동향 파악

3. 신규 진입 기회: Lip Care 카테고리 외 Face Powder, Toner 등 확장 가능성 검토
""")

    # ===== 푸터 =====
    doc.add_paragraph()
    footer = doc.add_paragraph()
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("© 2025 AMORE Pacific - Confidential").italic = True

    # BytesIO로 저장
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    # 파일명 생성
    filename = f"AMORE_Insight_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.docx"

    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


class AnalystReportRequest(BaseModel):
    """애널리스트 리포트 요청"""

    start_date: str  # Required: YYYY-MM-DD
    end_date: str  # Required: YYYY-MM-DD
    include_charts: bool = True
    include_external_signals: bool = True


@app.post("/api/export/excel")
async def export_excel(request: Request):
    """
    엑셀 데이터 내보내기 (JSON 파일 → Excel)

    데이터 소스:
    - Railway: /data/latest_crawl_result.json (Volume)
    - Local: ./data/latest_crawl_result.json
    """

    import pandas as pd

    try:
        # Parse request body
        body = await request.json()
        start_date = body.get("start_date")
        end_date = body.get("end_date")
        _include_metrics = body.get("include_metrics", True)  # reserved for future use

        # 데이터 디렉토리 경로 설정
        data_dir = Path("./data")

        # ========================================
        # 1차: SQLite에서 기간별 데이터 조회 (가장 빠름)
        # ========================================
        all_records = []
        data_source = None

        if start_date and end_date:
            # 1-1. SQLite 시도
            try:
                from src.tools.sqlite_storage import get_sqlite_storage

                sqlite = get_sqlite_storage()
                await sqlite.initialize()

                # limit을 크게 설정 (5개 카테고리 × 100개 × 기간일수)
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                days = (end_dt - start_dt).days + 1
                max_records = 500 * days  # 충분한 여유

                records = await sqlite.get_raw_data(
                    start_date=start_date, end_date=end_date, limit=max_records
                )

                if records:
                    all_records = records
                    data_source = "sqlite"
                    logging.info(
                        f"Excel export: loaded {len(all_records)} records from SQLite ({start_date} ~ {end_date})"
                    )

            except Exception as sqlite_err:
                logging.warning(f"Excel export: SQLite 조회 실패: {sqlite_err}")

            # 1-2. SQLite 실패 시 Google Sheets 시도
            if not all_records:
                try:
                    sheets_writer = get_sheets_writer()
                    if not sheets_writer._initialized:
                        await sheets_writer.initialize()

                    records = await sheets_writer.get_raw_data(days=days)

                    if records:
                        for record in records:
                            snapshot_date = record.get("snapshot_date", "")
                            if snapshot_date and start_date <= snapshot_date <= end_date:
                                all_records.append(record)

                        if all_records:
                            data_source = "sheets"
                            logging.info(
                                f"Excel export: loaded {len(all_records)} records from Google Sheets ({start_date} ~ {end_date})"
                            )

                except Exception as sheets_err:
                    logging.warning(f"Excel export: Google Sheets 조회 실패: {sheets_err}")

        # ========================================
        # 2차: 로컬 JSON 파일에서 데이터 로드 (폴백)
        # ========================================
        crawl_data = None
        json_path = None

        if not data_source:
            possible_paths = [
                data_dir / "latest_crawl_result.json",
                data_dir / "dashboard_data.json",
            ]

            for path in possible_paths:
                if path.exists():
                    json_path = path
                    break

            if json_path is None:
                raise HTTPException(
                    status_code=404, detail="크롤링 데이터가 없습니다. 먼저 크롤링을 실행해주세요."
                )

            with open(json_path, encoding="utf-8") as f:
                crawl_data = json.load(f)

            logging.info(f"Excel export: loaded data from {json_path}")

        # 데이터 소스 유형 판단
        # 1. data_source: SQLite 또는 Google Sheets에서 기간별 데이터 로드됨
        # 2. is_crawl_data: latest_crawl_result.json의 raw 데이터
        # 3. is_dashboard_data: dashboard_data.json의 집계 데이터
        is_crawl_data = False
        is_dashboard_data = False

        if crawl_data:
            if "categories" in crawl_data:
                first_cat = next(iter(crawl_data["categories"].values()), {})
                is_crawl_data = isinstance(first_cat, dict) and (
                    "rank_records" in first_cat or "products" in first_cat
                )
            is_dashboard_data = "metadata" in crawl_data and "brand" in crawl_data

        # 출력 경로 (Railway 환경 고려)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = data_dir / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"AMORE_Data_{timestamp}.xlsx"

        sheets_created = []
        total_rows = 0

        # 카테고리 매핑
        categories_info = {
            "beauty": "Beauty & Personal Care",
            "skin_care": "Skin Care",
            "lip_care": "Lip Care",
            "lip_makeup": "Lip Makeup",
            "face_powder": "Face Powder",
        }

        with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
            # Google Sheets RawData와 동일한 컬럼 순서
            RAWDATA_COLUMNS = [
                "snapshot_date",
                "category_id",
                "rank",
                "asin",
                "product_name",
                "brand",
                "price",
                "list_price",
                "discount_percent",
                "rating",
                "reviews_count",
                "badge",
                "coupon_text",
                "is_subscribe_save",
                "promo_badges",
                "product_url",
            ]

            # ========================================
            # Case 1: SQLite/Google Sheets에서 기간별 데이터 로드됨
            # ========================================
            if data_source and all_records:
                source_name = "SQLite" if data_source == "sqlite" else "Google Sheets"
                logging.info(
                    f"Excel export: using {source_name} data ({len(all_records)} records, {start_date} ~ {end_date})"
                )

                df_all = pd.DataFrame(all_records)

                if not df_all.empty:
                    # 1. RawData 시트 - 전체 데이터
                    available_cols = [c for c in RAWDATA_COLUMNS if c in df_all.columns]
                    df_raw = df_all[available_cols].copy()
                    df_raw = df_raw.sort_values(["snapshot_date", "category_id", "rank"])
                    df_raw.to_excel(writer, sheet_name="RawData", index=False)
                    sheets_created.append("RawData")
                    total_rows += len(df_raw)

                    # 2. 날짜별 요약 시트
                    if "snapshot_date" in df_all.columns:
                        date_summary = []
                        for date in sorted(df_all["snapshot_date"].unique()):
                            df_date = df_all[df_all["snapshot_date"] == date]
                            laneige_count = (
                                len(df_date[df_date["brand"].str.upper() == "LANEIGE"])
                                if "brand" in df_date.columns
                                else 0
                            )
                            date_summary.append(
                                {
                                    "날짜": date,
                                    "총 제품 수": len(df_date),
                                    "LANEIGE 제품 수": laneige_count,
                                    "LANEIGE SoS (%)": round(laneige_count / len(df_date) * 100, 1)
                                    if len(df_date) > 0
                                    else 0,
                                }
                            )
                        if date_summary:
                            df_summary = pd.DataFrame(date_summary)
                            df_summary.to_excel(writer, sheet_name="Daily Summary", index=False)
                            sheets_created.append("Daily Summary")
                            total_rows += len(df_summary)

                    # 3. 카테고리별 시트
                    if "category_id" in df_all.columns:
                        for cat_id in df_all["category_id"].unique():
                            df_cat = df_all[df_all["category_id"] == cat_id].copy()
                            if df_cat.empty:
                                continue

                            display_cols = [
                                "snapshot_date",
                                "rank",
                                "asin",
                                "product_name",
                                "brand",
                                "price",
                                "rating",
                                "reviews_count",
                                "badge",
                            ]
                            available_display = [c for c in display_cols if c in df_cat.columns]
                            df_display = df_cat[available_display].sort_values(
                                ["snapshot_date", "rank"]
                            )

                            sheet_name = categories_info.get(cat_id, cat_id)[:31]
                            df_display.to_excel(writer, sheet_name=sheet_name, index=False)
                            sheets_created.append(sheet_name)
                            total_rows += len(df_display)

                    # 4. LANEIGE 제품 전용 시트
                    if "brand" in df_all.columns:
                        df_laneige = df_all[df_all["brand"].str.upper() == "LANEIGE"].copy()
                        if not df_laneige.empty:
                            laneige_cols = [
                                "snapshot_date",
                                "category_id",
                                "rank",
                                "asin",
                                "product_name",
                                "price",
                                "rating",
                                "reviews_count",
                                "badge",
                            ]
                            available_laneige = [c for c in laneige_cols if c in df_laneige.columns]
                            df_laneige = df_laneige[available_laneige].sort_values(
                                ["snapshot_date", "category_id", "rank"]
                            )
                            df_laneige.to_excel(writer, sheet_name="LANEIGE Products", index=False)
                            sheets_created.append("LANEIGE Products")
                            total_rows += len(df_laneige)

            # ========================================
            # Case 2: 대시보드 데이터 형식 (집계 데이터만)
            # ========================================
            elif is_dashboard_data and not is_crawl_data:
                logging.info("Excel export: using dashboard_data.json (aggregated data only)")

                # 1. Overview 시트
                metadata = crawl_data.get("metadata", {})
                data_source = crawl_data.get("data_source", {})
                overview_data = [
                    {"항목": "데이터 날짜", "값": metadata.get("data_date", "N/A")},
                    {"항목": "생성 시각", "값": metadata.get("generated_at", "N/A")},
                    {"항목": "총 제품 수", "값": metadata.get("total_products", 0)},
                    {"항목": "LANEIGE 제품 수", "값": metadata.get("laneige_products", 0)},
                    {"항목": "플랫폼", "값": data_source.get("platform", "Amazon US")},
                ]
                df_overview = pd.DataFrame(overview_data)
                df_overview.to_excel(writer, sheet_name="Overview", index=False)
                sheets_created.append("Overview")
                total_rows += len(df_overview)

                # 2. Brand KPIs 시트
                brand_kpis = crawl_data.get("brand", {}).get("kpis", {})
                if brand_kpis:
                    kpi_data = [
                        {"KPI": "SoS (Share of Shelf)", "값": f"{brand_kpis.get('sos', 0)}%"},
                        {"KPI": "SoS 변화", "값": brand_kpis.get("sos_delta", "N/A")},
                        {"KPI": "Top 10 제품 수", "값": brand_kpis.get("top10_count", 0)},
                        {"KPI": "평균 순위", "값": brand_kpis.get("avg_rank", 0)},
                        {"KPI": "HHI (시장 집중도)", "값": brand_kpis.get("hhi", 0)},
                    ]
                    df_kpis = pd.DataFrame(kpi_data)
                    df_kpis.to_excel(writer, sheet_name="LANEIGE KPIs", index=False)
                    sheets_created.append("LANEIGE KPIs")
                    total_rows += len(df_kpis)

                # 3. Competitors 시트
                competitors = crawl_data.get("brand", {}).get("competitors", [])
                if competitors:
                    df_comp = pd.DataFrame(competitors)
                    column_mapping = {
                        "brand": "Brand",
                        "sos": "SoS (%)",
                        "avg_rank": "Avg Rank",
                        "product_count": "Product Count",
                        "avg_price": "Avg Price ($)",
                    }
                    existing_cols = {
                        k: v for k, v in column_mapping.items() if k in df_comp.columns
                    }
                    df_comp = df_comp.rename(columns=existing_cols)
                    df_comp.to_excel(writer, sheet_name="Competitors", index=False)
                    sheets_created.append("Competitors")
                    total_rows += len(df_comp)

                # 4. Action Items 시트
                action_items = crawl_data.get("home", {}).get("action_items", [])
                if action_items:
                    df_actions = pd.DataFrame(action_items)
                    df_actions.to_excel(writer, sheet_name="Action Items", index=False)
                    sheets_created.append("Action Items")
                    total_rows += len(df_actions)

                # 5. Category View 시트
                category_data = crawl_data.get("category", {})
                if category_data:
                    for cat_id, cat_info in category_data.items():
                        top_products = cat_info.get("top_products", [])
                        if top_products:
                            df_cat = pd.DataFrame(top_products)
                            sheet_name = categories_info.get(cat_id, cat_id)[:31]
                            df_cat.to_excel(writer, sheet_name=sheet_name, index=False)
                            sheets_created.append(sheet_name)
                            total_rows += len(df_cat)

            # ========================================
            # Case 3: 로컬 크롤링 원본 데이터
            # ========================================
            else:
                logging.info("Excel export: using latest_crawl_result.json (raw crawl data)")

                # 전체 RawData 수집 (카테고리별 rank_records)
                all_records = []
                for cat_id, cat_data in crawl_data.get("categories", {}).items():
                    records = cat_data.get("rank_records", cat_data.get("products", []))
                    for record in records:
                        # category_id 추가 (없는 경우)
                        if "category_id" not in record:
                            record["category_id"] = cat_id
                        all_records.append(record)

                if not all_records:
                    logging.warning("Excel export: no rank_records found in crawl data")

                if all_records:
                    df_all = pd.DataFrame(all_records)

                    # 날짜 필터 적용 (선택 기간)
                    if start_date and "snapshot_date" in df_all.columns:
                        df_all = df_all[df_all["snapshot_date"] >= start_date]
                    if end_date and "snapshot_date" in df_all.columns:
                        df_all = df_all[df_all["snapshot_date"] <= end_date]

                    if not df_all.empty:
                        # 1. RawData 시트 - Google Sheets와 동일한 전체 데이터
                        available_cols = [c for c in RAWDATA_COLUMNS if c in df_all.columns]
                        df_raw = df_all[available_cols].copy()
                        df_raw = df_raw.sort_values(["category_id", "rank"])
                        df_raw.to_excel(writer, sheet_name="RawData", index=False)
                        sheets_created.append("RawData")
                        total_rows += len(df_raw)

                        # 2. 카테고리별 시트 (요약 보기용)
                        for cat_id in df_all["category_id"].unique():
                            df_cat = df_all[df_all["category_id"] == cat_id].copy()
                            if df_cat.empty:
                                continue

                            # 핵심 컬럼만 선택하여 가독성 향상
                            display_cols = [
                                "rank",
                                "asin",
                                "product_name",
                                "brand",
                                "price",
                                "rating",
                                "reviews_count",
                                "badge",
                            ]
                            available_display = [c for c in display_cols if c in df_cat.columns]
                            df_display = df_cat[available_display].sort_values("rank")

                            # 시트 이름 (31자 제한)
                            sheet_name = categories_info.get(cat_id, cat_id)[:31]
                            df_display.to_excel(writer, sheet_name=sheet_name, index=False)
                            sheets_created.append(sheet_name)
                            total_rows += len(df_display)

                        # 3. LANEIGE 제품 전용 시트
                        df_laneige = df_all[df_all["brand"].str.upper() == "LANEIGE"].copy()
                        if not df_laneige.empty:
                            laneige_cols = [
                                "snapshot_date",
                                "category_id",
                                "rank",
                                "asin",
                                "product_name",
                                "price",
                                "rating",
                                "reviews_count",
                                "badge",
                            ]
                            available_laneige = [c for c in laneige_cols if c in df_laneige.columns]
                            df_laneige = df_laneige[available_laneige].sort_values(
                                ["category_id", "rank"]
                            )
                            df_laneige.to_excel(writer, sheet_name="LANEIGE Products", index=False)
                            sheets_created.append("LANEIGE Products")
                            total_rows += len(df_laneige)

                        # 4. Summary 시트 - 브랜드별 집계
                        if "brand" in df_all.columns:
                            agg_dict = {"asin": "count"}
                            if "rank" in df_all.columns:
                                agg_dict["rank"] = "mean"
                            if "price" in df_all.columns:
                                agg_dict["price"] = "mean"
                            if "rating" in df_all.columns:
                                agg_dict["rating"] = "mean"

                            summary = df_all.groupby("brand").agg(agg_dict).reset_index()
                            col_names = ["Brand", "Product Count"]
                            if "rank" in agg_dict:
                                col_names.append("Avg Rank")
                            if "price" in agg_dict:
                                col_names.append("Avg Price")
                            if "rating" in agg_dict:
                                col_names.append("Avg Rating")
                            summary.columns = col_names

                            summary = summary.sort_values("Product Count", ascending=False).head(30)
                            for col in ["Avg Rank", "Avg Price", "Avg Rating"]:
                                if col in summary.columns:
                                    summary[col] = summary[col].round(2)

                            summary.to_excel(writer, sheet_name="Summary", index=False)
                            sheets_created.append("Summary")
                            total_rows += len(summary)

            # 4. 시트가 하나도 없으면 안내 시트 생성
            if not sheets_created:
                data_source_info = (
                    "SQLite"
                    if data_source == "sqlite"
                    else (
                        "Google Sheets"
                        if data_source == "sheets"
                        else (str(json_path) if json_path else "N/A")
                    )
                )
                no_data_info = [
                    {"항목": "요청 기간", "값": f"{start_date or 'N/A'} ~ {end_date or 'N/A'}"},
                    {"항목": "결과", "값": "해당 기간에 데이터가 없습니다"},
                    {"항목": "데이터 소스", "값": data_source_info},
                    {"항목": "안내", "값": "크롤링 실행 후 다시 시도해주세요"},
                ]
                df_no_data = pd.DataFrame(no_data_info)
                df_no_data.to_excel(writer, sheet_name="No Data", index=False)
                sheets_created.append("No Data")

        logging.info(f"Excel exported: {output_path} ({total_rows} rows, sheets: {sheets_created})")

        # 파일 반환
        return FileResponse(
            path=str(output_path),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={output_path.name}"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Excel export error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Excel 내보내기 중 오류가 발생했습니다") from e


# ============= Competitor Comparison API =============


@app.get("/api/competitors")
async def get_competitor_data(brand: str | None = None):
    """
    경쟁사 추적 데이터 조회

    Args:
        brand: 브랜드 필터 (예: "Summer Fridays")

    Returns:
        경쟁사 제품 목록 및 LANEIGE 비교 데이터
    """
    try:
        import json
        from pathlib import Path

        from src.tools.sqlite_storage import get_sqlite_storage

        result = {"competitors": {}, "laneige_products": {}, "comparison": []}

        # 1. SQLite에서 경쟁사 데이터 조회 시도
        try:
            sqlite = get_sqlite_storage()
            await sqlite.initialize()
            comp_products = await sqlite.get_competitor_products(brand=brand)

            if comp_products:
                # 브랜드별로 그룹화
                for p in comp_products:
                    brand_name = p.get("brand", "Unknown")
                    if brand_name not in result["competitors"]:
                        result["competitors"][brand_name] = {
                            "brand": brand_name,
                            "products": [],
                            "product_count": 0,
                            "avg_price": 0,
                            "avg_rating": 0,
                        }
                    result["competitors"][brand_name]["products"].append(p)

                # 브랜드별 집계
                for _brand_name, brand_data in result["competitors"].items():
                    products = brand_data["products"]
                    brand_data["product_count"] = len(products)
                    prices = [p["price"] for p in products if p.get("price")]
                    ratings = [p["rating"] for p in products if p.get("rating")]
                    brand_data["avg_price"] = round(sum(prices) / len(prices), 2) if prices else 0
                    brand_data["avg_rating"] = (
                        round(sum(ratings) / len(ratings), 1) if ratings else 0
                    )

        except Exception as sqlite_err:
            logging.warning(f"SQLite competitor query failed: {sqlite_err}")

        # 2. JSON 파일에서 폴백
        if not result["competitors"]:
            json_path = Path("./data/competitor_products.json")
            if json_path.exists():
                with open(json_path, encoding="utf-8") as f:
                    json_data = json.load(f)
                    for p in json_data.get("products", []):
                        brand_name = p.get("brand", "Unknown")
                        if brand and brand_name != brand:
                            continue
                        if brand_name not in result["competitors"]:
                            result["competitors"][brand_name] = {
                                "brand": brand_name,
                                "products": [],
                                "product_count": 0,
                            }
                        result["competitors"][brand_name]["products"].append(p)

        # 3. LANEIGE 제품 데이터 로드 (최신 크롤링 데이터에서)
        data = load_dashboard_data()
        if data:
            # 카테고리별 LANEIGE 제품 추출
            for cat_id, cat_data in data.get("category", {}).items():
                for product in cat_data.get("top_products", []):
                    if "laneige" in product.get("brand", "").lower():
                        product_type = _detect_product_type(product.get("product_name", ""))
                        if product_type not in result["laneige_products"]:
                            result["laneige_products"][product_type] = []
                        result["laneige_products"][product_type].append(
                            {**product, "category_id": cat_id, "product_type": product_type}
                        )

        # 4. 제품 타입별 비교 데이터 생성
        for brand_name, brand_data in result["competitors"].items():
            for comp_product in brand_data["products"]:
                laneige_match = comp_product.get("laneige_competitor")
                product_type = comp_product.get("product_type", "")

                comparison_item = {
                    "competitor_brand": brand_name,
                    "competitor_product": comp_product.get("product_name", ""),
                    "competitor_price": comp_product.get("price"),
                    "competitor_rating": comp_product.get("rating"),
                    "competitor_reviews": comp_product.get("reviews_count"),
                    "product_type": product_type,
                    "laneige_product": laneige_match,
                    "laneige_price": None,
                    "laneige_rating": None,
                    "laneige_reviews": None,
                    "price_diff": None,
                    "rating_diff": None,
                }

                # LANEIGE 매칭 제품 찾기
                if product_type in result["laneige_products"]:
                    for lp in result["laneige_products"][product_type]:
                        comparison_item["laneige_price"] = lp.get("price")
                        comparison_item["laneige_rating"] = lp.get("rating")
                        comparison_item["laneige_reviews"] = lp.get("reviews_count")

                        # 차이 계산
                        if comparison_item["competitor_price"] and comparison_item["laneige_price"]:
                            comparison_item["price_diff"] = round(
                                comparison_item["laneige_price"]
                                - comparison_item["competitor_price"],
                                2,
                            )
                        if (
                            comparison_item["competitor_rating"]
                            and comparison_item["laneige_rating"]
                        ):
                            comparison_item["rating_diff"] = round(
                                comparison_item["laneige_rating"]
                                - comparison_item["competitor_rating"],
                                1,
                            )
                        break

                result["comparison"].append(comparison_item)

        return result

    except Exception as e:
        logger.error(f"Competitor data error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="경쟁사 데이터 조회 중 오류가 발생했습니다"
        ) from e


def _detect_product_type(product_name: str) -> str:
    """제품명에서 제품 타입 추출"""
    name_lower = product_name.lower()

    if "lip sleeping" in name_lower or "lip mask" in name_lower:
        return "lip_balm"
    elif "lip glowy" in name_lower or "lip butter" in name_lower or "lip balm" in name_lower:
        return "lip_balm"
    elif "water sleeping" in name_lower or "sleeping mask" in name_lower:
        return "sleeping_mask"
    elif "water bank" in name_lower or "cream" in name_lower or "moisturizer" in name_lower:
        return "moisturizer"
    elif "toner" in name_lower or "cream skin" in name_lower:
        return "toner"
    elif "serum" in name_lower:
        return "serum"
    else:
        return "other"


@app.get("/api/competitors/brands")
async def get_tracked_brands():
    """추적 중인 경쟁사 브랜드 목록"""
    try:
        import json
        from pathlib import Path

        config_path = Path("./config/tracked_competitors.json")
        if not config_path.exists():
            return {"brands": []}

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        brands = []
        for brand_name, brand_config in config.get("competitors", {}).items():
            brands.append(
                {
                    "name": brand_name,
                    "tier": brand_config.get("tier", ""),
                    "product_count": len(brand_config.get("products", [])),
                }
            )

        return {"brands": brands}

    except Exception as e:
        logging.error(f"Get tracked brands error: {e}")
        return {"brands": []}


# ============= Alert Settings API =============

from src.core.state_manager import StateManager, get_state_manager

# 싱글톤 State Manager
_state_manager: StateManager | None = None


def get_app_state_manager() -> StateManager:
    """앱 레벨 State Manager 반환"""
    global _state_manager
    if _state_manager is None:
        _state_manager = get_state_manager()
    return _state_manager


class AlertSettingsRequest(BaseModel):
    """알림 설정 요청"""

    email: str
    consent: bool
    alert_types: list[str] = []


class AlertSettingsResponse(BaseModel):
    """알림 설정 응답"""

    email: str
    consent: bool
    alert_types: list[str]
    consent_date: str | None = None


@app.get("/api/v3/alert-settings")
async def get_alert_settings():
    """
    현재 알림 설정 조회

    참고: 현재는 단일 사용자 설정만 지원 (첫 번째 등록된 이메일)
    """
    state_manager = get_app_state_manager()
    subscriptions = state_manager.get_all_subscriptions()

    if not subscriptions:
        return {"email": "", "consent": False, "alert_types": [], "consent_date": None}

    # 첫 번째 구독 반환
    email, sub = next(iter(subscriptions.items()))
    return {
        "email": email,
        "consent": sub.consent,
        "alert_types": sub.alert_types,
        "consent_date": sub.consent_date.isoformat() if sub.consent_date else None,
    }


@app.post("/api/v3/alert-settings", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")  # 분당 5회 제한 (스팸 방지)
async def save_alert_settings(request: Request, settings: AlertSettingsRequest):
    """
    알림 설정 저장

    보안: API Key + Rate Limiting (IP당 분당 5회)
    중요: consent가 True일 때만 이메일 등록
    """
    state_manager = get_app_state_manager()

    if not settings.email:
        raise HTTPException(status_code=400, detail="이메일 주소가 필요합니다.")

    if settings.consent:
        # 이메일 등록 (명시적 동의)
        success = state_manager.register_email(
            email=settings.email, consent=True, alert_types=settings.alert_types
        )

        if not success:
            raise HTTPException(status_code=400, detail="이메일 등록 실패")

        return {"status": "ok", "message": "알림 설정이 저장되었습니다."}
    else:
        # 동의 없으면 업데이트만 (알림 유형 변경)
        success = state_manager.update_email_subscription(
            email=settings.email, alert_types=settings.alert_types
        )

        return {"status": "ok", "message": "설정이 업데이트되었습니다."}


@app.post("/api/v3/alert-settings/revoke", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")  # 분당 5회 제한
async def revoke_alert_consent(request: Request):
    """
    알림 동의 철회

    보안: API Key + Rate Limiting
    첫 번째 등록된 이메일의 동의를 철회합니다.
    """
    state_manager = get_app_state_manager()
    subscriptions = state_manager.get_all_subscriptions()

    if not subscriptions:
        return {"status": "ok", "message": "철회할 동의가 없습니다."}

    # 첫 번째 이메일 철회
    email = next(iter(subscriptions.keys()))
    state_manager.revoke_email_consent(email)

    return {"status": "ok", "message": "동의가 철회되었습니다."}


@app.get("/api/v3/alerts")
async def get_alerts(limit: int = 50, alert_type: str | None = None):
    """
    알림 목록 조회

    Args:
        limit: 최대 개수
        alert_type: 필터할 알림 유형
    """
    from src.agents.alert_agent import AlertAgent

    state_manager = get_app_state_manager()
    alert_agent = AlertAgent(state_manager)

    return {
        "alerts": alert_agent.get_alerts(limit=limit, alert_type=alert_type),
        "pending_count": alert_agent.get_pending_count(),
        "stats": alert_agent.get_stats(),
    }


# ============= 대시보드 HTML 서빙 =============


@app.get("/dashboard")
async def serve_dashboard():
    """
    대시보드 HTML 페이지 서빙 (API 키 자동 주입)

    서버의 API_KEY를 HTML에 자동으로 주입하여
    프론트엔드에서 별도 설정 없이 인증된 API 호출 가능
    """
    dashboard_path = Path("./dashboard/amore_unified_dashboard_v4.html")
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")

    # API_KEY가 설정된 경우 HTML에 자동 주입
    if API_KEY:
        html_content = dashboard_path.read_text(encoding="utf-8")
        # </head> 직전에 API 키 설정 스크립트 삽입
        api_key_script = f'<script>window.DASHBOARD_API_KEY = "{API_KEY}";</script>\n</head>'
        html_content = html_content.replace("</head>", api_key_script)
        return HTMLResponse(content=html_content, media_type="text/html")

    return FileResponse(dashboard_path, media_type="text/html")


@app.get("/api/health")
async def health_check():
    """기본 헬스 체크 엔드포인트 (Railway healthcheck용)"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/health/deep")
async def deep_health_check():
    """
    Deep Health Check - 모든 서브시스템 상태 확인

    Returns:
        - database: SQLite 연결 상태
        - knowledge_graph: KG 로드 상태 및 트리플 수
        - llm: OpenAI API 연결 상태
        - scheduler: 자율 스케줄러 상태
        - memory: 시스템 메모리 사용량
        - disk: 디스크 사용량 (Railway Volume)
    """
    import os
    import sqlite3
    from pathlib import Path

    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {},
        "warnings": [],
    }

    # 1. SQLite 연결 확인
    try:
        db_path = (
            Path("/data/amore_data.db") if Path("/data").exists() else Path("data/amore_data.db")
        )
        if db_path.exists():
            conn = sqlite3.connect(str(db_path), timeout=5)
            cursor = conn.execute("SELECT COUNT(*) FROM raw_data")
            count = cursor.fetchone()[0]
            conn.close()
            health["checks"]["database"] = {
                "status": "healthy",
                "records": count,
                "path": str(db_path),
            }
        else:
            health["checks"]["database"] = {"status": "missing", "path": str(db_path)}
            health["warnings"].append("SQLite database not found")
    except Exception as e:
        health["checks"]["database"] = {"status": "unhealthy", "error": str(e)}
        health["status"] = "degraded"

    # 2. Knowledge Graph 상태
    try:
        from src.ontology.knowledge_graph import get_knowledge_graph

        kg = get_knowledge_graph()
        triple_count = len(kg.triples) if kg.triples else 0
        health["checks"]["knowledge_graph"] = {
            "status": "healthy" if triple_count > 0 else "empty",
            "triples": triple_count,
            "max_triples": kg.max_triples,
        }
        if triple_count == 0:
            health["warnings"].append("Knowledge Graph is empty")
    except Exception as e:
        health["checks"]["knowledge_graph"] = {"status": "unhealthy", "error": str(e)}
        health["status"] = "degraded"

    # 3. LLM API 연결 (OpenAI)
    try:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key and api_key.startswith("sk-"):
            health["checks"]["llm"] = {
                "status": "configured",
                "provider": "openai",
                "key_prefix": api_key[:10] + "...",
            }
        else:
            health["checks"]["llm"] = {"status": "not_configured"}
            health["warnings"].append("OPENAI_API_KEY not properly configured")
    except Exception as e:
        health["checks"]["llm"] = {"status": "error", "error": str(e)}

    # 4. 스케줄러 상태
    try:
        brain = await get_initialized_brain()
        scheduler_running = brain.scheduler.is_running if brain.scheduler else False
        health["checks"]["scheduler"] = {
            "status": "running" if scheduler_running else "stopped",
            "mode": brain.mode.value if brain.mode else "unknown",
        }
    except Exception as e:
        health["checks"]["scheduler"] = {"status": "error", "error": str(e)}

    # 5. 메모리 사용량
    try:
        import psutil

        memory = psutil.virtual_memory()
        health["checks"]["memory"] = {
            "status": "healthy" if memory.percent < 90 else "warning",
            "used_percent": round(memory.percent, 1),
            "available_gb": round(memory.available / (1024**3), 2),
        }
        if memory.percent > 90:
            health["warnings"].append(f"High memory usage: {memory.percent}%")
            health["status"] = "degraded"
    except ImportError:
        health["checks"]["memory"] = {"status": "unknown", "note": "psutil not installed"}

    # 6. 디스크 사용량 (Railway Volume)
    try:
        import shutil

        data_path = Path("/data") if Path("/data").exists() else Path("data")
        if data_path.exists():
            total, used, free = shutil.disk_usage(data_path)
            used_percent = (used / total) * 100
            health["checks"]["disk"] = {
                "status": "healthy" if used_percent < 90 else "warning",
                "used_percent": round(used_percent, 1),
                "free_gb": round(free / (1024**3), 2),
                "path": str(data_path),
            }
            if used_percent > 90:
                health["warnings"].append(f"Low disk space: {100-used_percent:.1f}% free")
                health["status"] = "degraded"
    except Exception as e:
        health["checks"]["disk"] = {"status": "error", "error": str(e)}

    # 최종 상태 결정
    unhealthy_checks = [k for k, v in health["checks"].items() if v.get("status") == "unhealthy"]
    if unhealthy_checks:
        health["status"] = "unhealthy"

    return health


# ============= Level 4 Brain API (v4) =============


class BrainChatRequest(BaseModel):
    """Brain 챗봇 요청"""

    message: str
    session_id: str | None = "default"
    skip_cache: bool = False


class BrainChatResponse(BaseModel):
    """Brain 챗봇 응답"""

    text: str
    confidence: float
    sources: list[str]
    reasoning: str | None = None
    tools_used: list[str]
    processing_time_ms: float
    from_cache: bool
    brain_mode: str


@app.post("/api/v4/chat", response_model=BrainChatResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")  # 분당 10회 제한 (보안 강화)
async def chat_v4(request: Request, body: BrainChatRequest):
    """
    Level 4 Brain 기반 챗봇 API (v4)

    LLM-First 접근:
    - 모든 판단을 LLM이 수행
    - 규칙 기반 빠른 경로 없음
    - RAG + KG 하이브리드 검색
    - 자율 스케줄러와 통합
    """
    import time

    start_time = time.time()

    message = body.message.strip()
    session_id = body.session_id or "default"

    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    try:
        # Brain 인스턴스 획득
        brain = await get_initialized_brain()

        # 현재 메트릭 데이터 로드
        data = load_dashboard_data()
        current_metrics = data if data else None

        # Brain으로 처리 (LLM-First)
        response = await brain.process_query(
            query=message,
            session_id=session_id,
            current_metrics=current_metrics,
            skip_cache=body.skip_cache,
        )

        processing_time = (time.time() - start_time) * 1000

        return BrainChatResponse(
            text=response.content,
            confidence=response.confidence,
            sources=response.sources,
            reasoning=response.reasoning,
            tools_used=response.tools_called if hasattr(response, "tools_called") else [],
            processing_time_ms=processing_time,
            from_cache=response.from_cache if hasattr(response, "from_cache") else False,
            brain_mode=brain.mode.value,
        )

    except Exception as e:
        logging.error(f"Brain error: {e}")
        return BrainChatResponse(
            text=f"처리 중 오류가 발생했습니다: {str(e)}",
            confidence=0.0,
            sources=[],
            reasoning=None,
            tools_used=[],
            processing_time_ms=(time.time() - start_time) * 1000,
            from_cache=False,
            brain_mode="error",
        )


@app.get("/api/v4/brain/status")
async def get_brain_status():
    """
    Brain 상태 조회

    Returns:
        - mode: 현재 Brain 모드
        - scheduler: 스케줄러 상태
        - pending_tasks: 대기 중 태스크
        - stats: 통계
    """
    try:
        brain = await get_initialized_brain()

        return {
            "mode": brain.mode.value,
            "scheduler_running": brain.scheduler.running if brain.scheduler else False,
            "pending_tasks": brain.scheduler.get_pending_count() if brain.scheduler else 0,
            "stats": brain.get_stats(),
            "initialized": True,
        }
    except Exception as e:
        return {
            "mode": "uninitialized",
            "scheduler_running": False,
            "pending_tasks": 0,
            "stats": {},
            "initialized": False,
            "error": str(e),
        }


@app.post("/api/v4/brain/scheduler/start", dependencies=[Depends(verify_api_key)])
async def start_brain_scheduler():
    """
    자율 스케줄러 시작 (API Key 필요)

    - 일일 크롤링 (09:00)
    - 주기적 알림 체크 (30분)
    - 백그라운드 분석
    """
    try:
        brain = await get_initialized_brain()

        if brain.scheduler and brain.scheduler.running:
            return {
                "started": False,
                "message": "스케줄러가 이미 실행 중입니다.",
                "status": "running",
            }

        await brain.start_scheduler()

        return {"started": True, "message": "자율 스케줄러가 시작되었습니다.", "status": "running"}
    except Exception as e:
        return {"started": False, "message": f"스케줄러 시작 실패: {str(e)}", "status": "error"}


@app.post("/api/v4/brain/scheduler/stop", dependencies=[Depends(verify_api_key)])
async def stop_brain_scheduler():
    """자율 스케줄러 중지 (API Key 필요)"""
    try:
        brain = await get_initialized_brain()

        if brain.scheduler:
            brain.scheduler.stop()

        return {"stopped": True, "message": "스케줄러가 중지되었습니다.", "status": "stopped"}
    except Exception as e:
        return {"stopped": False, "message": f"스케줄러 중지 실패: {str(e)}", "status": "error"}


@app.post("/api/v4/brain/autonomous-cycle", dependencies=[Depends(verify_api_key)])
async def run_autonomous_cycle():
    """
    자율 사이클 수동 실행 (API Key 필요)

    1. 데이터 신선도 확인
    2. 필요시 크롤링
    3. 지표 계산
    4. 알림 조건 체크
    5. 인사이트 생성
    """
    try:
        brain = await get_initialized_brain()
        result = await brain.run_autonomous_cycle()

        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/v4/brain/check-alerts")
async def check_brain_alerts():
    """
    알림 조건 수동 체크

    현재 메트릭 데이터를 기반으로 알림 조건을 체크합니다.
    """
    try:
        brain = await get_initialized_brain()
        data = load_dashboard_data()

        if not data:
            return {"alerts": [], "message": "데이터가 없습니다."}

        alerts = await brain.check_alerts(data)

        return {"alerts": alerts, "count": len(alerts), "checked_at": datetime.now().isoformat()}
    except Exception as e:
        return {"alerts": [], "error": str(e)}


@app.get("/api/v4/brain/stats")
async def get_brain_stats():
    """Brain 통계 조회"""
    try:
        brain = await get_initialized_brain()
        return brain.get_stats()
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/v4/brain/mode", dependencies=[Depends(verify_api_key)])
async def set_brain_mode(mode: str):
    """
    Brain 모드 변경 (API Key 필요)

    Args:
        mode: reactive, proactive, autonomous
    """
    try:
        brain = await get_initialized_brain()

        mode_map = {
            "reactive": BrainMode.REACTIVE,
            "proactive": BrainMode.PROACTIVE,
            "autonomous": BrainMode.AUTONOMOUS,
        }

        if mode not in mode_map:
            raise HTTPException(
                status_code=400, detail=f"Invalid mode. Valid modes: {list(mode_map.keys())}"
            )

        brain.mode = mode_map[mode]

        return {"mode": brain.mode.value, "message": f"Brain 모드가 {mode}(으)로 변경되었습니다."}
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}


# ============= Amazon Deals API =============

from src.tools.deals_scraper import get_deals_scraper


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


@app.get("/api/deals")
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


@app.get("/api/deals/summary")
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


@app.post("/api/deals/scrape", dependencies=[Depends(verify_api_key)])
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
            if result["deals"]:
                await storage.save_deals(result["deals"], is_competitor=False)

            # 경쟁사 딜은 is_competitor=True로 별도 저장
            if result["competitor_deals"]:
                await storage.save_deals(result["competitor_deals"], is_competitor=True)

                # 알림 서비스로 알림 처리
                try:
                    alert_service = get_alert_service()
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


@app.get("/api/deals/alerts")
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


@app.post("/api/deals/export")
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


# ============= 알림 서비스 API =============

from src.tools.alert_service import get_alert_service


class AlertSendRequest(BaseModel):
    """알림 발송 요청"""

    alert_ids: list[int] | None = None  # 발송할 알림 ID (없으면 미발송 전체)


@app.get("/api/alerts/status")
async def get_alert_service_status():
    """알림 서비스 상태 조회"""
    try:
        service = get_alert_service()
        return {"success": True, **service.get_status()}
    except Exception as e:
        logging.error(f"Alert service status error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/alerts/send")
async def send_pending_alerts(request: AlertSendRequest | None = None):
    """
    미발송 알림 발송

    특정 alert_ids를 지정하면 해당 알림만, 없으면 미발송 전체 발송
    """
    try:
        storage = get_sqlite_storage()
        await storage.initialize()

        alert_service = get_alert_service()

        # 미발송 알림 조회
        unsent_alerts = await storage.get_unsent_alerts(limit=50)

        if not unsent_alerts:
            return {"success": True, "message": "No pending alerts to send", "sent_count": 0}

        # 특정 ID 필터링
        if request and request.alert_ids:
            unsent_alerts = [a for a in unsent_alerts if a.get("id") in request.alert_ids]

        if not unsent_alerts:
            return {"success": True, "message": "No matching alerts found", "sent_count": 0}

        # 알림 발송
        sent_count = 0
        for alert in unsent_alerts:
            result = await alert_service.send_single_alert(alert)

            # 성공 시 발송 완료 표시
            if result.get("slack") or result.get("email"):
                await storage.mark_alert_sent(alert["id"])
                sent_count += 1

        return {
            "success": True,
            "sent_count": sent_count,
            "total_pending": len(unsent_alerts),
            "channels": {
                "slack": alert_service._slack_enabled,
                "email": alert_service._email_enabled,
            },
        }

    except Exception as e:
        logging.error(f"Alert send error: {e}")
        return {"success": False, "error": str(e), "sent_count": 0}


@app.post("/api/alerts/test")
async def send_test_alert():
    """테스트 알림 발송"""
    try:
        alert_service = get_alert_service()

        test_alert = {
            "alert_datetime": datetime.now().isoformat(),
            "brand": "TEST BRAND",
            "asin": "B000TEST01",
            "product_name": "Test Product - Alert System Verification",
            "deal_type": "lightning",
            "discount_percent": 50.0,
            "deal_price": 19.99,
            "original_price": 39.99,
            "time_remaining": "2h 30m",
            "claimed_percent": 45,
            "product_url": "https://amazon.com/dp/B000TEST01",
            "alert_type": "lightning_deal",
            "alert_message": "Test Alert - 시스템 테스트 알림입니다",
        }

        result = await alert_service.send_single_alert(test_alert)

        return {
            "success": True,
            "test_alert": test_alert,
            "send_result": result,
            "message": "Test alert sent successfully"
            if any(result.values())
            else "No channels enabled",
        }

    except Exception as e:
        logging.error(f"Test alert error: {e}")
        return {"success": False, "error": str(e)}


# ============= Email Verification API =============

import jwt

# JWT 설정
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
EMAIL_VERIFICATION_EXPIRES_MINUTES = 30  # 30분 만료


def create_email_verification_token(
    email: str, expires_minutes: int = EMAIL_VERIFICATION_EXPIRES_MINUTES
) -> str:
    """
    이메일 인증용 JWT 토큰 생성

    Args:
        email: 인증할 이메일 주소
        expires_minutes: 토큰 만료 시간 (분)

    Returns:
        JWT 토큰 문자열
    """
    if not JWT_SECRET_KEY:
        raise ValueError("JWT_SECRET_KEY 환경변수가 설정되지 않았습니다.")

    payload = {
        "email": email,
        "purpose": "email_verification",
        "exp": datetime.now(UTC) + timedelta(minutes=expires_minutes),
        "iat": datetime.now(UTC),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_jwt_email_token(token: str) -> dict:
    """
    JWT 이메일 인증 토큰 검증

    Args:
        token: JWT 토큰

    Returns:
        {"valid": True, "email": "..."} 또는 {"valid": False, "error": "..."}
    """
    if not JWT_SECRET_KEY:
        return {"valid": False, "error": "JWT_SECRET_KEY 환경변수가 설정되지 않았습니다."}

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        # purpose 검증
        if payload.get("purpose") != "email_verification":
            return {"valid": False, "error": "유효하지 않은 토큰입니다."}

        return {"valid": True, "email": payload["email"]}

    except jwt.ExpiredSignatureError:
        return {"valid": False, "error": "인증 토큰이 만료되었습니다. 다시 인증해주세요."}
    except jwt.InvalidTokenError:
        return {"valid": False, "error": "유효하지 않은 인증 토큰입니다."}


@app.post("/api/alerts/send-verification")
@limiter.limit("3/minute")  # 분당 3회 제한 (스팸 방지)
async def send_verification_email(request: Request):
    """
    이메일 인증 요청 - 인증 이메일 발송 (JWT 방식)

    보안: Rate Limit으로 스팸 방지 (분당 3회)
    사용자가 이메일을 입력하고 '인증하기' 버튼을 누르면
    해당 이메일로 JWT 토큰이 포함된 인증 링크를 발송합니다.

    JWT 토큰은 30분간 유효하며, 서버 재시작과 무관하게 검증 가능합니다.
    """
    try:
        body = await request.json()
        email = body.get("email", "").strip()

        # 이메일 형식 검증
        import re

        email_regex = r"^[^\s@]+@[^\s@]+\.[^\s@]+$"
        if not email or not re.match(email_regex, email):
            raise HTTPException(status_code=400, detail="올바른 이메일 주소를 입력해주세요.")

        # 이미 인증된 이메일인지 확인
        state_manager = get_state_manager()
        existing = state_manager.get_subscription(email)
        if existing and existing.verified:
            return {
                "success": True,
                "already_verified": True,
                "message": "이미 인증 완료된 이메일입니다.",
            }

        # JWT 토큰 생성 (30분 유효)
        token = create_email_verification_token(email)

        # 인증 전용 페이지 URL 생성 (대시보드 대신 전용 페이지로 리다이렉트)
        base_url = get_base_url()
        verify_url = f"{base_url}/api/alerts/confirm-email?token={token}&email={email}"

        # EmailSender 직접 사용
        from src.tools.email_sender import EmailSender

        email_sender = EmailSender()

        if not email_sender.is_enabled():
            raise HTTPException(status_code=503, detail="이메일 서비스가 설정되지 않았습니다.")

        # 인증 이메일 발송
        result = await email_sender.send_verification_email(
            recipient=email, verify_url=verify_url, token=token
        )

        if result.success:
            logging.info(
                f"Verification email sent to {email} (JWT, expires in {EMAIL_VERIFICATION_EXPIRES_MINUTES}min)"
            )
            return {
                "success": True,
                "message": "인증 이메일이 발송되었습니다. (30분 내 인증해주세요)",
            }
        else:
            raise HTTPException(status_code=500, detail=f"이메일 발송 실패: {result.message}")

    except ValueError as e:
        # JWT_SECRET_KEY 미설정 에러
        logging.error(f"JWT configuration error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Send verification email error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/alerts/verify-email")
@limiter.limit("10/minute")  # 분당 10회 제한 (brute force 방지)
async def verify_email_token_endpoint(request: Request):
    """
    이메일 인증 토큰 검증 (JWT 방식)

    보안: Rate Limit으로 brute force 방지 (분당 10회)
    사용자가 이메일의 인증 버튼을 클릭하면
    JWT 토큰을 검증하고 이메일 인증 상태를 StateManager에 영구 저장합니다.

    JWT 토큰은 stateless이므로 서버 재시작과 무관하게 검증 가능합니다.
    """
    try:
        body = await request.json()
        token = body.get("token", "")
        email = body.get("email", "").strip()

        if not token or not email:
            raise HTTPException(status_code=400, detail="토큰과 이메일이 필요합니다.")

        # JWT 토큰 검증
        result = verify_jwt_email_token(token)

        if not result["valid"]:
            raise HTTPException(status_code=400, detail=result["error"])

        # 토큰의 이메일과 요청 이메일 일치 확인
        token_email = result["email"]
        if token_email != email:
            raise HTTPException(status_code=400, detail="이메일이 일치하지 않습니다.")

        # StateManager에 인증 완료 상태 영구 저장
        try:
            state_manager = get_state_manager()

            # 기존 구독 정보 확인
            existing = state_manager.get_subscription(email)

            if existing:
                # 기존 구독이 있으면 verified 상태만 업데이트
                existing.verified = True
                existing.verified_at = datetime.now()
                state_manager._save_subscriptions()
            else:
                # 새 구독 등록 (verified=True로 생성)
                state_manager.register_email(
                    email=email,
                    consent=True,
                    alert_types=["rank_change", "important_insight", "error", "daily_summary"],
                )
                # verified 상태 추가 설정
                subscription = state_manager.get_subscription(email)
                if subscription:
                    subscription.verified = True
                    subscription.verified_at = datetime.now()
                    state_manager._save_subscriptions()

            logging.info(f"Email verified and saved to StateManager: {email}")
        except Exception as e:
            logging.warning(f"Failed to save verification status: {e}")

        return {"verified": True, "email": email, "message": "이메일 인증이 완료되었습니다!"}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Verify email error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/alerts/confirm-email")
async def confirm_email_page(token: str, email: str):
    """
    이메일 인증 확인 페이지 (GET 요청으로 접근)

    사용자가 이메일의 인증 링크를 클릭하면 이 페이지가 표시됩니다.
    토큰을 검증하고 인증 완료 상태를 저장한 후, 창을 닫아도 되는 안내 페이지를 반환합니다.
    원래 대시보드 탭은 폴링으로 인증 완료를 감지하여 자동으로 다음 단계로 이동합니다.
    """
    from fastapi.responses import HTMLResponse

    # JWT 토큰 검증
    result = verify_jwt_email_token(token)

    if not result["valid"]:
        error_html = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>인증 실패 - AMORE Pacific</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #001C58 0%, #1F5795 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 20px;
                }}
                .card {{
                    background: white;
                    border-radius: 20px;
                    padding: 48px;
                    max-width: 420px;
                    width: 100%;
                    text-align: center;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                }}
                .icon {{
                    width: 80px;
                    height: 80px;
                    background: #fee2e2;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 24px;
                }}
                .icon svg {{ width: 40px; height: 40px; color: #ef4444; }}
                h1 {{ color: #001C58; font-size: 24px; margin-bottom: 12px; }}
                p {{ color: #64748b; font-size: 15px; line-height: 1.6; }}
                .error-msg {{ color: #ef4444; font-size: 13px; margin-top: 16px; padding: 12px; background: #fef2f2; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <div class="card">
                <div class="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                    </svg>
                </div>
                <h1>인증 실패</h1>
                <p>이메일 인증 링크가 만료되었거나 유효하지 않습니다.</p>
                <div class="error-msg">{result.get('error', '토큰이 유효하지 않습니다.')}</div>
                <p style="margin-top: 20px; font-size: 13px;">대시보드에서 다시 인증을 요청해주세요.</p>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=400)

    # 토큰의 이메일과 요청 이메일 일치 확인
    token_email = result["email"]
    if token_email != email:
        return HTMLResponse(content="이메일이 일치하지 않습니다.", status_code=400)

    # StateManager에 인증 완료 상태 저장
    try:
        state_manager = get_state_manager()
        existing = state_manager.get_subscription(email)

        if existing:
            existing.verified = True
            existing.verified_at = datetime.now()
            state_manager._save_subscriptions()
        else:
            state_manager.register_email(
                email=email,
                consent=True,
                alert_types=["rank_change", "important_insight", "error", "daily_summary"],
            )
            subscription = state_manager.get_subscription(email)
            if subscription:
                subscription.verified = True
                subscription.verified_at = datetime.now()
                state_manager._save_subscriptions()

        logging.info(f"Email verified via confirm page: {email}")
    except Exception as e:
        logging.warning(f"Failed to save verification status: {e}")

    # 인증 성공 페이지 반환
    success_html = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>인증 완료 - AMORE Pacific</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #001C58 0%, #1F5795 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }}
            .card {{
                background: white;
                border-radius: 20px;
                padding: 48px;
                max-width: 420px;
                width: 100%;
                text-align: center;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }}
            .icon {{
                width: 80px;
                height: 80px;
                background: #d1fae5;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 24px;
                animation: pulse 2s infinite;
            }}
            @keyframes pulse {{
                0%, 100% {{ transform: scale(1); }}
                50% {{ transform: scale(1.05); }}
            }}
            .icon svg {{ width: 40px; height: 40px; color: #10b981; }}
            h1 {{ color: #001C58; font-size: 24px; margin-bottom: 12px; }}
            p {{ color: #64748b; font-size: 15px; line-height: 1.6; }}
            .email {{
                color: #1F5795;
                font-weight: 600;
                background: #f0f9ff;
                padding: 8px 16px;
                border-radius: 8px;
                display: inline-block;
                margin: 16px 0;
            }}
            .hint {{
                margin-top: 24px;
                padding: 16px;
                background: #f8fafc;
                border-radius: 12px;
                font-size: 13px;
                color: #475569;
            }}
            .close-btn {{
                margin-top: 24px;
                padding: 14px 32px;
                background: #001C58;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.2s;
            }}
            .close-btn:hover {{ background: #1F5795; }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="icon">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
                </svg>
            </div>
            <h1>이메일 인증 완료!</h1>
            <div class="email">{email}</div>
            <p>이메일 주소가 성공적으로 인증되었습니다.</p>
            <div class="hint">
                이 창은 닫아도 됩니다.<br>
                원래 대시보드 화면에서 자동으로 다음 단계로 이동합니다.
            </div>
            <button class="close-btn" onclick="window.close()">이 창 닫기</button>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=success_html)


@app.get("/api/alerts/verification-status")
async def get_verification_status(email: str):
    """
    이메일 인증 상태 확인 (StateManager 기반)

    JWT 방식으로 변경되어 인증 완료 상태는 StateManager에 영구 저장됩니다.
    """
    try:
        state_manager = get_state_manager()
        subscription = state_manager.get_subscription(email)

        if subscription:
            return {
                "verified": subscription.verified,
                "status": "verified" if subscription.verified else "pending",
                "verified_at": subscription.verified_at.isoformat()
                if subscription.verified_at
                else None,
            }

        return {"verified": False, "status": "not_found"}

    except Exception as e:
        logging.error(f"Get verification status error: {e}")
        return {"verified": False, "status": "error", "error": str(e)}


# ============= Insight Email API =============


@app.post("/api/alerts/send-insight-report")
async def send_insight_report_email(request: Request):
    """
    인사이트 리포트 이메일 발송 (수동)

    대시보드에서 '이메일로 보내기' 버튼 클릭 시 호출됩니다.
    현재 인사이트와 KPI 데이터를 이메일로 발송합니다.

    StateManager 기반 인증 상태 확인 (JWT 방식 변경에 따른 업데이트)
    """
    try:
        body = await request.json()
        recipient_email = body.get("email", "").strip()

        if not recipient_email:
            raise HTTPException(status_code=400, detail="이메일 주소가 필요합니다.")

        # StateManager에서 이메일 인증 상태 확인
        state_manager = get_state_manager()
        subscription = state_manager.get_subscription(recipient_email)

        if not subscription or not subscription.verified:
            raise HTTPException(
                status_code=403, detail="이메일 인증이 필요합니다. 먼저 이메일을 인증해주세요."
            )

        # EmailSender 초기화
        from src.tools.email_sender import EmailSender

        email_sender = EmailSender()

        if not email_sender.is_enabled():
            raise HTTPException(status_code=503, detail="이메일 서비스가 설정되지 않았습니다.")

        # 현재 대시보드 데이터 로드
        dashboard_data = load_dashboard_data()
        if not dashboard_data:
            raise HTTPException(status_code=404, detail="대시보드 데이터가 없습니다.")

        # KPI 계산
        products = dashboard_data.get("products", [])
        laneige_products = [p for p in products if p.get("brand") == "LANEIGE"]
        avg_rank = (
            sum(p.get("rank", 100) for p in laneige_products) / len(laneige_products)
            if laneige_products
            else 0
        )

        # SoS 계산 (Top 100 기준)
        top100 = products[:100]
        laneige_in_top100 = len([p for p in top100 if p.get("brand") == "LANEIGE"])
        sos = (laneige_in_top100 / len(top100) * 100) if top100 else 0

        # HHI 계산
        brand_counts = {}
        for p in top100:
            brand = p.get("brand", "Unknown")
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        hhi = (
            sum((count / len(top100) * 100) ** 2 for count in brand_counts.values())
            if top100
            else 0
        )

        # 인사이트 가져오기 (캐시된 것 또는 새로 생성)
        insight_content = dashboard_data.get("latest_insight", "")
        if not insight_content:
            insight_content = (
                "<p>현재 생성된 인사이트가 없습니다. 대시보드에서 인사이트를 먼저 생성해주세요.</p>"
            )
        else:
            # 마크다운을 HTML로 간단 변환
            insight_content = insight_content.replace("\n\n", "</p><p>").replace("\n", "<br>")
            insight_content = f"<p>{insight_content}</p>"

        # Top 10 제품 데이터
        top10_products = []
        for i, p in enumerate(products[:10]):
            top10_products.append(
                {
                    "rank": i + 1,
                    "name": p.get("title", "N/A"),
                    "brand": p.get("brand", "Unknown"),
                    "change": p.get("rank_change", 0),
                }
            )

        # 브랜드별 변동
        brand_changes = []
        for brand in ["LANEIGE", "e.l.f.", "Maybelline", "Summer Fridays", "COSRX"]:
            brand_products = [p for p in products if p.get("brand") == brand]
            if brand_products:
                avg_change = sum(p.get("rank_change", 0) for p in brand_products) / len(
                    brand_products
                )
                if avg_change > 0:
                    brand_changes.append(
                        {
                            "brand": brand,
                            "change_text": f"평균 ▲{avg_change:.1f} 상승",
                            "color": "#28a745",
                        }
                    )
                elif avg_change < 0:
                    brand_changes.append(
                        {
                            "brand": brand,
                            "change_text": f"평균 ▼{abs(avg_change):.1f} 하락",
                            "color": "#dc3545",
                        }
                    )

        # 리포트 날짜
        report_date = datetime.now().strftime("%Y년 %m월 %d일")

        # 대시보드 URL (Railway 자동 감지)
        dashboard_url = get_base_url() + "/dashboard"

        # 이메일 발송
        result = await email_sender.send_insight_report(
            recipients=[recipient_email],
            report_date=report_date,
            avg_rank=avg_rank,
            sos=sos,
            hhi=hhi,
            insight_content=insight_content,
            top10_products=top10_products,
            brand_changes=brand_changes,
            dashboard_url=dashboard_url,
        )

        if result.success:
            logging.info(f"Insight report sent to {recipient_email}")
            return {
                "success": True,
                "message": f"인사이트 리포트가 {recipient_email}로 발송되었습니다.",
                "sent_to": result.sent_to,
            }
        else:
            raise HTTPException(status_code=500, detail=f"이메일 발송 실패: {result.message}")

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Send insight report error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============= Category KPI API =============


@app.get("/api/category/kpi")
async def get_category_kpi(
    category_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    brand: str = "LANEIGE",
):
    """
    카테고리별 KPI 데이터 조회 (기간 필터링 지원)

    Args:
        category_id: 카테고리 ID (beauty_personal_care, skin_care, lip_care, lip_makeup, face_powder)
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        brand: 타겟 브랜드 (기본값: LANEIGE)

    Returns:
        KPI 데이터: sos, best_rank, cpi, new_competitors
    """
    try:
        # 날짜 범위 설정
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        rows = []

        # SQLite에서 데이터 조회
        try:
            from src.tools.sqlite_storage import get_sqlite_storage

            sqlite = get_sqlite_storage()
            await sqlite.initialize()

            query = """
                SELECT snapshot_date, rank, brand, price
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                AND category_id = ?
                ORDER BY snapshot_date DESC, rank ASC
            """
            with sqlite.get_connection() as conn:
                cursor = conn.execute(query, (start_date, end_date, category_id))
                rows = cursor.fetchall()
        except Exception as db_err:
            logging.warning(f"SQLite query failed for category KPI: {db_err}")

        # JSON fallback
        if not rows:
            crawl_data = _load_crawl_data_for_sos()
            if crawl_data and crawl_data.get("categories", {}).get(category_id):
                cat_data = crawl_data["categories"][category_id]
                snapshot_date = crawl_data.get("snapshot_date", end_date)
                for product in cat_data.get("products", []):
                    rows.append(
                        (
                            snapshot_date,
                            product.get("rank", 100),
                            product.get("brand", "Unknown"),
                            product.get("price"),
                        )
                    )

        if not rows:
            return {
                "success": True,
                "message": f"해당 기간({start_date} ~ {end_date})에 데이터가 없습니다.",
                "data": None,
                "period": {"start": start_date, "end": end_date},
            }

        # KPI 계산
        total_products = len(rows)
        brand_products = [r for r in rows if r[2] and brand.lower() in r[2].lower()]
        brand_count = len(brand_products)

        # SoS (Share of Shelf)
        sos = (brand_count / total_products * 100) if total_products > 0 else 0

        # Best Rank
        brand_ranks = [r[1] for r in brand_products if r[1]]
        best_rank = min(brand_ranks) if brand_ranks else None

        # CPI (Competitive Price Index) - 브랜드 평균가 / 전체 평균가 * 100
        brand_prices = [r[3] for r in brand_products if r[3] and r[3] > 0]
        all_prices = [r[3] for r in rows if r[3] and r[3] > 0]

        if brand_prices and all_prices:
            brand_avg_price = sum(brand_prices) / len(brand_prices)
            all_avg_price = sum(all_prices) / len(all_prices)
            cpi = (brand_avg_price / all_avg_price * 100) if all_avg_price > 0 else 100
        else:
            cpi = 100

        # New Competitors (최근 7일 내 신규 진입 - 간소화된 계산)
        # 실제로는 이전 기간 데이터와 비교 필요, 여기서는 추정값
        new_competitors = max(0, total_products - brand_count - 50)  # 간소화된 추정

        return {
            "success": True,
            "data": {
                "category_id": category_id,
                "sos": round(sos, 1),
                "best_rank": best_rank,
                "cpi": round(cpi, 0),
                "new_competitors": new_competitors,
                "brand": brand,
                "product_count": brand_count,
                "total_products": total_products,
            },
            "period": {"start": start_date, "end": end_date},
        }

    except Exception as e:
        logging.error(f"Category KPI API error: {e}")
        return {"success": False, "error": str(e), "data": None}


# ============= SoS (Share of Shelf) API =============


def _load_crawl_data_for_sos():
    """JSON 파일에서 크롤링 데이터 로드 (SQLite fallback)"""
    import json
    from pathlib import Path

    # latest_crawl_result.json에서 데이터 로드
    crawl_path = Path("./data/latest_crawl_result.json")
    if crawl_path.exists():
        with open(crawl_path, encoding="utf-8") as f:
            return json.load(f)
    return None


@app.get("/api/sos/category")
async def get_sos_by_category(
    start_date: str | None = None,
    end_date: str | None = None,
    compare_brands: str | None = None,  # comma-separated brand names
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
    try:
        # 비교 브랜드 파싱
        compare_brand_list = []
        if compare_brands:
            compare_brand_list = [b.strip() for b in compare_brands.split(",") if b.strip()]

        # 날짜 범위 설정
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = end_date

        # SQLite 먼저 시도
        rows = []
        try:
            from src.tools.sqlite_storage import get_sqlite_storage

            sqlite = get_sqlite_storage()
            await sqlite.initialize()

            query = """
                SELECT snapshot_date, category_id, brand, COUNT(*) as product_count
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                GROUP BY snapshot_date, category_id, brand
                ORDER BY snapshot_date DESC, category_id, product_count DESC
            """
            with sqlite.get_connection() as conn:
                cursor = conn.execute(query, (start_date, end_date))
                rows = cursor.fetchall()
        except Exception as db_err:
            logging.warning(f"SQLite query failed, using JSON fallback: {db_err}")

        # SQLite 데이터 없으면 JSON fallback
        if not rows:
            crawl_data = _load_crawl_data_for_sos()
            if crawl_data and crawl_data.get("categories"):
                # JSON에서 데이터 추출
                snapshot_date = crawl_data.get("snapshot_date", end_date)
                for cat_id, cat_data in crawl_data.get("categories", {}).items():
                    for product in cat_data.get("products", []):
                        brand = product.get("brand", "Unknown")
                        rows.append((snapshot_date, cat_id, brand, 1))

        if not rows:
            return {
                "success": True,
                "message": f"해당 기간({start_date} ~ {end_date})에 데이터가 없습니다.",
                "data": [],
                "period": {"start": start_date, "end": end_date},
            }

        # 데이터 집계
        # 구조: {category_id: {brand: {dates: [count, ...], total_count: N}}}
        category_data = {}
        dates_set = set()

        for row in rows:
            if len(row) == 4:
                snapshot_date, category_id, brand, count = row
            else:
                snapshot_date, category_id, brand, count = row[0], row[1], row[2], row[3]
            dates_set.add(snapshot_date)

            if category_id not in category_data:
                category_data[category_id] = {}
            if brand not in category_data[category_id]:
                category_data[category_id][brand] = {"dates": {}, "total_count": 0}

            category_data[category_id][brand]["dates"][snapshot_date] = count
            category_data[category_id][brand]["total_count"] += count

        # SoS 계산 (기간 평균)
        num_dates = len(dates_set)
        result_data = []

        # 카테고리 계층 구조 로드
        hierarchy_path = Path("./config/category_hierarchy.json")
        hierarchy_data = {}
        if hierarchy_path.exists():
            with open(hierarchy_path, encoding="utf-8") as f:
                hierarchy_data = json.load(f).get("categories", {})

        # 카테고리 메타 정보 (계층 구조 포함)
        category_meta = {
            "beauty": {
                "name": "Beauty & Personal Care",
                "level": 0,
                "parent_id": None,
                "indent": 0,
                "order": 0,
            },
            "skin_care": {
                "name": "Skin Care",
                "level": 1,
                "parent_id": "beauty",
                "indent": 1,
                "order": 1,
            },
            "lip_care": {
                "name": "Lip Care",
                "level": 2,
                "parent_id": "skin_care",
                "indent": 2,
                "order": 2,
            },
            "lip_makeup": {
                "name": "Lip Makeup",
                "level": 2,
                "parent_id": "makeup",
                "indent": 1,
                "order": 3,
            },
            "face_powder": {
                "name": "Face Powder",
                "level": 3,
                "parent_id": "face_makeup",
                "indent": 2,
                "order": 4,
            },
        }

        # hierarchy_data에서 정보 업데이트
        for cat_id, meta in category_meta.items():
            if cat_id in hierarchy_data:
                meta["name"] = hierarchy_data[cat_id].get("name", meta["name"])
                meta["level"] = hierarchy_data[cat_id].get("level", meta["level"])
                meta["parent_id"] = hierarchy_data[cat_id].get("parent_id", meta["parent_id"])

        for category_id, brands in category_data.items():
            # 해당 카테고리의 총 제품 수 (기간 합계)
            total_products_in_category = sum(b["total_count"] for b in brands.values())

            # LANEIGE SoS
            laneige_count = 0
            laneige_appearance_days = 0  # Top 100에 진입한 고유 날짜 수
            laneige_dates = set()  # 중복 제거용
            laneige_variants = ["LANEIGE", "Laneige", "laneige"]
            for variant in laneige_variants:
                if variant in brands:
                    laneige_count += brands[variant]["total_count"]
                    # 출현 날짜 수집
                    if "dates" in brands[variant]:
                        laneige_dates.update(brands[variant]["dates"])
            laneige_appearance_days = len(laneige_dates)

            laneige_sos = (
                (laneige_count / total_products_in_category * 100)
                if total_products_in_category > 0
                else 0
            )

            # 평균 SoS (전체 브랜드 수 기준)
            num_brands = len(brands)
            avg_sos = (100 / num_brands) if num_brands > 0 else 0

            # 비교 브랜드 SoS
            compare_sos = {}
            for compare_brand in compare_brand_list:
                brand_count = 0
                for brand_name, brand_data in brands.items():
                    if compare_brand.lower() in brand_name.lower():
                        brand_count += brand_data["total_count"]
                compare_sos[compare_brand] = (
                    (brand_count / total_products_in_category * 100)
                    if total_products_in_category > 0
                    else 0
                )

            # LANEIGE 개별 제품 데이터 (해당 카테고리 내)
            # 제품별 상세는 별도 쿼리 필요 - 여기서는 브랜드 레벨만

            # 카테고리 메타 정보 가져오기
            meta = category_meta.get(
                category_id,
                {"name": category_id, "level": 0, "parent_id": None, "indent": 0, "order": 99},
            )

            result_data.append(
                {
                    "category_id": category_id,
                    "category_name": meta["name"],
                    "level": meta["level"],
                    "parent_id": meta["parent_id"],
                    "indent": meta["indent"],
                    "order": meta["order"],
                    "total_products": total_products_in_category // num_dates
                    if num_dates > 0
                    else 0,
                    "laneige_sos": round(laneige_sos, 2),
                    "laneige_count": round(laneige_count / num_dates, 1)
                    if num_dates > 0
                    else 0,  # 소수점 1자리 (일 평균)
                    "laneige_appearance_days": laneige_appearance_days,  # 출현 일수
                    "laneige_appearance_rate": round(laneige_appearance_days / num_dates * 100, 1)
                    if num_dates > 0
                    else 0,  # 출현율 %
                    "avg_sos": round(avg_sos, 2),
                    "compare_brands": compare_sos,
                    "num_dates": num_dates,
                }
            )

        # 계층 구조 순서대로 정렬
        result_data.sort(key=lambda x: x.get("order", 99))

        return {
            "success": True,
            "period": {"start": start_date, "end": end_date, "days": num_dates},
            "data": result_data,
            "compare_brands": compare_brand_list,
            "hierarchy_info": {
                "description": "각 카테고리는 자체 Top 100 기준으로 독립 계산됩니다.",
                "note": "상위 카테고리와 하위 카테고리의 SoS는 서로 다른 랭킹에서 계산됩니다.",
            },
        }

    except Exception as e:
        logging.error(f"SoS category API error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/sos/brands")
async def get_available_brands(category_id: str | None = None, min_count: int = 1):
    """
    비교 가능한 브랜드 목록 조회 (Top 100에 포함된 브랜드들)

    Args:
        category_id: 특정 카테고리만 조회 (선택)
        min_count: 최소 제품 수 (기본: 1)

    Returns:
        브랜드 목록 (제품 수 기준 정렬)
    """
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        rows = []
        # SQLite 먼저 시도
        try:
            from src.tools.sqlite_storage import get_sqlite_storage

            sqlite = get_sqlite_storage()
            await sqlite.initialize()

            if category_id:
                query = """
                    SELECT brand, COUNT(DISTINCT asin) as product_count,
                           COUNT(DISTINCT snapshot_date) as days_present
                    FROM raw_data
                    WHERE snapshot_date BETWEEN ? AND ?
                    AND category_id = ?
                    AND LOWER(brand) != 'unknown'
                    GROUP BY brand
                    HAVING product_count >= ?
                    ORDER BY product_count DESC
                """
                params = (start_date, end_date, category_id, min_count)
            else:
                query = """
                    SELECT brand, COUNT(DISTINCT asin) as product_count,
                           COUNT(DISTINCT snapshot_date) as days_present
                    FROM raw_data
                    WHERE snapshot_date BETWEEN ? AND ?
                    AND LOWER(brand) != 'unknown'
                    GROUP BY brand
                    HAVING product_count >= ?
                    ORDER BY product_count DESC
                """
                params = (start_date, end_date, min_count)

            with sqlite.get_connection() as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
        except Exception as db_err:
            logging.warning(f"SQLite query failed for brands: {db_err}")

        # SQLite 데이터 없으면 JSON fallback
        brands = []
        if not rows:
            crawl_data = _load_crawl_data_for_sos()
            if crawl_data and crawl_data.get("categories"):
                brand_counts = {}
                for cat_id, cat_data in crawl_data.get("categories", {}).items():
                    if category_id and cat_id != category_id:
                        continue
                    for product in cat_data.get("products", []):
                        brand = product.get("brand", "Unknown")
                        if brand:
                            brand_counts[brand] = brand_counts.get(brand, 0) + 1

                for brand_name, count in sorted(brand_counts.items(), key=lambda x: -x[1]):
                    # Unknown 브랜드 제외
                    if (
                        count >= min_count
                        and brand_name.strip()
                        and brand_name.lower() != "unknown"
                    ):
                        brands.append(
                            {
                                "name": brand_name,
                                "product_count": count,
                                "days_present": 1,
                                "is_laneige": "laneige" in brand_name.lower(),
                            }
                        )
        else:
            for row in rows:
                brand_name, product_count, days_present = row
                # Unknown 브랜드 제외 (SQL에서도 필터링하지만 이중 체크)
                if brand_name and brand_name.strip() and brand_name.lower() != "unknown":
                    brands.append(
                        {
                            "name": brand_name,
                            "product_count": product_count,
                            "days_present": days_present,
                            "is_laneige": "laneige" in brand_name.lower(),
                        }
                    )

        return {
            "success": True,
            "period": {"start": start_date, "end": end_date},
            "category_id": category_id,
            "brands": brands,
            "total_brands": len(brands),
        }

    except Exception as e:
        logging.error(f"SoS brands API error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/sos/trend")
async def get_sos_trend(
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
    try:
        from src.tools.sqlite_storage import get_sqlite_storage

        sqlite = get_sqlite_storage()
        await sqlite.initialize()

        # start_date/end_date가 제공되면 사용, 아니면 days 기반으로 계산
        if start_date and end_date:
            pass  # 그대로 사용
        else:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        # 일별 전체 제품 수
        if category_id:
            total_query = """
                SELECT snapshot_date, COUNT(*) as total_count
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                AND category_id = ?
                GROUP BY snapshot_date
                ORDER BY snapshot_date
            """
            total_params = (start_date, end_date, category_id)

            brand_query = """
                SELECT snapshot_date, COUNT(*) as brand_count
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                AND category_id = ?
                AND LOWER(brand) LIKE ?
                GROUP BY snapshot_date
                ORDER BY snapshot_date
            """
            brand_params = (start_date, end_date, category_id, f"%{brand.lower()}%")
        else:
            total_query = """
                SELECT snapshot_date, COUNT(*) as total_count
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                GROUP BY snapshot_date
                ORDER BY snapshot_date
            """
            total_params = (start_date, end_date)

            brand_query = """
                SELECT snapshot_date, COUNT(*) as brand_count
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                AND LOWER(brand) LIKE ?
                GROUP BY snapshot_date
                ORDER BY snapshot_date
            """
            brand_params = (start_date, end_date, f"%{brand.lower()}%")

        with sqlite.get_connection() as conn:
            # 전체 카운트
            cursor = conn.execute(total_query, total_params)
            total_rows = cursor.fetchall()
            total_by_date = {row[0]: row[1] for row in total_rows}

            # 브랜드 카운트
            cursor = conn.execute(brand_query, brand_params)
            brand_rows = cursor.fetchall()
            brand_by_date = {row[0]: row[1] for row in brand_rows}

        # SoS 계산
        trend_data = []
        for date, total in sorted(total_by_date.items()):
            brand_count = brand_by_date.get(date, 0)
            sos = (brand_count / total * 100) if total > 0 else 0
            trend_data.append(
                {
                    "date": date,
                    "total_products": total,
                    "brand_count": brand_count,
                    "sos": round(sos, 2),
                }
            )

        return {
            "success": True,
            "brand": brand,
            "category_id": category_id,
            "period": {"start": start_date, "end": end_date, "days": days},
            "trend": trend_data,
        }

    except Exception as e:
        logging.error(f"SoS trend API error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/sos/trend/competitors-avg")
async def get_competitors_avg_sos_trend(
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
    try:
        from src.tools.sqlite_storage import get_sqlite_storage

        sqlite = get_sqlite_storage()
        await sqlite.initialize()

        # 날짜 범위 결정
        if start_date and end_date:
            pass
        else:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        # 일별 전체 제품 수 쿼리
        if category_id:
            total_query = """
                SELECT snapshot_date, COUNT(*) as total_count
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                AND category_id = ?
                GROUP BY snapshot_date
                ORDER BY snapshot_date
            """
            total_params = (start_date, end_date, category_id)

            # 일별/브랜드별 제품 수 (LANEIGE 제외)
            brand_daily_query = """
                SELECT snapshot_date, brand, COUNT(*) as brand_count
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                AND category_id = ?
                AND LOWER(brand) NOT LIKE ?
                AND brand IS NOT NULL
                AND brand != ''
                GROUP BY snapshot_date, brand
                ORDER BY snapshot_date, brand_count DESC
            """
            brand_daily_params = (start_date, end_date, category_id, f"%{exclude_brand.lower()}%")
        else:
            total_query = """
                SELECT snapshot_date, COUNT(*) as total_count
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                GROUP BY snapshot_date
                ORDER BY snapshot_date
            """
            total_params = (start_date, end_date)

            brand_daily_query = """
                SELECT snapshot_date, brand, COUNT(*) as brand_count
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                AND LOWER(brand) NOT LIKE ?
                AND brand IS NOT NULL
                AND brand != ''
                GROUP BY snapshot_date, brand
                ORDER BY snapshot_date, brand_count DESC
            """
            brand_daily_params = (start_date, end_date, f"%{exclude_brand.lower()}%")

        with sqlite.get_connection() as conn:
            # 전체 카운트
            cursor = conn.execute(total_query, total_params)
            total_rows = cursor.fetchall()
            total_by_date = {row[0]: row[1] for row in total_rows}

            # 일별/브랜드별 카운트
            cursor = conn.execute(brand_daily_query, brand_daily_params)
            brand_rows = cursor.fetchall()

        # 일별로 Top N 브랜드의 평균 SoS 계산
        from collections import defaultdict

        # 일별 브랜드 데이터 그룹화
        daily_brands: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for date, brand, count in brand_rows:
            daily_brands[date].append((brand, count))

        # 일별 경쟁 브랜드 평균 SoS 계산
        trend_data = []
        for date, total in sorted(total_by_date.items()):
            brands_for_date = daily_brands.get(date, [])
            # 이미 brand_count DESC로 정렬되어 있으므로 상위 N개 선택
            top_brands = brands_for_date[:top_n]

            if top_brands and total > 0:
                # 각 브랜드의 SoS 계산
                sos_values = [(count / total * 100) for _, count in top_brands]
                avg_sos = sum(sos_values) / len(sos_values)
            else:
                avg_sos = 0

            trend_data.append(
                {
                    "date": date,
                    "total_products": total,
                    "top_brands_count": len(top_brands),
                    "avg_sos": round(avg_sos, 2),
                }
            )

        return {
            "success": True,
            "category_id": category_id,
            "excluded_brand": exclude_brand,
            "top_n": top_n,
            "period": {"start": start_date, "end": end_date, "days": days},
            "trend": trend_data,
        }

    except Exception as e:
        logging.error(f"Competitors avg SoS trend API error: {e}")
        return {"success": False, "error": str(e)}


# ============= 데이터 동기화 API =============


@app.get("/api/sync/status")
async def sync_status():
    """
    Railway Volume의 데이터 현황 반환

    Returns:
        - latest: 최신 데이터 날짜
        - oldest: 가장 오래된 데이터 날짜
        - total_days: 총 일수
        - total_records: SQLite raw_data 총 레코드 수
    """
    try:
        sqlite = get_sqlite_storage()
        if not sqlite:
            raise HTTPException(status_code=500, detail="SQLite not available")

        await sqlite.initialize()

        # raw_data 테이블에서 날짜 범위 조회
        with sqlite.get_connection() as conn:
            cursor = conn.execute("""
                SELECT
                    MIN(snapshot_date) as oldest,
                    MAX(snapshot_date) as latest,
                    COUNT(DISTINCT snapshot_date) as total_days,
                    COUNT(*) as total_records
                FROM raw_data
            """)
            row = cursor.fetchone()

            if not row or not row[0]:
                return {
                    "success": True,
                    "latest": None,
                    "oldest": None,
                    "total_days": 0,
                    "total_records": 0,
                    "message": "No data available",
                }

            return {
                "success": True,
                "latest": row[1],
                "oldest": row[0],
                "total_days": row[2],
                "total_records": row[3],
            }
    except Exception as e:
        logging.error(f"Sync status error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/sync/dates")
async def sync_dates():
    """
    사용 가능한 모든 날짜 목록 반환 (정렬됨)

    Returns:
        - dates: ["2026-01-17", "2026-01-18", ..., "2026-01-25"]
    """
    try:
        sqlite = get_sqlite_storage()
        if not sqlite:
            raise HTTPException(status_code=500, detail="SQLite not available")

        await sqlite.initialize()

        with sqlite.get_connection() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT snapshot_date
                FROM raw_data
                ORDER BY snapshot_date
            """)
            dates = [row[0] for row in cursor.fetchall()]

        return {"success": True, "dates": dates, "count": len(dates)}
    except Exception as e:
        logging.error(f"Sync dates error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/sync/download/{date}")
async def sync_download(date: str):
    """
    특정 날짜의 raw_data를 JSON으로 다운로드

    Args:
        date: 날짜 (YYYY-MM-DD 형식)

    Returns:
        JSON array of raw_data records for the specified date
    """
    import re

    # 날짜 형식 검증
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    try:
        sqlite = get_sqlite_storage()
        if not sqlite:
            raise HTTPException(status_code=500, detail="SQLite not available")

        await sqlite.initialize()

        with sqlite.get_connection() as conn:
            # 컬럼명 가져오기
            cursor = conn.execute("PRAGMA table_info(raw_data)")
            columns = [row[1] for row in cursor.fetchall()]

            # 해당 날짜 데이터 조회
            cursor = conn.execute(
                """
                SELECT * FROM raw_data
                WHERE snapshot_date = ?
                ORDER BY category_id, rank
            """,
                (date,),
            )
            rows = cursor.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail=f"No data found for date: {date}")

        # 딕셔너리 변환
        records = []
        for row in rows:
            record = dict(zip(columns, row, strict=False))
            records.append(record)

        return {"success": True, "date": date, "count": len(records), "records": records}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Sync download error for {date}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/sync/upload")
async def sync_upload(request: Request):
    """
    로컬에서 Railway로 raw_data 업로드

    Request Body:
        {
            "records": [...],  # raw_data 레코드 배열
            "api_key": "..."   # 인증 키 (선택)
        }

    Returns:
        {"success": True, "inserted": N, "updated": M}
    """
    import os

    try:
        body = await request.json()
        records = body.get("records", [])
        api_key = body.get("api_key", "")

        # API 키 검증 (설정된 경우)
        expected_key = os.getenv("API_KEY", "")
        if expected_key and api_key != expected_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

        if not records:
            raise HTTPException(status_code=400, detail="No records provided")

        sqlite = get_sqlite_storage()
        if not sqlite:
            raise HTTPException(status_code=500, detail="SQLite not available")

        await sqlite.initialize()

        inserted = 0
        updated = 0

        with sqlite.get_connection() as conn:
            # 스키마 마이그레이션: 누락된 컬럼 추가
            migration_columns = [
                ("image_url", "TEXT"),
                ("is_best_seller", "INTEGER DEFAULT 0"),
                ("is_amazon_choice", "INTEGER DEFAULT 0"),
            ]
            for col_name, col_type in migration_columns:
                try:
                    conn.execute(f"ALTER TABLE raw_data ADD COLUMN {col_name} {col_type}")
                    conn.commit()
                    logging.info(f"Added {col_name} column to raw_data table")
                except Exception:
                    pass  # 이미 존재하면 무시

            for record in records:
                # UPSERT: snapshot_date + category_id + asin 조합이 unique key
                cursor = conn.execute(
                    """
                    SELECT id FROM raw_data
                    WHERE snapshot_date = ? AND category_id = ? AND asin = ?
                """,
                    (
                        record.get("snapshot_date"),
                        record.get("category_id"),
                        record.get("asin"),
                    ),
                )
                existing = cursor.fetchone()

                if existing:
                    # UPDATE
                    conn.execute(
                        """
                        UPDATE raw_data SET
                            rank = ?, product_name = ?, brand = ?, price = ?,
                            rating = ?, reviews_count = ?, product_url = ?,
                            image_url = ?, is_best_seller = ?, is_amazon_choice = ?
                        WHERE id = ?
                    """,
                        (
                            record.get("rank"),
                            record.get("product_name"),
                            record.get("brand"),
                            record.get("price"),
                            record.get("rating"),
                            record.get("reviews_count"),
                            record.get("product_url"),
                            record.get("image_url"),
                            record.get("is_best_seller", 0),
                            record.get("is_amazon_choice", 0),
                            existing[0],
                        ),
                    )
                    updated += 1
                else:
                    # INSERT
                    conn.execute(
                        """
                        INSERT INTO raw_data (
                            snapshot_date, category_id, asin, rank, product_name,
                            brand, price, rating, reviews_count, product_url,
                            image_url, is_best_seller, is_amazon_choice
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            record.get("snapshot_date"),
                            record.get("category_id"),
                            record.get("asin"),
                            record.get("rank"),
                            record.get("product_name"),
                            record.get("brand"),
                            record.get("price"),
                            record.get("rating"),
                            record.get("reviews_count"),
                            record.get("product_url"),
                            record.get("image_url"),
                            record.get("is_best_seller", 0),
                            record.get("is_amazon_choice", 0),
                        ),
                    )
                    inserted += 1

            conn.commit()

        logging.info(f"Sync upload: inserted={inserted}, updated={updated}")
        return {"success": True, "inserted": inserted, "updated": updated}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Sync upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Market Intelligence API (v2026.01.26)
# =============================================================================


@app.get("/api/market-intelligence/status", response_model=MarketIntelligenceStatusResponse)
async def get_market_intelligence_status():
    """
    Market Intelligence 시스템 상태 조회

    Returns:
        초기화 상태, 수집된 레이어, 통계
    """
    try:
        engine = await get_market_intelligence()
        stats = engine.get_stats()

        # 마지막 수집 시간
        last_collection = None
        if engine.layer_data:
            times = [ld.collected_at for ld in engine.layer_data.values()]
            if times:
                last_collection = max(times)

        return MarketIntelligenceStatusResponse(
            initialized=engine._initialized,
            layers_collected=list(engine.layer_data.keys()),
            last_collection=last_collection,
            stats=stats,
        )
    except Exception as e:
        logger.error(f"Market Intelligence status error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/market-intelligence/layers")
async def get_market_intelligence_layers(layer: int | None = None):
    """
    4-Layer 데이터 조회

    Args:
        layer: 특정 레이어만 조회 (1-4, None이면 전체)

    Returns:
        레이어별 데이터
    """
    try:
        engine = await get_market_intelligence()

        if layer is not None:
            layer_data = engine.layer_data.get(layer)
            if not layer_data:
                return {
                    "error": f"Layer {layer} 데이터가 없습니다.",
                    "available_layers": list(engine.layer_data.keys()),
                }

            return {
                "layer": layer_data.layer,
                "layer_name": layer_data.layer_name,
                "collected_at": layer_data.collected_at,
                "data": layer_data.data,
                "sources": layer_data.sources,
            }

        # 전체 레이어
        result = {}
        for layer_num, layer_data in engine.layer_data.items():
            result[f"layer_{layer_num}"] = {
                "layer": layer_data.layer,
                "layer_name": layer_data.layer_name,
                "collected_at": layer_data.collected_at,
                "data": layer_data.data,
                "sources": layer_data.sources,
            }

        return result
    except Exception as e:
        logger.error(f"Market Intelligence layers error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/market-intelligence/collect", dependencies=[Depends(verify_api_key)])
async def collect_market_intelligence(layers: list[int] | None = None):
    """
    Market Intelligence 데이터 수집 트리거

    Args:
        layers: 수집할 레이어 목록 (None이면 전체)

    Returns:
        수집 결과
    """
    try:
        engine = await get_market_intelligence()

        if layers:
            # 특정 레이어만 수집
            results = {}
            for layer_num in layers:
                layer_data = await engine.collect_layer(layer_num)
                if layer_data:
                    results[f"layer_{layer_num}"] = {
                        "status": "collected",
                        "collected_at": layer_data.collected_at,
                        "sources_count": len(layer_data.sources),
                    }
                else:
                    results[f"layer_{layer_num}"] = {"status": "skipped"}
        else:
            # 전체 수집
            await engine.collect_all_layers()
            results = {
                f"layer_{k}": {
                    "status": "collected",
                    "collected_at": v.collected_at,
                    "sources_count": len(v.sources),
                }
                for k, v in engine.layer_data.items()
            }

        # 데이터 저장
        engine.save_data()

        return {"status": "success", "collected": results, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Market Intelligence collection error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/market-intelligence/insight")
async def get_market_intelligence_insight(include_amazon: bool = False):
    """
    4-Layer 기반 인사이트 생성

    Args:
        include_amazon: Layer 1 Amazon 데이터 포함 여부

    Returns:
        생성된 인사이트 텍스트
    """
    try:
        engine = await get_market_intelligence()

        # 데이터가 없으면 먼저 수집
        if not engine.layer_data:
            await engine.collect_all_layers()

        # Amazon 데이터 가져오기 (선택)
        amazon_data = None
        if include_amazon:
            try:
                await get_sqlite_storage()
                # 최신 LANEIGE 데이터 조회
                # (실제 구현은 storage의 메서드에 따라 다름)
                amazon_data = {"sos": 5.2, "laneige_rank": 15}  # placeholder
            except Exception:
                pass

        insight = engine.generate_layered_insight(amazon_data=amazon_data)

        return {
            "insight": insight,
            "generated_at": datetime.now().isoformat(),
            "layers_used": list(engine.layer_data.keys()),
        }
    except Exception as e:
        logger.error(f"Market Intelligence insight error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/insights/sources")
async def get_insight_sources():
    """
    인사이트 출처 정보 조회

    Returns:
        출처 목록 및 통계
    """
    try:
        engine = await get_market_intelligence()

        all_sources = []
        for layer_data in engine.layer_data.values():
            all_sources.extend(layer_data.sources)

        # 출처 유형별 통계
        by_type = {}
        for source in all_sources:
            source_type = source.get("source_type", "unknown")
            by_type[source_type] = by_type.get(source_type, 0) + 1

        return {
            "total_sources": len(all_sources),
            "by_type": by_type,
            "sources": all_sources[:20],  # 최근 20개
            "source_manager_stats": engine.source_manager.get_stats(),
        }
    except Exception as e:
        logger.error(f"Insight sources error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============= 서버 실행 =============

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)

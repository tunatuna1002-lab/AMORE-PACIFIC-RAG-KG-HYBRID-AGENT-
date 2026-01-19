"""
Dashboard API Server
대시보드용 FastAPI 백엔드 서버

- 챗봇 API (ChatGPT + RAG + Ontology 연동)
- DOCX 인사이트 리포트 생성
- 대화 메모리 지원
- Audit Trail 로깅
"""

import json
import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from io import BytesIO
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
from litellm import acompletion

# Rate Limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

# RAG 시스템 연동
from src.rag.router import RAGRouter, QueryType
from src.rag.retriever import DocumentRetriever

# Ontology 스키마
from src.ontology.schema import ProductMetrics, BrandMetrics, MarketMetrics

# 통합 오케스트레이터 (신규)
from src.core.unified_orchestrator import UnifiedOrchestrator, get_unified_orchestrator
from src.core.crawl_manager import get_crawl_manager, CrawlStatus

# Level 4 Brain (LLM-First Autonomous Agent)
from src.core.brain import UnifiedBrain, get_brain, get_initialized_brain, BrainMode, TaskPriority

# 환경 변수 로드
load_dotenv()

# Rate Limiter 설정 (IP 기반)
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="AMORE Dashboard API",
    description="LANEIGE Amazon 대시보드 백엔드 API (RAG + Ontology 통합)",
    version="2.0.0"
)

# Rate Limit 초과 시 에러 핸들러
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS 설정 (환경변수로 허용 도메인 설정 가능)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8001,http://127.0.0.1:8001").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)

# ============= 서버 시작 시 자동 스케줄러 =============

AUTO_START_SCHEDULER = os.getenv("AUTO_START_SCHEDULER", "true").lower() == "true"


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 자동 스케줄러 시작 및 즉시 크롤링 체크"""
    # 1. 즉시 크롤링 필요 여부 체크 (서버 재시작 후 오늘 데이터가 없으면 바로 크롤링)
    try:
        crawl_manager = get_crawl_manager()
        if crawl_manager.needs_crawl():
            logging.info(f"서버 시작: 오늘({crawl_manager.get_kst_today()}) 데이터 없음 → 크롤링 시작")
            await crawl_manager.start_crawl()
        else:
            logging.info(f"서버 시작: 오늘 데이터 있음 또는 크롤링 중 (data_date={crawl_manager.get_data_date()})")
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

# 데이터 경로
DATA_PATH = "./data/dashboard_data.json"
DOCS_PATH = "./"  # MD 파일들이 루트에 있음
AUDIT_LOG_DIR = "./logs"

# ============= API Key 인증 설정 =============

# API_KEY: 환경변수 필수 - 기본값 없음 (보안상 하드코딩 금지)
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logging.warning("⚠️ API_KEY 환경변수가 설정되지 않았습니다. 보호된 엔드포인트에 접근할 수 없습니다.")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    API Key 검증 (민감한 엔드포인트용)

    사용법: 엔드포인트에 dependencies=[Depends(verify_api_key)] 추가
    """
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API Key가 필요합니다. 헤더에 X-API-Key를 추가하세요."
        )
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="유효하지 않은 API Key입니다."
        )
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
    formatter = logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
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
    entities: Dict,
    sources: List[str],
    response_time_ms: float
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
        "response_time_ms": round(response_time_ms, 2)
    }

    # JSON 형식으로 로그 기록
    audit_logger.info(json.dumps(audit_entry, ensure_ascii=False))

# ============= Global Instances =============

# RAG 시스템
rag_router = RAGRouter()
doc_retriever = DocumentRetriever(DOCS_PATH)

# 세션별 대화 메모리 (TTL 기반 자동 정리)
conversation_memory: Dict[str, List[Dict[str, str]]] = defaultdict(list)
session_last_activity: Dict[str, datetime] = {}  # 세션별 마지막 활동 시간
MAX_MEMORY_TURNS = 10
SESSION_TTL_HOURS = 1  # 세션 만료 시간 (1시간)
MAX_SESSIONS = 1000  # 최대 세션 수


def cleanup_expired_sessions() -> int:
    """만료된 세션 정리 (TTL 기반)"""
    now = datetime.now()
    expired = [
        sid for sid, last_time in session_last_activity.items()
        if (now - last_time).total_seconds() > SESSION_TTL_HOURS * 3600
    ]
    for sid in expired:
        if sid in conversation_memory:
            del conversation_memory[sid]
        if sid in session_last_activity:
            del session_last_activity[sid]
    return len(expired)

# 통합 오케스트레이터 인스턴스는 get_unified_orchestrator()로 관리됨


# ============= Pydantic Models =============

class ChatRequest(BaseModel):
    """챗봇 요청"""
    message: str
    session_id: Optional[str] = "default"
    context: Optional[Dict] = None


class ChatResponse(BaseModel):
    """챗봇 응답"""
    response: str
    query_type: str
    confidence: float
    sources: List[str]
    suggestions: List[str]
    entities: Dict[str, Any]


class ExportRequest(BaseModel):
    """내보내기 요청"""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    include_strategy: bool = True


# ============= Helper Functions =============

def load_dashboard_data() -> Dict[str, Any]:
    """대시보드 데이터 로드"""
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
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

    conversation_memory[session_id].append({
        "role": role,
        "content": content,
        "timestamp": now.isoformat()
    })
    # 최대 개수 유지
    if len(conversation_memory[session_id]) > MAX_MEMORY_TURNS * 2:
        conversation_memory[session_id] = conversation_memory[session_id][-MAX_MEMORY_TURNS * 2:]


def build_data_context(data: Dict, query_type: QueryType, entities: Dict) -> str:
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
    if query_type in [QueryType.DEFINITION, QueryType.INTERPRETATION, QueryType.ANALYSIS, QueryType.COMBINATION]:
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

    if query_type == QueryType.ANALYSIS or any(b for b in brands_mentioned if b.lower() != "laneige"):
        if competitors:
            top_comps = competitors[:5]
            comp_lines = [f"  - {c['brand']}: SoS {c['sos']}%, 평균 순위 {c['avg_rank']}위, 제품 {c['product_count']}개" for c in top_comps]
            context_parts.append("[경쟁사 현황]\n" + "\n".join(comp_lines))

    # 제품 정보 (DATA_QUERY, 특정 제품 언급 시)
    products = data.get("products", {})
    products_mentioned = entities.get("products", [])

    if query_type == QueryType.DATA_QUERY or products_mentioned:
        if products:
            prod_lines = []
            for asin, p in list(products.items())[:5]:
                prod_lines.append(f"""  - {p['name'][:40]}
    순위: #{p['rank']} ({p['rank_delta']}), 평점: {p['rating']}, 변동성: {p.get('volatility_status', 'N/A')}""")
            context_parts.append("[LANEIGE 제품 현황] (Ontology: ProductMetrics)\n" + "\n".join(prod_lines))

    # 카테고리 정보
    categories = data.get("categories", {})
    categories_mentioned = entities.get("categories", [])

    if categories_mentioned or query_type in [QueryType.ANALYSIS, QueryType.INTERPRETATION]:
        if categories:
            cat_lines = []
            for cat_id, cat in categories.items():
                cat_lines.append(f"  - {cat['name']}: SoS {cat['sos']}%, 최고 순위 #{cat['best_rank']}, CPI {cat.get('cpi', 100)}")
            context_parts.append("[카테고리 현황] (Ontology: MarketMetrics)\n" + "\n".join(cat_lines))

    # 액션 아이템 (전략 질문)
    if query_type == QueryType.ANALYSIS:
        action_items = data.get("home", {}).get("action_items", [])
        if action_items:
            action_lines = [f"  - [{a['priority']}] {a['product_name']}: {a['signal']} → {a['action_tag']}" for a in action_items[:4]]
            context_parts.append("[현재 액션 아이템]\n" + "\n".join(action_lines))

    return "\n\n".join(context_parts)


async def get_rag_context(query: str, query_type: QueryType) -> tuple[str, List[str]]:
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
            "home_insight_rules": "Home Page Insight Rules"
        }
        if doc_id in doc_name_map and doc_name_map[doc_id] not in sources:
            sources.append(doc_name_map[doc_id])

    return "\n\n---\n\n".join(context_parts), sources


def get_dynamic_suggestions(query_type: QueryType, entities: Dict, response: str) -> List[str]:
    """
    동적 후속 질문 제안

    응답 내용과 컨텍스트를 기반으로 맞춤형 제안 생성
    """
    suggestions = []

    # 엔티티 기반 제안
    brands = entities.get("brands", [])
    indicators = entities.get("indicators", [])
    categories = entities.get("categories", [])

    if query_type == QueryType.DEFINITION:
        # 정의 질문 → 해석/활용 질문 제안
        if indicators:
            ind = indicators[0].upper()
            suggestions.append(f"{ind}가 높으면 어떤 의미인가요?")
            suggestions.append(f"{ind} 개선을 위한 전략은?")
        suggestions.append("다른 지표와 함께 해석하면 어떤가요?")

    elif query_type == QueryType.INTERPRETATION:
        # 해석 질문 → 액션/조합 제안
        suggestions.append("현재 LANEIGE 수치를 분석해주세요")
        suggestions.append("경쟁사와 비교하면 어떤가요?")
        suggestions.append("개선을 위한 액션 아이템은?")

    elif query_type == QueryType.DATA_QUERY:
        # 데이터 조회 → 심화 분석 제안
        suggestions.append("이 수치가 좋은 건가요?")
        suggestions.append("최근 7일 추이를 알려주세요")
        suggestions.append("경쟁사 대비 어떤가요?")

    elif query_type == QueryType.ANALYSIS:
        # 분석 → 전략/액션 제안
        suggestions.append("가장 시급한 액션은 무엇인가요?")
        suggestions.append("Top 10 진입을 위한 전략은?")
        suggestions.append("리스크 요인이 있나요?")

    elif query_type == QueryType.COMBINATION:
        # 조합 질문 → 다른 조합 제안
        suggestions.append("다른 시나리오도 분석해주세요")
        suggestions.append("현재 데이터에서 해당 상황이 있나요?")

    else:
        # 기본 제안
        suggestions = [
            "SoS(점유율)에 대해 알려주세요",
            "현재 LANEIGE 순위는?",
            "전략적 권고사항이 있나요?"
        ]

    return suggestions[:3]


# ============= API Endpoints =============

@app.get("/")
async def root():
    """헬스 체크"""
    return {
        "status": "ok",
        "message": "AMORE Dashboard API v2.0 (RAG + Ontology)",
        "features": ["chatbot", "rag", "ontology", "memory", "docx_export"]
    }


@app.get("/api/data")
async def get_data():
    """대시보드 데이터 조회"""
    data = load_dashboard_data()
    if not data:
        raise HTTPException(status_code=404, detail="Dashboard data not found")
    return data


@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit("30/minute")  # 분당 30회 제한
async def chat(request: ChatRequest, req: Request):
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

    message = request.message.strip()
    session_id = request.session_id or "default"

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
            query_type=query_type.value if hasattr(query_type, 'value') else str(query_type),
            confidence=confidence,
            sources=[],
            suggestions=["예, 전체 브랜드 분석해주세요", "LANEIGE만 분석해주세요", "Lip Care 카테고리만"],
            entities=entities
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
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        answer = response.choices[0].message.content

        # 9. 대화 메모리에 저장
        add_to_memory(session_id, "user", message)
        add_to_memory(session_id, "assistant", answer)

        # 10. 동적 후속 질문 제안
        suggestions = get_dynamic_suggestions(query_type, entities, answer)

        # 11. Audit Trail 로깅
        response_time_ms = (time.time() - start_time) * 1000
        log_chat_interaction(
            session_id=session_id,
            user_query=message,
            ai_response=answer,
            query_type=query_type.value if hasattr(query_type, 'value') else str(query_type),
            confidence=confidence,
            entities=entities,
            sources=sources,
            response_time_ms=response_time_ms
        )

        return ChatResponse(
            response=answer,
            query_type=query_type.value if hasattr(query_type, 'value') else str(query_type),
            confidence=confidence,
            sources=sources,
            suggestions=suggestions,
            entities=entities
        )

    except Exception as e:
        print(f"LLM Error: {e}")

        # Fallback 응답
        fallback = route_result.get("fallback_message") or rag_router.get_fallback_response("unknown")

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
            query_type=query_type.value if hasattr(query_type, 'value') else str(query_type),
            confidence=0.0,
            entities=entities,
            sources=["fallback"],
            response_time_ms=response_time_ms
        )

        return ChatResponse(
            response=fallback,
            query_type=query_type.value if hasattr(query_type, 'value') else str(query_type),
            confidence=0.0,
            sources=[],
            suggestions=["다시 질문해주세요", "SoS가 뭔가요?", "현재 순위 알려주세요"],
            entities=entities
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
    session_id: Optional[str] = "default"


class SimpleChatResponse(BaseModel):
    """Simple Chat 응답"""
    text: str
    suggestions: List[str]
    tools_used: List[str]
    sources: List[Dict[str, Any]] = []  # AI 출처 정보 추가
    data_date: str
    processing_time_ms: float


@app.post("/api/v3/chat", response_model=SimpleChatResponse)
@limiter.limit("30/minute")  # 분당 30회 제한
async def chat_v3(request: SimpleChatRequest, req: Request):
    """
    Simple LLM Chat API (v3)

    단순화된 구조:
    - LLM이 모든 판단 담당
    - Function Calling으로 도구 사용
    - 불필요한 레이어 제거
    """
    message = request.message.strip()
    session_id = request.session_id or "default"

    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    # 크롤링 상태 체크
    crawl_manager = get_crawl_manager()
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
        processing_time_ms=result.get("processing_time_ms", 0)
    )


# ============= LLM Orchestrator API (v2 - 기존, deprecated) =============

class OrchestratorChatRequest(BaseModel):
    """LLM Orchestrator 챗봇 요청"""
    message: str
    session_id: Optional[str] = "default"
    skip_cache: bool = False


class OrchestratorChatResponse(BaseModel):
    """LLM Orchestrator 챗봇 응답"""
    text: str
    query_type: str
    confidence_level: str
    confidence_score: float
    sources: List[str]
    entities: Dict[str, Any]
    tools_called: List[str]
    suggestions: List[str]
    is_fallback: bool
    is_clarification: bool
    processing_time_ms: float


@app.post("/api/v2/chat", response_model=OrchestratorChatResponse)
@limiter.limit("30/minute")  # 분당 30회 제한
async def chat_v2(request: OrchestratorChatRequest, req: Request):
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

    message = request.message.strip()
    session_id = request.session_id or "default"

    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    # === 크롤링 상태 체크 ===
    crawl_manager = get_crawl_manager()
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
        # 통합 오케스트레이터로 처리
        orchestrator = get_unified_orchestrator()

        # 현재 메트릭 데이터 로드
        data = load_dashboard_data()
        current_metrics = data if data else None

        # 처리
        response = await orchestrator.process(
            query=message,
            session_id=session_id,
            current_metrics=current_metrics,
            skip_cache=request.skip_cache
        )

        # 응답 텍스트 구성
        response_text = response.text

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
            query_type=response.query_type,
            confidence_level=response.confidence_level.value,
            confidence_score=response.confidence_score,
            sources=response.sources,
            entities=response.entities,
            tools_called=response.tools_called,
            suggestions=response.suggestions,
            is_fallback=response.is_fallback,
            is_clarification=response.is_clarification,
            processing_time_ms=response.processing_time_ms
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
            processing_time_ms=(time.time() - start_time) * 1000
        )


@app.get("/api/v2/stats")
async def get_orchestrator_stats():
    """통합 오케스트레이터 통계 조회"""
    orchestrator = get_unified_orchestrator()
    return orchestrator.get_stats()


@app.get("/api/v2/state")
async def get_orchestrator_state():
    """통합 오케스트레이터 상태 조회"""
    orchestrator = get_unified_orchestrator()
    return {
        "summary": orchestrator.get_state_summary(),
        "state": orchestrator.state.to_dict()
    }


@app.get("/api/v2/errors")
async def get_orchestrator_errors():
    """통합 오케스트레이터 최근 에러 조회"""
    orchestrator = get_unified_orchestrator()
    return {
        "recent_errors": orchestrator.get_recent_errors(limit=20),
        "stats": orchestrator.get_stats()
    }


@app.post("/api/v2/reset-errors")
async def reset_orchestrator_errors():
    """실패한 에이전트 목록 초기화"""
    orchestrator = get_unified_orchestrator()
    orchestrator.reset_failed_agents()
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
    crawl_manager = get_crawl_manager()
    return {
        **crawl_manager.state.to_dict(),
        "data_date": crawl_manager.get_data_date(),
        "needs_crawl": crawl_manager.needs_crawl(),
        "is_today_available": crawl_manager.is_today_data_available(),
        "status_message": crawl_manager.get_status_message()
    }


@app.post("/api/crawl/start", dependencies=[Depends(verify_api_key)])
async def start_crawl():
    """
    수동으로 크롤링 시작 (API Key 필요)

    Returns:
        - started: 크롤링 시작 여부
        - message: 상태 메시지
    """
    crawl_manager = get_crawl_manager()

    if crawl_manager.is_crawling():
        return {
            "started": False,
            "message": "크롤링이 이미 진행 중입니다.",
            "status": crawl_manager.state.to_dict()
        }

    if crawl_manager.is_today_data_available():
        return {
            "started": False,
            "message": "오늘 데이터가 이미 존재합니다.",
            "status": crawl_manager.state.to_dict()
        }

    started = await crawl_manager.start_crawl()
    return {
        "started": started,
        "message": "크롤링을 시작했습니다." if started else "크롤링 시작 실패",
        "status": crawl_manager.state.to_dict()
    }


# ============= Historical Data API =============

from src.tools.sheets_writer import SheetsWriter
from datetime import timedelta

# SheetsWriter 싱글톤 인스턴스
_sheets_writer: Optional[SheetsWriter] = None


def get_sheets_writer() -> SheetsWriter:
    """SheetsWriter 싱글톤 인스턴스 반환"""
    global _sheets_writer
    if _sheets_writer is None:
        _sheets_writer = SheetsWriter()
    return _sheets_writer


@app.get("/api/historical")
async def get_historical_data(
    start_date: str,
    end_date: str,
    category_id: Optional[str] = None,
    brand: Optional[str] = "LANEIGE"
):
    """
    히스토리컬 데이터 조회 (Google Sheets에서)

    Args:
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (YYYY-MM-DD)
        category_id: 카테고리 필터 (선택)
        brand: 브랜드 필터 (기본값: LANEIGE)

    Returns:
        - data: 날짜별 지표 데이터
        - sos_history: SoS 추이 데이터
        - rank_history: 순위 추이 데이터
    """
    try:
        sheets_writer = get_sheets_writer()
        if not sheets_writer._initialized:
            await sheets_writer.initialize()

        # 날짜 범위 계산
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end_dt - start_dt).days + 1

        # Google Sheets에서 히스토리컬 데이터 조회
        records = await sheets_writer.get_rank_history(
            category_id=category_id,
            brand=brand,
            days=days
        )

        if not records:
            # Google Sheets에 데이터가 없으면 로컬 JSON 파일에서 시도
            return await _get_historical_from_local(start_date, end_date, brand)

        # 날짜별 데이터 집계
        daily_data = {}
        for record in records:
            snapshot_date = record.get("snapshot_date", "")
            if not snapshot_date or snapshot_date < start_date or snapshot_date > end_date:
                continue

            if snapshot_date not in daily_data:
                daily_data[snapshot_date] = {
                    "date": snapshot_date,
                    "products": [],
                    "total_count": 0,
                    "top10_count": 0
                }

            rank = int(record.get("rank", 0)) if record.get("rank") else 0
            daily_data[snapshot_date]["products"].append({
                "asin": record.get("asin", ""),
                "product_name": record.get("product_name", ""),
                "rank": rank,
                "price": record.get("price", ""),
                "rating": record.get("rating", "")
            })
            daily_data[snapshot_date]["total_count"] += 1
            if rank <= 10:
                daily_data[snapshot_date]["top10_count"] += 1

        # SoS 추이 계산 (Top 100 기준)
        sos_history = []
        rank_history = []
        for date_str in sorted(daily_data.keys()):
            day_data = daily_data[date_str]
            products = day_data["products"]

            # SoS = (LANEIGE 제품 수 / 100) * 100
            sos = round(len(products) / 100 * 100, 1) if products else 0
            sos_history.append({
                "date": date_str,
                "sos": sos,
                "product_count": len(products),
                "top10_count": day_data["top10_count"]
            })

            # 평균 순위 (있는 경우)
            if products:
                avg_rank = round(sum(p["rank"] for p in products) / len(products), 1)
                rank_history.append({
                    "date": date_str,
                    "rank": avg_rank,
                    "best_rank": min(p["rank"] for p in products),
                    "worst_rank": max(p["rank"] for p in products)
                })

        # available_dates 계산
        available_dates = sorted(daily_data.keys())

        # brand_metrics 계산 (전체 기간 통합 - 모든 브랜드 포함)
        brand_metrics = await _calculate_brand_metrics_for_period(records, daily_data, brand)

        return {
            "success": True,
            "available_dates": available_dates,
            "brand_metrics": brand_metrics,
            "data": {
                "sos_history": sos_history,
                "rank_history": rank_history,
                "daily_data": list(daily_data.values()),
                "period": {
                    "start": start_date,
                    "end": end_date,
                    "days": days
                },
                "brand": brand
            }
        }

    except Exception as e:
        logging.error(f"Historical data error: {e}")
        # 폴백: 로컬 데이터에서 시도
        return await _get_historical_from_local(start_date, end_date, brand)


async def _calculate_brand_metrics_for_period(
    records: List[Dict],
    daily_data: Dict,
    target_brand: str
) -> List[Dict]:
    """
    기간 내 모든 브랜드의 메트릭 계산 (SoS × Avg Rank 차트용)

    Returns:
        브랜드별 SoS, 평균 순위, 제품 수 등
    """
    # 전체 제품 데이터 집계 (모든 브랜드)
    brand_data = {}

    for record in records:
        brand_name = record.get("brand", "Unknown")
        rank = int(record.get("rank", 0)) if record.get("rank") else 0

        if not brand_name or rank == 0:
            continue

        if brand_name not in brand_data:
            brand_data[brand_name] = {
                "brand": brand_name,
                "ranks": [],
                "product_count": 0
            }

        brand_data[brand_name]["ranks"].append(rank)
        brand_data[brand_name]["product_count"] += 1

    # 총 제품 수 (모든 브랜드)
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

        brand_metrics.append({
            "brand": brand_name,
            "sos": sos,
            "avg_rank": avg_rank,
            "product_count": data["product_count"],
            "bubble_size": bubble_size,
            "is_laneige": target_brand.upper() in brand_name.upper()
        })

    # SoS 기준 내림차순 정렬, 상위 10개만
    brand_metrics.sort(key=lambda x: x["sos"], reverse=True)
    return brand_metrics[:10]


def _get_brand_metrics_from_dashboard(
    dashboard_data: Optional[Dict],
    target_brand: str
) -> List[Dict]:
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
        brand_metrics.append({
            "brand": comp.get("brand", "Unknown"),
            "sos": comp.get("sos", 0),
            "avg_rank": comp.get("avg_rank", 50),
            "product_count": comp.get("product_count", 0),
            "bubble_size": max(5, min(25, comp.get("product_count", 0) * 2)),
            "is_laneige": target_brand.upper() in comp.get("brand", "").upper()
        })

    return brand_metrics


async def _get_historical_from_local(
    start_date: str,
    end_date: str,
    brand: str = "LANEIGE"
) -> Dict[str, Any]:
    """
    로컬 JSON 파일에서 히스토리컬 데이터 조회 (폴백)

    data/ 폴더의 날짜별 JSON 파일이나 dashboard_data.json의 히스토리 데이터 활용
    """
    try:
        # 메인 대시보드 데이터 로드
        data = load_dashboard_data()
        sos_history = []
        rank_history = []

        # 1. 대시보드 데이터에서 현재 SoS/순위 정보 추출
        if data:
            brand_kpis = data.get("brand", {}).get("kpis", {})
            current_sos = brand_kpis.get("sos", 0)
            data_date = data.get("metadata", {}).get("data_date", datetime.now().strftime("%Y-%m-%d"))

            # 현재 날짜가 요청 범위에 포함되면 추가
            if start_date <= data_date <= end_date:
                sos_history.append({
                    "date": data_date,
                    "sos": current_sos,
                    "product_count": brand_kpis.get("product_count", 0),
                    "top10_count": brand_kpis.get("top10_count", 0)
                })

                avg_rank = brand_kpis.get("avg_rank", 0)
                if avg_rank:
                    rank_history.append({
                        "date": data_date,
                        "rank": avg_rank,
                        "best_rank": brand_kpis.get("best_rank", avg_rank),
                        "worst_rank": brand_kpis.get("worst_rank", avg_rank)
                    })

        # 2. latest_crawl_result.json에서 데이터 추출
        latest_crawl_path = Path("./data/latest_crawl_result.json")
        if latest_crawl_path.exists():
            try:
                with open(latest_crawl_path, "r", encoding="utf-8") as f:
                    crawl_data = json.load(f)

                # 모든 카테고리에서 브랜드 제품 찾기
                brand_products = []
                crawl_date = None

                for cat_id, cat_data in crawl_data.get("categories", {}).items():
                    for product in cat_data.get("products", []):
                        product_brand = product.get("brand", "")
                        product_name = product.get("product_name", "")

                        # 브랜드 매칭 (대소문자 무시, 부분 매칭)
                        if brand.upper() in product_brand.upper() or brand.upper() in product_name.upper():
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
                        avg_rank = round(sum(p.get("rank", 0) for p in brand_products) / len(brand_products), 1)

                        sos_history.append({
                            "date": crawl_date,
                            "sos": sos,
                            "product_count": len(brand_products),
                            "top10_count": sum(1 for p in brand_products if p.get("rank", 100) <= 10)
                        })
                        rank_history.append({
                            "date": crawl_date,
                            "rank": avg_rank,
                            "best_rank": min(p.get("rank", 100) for p in brand_products),
                            "worst_rank": max(p.get("rank", 100) for p in brand_products)
                        })

            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"Failed to parse latest_crawl_result.json: {e}")

        # 3. raw_products 폴더에서 날짜별 데이터 검색 (기존 로직)
        raw_data_dir = Path("./data/raw_products")
        if raw_data_dir.exists():
            for json_file in raw_data_dir.glob("*.json"):
                try:
                    file_date = json_file.stem  # 파일명이 YYYY-MM-DD 형식이라고 가정
                    if start_date <= file_date <= end_date:
                        with open(json_file, "r", encoding="utf-8") as f:
                            daily_raw = json.load(f)

                        # 브랜드 제품만 필터링
                        brand_products = [
                            p for p in daily_raw
                            if brand.upper() in p.get("brand", "").upper() or brand.upper() in p.get("product_name", "").upper()
                        ]

                        if brand_products:
                            sos = round(len(brand_products) / 100 * 100, 1)
                            avg_rank = round(sum(p.get("rank", 0) for p in brand_products) / len(brand_products), 1)

                            # 중복 제거
                            if not any(h["date"] == file_date for h in sos_history):
                                sos_history.append({
                                    "date": file_date,
                                    "sos": sos,
                                    "product_count": len(brand_products),
                                    "top10_count": sum(1 for p in brand_products if p.get("rank", 100) <= 10)
                                })
                                rank_history.append({
                                    "date": file_date,
                                    "rank": avg_rank,
                                    "best_rank": min(p.get("rank", 100) for p in brand_products),
                                    "worst_rank": max(p.get("rank", 100) for p in brand_products)
                                })
                except (json.JSONDecodeError, ValueError):
                    continue

        # 날짜순 정렬
        sos_history.sort(key=lambda x: x["date"])
        rank_history.sort(key=lambda x: x["date"])

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
                "data": None
            }

        return {
            "success": True,
            "available_dates": available_dates,
            "brand_metrics": brand_metrics,
            "data": {
                "sos_history": sos_history,
                "rank_history": rank_history,
                "period": {
                    "start": start_date,
                    "end": end_date
                },
                "brand": brand,
                "source": "local"
            }
        }

    except Exception as e:
        logging.error(f"Local historical data error: {e}")
        return {
            "success": False,
            "error": str(e),
            "available_dates": [],
            "brand_metrics": [],
            "data": None
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
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Arial'
    font.size = Pt(11)

    # ===== 표지 =====
    title = doc.add_heading('AMORE INSIGHT Report', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph('LANEIGE Amazon US 분석 리포트')
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 날짜
    metadata = data.get("metadata", {})
    date_para = doc.add_paragraph()
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_para.add_run(f"분석 기준일: {metadata.get('data_date', datetime.now().strftime('%Y-%m-%d'))}")
    date_para.add_run(f"\n생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    doc.add_page_break()

    # ===== 1. Executive Summary =====
    doc.add_heading('1. Executive Summary', level=1)

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
    doc.add_heading('2. LANEIGE 제품 현황', level=1)

    products = data.get("products", {})
    if products:
        # 테이블 생성
        table = doc.add_table(rows=1, cols=5)
        table.style = 'Table Grid'
        table.alignment = WD_TABLE_ALIGNMENT.CENTER

        # 헤더
        header_cells = table.rows[0].cells
        headers = ['제품명', '순위', '변동', '평점', '변동성']
        for i, header in enumerate(headers):
            header_cells[i].text = header
            header_cells[i].paragraphs[0].runs[0].bold = True

        # 데이터 행
        for asin, product in products.items():
            row = table.add_row().cells
            row[0].text = product.get('name', '')[:40]
            row[1].text = f"#{product.get('rank', 'N/A')}"
            row[2].text = product.get('rank_delta', '-')
            row[3].text = str(product.get('rating', '-'))
            row[4].text = product.get('volatility_status', '-')

    doc.add_paragraph()

    # ===== 3. 경쟁사 분석 =====
    doc.add_heading('3. 경쟁사 분석', level=1)

    competitors = data.get("brand", {}).get("competitors", [])
    if competitors:
        table = doc.add_table(rows=1, cols=4)
        table.style = 'Table Grid'

        header_cells = table.rows[0].cells
        headers = ['브랜드', 'SoS (%)', '평균 순위', '제품 수']
        for i, header in enumerate(headers):
            header_cells[i].text = header
            header_cells[i].paragraphs[0].runs[0].bold = True

        for comp in competitors[:10]:
            row = table.add_row().cells
            row[0].text = comp.get('brand', '')
            row[1].text = str(comp.get('sos', 0))
            row[2].text = str(comp.get('avg_rank', '-'))
            row[3].text = str(comp.get('product_count', 0))

    doc.add_paragraph()

    # ===== 4. 액션 아이템 =====
    doc.add_heading('4. 액션 아이템', level=1)

    action_items = data.get("home", {}).get("action_items", [])
    if action_items:
        for item in action_items:
            priority_marker = "🔴" if item.get('priority') == 'P1' else "🟠"
            para = doc.add_paragraph()
            para.add_run(f"{priority_marker} [{item.get('priority')}] ").bold = True
            para.add_run(f"{item.get('product_name', '')}\n")
            para.add_run(f"   신호: {item.get('signal', '')}\n")
            para.add_run(f"   권장 액션: {item.get('action_tag', '')}")
    else:
        doc.add_paragraph("현재 특별한 액션 아이템이 없습니다.")

    # ===== 5. 전략적 권고사항 =====
    if request.include_strategy:
        doc.add_heading('5. 전략적 권고사항', level=1)

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
                    {"role": "system", "content": "당신은 뷰티 이커머스 전문 컨설턴트입니다. 간결하고 실행 가능한 전략을 제안합니다."},
                    {"role": "user", "content": strategy_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            strategy_text = response.choices[0].message.content
            doc.add_paragraph(strategy_text)

        except Exception as e:
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
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ============= Alert Settings API =============

from src.core.state_manager import StateManager, get_state_manager

# 싱글톤 State Manager
_state_manager: Optional[StateManager] = None

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
    alert_types: List[str] = []


class AlertSettingsResponse(BaseModel):
    """알림 설정 응답"""
    email: str
    consent: bool
    alert_types: List[str]
    consent_date: Optional[str] = None


@app.get("/api/v3/alert-settings")
async def get_alert_settings():
    """
    현재 알림 설정 조회

    참고: 현재는 단일 사용자 설정만 지원 (첫 번째 등록된 이메일)
    """
    state_manager = get_app_state_manager()
    subscriptions = state_manager.get_all_subscriptions()

    if not subscriptions:
        return {
            "email": "",
            "consent": False,
            "alert_types": [],
            "consent_date": None
        }

    # 첫 번째 구독 반환
    email, sub = next(iter(subscriptions.items()))
    return {
        "email": email,
        "consent": sub.consent,
        "alert_types": sub.alert_types,
        "consent_date": sub.consent_date.isoformat() if sub.consent_date else None
    }


@app.post("/api/v3/alert-settings", dependencies=[Depends(verify_api_key)])
async def save_alert_settings(request: AlertSettingsRequest):
    """
    알림 설정 저장 (API Key 필요)

    중요: consent가 True일 때만 이메일 등록
    """
    state_manager = get_app_state_manager()

    if not request.email:
        raise HTTPException(status_code=400, detail="이메일 주소가 필요합니다.")

    if request.consent:
        # 이메일 등록 (명시적 동의)
        success = state_manager.register_email(
            email=request.email,
            consent=True,
            alert_types=request.alert_types
        )

        if not success:
            raise HTTPException(status_code=400, detail="이메일 등록 실패")

        return {"status": "ok", "message": "알림 설정이 저장되었습니다."}
    else:
        # 동의 없으면 업데이트만 (알림 유형 변경)
        success = state_manager.update_email_subscription(
            email=request.email,
            alert_types=request.alert_types
        )

        return {"status": "ok", "message": "설정이 업데이트되었습니다."}


@app.post("/api/v3/alert-settings/revoke", dependencies=[Depends(verify_api_key)])
async def revoke_alert_consent():
    """
    알림 동의 철회 (API Key 필요)

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
async def get_alerts(limit: int = 50, alert_type: Optional[str] = None):
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
        "stats": alert_agent.get_stats()
    }


# ============= 대시보드 HTML 서빙 =============

@app.get("/dashboard")
async def serve_dashboard():
    """대시보드 HTML 페이지 서빙"""
    dashboard_path = Path("./dashboard/amore_unified_dashboard_v4.html")
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(dashboard_path, media_type="text/html")


@app.get("/api/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ============= Level 4 Brain API (v4) =============

class BrainChatRequest(BaseModel):
    """Brain 챗봇 요청"""
    message: str
    session_id: Optional[str] = "default"
    skip_cache: bool = False


class BrainChatResponse(BaseModel):
    """Brain 챗봇 응답"""
    text: str
    confidence: float
    sources: List[str]
    reasoning: Optional[str] = None
    tools_used: List[str]
    processing_time_ms: float
    from_cache: bool
    brain_mode: str


@app.post("/api/v4/chat", response_model=BrainChatResponse)
@limiter.limit("30/minute")  # 분당 30회 제한
async def chat_v4(request: BrainChatRequest, req: Request):
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

    message = request.message.strip()
    session_id = request.session_id or "default"

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
            skip_cache=request.skip_cache
        )

        processing_time = (time.time() - start_time) * 1000

        return BrainChatResponse(
            text=response.content,
            confidence=response.confidence,
            sources=response.sources,
            reasoning=response.reasoning,
            tools_used=response.tools_called if hasattr(response, 'tools_called') else [],
            processing_time_ms=processing_time,
            from_cache=response.from_cache if hasattr(response, 'from_cache') else False,
            brain_mode=brain.mode.value
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
            brain_mode="error"
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
            "initialized": True
        }
    except Exception as e:
        return {
            "mode": "uninitialized",
            "scheduler_running": False,
            "pending_tasks": 0,
            "stats": {},
            "initialized": False,
            "error": str(e)
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
                "status": "running"
            }

        await brain.start_scheduler()

        return {
            "started": True,
            "message": "자율 스케줄러가 시작되었습니다.",
            "status": "running"
        }
    except Exception as e:
        return {
            "started": False,
            "message": f"스케줄러 시작 실패: {str(e)}",
            "status": "error"
        }


@app.post("/api/v4/brain/scheduler/stop", dependencies=[Depends(verify_api_key)])
async def stop_brain_scheduler():
    """자율 스케줄러 중지 (API Key 필요)"""
    try:
        brain = await get_initialized_brain()

        if brain.scheduler:
            brain.scheduler.stop()

        return {
            "stopped": True,
            "message": "스케줄러가 중지되었습니다.",
            "status": "stopped"
        }
    except Exception as e:
        return {
            "stopped": False,
            "message": f"스케줄러 중지 실패: {str(e)}",
            "status": "error"
        }


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

        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


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
            return {
                "alerts": [],
                "message": "데이터가 없습니다."
            }

        alerts = await brain.check_alerts(data)

        return {
            "alerts": alerts,
            "count": len(alerts),
            "checked_at": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "alerts": [],
            "error": str(e)
        }


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
            "autonomous": BrainMode.AUTONOMOUS
        }

        if mode not in mode_map:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode. Valid modes: {list(mode_map.keys())}"
            )

        brain.mode = mode_map[mode]

        return {
            "mode": brain.mode.value,
            "message": f"Brain 모드가 {mode}(으)로 변경되었습니다."
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}


# ============= 서버 실행 =============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

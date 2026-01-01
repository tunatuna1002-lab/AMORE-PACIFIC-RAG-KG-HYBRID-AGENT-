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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from litellm import acompletion

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

# RAG 시스템 연동
from rag.router import RAGRouter, QueryType
from rag.retriever import DocumentRetriever

# Ontology 스키마
from ontology.schema import ProductMetrics, BrandMetrics, MarketMetrics

# 통합 오케스트레이터 (신규)
from core.unified_orchestrator import UnifiedOrchestrator, get_unified_orchestrator
from core.crawl_manager import get_crawl_manager, CrawlStatus

# 환경 변수 로드
load_dotenv()

app = FastAPI(
    title="AMORE Dashboard API",
    description="LANEIGE Amazon 대시보드 백엔드 API (RAG + Ontology 통합)",
    version="2.0.0"
)

# CORS 설정 (로컬 개발용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 경로
DATA_PATH = "./data/dashboard_data.json"
DOCS_PATH = "./"  # MD 파일들이 루트에 있음
AUDIT_LOG_DIR = "./logs"

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

# 세션별 대화 메모리 (간단한 인메모리 구현)
conversation_memory: Dict[str, List[Dict[str, str]]] = defaultdict(list)
MAX_MEMORY_TURNS = 10

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
    conversation_memory[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
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
async def chat(request: ChatRequest):
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

from core.simple_chat import get_chat_service


class SimpleChatRequest(BaseModel):
    """Simple Chat 요청"""
    message: str
    session_id: Optional[str] = "default"


class SimpleChatResponse(BaseModel):
    """Simple Chat 응답"""
    text: str
    suggestions: List[str]
    tools_used: List[str]
    data_date: str
    processing_time_ms: float


@app.post("/api/v3/chat", response_model=SimpleChatResponse)
async def chat_v3(request: SimpleChatRequest):
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
async def chat_v2(request: OrchestratorChatRequest):
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


@app.post("/api/crawl/start")
async def start_crawl():
    """
    수동으로 크롤링 시작

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

from core.state_manager import StateManager, get_state_manager

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


@app.post("/api/v3/alert-settings")
async def save_alert_settings(request: AlertSettingsRequest):
    """
    알림 설정 저장

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


@app.post("/api/v3/alert-settings/revoke")
async def revoke_alert_consent():
    """
    알림 동의 철회

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
    from agents.alert_agent import AlertAgent

    state_manager = get_app_state_manager()
    alert_agent = AlertAgent(state_manager)

    return {
        "alerts": alert_agent.get_alerts(limit=limit, alert_type=alert_type),
        "pending_count": alert_agent.get_pending_count(),
        "stats": alert_agent.get_stats()
    }


# ============= 서버 실행 =============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

"""
Chat Routes
===========
챗봇 관련 API 엔드포인트 (v1, v2, v3, v4)
"""

import time
import logging

logger = logging.getLogger(__name__)
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from litellm import acompletion

# Dependencies
from src.api.dependencies import (
    limiter,
    conversation_memory,
    get_conversation_history,
    add_to_memory,
    log_chat_interaction,
    load_dashboard_data
)

# RAG & Router
from src.rag.router import RAGRouter, QueryType
from src.rag.retriever import DocumentRetriever

# Core Services
from src.core.simple_chat import get_chat_service
from src.core.unified_orchestrator import get_unified_orchestrator
from src.core.brain import get_initialized_brain
from src.core.crawl_manager import get_crawl_manager

# ============= Router Setup =============

router = APIRouter()

# RAG System
rag_router = RAGRouter()
DOCS_PATH = "./docs/guides"
doc_retriever = DocumentRetriever(DOCS_PATH)


# ============= Pydantic Models =============

class ChatRequest(BaseModel):
    """챗봇 요청 (v1)"""
    message: str
    session_id: Optional[str] = "default"
    context: Optional[Dict] = None


class ChatResponse(BaseModel):
    """챗봇 응답 (v1)"""
    response: str
    query_type: str
    confidence: float
    sources: List[str]
    suggestions: List[str]
    entities: Dict[str, Any]


class SimpleChatRequest(BaseModel):
    """Simple Chat 요청 (v3)"""
    message: str
    session_id: Optional[str] = "default"


class SimpleChatResponse(BaseModel):
    """Simple Chat 응답 (v3)"""
    text: str
    suggestions: List[str]
    tools_used: List[str]
    sources: List[Dict[str, Any]] = []
    data_date: str
    processing_time_ms: float


class OrchestratorChatRequest(BaseModel):
    """LLM Orchestrator 챗봇 요청 (v2)"""
    message: str
    session_id: Optional[str] = "default"
    skip_cache: bool = False


class OrchestratorChatResponse(BaseModel):
    """LLM Orchestrator 챗봇 응답 (v2)"""
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


class BrainChatRequest(BaseModel):
    """Brain 챗봇 요청 (v4)"""
    message: str
    session_id: Optional[str] = "default"
    skip_cache: bool = False


class BrainChatResponse(BaseModel):
    """Brain 챗봇 응답 (v4)"""
    text: str
    confidence: float
    sources: List[str]
    reasoning: Optional[str] = None
    tools_used: List[str]
    processing_time_ms: float
    from_cache: bool
    brain_mode: str


# ============= Helper Functions =============

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


# ============= Chat API v1 (RAG + Ontology) =============

@router.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
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
        logger.error(f"LLM Error: {e}")

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


@router.delete("/chat/memory/{session_id}")
async def clear_memory(session_id: str):
    """세션 대화 기록 초기화"""
    if session_id in conversation_memory:
        del conversation_memory[session_id]
    return {"status": "ok", "message": f"Session {session_id} memory cleared"}


# ============= Chat API v3 (Simple Chat) =============

@router.post("/v3/chat", response_model=SimpleChatResponse)
@limiter.limit("30/minute")
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
        sources=result.get("sources", []),
        data_date=result.get("data_date", "N/A"),
        processing_time_ms=result.get("processing_time_ms", 0)
    )


# ============= Chat API v2 (Orchestrator - deprecated) =============

@router.post("/v2/chat", response_model=OrchestratorChatResponse)
@limiter.limit("30/minute")
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
    start_time = time.time()

    message = request.message.strip()
    session_id = request.session_id or "default"

    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    # 크롤링 상태 체크
    crawl_manager = await get_crawl_manager()
    crawl_notification = None
    crawl_started = False

    if crawl_manager.needs_crawl():
        crawl_started = await crawl_manager.start_crawl()
        if crawl_started:
            logging.info("Background crawl started for today's data")

    if crawl_manager.should_notify(session_id):
        crawl_notification = crawl_manager.get_notification_message()
        crawl_manager.mark_notified(session_id)

    try:
        # 통합 오케스트레이터로 처리
        orchestrator = await get_unified_orchestrator()

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
            confidence_level="low",
            confidence_score=0.0,
            sources=[],
            entities={},
            tools_called=[],
            suggestions=["다시 시도해주세요"],
            is_fallback=True,
            is_clarification=False,
            processing_time_ms=(time.time() - start_time) * 1000
        )


# ============= Chat API v4 (Brain - LLM First) =============

@router.post("/v4/chat", response_model=BrainChatResponse)
@limiter.limit("30/minute")
async def chat_v4(request: BrainChatRequest, req: Request):
    """
    Level 4 Brain 기반 챗봇 API (v4)

    LLM-First 접근:
    - 모든 판단을 LLM이 수행
    - 규칙 기반 빠른 경로 없음
    - RAG + KG 하이브리드 검색
    - 자율 스케줄러와 통합
    """
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

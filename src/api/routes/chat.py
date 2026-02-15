"""
Chat API Routes
===============
챗봇 엔드포인트 (v1, v4, v4/stream)

## 엔드포인트
- POST /api/chat: v1 RAG + Ontology 챗봇
- DELETE /api/chat/memory/{session_id}: 세션 메모리 삭제
- POST /api/v4/chat: v4 Brain 챗봇
- POST /api/v4/chat/stream: v4 Brain SSE 스트리밍
"""

import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from litellm import acompletion

from src.api.dependencies import (
    add_to_memory,
    build_data_context,
    conversation_memory,
    generate_dynamic_suggestions,
    get_conversation_history,
    get_rag_context,
    limiter,
    load_dashboard_data,
    log_chat_interaction,
    rag_router,
    verify_api_key,
)
from src.api.models import BrainChatRequest, BrainChatResponse, ChatRequest, ChatResponse
from src.core.brain import get_initialized_brain
from src.rag.router import QueryType

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
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
{query_type.value if hasattr(query_type, "value") else str(query_type)} (신뢰도: {confidence:.1%})

## 추출된 엔티티
- 브랜드: {", ".join(entities.get("brands", [])) or "없음"}
- 카테고리: {", ".join(entities.get("categories", [])) or "없음"}
- 지표: {", ".join(entities.get("indicators", [])) or "없음"}
- 기간: {entities.get("time_range") or "없음"}

## RAG 참조 문서
{rag_context if rag_context else "관련 문서 없음"}

## 현재 데이터
{data_context}

## 이전 대화
{conversation_history if conversation_history else "이전 대화 없음"}

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
- SoS: {brand_kpis.get("sos", 0)}%
- Top 10 제품: {brand_kpis.get("top10_count", 0)}개
- 평균 순위: {brand_kpis.get("avg_rank", 0)}위

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


@router.delete("/api/chat/memory/{session_id}")
async def clear_memory(session_id: str):
    """세션 대화 기록 초기화"""
    if session_id in conversation_memory:
        del conversation_memory[session_id]
    return {"status": "ok", "message": f"Session {session_id} memory cleared"}


@router.post(
    "/api/v4/chat", response_model=BrainChatResponse, dependencies=[Depends(verify_api_key)]
)
@limiter.limit("10/minute")
async def chat_v4(request: Request, body: BrainChatRequest):
    """
    Level 4 Brain 기반 챗봇 API (v4)

    LLM-First 접근:
    - 모든 판단을 LLM이 수행
    - 규칙 기반 빠른 경로 없음
    - RAG + KG 하이브리드 검색
    - 자율 스케줄러와 통합
    """
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
            text=response.text,
            confidence=response.confidence_score,
            sources=response.sources if isinstance(response.sources, list) else [],
            reasoning=response.query_type,
            tools_used=response.tools_called,
            processing_time_ms=processing_time,
            from_cache=False,
            brain_mode=brain.mode.value,
            suggestions=response.suggestions,
            query_type=response.query_type,
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


@router.post("/api/v4/chat/stream")
@limiter.limit("10/minute")
async def chat_v4_stream(request: Request, body: BrainChatRequest):
    """
    Level 4 Brain 기반 SSE 스트리밍 챗봇 API (v4)

    v3의 SSE 스트리밍과 동일한 인터페이스로 v4 Brain의 처리 결과를 반환합니다.
    ReAct + OWL + PromptGuard + 도구 호출을 모두 지원합니다.

    이벤트 타입:
    - status: 처리 단계 알림
    - tool_call: 도구 호출 정보
    - text: 응답 텍스트
    - done: 완료 (메타데이터 포함)
    - error: 오류 발생
    """
    message = body.message.strip()
    session_id = body.session_id or "default"

    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    try:
        brain = await get_initialized_brain()
        data = load_dashboard_data()
        current_metrics = data if data else None

        async def generate():
            try:
                async for chunk in brain.process_query_stream(
                    query=message,
                    session_id=session_id,
                    current_metrics=current_metrics,
                ):
                    event_data = json.dumps(chunk, ensure_ascii=False)
                    yield f"data: {event_data}\n\n"
            except Exception as e:
                logger.error(f"v4 SSE stream error: {e}")
                error_data = json.dumps({"type": "error", "content": str(e)}, ensure_ascii=False)
                yield f"data: {error_data}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        logger.error(f"v4 stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

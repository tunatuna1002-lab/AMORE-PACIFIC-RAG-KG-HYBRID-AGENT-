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

from src.api.dependencies import (
    add_to_memory,
    conversation_memory,
    limiter,
    load_dashboard_data,
    log_chat_interaction,
    verify_api_key,
)
from src.api.models import BrainChatRequest, BrainChatResponse, ChatRequest, ChatResponse
from src.api.validators.input_validator import get_validator
from src.core.brain import get_initialized_brain
from src.domain.exceptions import DataValidationError
from src.infrastructure.container import Container

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest):
    """
    ChatGPT + RAG + Ontology 통합 챗봇 API (via ChatWorkflow)

    1. 입력 검증 (Prompt Injection 방어)
    2. ChatWorkflow 실행 (RAG + KG + LLM)
    3. Audit Trail 로깅
    """
    start_time = time.time()

    message = body.message.strip()
    session_id = body.session_id or "default"

    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    # 0. 입력 검증 (Prompt Injection 방어)
    try:
        _, message = get_validator().validate(message)
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # 1. 데이터 컨텍스트 로드
    data = load_dashboard_data()

    try:
        # 2. ChatWorkflow 실행 (Container에서 의존성 주입)
        workflow = Container.get_chat_workflow()
        result = await workflow.execute(
            query=message,
            session_id=session_id,
            current_metrics=data if data else None,
        )

        if result.error:
            raise RuntimeError(result.error)

        answer = result.response or ""
        sources = [s if isinstance(s, str) else str(s) for s in result.sources]
        confidence = result.confidence
        suggestions = result.suggestions
        entities = result.entities
        query_type_str = result.intent

        # 3. 대화 메모리에 저장
        add_to_memory(session_id, "user", message)
        add_to_memory(session_id, "assistant", answer)

        # 4. Audit Trail 로깅
        response_time_ms = (time.time() - start_time) * 1000
        log_chat_interaction(
            session_id=session_id,
            user_query=message,
            ai_response=answer,
            query_type=query_type_str,
            confidence=confidence,
            entities=entities,
            sources=sources,
            response_time_ms=response_time_ms,
        )

        return ChatResponse(
            response=answer,
            query_type=query_type_str,
            confidence=confidence,
            sources=sources,
            suggestions=suggestions,
            entities=entities,
        )

    except Exception as e:
        logger.error(f"ChatWorkflow Error: {e}")

        fallback = "질문을 처리하는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

        response_time_ms = (time.time() - start_time) * 1000
        log_chat_interaction(
            session_id=session_id,
            user_query=message,
            ai_response=f"[ERROR] {str(e)[:100]} | Fallback: {fallback[:200]}",
            query_type="unknown",
            confidence=0.0,
            entities={},
            sources=["fallback"],
            response_time_ms=response_time_ms,
        )

        return ChatResponse(
            response=fallback,
            query_type="unknown",
            confidence=0.0,
            sources=[],
            suggestions=["다시 질문해주세요", "SoS가 뭔가요?", "현재 순위 알려주세요"],
            entities={},
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

    # 입력 검증 (Prompt Injection 방어)
    try:
        _, message = get_validator().validate(message)
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

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


@router.post("/api/v4/chat/stream", dependencies=[Depends(verify_api_key)])
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

    # 입력 검증 (Prompt Injection 방어)
    try:
        _, message = get_validator().validate(message)
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

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

"""
Chat Routes
===========
챗봇 관련 API 엔드포인트 (v4)
"""

import logging
import time

logger = logging.getLogger(__name__)

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.api.dependencies import limiter, load_dashboard_data

# Input Validation
from src.api.validators.input_validator import get_validator

# Core Services
from src.core.brain import get_initialized_brain
from src.domain.exceptions import DataValidationError

# ============= Router Setup =============

router = APIRouter()


# ============= Pydantic Models =============


class BrainChatRequest(BaseModel):
    """Brain 챗봇 요청 (v4)"""

    message: str
    session_id: str | None = "default"
    skip_cache: bool = False


class BrainChatResponse(BaseModel):
    """Brain 챗봇 응답 (v4)"""

    text: str
    confidence: float
    sources: list[str]
    reasoning: str | None = None
    tools_used: list[str]
    processing_time_ms: float
    from_cache: bool
    brain_mode: str


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
    session_id = request.session_id or "default"

    # 입력 검증
    try:
        validator = get_validator()
        is_valid, message = validator.validate(request.message)
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=f"입력 검증 실패: {e}") from e

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
            skip_cache=request.skip_cache,
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

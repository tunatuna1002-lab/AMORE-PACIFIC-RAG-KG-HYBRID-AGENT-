"""
Query Processing State
======================
LangGraph-style TypedDict state for query processing pipeline.
프레임워크 의존 없이 동일 패턴의 경량 구현.
"""

from dataclasses import dataclass, field
from typing import Any

from .models import ConfidenceLevel, Context, Decision, Response, ToolResult


@dataclass
class QueryState:
    """
    쿼리 처리 상태

    각 노드가 상태를 읽고 수정하며, 조건부 엣지가 다음 노드를 결정합니다.

    Attributes:
        query: 원본 사용자 질문
        original_query: 최초 질문 (재작성 전 보존)
        session_id: 세션 ID
        current_metrics: 현재 지표 데이터
        skip_cache: 캐시 스킵 여부
        context: 수집된 컨텍스트
        confidence_level: 신뢰도 레벨
        decision: LLM 판단 결과
        tool_result: 도구 실행 결과
        response: 최종 응답
        system_state: 시스템 상태
        rewrite_count: 쿼리 재작성 횟수
        max_rewrites: 최대 재작성 횟수
        is_complex: 복잡한 질문 여부
        is_blocked: PromptGuard 차단 여부
        block_reason: 차단 사유
        error: 에러 메시지
        metadata: 추가 메타데이터 (트레이싱 등)
    """

    # Input
    query: str = ""
    original_query: str = ""
    session_id: str | None = None
    current_metrics: dict[str, Any] | None = None
    skip_cache: bool = False

    # Processing state
    context: Context | None = None
    confidence_level: ConfidenceLevel | None = None
    decision: Decision | None = None
    tool_result: ToolResult | None = None
    response: Response | None = None
    system_state: dict[str, Any] = field(default_factory=dict)

    # Control flow
    rewrite_count: int = 0
    max_rewrites: int = 2
    is_complex: bool = False
    is_blocked: bool = False
    block_reason: str | None = None
    error: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

"""
데이터 모델 정의
================
LLM Orchestrator에서 사용하는 핵심 데이터 구조

Clean Architecture: 이 모듈은 Domain 레이어의 모델을 상속/재export
- ConfidenceLevel, KGFact, SystemState → domain.entities.brain_models
- Context, ToolResult, Response → domain.entities.brain_models의 Base 클래스 상속

주요 모델:
- ConfidenceLevel: 신뢰도 레벨 (HIGH/MEDIUM/LOW/UNKNOWN)
- Context: RAG + KG 컨텍스트 통합 구조
- Decision: LLM 판단 결과
- ToolResult: 도구 실행 결과
- Response: 최종 응답
"""

from dataclasses import dataclass, field
from typing import Any

# Domain 모델 재export (Backward Compatibility)
from src.domain.entities.brain_models import (
    ConfidenceLevel,
    ContextBase,
    KGFact,
    ResponseBase,
    SystemState,
    ToolResultBase,
)

# Re-export for backward compatibility
__all__ = [
    "ConfidenceLevel",
    "KGFact",
    "SystemState",
    "Context",
    "Decision",
    "ToolResult",
    "Response",
]


# =============================================================================
# 컨텍스트 모델 (Domain 상속)
# =============================================================================


@dataclass
class Context(ContextBase):
    """
    RAG + KG 통합 컨텍스트

    LLM 판단 및 응답 생성에 사용되는 모든 컨텍스트 정보를 담는다.
    Domain 레이어의 ContextBase를 상속하여 추가 기능 제공.

    Attributes:
        query: 원본 사용자 질문
        entities: 추출된 엔티티 (brands, categories, indicators 등)
        rag_docs: RAG 검색 결과 문서들
        kg_facts: KG에서 조회한 사실들
        kg_inferences: KG 추론 결과들
        system_state: 시스템 상태
        summary: 컨텍스트 요약 문자열 (LLM 프롬프트용)
        gathered_at: 수집 시간

    Usage:
        context = await context_gatherer.gather(query, entities)
        response = await response_pipeline.generate(query, context)
    """

    pass  # ContextBase의 모든 기능 상속


# =============================================================================
# LLM 판단 결과
# =============================================================================


@dataclass
class Decision:
    """
    LLM 판단 결과

    Attributes:
        tool: 사용할 도구 (None이면 direct_answer)
        tool_params: 도구 파라미터
        reason: 판단 이유
        key_points: 응답에 포함할 핵심 포인트
        confidence: 판단 신뢰도
    """

    tool: str | None = None  # None = direct_answer
    tool_params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    key_points: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def requires_tool(self) -> bool:
        """도구 실행이 필요한지 확인"""
        return self.tool is not None and self.tool != "direct_answer"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "tool_params": self.tool_params,
            "reason": self.reason,
            "key_points": self.key_points,
            "confidence": self.confidence,
        }


# =============================================================================
# 도구 실행 결과 (Domain 상속)
# =============================================================================


@dataclass
class ToolResult(ToolResultBase):
    """
    도구 실행 결과

    Domain 레이어의 ToolResultBase를 상속하여 추가 기능 제공.

    Attributes:
        tool_name: 실행된 도구명
        success: 성공 여부
        data: 결과 데이터
        error: 에러 메시지 (실패 시)
        execution_time_ms: 실행 시간 (밀리초)
    """

    def to_summary(self) -> str:
        """응답에 포함할 요약 생성"""
        if not self.success:
            return f"[{self.tool_name}] 실행 실패: {self.error}"

        # 도구별 요약 생성
        if self.tool_name == "crawl_amazon":
            total = self.data.get("total_products", 0)
            laneige = self.data.get("laneige_count", 0)
            return f"크롤링 완료: {total}개 제품 수집, LANEIGE {laneige}개"

        elif self.tool_name == "calculate_metrics":
            brands = len(self.data.get("brand_metrics", []))
            products = len(self.data.get("product_metrics", []))
            return f"지표 계산 완료: {brands}개 브랜드, {products}개 제품"

        elif self.tool_name == "query_data":
            return "데이터 조회 완료"

        else:
            return f"[{self.tool_name}] 실행 완료"


# =============================================================================
# 최종 응답 (Domain 상속)
# =============================================================================


@dataclass
class Response(ResponseBase):
    """
    최종 응답

    Domain 레이어의 ResponseBase를 상속하여 추가 기능 제공.

    Attributes:
        text: 응답 텍스트
        query_type: 질문 유형
        confidence_level: 신뢰도 레벨
        confidence_score: 신뢰도 점수
        sources: 참조 출처
        entities: 추출된 엔티티
        tools_called: 호출된 도구 목록
        suggestions: 후속 질문 제안
        is_fallback: Fallback 응답 여부
        is_clarification: 명확화 요청 여부
        processing_time_ms: 처리 시간
        created_at: 생성 시간
    """

    @classmethod
    def clarification(cls, message: str, suggestions: list[str] = None) -> "Response":
        """명확화 요청 응답 생성 팩토리 메서드"""
        return cls(
            text=message,
            query_type="clarification",
            confidence_level=ConfidenceLevel.UNKNOWN,
            is_clarification=True,
            suggestions=suggestions or [],
        )

    @classmethod
    def fallback(cls, message: str) -> "Response":
        """Fallback 응답 생성 팩토리 메서드"""
        return cls(
            text=message,
            query_type="fallback",
            confidence_level=ConfidenceLevel.UNKNOWN,
            is_fallback=True,
            suggestions=["다시 질문해주세요", "SoS가 뭔가요?", "현재 순위 알려주세요"],
        )

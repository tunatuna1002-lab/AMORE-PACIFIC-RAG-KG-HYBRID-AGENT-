"""
Brain 도메인 모델
=================
LLM Orchestrator에서 사용하는 순수 도메인 모델

Clean Architecture Layer 1: 외부 의존성 없음

주요 모델:
- ConfidenceLevel: 신뢰도 레벨 (HIGH/MEDIUM/LOW/UNKNOWN)
- ContextBase: RAG + KG 컨텍스트 기본 구조 (Protocol)
- ToolResultBase: 도구 실행 결과 기본 구조 (Protocol)
- ResponseBase: 최종 응답 기본 구조 (Protocol)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# =============================================================================
# 신뢰도 레벨
# =============================================================================


class ConfidenceLevel(Enum):
    """
    신뢰도 레벨 정의

    점수 기준:
    - HIGH: 5.0+ (Rule 결과 신뢰, LLM 판단 스킵)
    - MEDIUM: 3.0~4.9 (LLM에게 도구 선택 위임)
    - LOW: 1.5~2.9 (LLM에게 전체 판단 위임)
    - UNKNOWN: <1.5 (명확화 요청)
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


# =============================================================================
# KG Fact 모델
# =============================================================================


@dataclass
class KGFact:
    """
    Knowledge Graph에서 조회한 사실

    Attributes:
        fact_type: 사실 유형 (brand_info, competitors, category_brands 등)
        entity: 관련 엔티티
        data: 사실 데이터
    """

    fact_type: str
    entity: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.fact_type, "entity": self.entity, "data": self.data}


# =============================================================================
# 시스템 상태
# =============================================================================


@dataclass
class SystemState:
    """
    시스템 상태 스냅샷

    Attributes:
        last_crawl_time: 마지막 크롤링 시간
        data_freshness: 데이터 신선도 (fresh/stale/unknown)
        kg_triple_count: KG 트리플 수
        kg_initialized: KG 초기화 여부
    """

    last_crawl_time: datetime | None = None
    data_freshness: str = "unknown"
    kg_triple_count: int = 0
    kg_initialized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_crawl_time": (self.last_crawl_time.isoformat() if self.last_crawl_time else None),
            "data_freshness": self.data_freshness,
            "kg_triple_count": self.kg_triple_count,
            "kg_initialized": self.kg_initialized,
        }


# =============================================================================
# Context Protocol (Domain Layer Interface)
# =============================================================================


@runtime_checkable
class ContextProtocol(Protocol):
    """
    RAG + KG 통합 컨텍스트 인터페이스

    LLM 판단 및 응답 생성에 사용되는 컨텍스트의 필수 속성
    """

    query: str
    entities: dict[str, list[str]]
    rag_docs: list[dict[str, Any]]
    kg_facts: list[KGFact]
    kg_inferences: list[dict[str, Any]]
    system_state: SystemState | None
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 변환"""
        ...

    def has_sufficient_context(self) -> bool:
        """응답 생성에 충분한 컨텍스트가 있는지 확인"""
        ...


# =============================================================================
# ToolResult Protocol (Domain Layer Interface)
# =============================================================================


@runtime_checkable
class ToolResultProtocol(Protocol):
    """
    도구 실행 결과 인터페이스
    """

    tool_name: str
    success: bool
    data: dict[str, Any]
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 변환"""
        ...


# =============================================================================
# Response Protocol (Domain Layer Interface)
# =============================================================================


@runtime_checkable
class ResponseProtocol(Protocol):
    """
    최종 응답 인터페이스
    """

    text: str
    query_type: str
    confidence_level: ConfidenceLevel
    confidence_score: float
    sources: list[str]
    is_fallback: bool
    is_clarification: bool

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 변환"""
        ...


# =============================================================================
# Context 기본 구현 (Domain Layer)
# =============================================================================


@dataclass
class ContextBase:
    """
    RAG + KG 통합 컨텍스트 기본 구현

    LLM 판단 및 응답 생성에 사용되는 모든 컨텍스트 정보를 담는다.
    """

    query: str
    entities: dict[str, list[str]] = field(default_factory=dict)
    rag_docs: list[dict[str, Any]] = field(default_factory=list)
    kg_facts: list[KGFact] = field(default_factory=list)
    kg_inferences: list[dict[str, Any]] = field(default_factory=list)
    system_state: SystemState | None = None
    summary: str = ""
    gathered_at: datetime | None = None

    def __post_init__(self):
        """생성 시 시간 자동 기록"""
        if self.gathered_at is None:
            self.gathered_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "entities": self.entities,
            "rag_docs_count": len(self.rag_docs),
            "kg_facts_count": len(self.kg_facts),
            "kg_inferences_count": len(self.kg_inferences),
            "system_state": (self.system_state.to_dict() if self.system_state else None),
            "summary": (self.summary[:200] + "..." if len(self.summary) > 200 else self.summary),
            "gathered_at": (self.gathered_at.isoformat() if self.gathered_at else None),
        }

    def has_sufficient_context(self) -> bool:
        """응답 생성에 충분한 컨텍스트가 있는지 확인"""
        return bool(self.rag_docs or self.kg_facts or self.kg_inferences)


# =============================================================================
# ToolResult 기본 구현 (Domain Layer)
# =============================================================================


@dataclass
class ToolResultBase:
    """
    도구 실행 결과 기본 구현
    """

    tool_name: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
        }


# =============================================================================
# Response 기본 구현 (Domain Layer)
# =============================================================================


@dataclass
class ResponseBase:
    """
    최종 응답 기본 구현
    """

    text: str
    query_type: str = "unknown"
    confidence_level: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    confidence_score: float = 0.0
    sources: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)
    tools_called: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    is_fallback: bool = False
    is_clarification: bool = False
    processing_time_ms: float = 0.0
    created_at: datetime | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "query_type": self.query_type,
            "confidence_level": self.confidence_level.value,
            "confidence_score": self.confidence_score,
            "sources": self.sources,
            "entities": self.entities,
            "tools_called": self.tools_called,
            "suggestions": self.suggestions,
            "is_fallback": self.is_fallback,
            "is_clarification": self.is_clarification,
            "processing_time_ms": self.processing_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def clarification(cls, message: str, suggestions: list[str] = None) -> "ResponseBase":
        """명확화 요청 응답 생성 팩토리 메서드"""
        return cls(
            text=message,
            query_type="clarification",
            confidence_level=ConfidenceLevel.UNKNOWN,
            is_clarification=True,
            suggestions=suggestions or [],
        )

    @classmethod
    def fallback(cls, message: str) -> "ResponseBase":
        """Fallback 응답 생성 팩토리 메서드"""
        return cls(
            text=message,
            query_type="fallback",
            confidence_level=ConfidenceLevel.UNKNOWN,
            is_fallback=True,
            suggestions=["다시 질문해주세요", "SoS가 뭔가요?", "현재 순위 알려주세요"],
        )

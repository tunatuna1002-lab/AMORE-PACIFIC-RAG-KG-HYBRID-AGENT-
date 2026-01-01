"""
데이터 모델 정의
================
LLM Orchestrator에서 사용하는 핵심 데이터 구조

주요 모델:
- ConfidenceLevel: 신뢰도 레벨 (HIGH/MEDIUM/LOW/UNKNOWN)
- Context: RAG + KG 컨텍스트 통합 구조
- Decision: LLM 판단 결과
- ToolResult: 도구 실행 결과
- Response: 최종 응답
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional


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
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.fact_type,
            "entity": self.entity,
            "data": self.data
        }


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
    last_crawl_time: Optional[datetime] = None
    data_freshness: str = "unknown"
    kg_triple_count: int = 0
    kg_initialized: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_crawl_time": self.last_crawl_time.isoformat() if self.last_crawl_time else None,
            "data_freshness": self.data_freshness,
            "kg_triple_count": self.kg_triple_count,
            "kg_initialized": self.kg_initialized
        }


# =============================================================================
# 컨텍스트 모델
# =============================================================================

@dataclass
class Context:
    """
    RAG + KG 통합 컨텍스트

    LLM 판단 및 응답 생성에 사용되는 모든 컨텍스트 정보를 담는다.

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
    query: str
    entities: Dict[str, List[str]] = field(default_factory=dict)
    rag_docs: List[Dict[str, Any]] = field(default_factory=list)
    kg_facts: List[KGFact] = field(default_factory=list)
    kg_inferences: List[Dict[str, Any]] = field(default_factory=list)
    system_state: Optional[SystemState] = None
    summary: str = ""
    gathered_at: Optional[datetime] = None

    def __post_init__(self):
        """생성 시 시간 자동 기록"""
        if self.gathered_at is None:
            self.gathered_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "entities": self.entities,
            "rag_docs_count": len(self.rag_docs),
            "kg_facts_count": len(self.kg_facts),
            "kg_inferences_count": len(self.kg_inferences),
            "system_state": self.system_state.to_dict() if self.system_state else None,
            "summary": self.summary[:200] + "..." if len(self.summary) > 200 else self.summary,
            "gathered_at": self.gathered_at.isoformat() if self.gathered_at else None
        }

    def has_sufficient_context(self) -> bool:
        """응답 생성에 충분한 컨텍스트가 있는지 확인"""
        return bool(self.rag_docs or self.kg_facts or self.kg_inferences)


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
    tool: Optional[str] = None  # None = direct_answer
    tool_params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    key_points: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def requires_tool(self) -> bool:
        """도구 실행이 필요한지 확인"""
        return self.tool is not None and self.tool != "direct_answer"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "tool_params": self.tool_params,
            "reason": self.reason,
            "key_points": self.key_points,
            "confidence": self.confidence
        }


# =============================================================================
# 도구 실행 결과
# =============================================================================

@dataclass
class ToolResult:
    """
    도구 실행 결과

    Attributes:
        tool_name: 실행된 도구명
        success: 성공 여부
        data: 결과 데이터
        error: 에러 메시지 (실패 시)
        execution_time_ms: 실행 시간 (밀리초)
    """
    tool_name: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms
        }

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
            return f"데이터 조회 완료"

        else:
            return f"[{self.tool_name}] 실행 완료"


# =============================================================================
# 최종 응답
# =============================================================================

@dataclass
class Response:
    """
    최종 응답

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
    text: str
    query_type: str = "unknown"
    confidence_level: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    confidence_score: float = 0.0
    sources: List[str] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    tools_called: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    is_fallback: bool = False
    is_clarification: bool = False
    processing_time_ms: float = 0.0
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
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
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

    @classmethod
    def clarification(cls, message: str, suggestions: List[str] = None) -> "Response":
        """명확화 요청 응답 생성 팩토리 메서드"""
        return cls(
            text=message,
            query_type="clarification",
            confidence_level=ConfidenceLevel.UNKNOWN,
            is_clarification=True,
            suggestions=suggestions or []
        )

    @classmethod
    def fallback(cls, message: str) -> "Response":
        """Fallback 응답 생성 팩토리 메서드"""
        return cls(
            text=message,
            query_type="fallback",
            confidence_level=ConfidenceLevel.UNKNOWN,
            is_fallback=True,
            suggestions=["다시 질문해주세요", "SoS가 뭔가요?", "현재 순위 알려주세요"]
        )

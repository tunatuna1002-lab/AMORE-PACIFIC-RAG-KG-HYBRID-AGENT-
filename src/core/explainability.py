"""
응답 설명성 모듈
================
응답이 어떻게 생성되었는지 투명하게 설명

역할:
- 소스 추적 (RAG, KG, Ontology, Crawled)
- 추론 경로 설명
- 신뢰도 분해
- 사용자 친화적 설명 생성

연결 파일:
- core/models.py: Context, Decision, Response
- core/confidence.py: ConfidenceAssessor
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from .models import ConfidenceLevel, Context, Decision, Response

logger = logging.getLogger(__name__)


@dataclass
class ExplanationTrace:
    """응답 생성 과정 추적"""

    # 소스 정보
    sources_used: list[str] = field(default_factory=list)  # ["RAG", "KG", "Ontology", "Crawled"]
    rag_doc_count: int = 0
    kg_fact_count: int = 0
    kg_inference_count: int = 0

    # 판단 경로
    confidence_level: str = "unknown"
    confidence_score: float = 0.0
    decision_tool: str = ""
    decision_reason: str = ""
    routing_path: str = ""  # "HIGH→direct" | "MEDIUM→LLM→tool" | "LOW→LLM→full"

    # 핵심 근거
    key_evidence: list[str] = field(default_factory=list)

    # 처리 시간
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "sources_used": self.sources_used,
            "rag_doc_count": self.rag_doc_count,
            "kg_fact_count": self.kg_fact_count,
            "kg_inference_count": self.kg_inference_count,
            "confidence_level": self.confidence_level,
            "confidence_score": self.confidence_score,
            "decision_tool": self.decision_tool,
            "decision_reason": self.decision_reason,
            "routing_path": self.routing_path,
            "key_evidence": self.key_evidence,
            "processing_time_ms": self.processing_time_ms,
        }

    def to_human_readable(self) -> str:
        """사용자 친화적 설명 생성"""
        parts = []

        # 1. 소스 요약
        if self.sources_used:
            source_str = ", ".join(self.sources_used)
            parts.append(f"📊 **활용 소스**: {source_str}")

        # 2. 데이터 규모
        data_parts = []
        if self.rag_doc_count > 0:
            data_parts.append(f"참조 문서 {self.rag_doc_count}개")
        if self.kg_fact_count > 0:
            data_parts.append(f"KG 사실 {self.kg_fact_count}개")
        if self.kg_inference_count > 0:
            data_parts.append(f"추론 결과 {self.kg_inference_count}개")
        if data_parts:
            parts.append(f"📁 **참조 데이터**: {', '.join(data_parts)}")

        # 3. 신뢰도
        confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🟠", "unknown": "🔴"}
        emoji = confidence_emoji.get(self.confidence_level.lower(), "⚪")
        parts.append(
            f"{emoji} **신뢰도**: {self.confidence_level.upper()} ({self.confidence_score:.0%})"
        )

        # 4. 처리 경로
        if self.routing_path:
            parts.append(f"🔄 **처리 경로**: {self.routing_path}")

        # 5. 핵심 근거
        if self.key_evidence:
            parts.append("📌 **핵심 근거**:")
            for i, evidence in enumerate(self.key_evidence[:3], 1):
                parts.append(f"   {i}. {evidence}")

        return "\n".join(parts)


class ExplainabilityEngine:
    """
    응답 설명성 엔진

    응답 생성 과정을 추적하고 사용자에게 설명을 제공.

    Usage:
        engine = ExplainabilityEngine()
        trace = engine.build_trace(context, decision, response)
        explanation = trace.to_human_readable()
    """

    def build_trace(
        self,
        context: Context,
        decision: Decision | None = None,
        response: Response | None = None,
        confidence_level: ConfidenceLevel | None = None,
    ) -> ExplanationTrace:
        """
        응답 생성 과정 추적 빌드

        Args:
            context: 수집된 컨텍스트
            decision: LLM 판단 결과
            response: 생성된 응답
            confidence_level: 신뢰도 레벨

        Returns:
            ExplanationTrace
        """
        trace = ExplanationTrace()

        # 1. 소스 추적
        trace.sources_used = self._identify_sources(context)
        trace.rag_doc_count = len(context.rag_docs) if context.rag_docs else 0
        trace.kg_fact_count = len(context.kg_facts) if context.kg_facts else 0
        trace.kg_inference_count = len(context.kg_inferences) if context.kg_inferences else 0

        # 2. 판단 경로
        if decision:
            trace.decision_tool = decision.tool or ""
            trace.decision_reason = decision.reason or ""
            trace.confidence_score = decision.confidence or 0.0

        # 3. 신뢰도
        if confidence_level:
            trace.confidence_level = confidence_level.value
            trace.routing_path = self._determine_routing_path(confidence_level, decision)
        elif decision:
            # Decision의 confidence에서 추정
            if decision.confidence >= 0.85:
                trace.confidence_level = "high"
                trace.routing_path = "HIGH → 직접 응답"
            elif decision.confidence >= 0.5:
                trace.confidence_level = "medium"
                trace.routing_path = "MEDIUM → LLM 판단"
            else:
                trace.confidence_level = "low"
                trace.routing_path = "LOW → LLM 전체 판단"

        # 4. 응답 정보
        if response:
            trace.confidence_score = response.confidence_score or trace.confidence_score
            trace.processing_time_ms = response.processing_time_ms or 0.0

        # 5. 핵심 근거 추출
        trace.key_evidence = self._extract_key_evidence(context, decision)

        return trace

    def _identify_sources(self, context: Context) -> list[str]:
        """사용된 소스 식별"""
        sources = []

        if context.rag_docs:
            sources.append("RAG")

        if context.kg_facts:
            sources.append("KG")

        if context.kg_inferences:
            sources.append("Ontology")

        # 시스템 상태에서 크롤링 데이터 확인
        if context.system_state and hasattr(context.system_state, "last_crawl_time"):
            if context.system_state.last_crawl_time:
                sources.append("Crawled")

        return sources if sources else ["None"]

    def _determine_routing_path(
        self, confidence_level: ConfidenceLevel, decision: Decision | None
    ) -> str:
        """처리 경로 결정"""
        level_str = confidence_level.value.upper()

        if confidence_level == ConfidenceLevel.HIGH:
            return f"{level_str} → 직접 응답 (LLM 스킵)"
        elif confidence_level == ConfidenceLevel.MEDIUM:
            tool = decision.tool if decision else "unknown"
            if tool == "direct_answer":
                return f"{level_str} → LLM 판단 → 직접 응답"
            else:
                return f"{level_str} → LLM 판단 → {tool}"
        elif confidence_level == ConfidenceLevel.LOW:
            tool = decision.tool if decision else "unknown"
            return f"{level_str} → LLM 전체 판단 → {tool}"
        else:
            return f"{level_str} → 명확화 요청"

    def _extract_key_evidence(self, context: Context, decision: Decision | None) -> list[str]:
        """핵심 근거 추출"""
        evidence = []

        # KG 사실에서
        if context.kg_facts:
            for fact in context.kg_facts[:2]:
                if hasattr(fact, "entity") and hasattr(fact, "fact_type"):
                    evidence.append(f"[KG] {fact.entity} - {fact.fact_type}")

        # KG 추론에서
        if context.kg_inferences:
            for inf in context.kg_inferences[:2]:
                if isinstance(inf, dict):
                    insight = inf.get("insight", inf.get("type", ""))
                    if insight:
                        evidence.append(f"[Ontology] {insight}")

        # RAG 문서에서
        if context.rag_docs:
            for doc in context.rag_docs[:1]:
                title = doc.get("metadata", {}).get("title", "")
                if title:
                    evidence.append(f"[RAG] {title}")

        # Decision key_points에서
        if decision and decision.key_points:
            for point in decision.key_points[:2]:
                evidence.append(f"[분석] {point}")

        return evidence[:5]  # 최대 5개

    def format_for_response(self, trace: ExplanationTrace, include_details: bool = False) -> str:
        """
        응답에 포함할 설명 포맷팅

        Args:
            trace: 추적 정보
            include_details: 상세 정보 포함 여부

        Returns:
            포맷된 설명 문자열
        """
        if include_details:
            return trace.to_human_readable()

        # 간략 버전
        sources = ", ".join(trace.sources_used)
        return f"\n\n---\n_소스: {sources} | 신뢰도: {trace.confidence_level.upper()} ({trace.confidence_score:.0%})_"

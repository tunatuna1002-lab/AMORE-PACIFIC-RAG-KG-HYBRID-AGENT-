"""
컨텍스트 수집기
==============
RAG + KG 통합 컨텍스트를 LLM 판단용으로 수집

역할:
- RAG 검색 결과 수집
- KG 사실 및 추론 결과 수집
- 시스템 상태 수집
- LLM 프롬프트용 요약 생성

연결 파일:
- rag/hybrid_retriever.py: HybridRetriever 활용
- ontology/knowledge_graph.py: KG 직접 조회
- ontology/reasoner.py: 추론 결과 조회
- core/models.py: Context, KGFact, SystemState
- core/state.py: OrchestratorState
"""

import logging
from datetime import datetime
from typing import Any

from .models import Context, KGFact, SystemState
from .state import OrchestratorState

logger = logging.getLogger(__name__)


class ContextGatherer:
    """
    RAG + KG 통합 컨텍스트 수집기

    LLM이 판단 및 응답 생성에 필요한 모든 컨텍스트를 수집한다.
    HybridRetriever를 래핑하여 core 모듈의 데이터 모델로 변환.

    Usage:
        gatherer = ContextGatherer(hybrid_retriever, orchestrator_state)
        context = await gatherer.gather(query, entities)
    """

    def __init__(
        self,
        hybrid_retriever: Any | None = None,
        orchestrator_state: OrchestratorState | None = None,
        max_rag_docs: int = 5,
        max_kg_facts: int = 10,
    ):
        """
        Args:
            hybrid_retriever: HybridRetriever 인스턴스
            orchestrator_state: 오케스트레이터 상태
            max_rag_docs: RAG 문서 최대 수
            max_kg_facts: KG 사실 최대 수
        """
        self.retriever = hybrid_retriever
        self.state = orchestrator_state or OrchestratorState()
        self.max_rag_docs = max_rag_docs
        self.max_kg_facts = max_kg_facts

        # 초기화 플래그
        self._initialized = False

    # =========================================================================
    # 초기화
    # =========================================================================

    async def initialize(self) -> None:
        """비동기 초기화"""
        if self._initialized:
            return

        # HybridRetriever 초기화
        if self.retriever and hasattr(self.retriever, "initialize"):
            await self.retriever.initialize()

        self._initialized = True
        logger.info("ContextGatherer initialized")

    def set_retriever(self, retriever: Any) -> None:
        """Retriever 설정 (지연 주입용)"""
        self.retriever = retriever
        self._initialized = False

    # =========================================================================
    # 메인 수집 메서드
    # =========================================================================

    async def gather(
        self,
        query: str,
        entities: dict[str, list[str]] | None = None,
        current_metrics: dict[str, Any] | None = None,
        include_system_state: bool = True,
    ) -> Context:
        """
        통합 컨텍스트 수집

        Args:
            query: 사용자 질문
            entities: 추출된 엔티티 (없으면 retriever가 추출)
            current_metrics: 현재 계산된 지표
            include_system_state: 시스템 상태 포함 여부

        Returns:
            Context 객체
        """
        # 초기화 확인
        if not self._initialized:
            await self.initialize()

        # 기본 Context 생성
        context = Context(query=query, entities=entities or {})

        try:
            # 1. HybridRetriever를 통한 RAG + KG 조회
            if self.retriever:
                # TrueHybridRetriever vs HybridRetriever 시그니처 분기
                if hasattr(self.retriever, "owl_reasoner"):
                    # TrueHybridRetriever (OWL + Entity Linking + Confidence Fusion)
                    hybrid_context = await self.retriever.retrieve(
                        query=query, current_metrics=current_metrics, top_k=self.max_rag_docs
                    )
                else:
                    # 기존 HybridRetriever
                    hybrid_context = await self.retriever.retrieve(
                        query=query, current_metrics=current_metrics, include_explanations=True
                    )

                # HybridResult (TrueHybridRetriever) vs HybridContext (HybridRetriever) 감지
                if hasattr(hybrid_context, "entity_links"):
                    # TrueHybridRetriever → to_dict()로 호환 포맷 변환
                    compat = hybrid_context.to_dict()

                    if not entities:
                        context.entities = compat.get("entities", {})

                    context.rag_docs = compat.get("rag_chunks", [])[: self.max_rag_docs]

                    context.kg_facts = self._convert_kg_facts(compat.get("ontology_facts", []))

                    # inferences는 이미 dict 리스트 (to_dict()에서 변환됨)
                    context.kg_inferences = compat.get("inferences", [])

                    # TrueHybridRetriever의 combined_context를 요약으로 활용
                    if hybrid_context.combined_context:
                        context.summary = hybrid_context.combined_context

                    logger.debug(
                        f"TrueHybridRetriever: confidence={hybrid_context.confidence:.2f}, "
                        f"entities={len(hybrid_context.entity_links)}"
                    )
                else:
                    # 기존 HybridRetriever → HybridContext 처리
                    if hybrid_context.entities and not entities:
                        context.entities = hybrid_context.entities

                    context.rag_docs = hybrid_context.rag_chunks[: self.max_rag_docs]

                    context.kg_facts = self._convert_kg_facts(hybrid_context.ontology_facts)

                    context.kg_inferences = [inf.to_dict() for inf in hybrid_context.inferences]

            # 2. 시스템 상태
            if include_system_state:
                context.system_state = self._get_system_state()

            # 3. 요약 생성
            # TrueHybridRetriever가 이미 combined_context를 설정한 경우 그것을 유지.
            # 두 요약을 이어붙이면 같은 데이터가 중복되어 LLM 토큰 낭비 + 답변 혼란.
            if context.summary:
                # TrueHybridRetriever의 구조화된 요약 유지, 시스템 상태만 보충
                if context.system_state and "[시스템 상태]" not in context.summary:
                    state_str = self._format_system_state(context.system_state)
                    if state_str:
                        context.summary = f"[시스템 상태] {state_str}\n\n{context.summary}"
            else:
                context.summary = self._build_summary(context)

            # 수집 시간 기록
            context.gathered_at = datetime.now()

            logger.debug(
                f"Context gathered: {len(context.rag_docs)} docs, "
                f"{len(context.kg_facts)} facts, "
                f"{len(context.kg_inferences)} inferences"
            )

        except Exception as e:
            logger.error(f"Context gathering failed: {e}")
            # 최소한의 컨텍스트 반환
            context.summary = f"컨텍스트 수집 중 오류: {str(e)}"

        return context

    async def gather_for_decision(
        self,
        query: str,
        entities: dict[str, list[str]],
        current_metrics: dict[str, Any] | None = None,
    ) -> Context:
        """
        LLM 판단용 경량 컨텍스트 수집

        전체 컨텍스트 대신 판단에 필요한 핵심 정보만 수집.

        Args:
            query: 질문
            entities: 엔티티
            current_metrics: 지표

        Returns:
            경량 Context
        """
        context = Context(query=query, entities=entities)

        # KG에서 핵심 정보만
        if self.retriever and hasattr(self.retriever, "kg"):
            kg = self.retriever.kg
            facts = []

            # 브랜드 정보
            for brand in entities.get("brands", []):
                meta = kg.get_entity_metadata(brand)
                if meta:
                    facts.append(KGFact(fact_type="brand_info", entity=brand, data=meta))

            context.kg_facts = facts[:5]

        # 시스템 상태
        context.system_state = self._get_system_state()

        # 간략 요약
        context.summary = self._build_decision_summary(context)

        return context

    # =========================================================================
    # 변환 헬퍼
    # =========================================================================

    def _convert_kg_facts(self, ontology_facts: list[dict[str, Any]]) -> list[KGFact]:
        """
        HybridRetriever의 ontology_facts를 KGFact로 변환

        Args:
            ontology_facts: raw 사실 리스트

        Returns:
            KGFact 리스트
        """
        kg_facts = []

        for fact in ontology_facts[: self.max_kg_facts]:
            kg_facts.append(
                KGFact(
                    fact_type=fact.get("type", "unknown"),
                    entity=fact.get("entity", ""),
                    data=fact.get("data", {}),
                )
            )

        return kg_facts

    def _get_system_state(self) -> SystemState:
        """현재 시스템 상태 수집"""
        return SystemState(
            last_crawl_time=self.state.last_crawl_time,
            data_freshness=self.state.data_freshness,
            kg_triple_count=self.state.kg_triple_count,
            kg_initialized=self.state.kg_initialized,
        )

    # =========================================================================
    # 요약 생성
    # =========================================================================

    def _build_summary(self, context: Context) -> str:
        """
        LLM 프롬프트용 컨텍스트 요약 생성

        Args:
            context: 수집된 컨텍스트

        Returns:
            요약 문자열
        """
        parts = []

        # 1. 시스템 상태
        if context.system_state:
            state_str = self._format_system_state(context.system_state)
            if state_str:
                parts.append(f"[시스템 상태] {state_str}")

        # 2. KG 추론 인사이트 (가장 중요)
        if context.kg_inferences:
            parts.append("\n[분석 인사이트]")
            for i, inf in enumerate(context.kg_inferences[:3], 1):
                insight = inf.get("insight", "")
                rec = inf.get("recommendation", "")
                parts.append(f"{i}. {insight}")
                if rec:
                    parts.append(f"   → {rec}")

        # 3. KG 사실
        if context.kg_facts:
            parts.append("\n[관련 정보]")
            for fact in context.kg_facts[:5]:
                fact_str = self._format_kg_fact(fact)
                if fact_str:
                    parts.append(f"- {fact_str}")

        # 4. RAG 문서 요약
        if context.rag_docs:
            parts.append("\n[참조 문서]")
            for doc in context.rag_docs[:3]:
                title = doc.get("metadata", {}).get("title", "")
                content = doc.get("content", "")[:100]
                if title:
                    parts.append(f"- {title}: {content}...")

        return "\n".join(parts)

    def _build_decision_summary(self, context: Context) -> str:
        """LLM 판단용 간략 요약"""
        parts = []

        # 시스템 상태
        if context.system_state:
            if context.system_state.data_freshness == "fresh":
                parts.append("데이터: 최신")
            elif context.system_state.last_crawl_time:
                hours = (
                    datetime.now() - context.system_state.last_crawl_time
                ).total_seconds() / 3600
                parts.append(f"데이터: {hours:.1f}시간 전 수집")
            else:
                parts.append("데이터: 없음 (크롤링 필요)")

        # KG 상태
        if context.system_state and context.system_state.kg_initialized:
            parts.append(f"KG: {context.system_state.kg_triple_count} 트리플")
        else:
            parts.append("KG: 미초기화")

        # 핵심 사실
        for fact in context.kg_facts[:2]:
            if fact.fact_type == "brand_info" and fact.data.get("sos"):
                parts.append(f"{fact.entity} SoS: {fact.data['sos']*100:.1f}%")

        return " | ".join(parts)

    def _format_system_state(self, state: SystemState) -> str:
        """시스템 상태 포맷팅"""
        parts = []

        if state.last_crawl_time:
            age = (datetime.now() - state.last_crawl_time).total_seconds() / 3600
            parts.append(f"마지막 크롤링: {age:.1f}시간 전")
        else:
            parts.append("크롤링 기록 없음")

        parts.append(f"데이터 상태: {state.data_freshness}")

        if state.kg_initialized:
            parts.append(f"KG: {state.kg_triple_count} 트리플")

        return " | ".join(parts)

    def _format_kg_fact(self, fact: KGFact) -> str:
        """KG 사실 포맷팅"""
        if fact.fact_type == "brand_info":
            sos = fact.data.get("sos", 0)
            avg_rank = fact.data.get("avg_rank")
            info = f"{fact.entity}"
            if sos:
                info += f" SoS {sos*100:.1f}%"
            if avg_rank:
                info += f" 평균순위 {avg_rank:.1f}"
            return info

        elif fact.fact_type == "brand_products":
            count = fact.data.get("product_count", 0)
            return f"{fact.entity} 제품 {count}개"

        elif fact.fact_type == "competitors":
            comps = (
                [c.get("brand", "") for c in fact.data[:3]] if isinstance(fact.data, list) else []
            )
            if comps:
                return f"{fact.entity} 경쟁사: {', '.join(comps)}"

        elif fact.fact_type == "category_brands":
            top = fact.data.get("top_brands", [])[:3]
            brands = [b.get("brand", "") for b in top]
            if brands:
                return f"{fact.entity} Top 브랜드: {', '.join(brands)}"

        return ""

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """수집기 통계"""
        stats = {
            "initialized": self._initialized,
            "max_rag_docs": self.max_rag_docs,
            "max_kg_facts": self.max_kg_facts,
            "has_retriever": self.retriever is not None,
        }

        if self.retriever and hasattr(self.retriever, "get_stats"):
            stats["retriever_stats"] = self.retriever.get_stats()

        return stats

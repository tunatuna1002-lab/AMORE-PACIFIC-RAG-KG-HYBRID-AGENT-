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

from src.domain.entities.evidence import Evidence
from src.rag.evidence_renderer import render_for_prompt

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
                from ..domain.value_objects.retrieval_result import UnifiedRetrievalResult

                # retrieve_unified()가 있으면 사용 (통합 경로)
                if hasattr(self.retriever, "retrieve_unified"):
                    result = await self.retriever.retrieve_unified(
                        query=query, current_metrics=current_metrics, top_k=self.max_rag_docs
                    )

                    if isinstance(result, UnifiedRetrievalResult):
                        if not entities:
                            context.entities = result.entities

                        context.rag_docs = result.rag_chunks[: self.max_rag_docs]
                        context.kg_facts = self._convert_kg_facts(result.ontology_facts)
                        context.kg_inferences = result.inferences
                        context.evidence = self._card_list(result.evidence)
                        context.prompt_evidence = self._card_list(result.prompt_evidence)

                        if result.combined_context:
                            context.summary = result.combined_context

                        logger.debug(
                            f"HybridRetriever ({result.retriever_type}): "
                            f"confidence={result.confidence:.2f}, "
                            f"entities={len(result.entity_links)}"
                        )
                    else:
                        logger.warning(
                            f"Expected UnifiedRetrievalResult but got {type(result).__name__}"
                        )
                else:
                    # Fallback: legacy retrieve() → HybridContext
                    hybrid_context = await self.retriever.retrieve(
                        query=query, current_metrics=current_metrics, include_explanations=True
                    )

                    if hybrid_context.entities and not entities:
                        context.entities = hybrid_context.entities

                    context.rag_docs = hybrid_context.rag_chunks[: self.max_rag_docs]
                    context.kg_facts = self._convert_kg_facts(hybrid_context.ontology_facts)
                    context.kg_inferences = [
                        inf.to_dict() if hasattr(inf, "to_dict") else inf
                        for inf in hybrid_context.inferences
                    ]
                    context.evidence = self._card_list(getattr(hybrid_context, "evidence", None))
                    context.prompt_evidence = self._card_list(
                        getattr(hybrid_context, "prompt_evidence", None)
                    )
                    combined = getattr(hybrid_context, "combined_context", "")
                    if isinstance(combined, str) and combined:
                        context.summary = combined

            # 2. 시스템 상태
            if include_system_state:
                context.system_state = self._get_system_state()

            # 3. 요약 생성
            # retriever의 combined_context(= 프롬프트 카드 렌더링)를 그대로 쓰고 시스템 상태만
            # 앞에 붙인다. KG 사실·추론·문서를 카드 밖 형식으로 다시 렌더링하지 않는다 (E1).
            if context.summary:
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

    @staticmethod
    def _card_list(cards: Any) -> list[Evidence]:
        """검색 결과의 카드 필드 → 카드 리스트 (카드 필드가 없는 결과는 빈 리스트)."""
        if not isinstance(cards, list | tuple):
            return []
        return [card for card in cards if isinstance(card, Evidence)]

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
        LLM 프롬프트용 컨텍스트 요약 (retriever가 combined_context를 주지 않았을 때)

        시스템 상태 한 줄 + 프롬프트 카드 렌더링만 싣는다. ``kg_facts``·``kg_inferences``·
        ``rag_docs`` 원자료는 카드로만 프롬프트에 간다 (E1) — 특히 KG 엔티티 메타데이터의
        날짜 없는 SoS·평균 순위는 증거가 아니다 (E2).

        Args:
            context: 수집된 컨텍스트

        Returns:
            요약 문자열
        """
        parts = []

        if context.system_state:
            state_str = self._format_system_state(context.system_state)
            if state_str:
                parts.append(f"[시스템 상태] {state_str}")

        cards = render_for_prompt(context.prompt_evidence)
        if cards:
            parts.append(cards)

        return "\n\n".join(parts)

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
                parts.append(f"{fact.entity} SoS: {fact.data['sos'] * 100:.1f}%")

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

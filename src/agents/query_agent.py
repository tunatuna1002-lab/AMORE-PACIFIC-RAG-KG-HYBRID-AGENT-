"""
쿼리 에이전트 (Query Agent)
============================
사용자 질문 처리 전담 에이전트

역할:
- 사용자 질문 분석 및 처리
- RAG + Ontology 하이브리드 검색
- LLM 기반 응답 생성
- 컨텍스트 관리

이 에이전트는 core/brain.py에서 호출되어 작동합니다.

Usage:
    query_agent = QueryAgent()
    await query_agent.initialize()
    response = await query_agent.process("LANEIGE 순위가 어떻게 되나요?")
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from src.core.context_gatherer import ContextGatherer
from src.core.response_pipeline import ResponsePipeline
from src.core.cache import ResponseCache
from src.core.models import Context, Response
from src.core.confidence import ConfidenceAssessor
from src.core.state import OrchestratorState

from src.rag.router import RAGRouter
from src.rag.hybrid_retriever import HybridRetriever

from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner

from src.memory.context import ContextManager

logger = logging.getLogger(__name__)


# =============================================================================
# 쿼리 결과 정의
# =============================================================================

@dataclass
class QueryResult:
    """쿼리 처리 결과"""
    query: str
    response: str
    confidence: float
    sources: List[str] = field(default_factory=list)
    reasoning: Optional[str] = None
    processing_time_ms: float = 0
    from_cache: bool = False
    entities: Dict[str, List[str]] = field(default_factory=dict)
    inferences: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "confidence": self.confidence,
            "sources": self.sources,
            "reasoning": self.reasoning,
            "processing_time_ms": self.processing_time_ms,
            "from_cache": self.from_cache,
            "entities": self.entities,
            "inferences": self.inferences
        }


# =============================================================================
# 쿼리 에이전트
# =============================================================================

class QueryAgent:
    """
    사용자 질문 처리 전담 에이전트

    RAG와 Ontology를 결합한 하이브리드 검색으로
    정확하고 풍부한 응답을 생성합니다.

    처리 흐름:
    1. 질문 분석 (엔티티 추출)
    2. 컨텍스트 수집 (RAG + KG)
    3. 응답 생성 (LLM)
    4. 신뢰도 평가
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        kg_persist_path: str = "./data/knowledge_graph.json",
        cache_ttl: int = 3600
    ):
        """
        Args:
            model: LLM 모델
            kg_persist_path: Knowledge Graph 경로
            cache_ttl: 캐시 TTL (초)
        """
        self.model = model
        self.kg_persist_path = kg_persist_path
        self.cache_ttl = cache_ttl

        # 핵심 컴포넌트 (lazy init)
        self._router: Optional[RAGRouter] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._reasoner: Optional[OntologyReasoner] = None
        self._hybrid_retriever: Optional[HybridRetriever] = None
        self._context_gatherer: Optional[ContextGatherer] = None
        self._response_pipeline: Optional[ResponsePipeline] = None
        self._cache: Optional[ResponseCache] = None
        self._context_manager: Optional[ContextManager] = None
        self._confidence_assessor: Optional[ConfidenceAssessor] = None

        # 현재 데이터 컨텍스트
        self._current_metrics: Optional[Dict[str, Any]] = None
        self._session_id: Optional[str] = None

        # 초기화 플래그
        self._initialized = False

        # 통계
        self._stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "avg_confidence": 0.0,
            "avg_processing_time": 0.0
        }

    # =========================================================================
    # 초기화
    # =========================================================================

    async def initialize(self) -> None:
        """비동기 초기화"""
        if self._initialized:
            return

        # Knowledge Graph
        self._knowledge_graph = KnowledgeGraph(persist_path=self.kg_persist_path)

        # Reasoner
        self._reasoner = OntologyReasoner(self._knowledge_graph)

        # Hybrid Retriever
        self._hybrid_retriever = HybridRetriever(
            knowledge_graph=self._knowledge_graph,
            reasoner=self._reasoner
        )

        # RAG Router
        self._router = RAGRouter()

        # Context Gatherer
        self._context_gatherer = ContextGatherer(
            hybrid_retriever=self._hybrid_retriever,
            orchestrator_state=OrchestratorState()
        )
        await self._context_gatherer.initialize()

        # Response Pipeline
        self._response_pipeline = ResponsePipeline()

        # Cache
        self._cache = ResponseCache()

        # Context Manager
        self._context_manager = ContextManager()

        # Confidence Assessor
        self._confidence_assessor = ConfidenceAssessor()

        self._initialized = True
        logger.info("QueryAgent initialized")

    # =========================================================================
    # 데이터 컨텍스트 설정
    # =========================================================================

    def set_data_context(self, metrics_data: Dict[str, Any]) -> None:
        """현재 지표 데이터 컨텍스트 설정"""
        self._current_metrics = metrics_data

    def set_session(self, session_id: str) -> None:
        """세션 설정"""
        self._session_id = session_id

    # =========================================================================
    # 질문 처리
    # =========================================================================

    async def process(
        self,
        query: str,
        skip_cache: bool = False
    ) -> QueryResult:
        """
        사용자 질문 처리

        Args:
            query: 사용자 질문
            skip_cache: 캐시 스킵 여부

        Returns:
            QueryResult
        """
        start_time = datetime.now()
        self._stats["total_queries"] += 1

        # 초기화 확인
        if not self._initialized:
            await self.initialize()

        try:
            # 1. 캐시 확인
            if not skip_cache and self._cache:
                cached = self._cache.get(query, "query")
                if cached:
                    self._stats["cache_hits"] += 1
                    return QueryResult(
                        query=query,
                        response=cached.content,
                        confidence=cached.confidence,
                        sources=cached.sources,
                        processing_time_ms=0,
                        from_cache=True
                    )

            # 2. 엔티티 추출
            entities = self._extract_entities(query)

            # 3. 컨텍스트 수집
            context = await self._gather_context(query, entities)

            # 4. 추론 수행
            inferences = await self._perform_inferences(entities)

            # 5. 응답 생성
            response = await self._generate_response(query, context, inferences)

            # 6. 신뢰도 평가
            confidence = self._assess_confidence(query, context, response)

            # 처리 시간
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # 결과 생성
            result = QueryResult(
                query=query,
                response=response.content,
                confidence=confidence,
                sources=response.sources,
                reasoning=response.reasoning,
                processing_time_ms=processing_time,
                from_cache=False,
                entities=entities,
                inferences=inferences
            )

            # 캐시 저장
            if not skip_cache and self._cache and not response.is_fallback:
                self._cache.set(query, response, "query")

            # 통계 업데이트
            self._update_stats(confidence, processing_time)

            return result

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return QueryResult(
                query=query,
                response=f"처리 중 오류가 발생했습니다: {str(e)}",
                confidence=0.0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """엔티티 추출"""
        if self._router:
            return self._router.extract_entities(query)

        # 간단한 폴백 추출
        entities = {"brands": [], "products": [], "categories": [], "indicators": []}

        # 알려진 브랜드 매칭
        known_brands = ["LANEIGE", "라네즈", "설화수", "Sulwhasoo", "이니스프리", "Innisfree"]
        query_upper = query.upper()
        for brand in known_brands:
            if brand.upper() in query_upper:
                entities["brands"].append(brand)

        # 지표 키워드 매칭
        indicators = ["SoS", "점유율", "순위", "HHI", "CPI", "변동성"]
        for ind in indicators:
            if ind.lower() in query.lower():
                entities["indicators"].append(ind)

        return entities

    async def _gather_context(self, query: str, entities: Dict[str, List[str]]) -> Context:
        """컨텍스트 수집"""
        if self._context_gatherer:
            return await self._context_gatherer.gather(
                query=query,
                entities=entities,
                current_metrics=self._current_metrics
            )

        # 폴백 컨텍스트
        return Context(
            query=query,
            entities=entities,
            summary="컨텍스트 수집 불가"
        )

    async def _perform_inferences(self, entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """엔티티에 대한 추론 수행"""
        inferences = []

        if not self._reasoner:
            return inferences

        # 브랜드에 대한 추론
        for brand in entities.get("brands", []):
            brand_inferences = self._reasoner.infer(brand)
            inferences.extend([inf.to_dict() for inf in brand_inferences])

        return inferences

    async def _generate_response(
        self,
        query: str,
        context: Context,
        inferences: List[Dict[str, Any]]
    ) -> Response:
        """응답 생성"""
        if self._response_pipeline:
            # 추론 결과를 컨텍스트에 추가
            if inferences:
                context.kg_inferences = inferences

            decision = {"tool": "direct_answer", "reason": "쿼리 응답", "confidence": 0.8}
            return await self._response_pipeline.generate(
                query=query,
                context=context,
                decision=decision
            )

        # 폴백 응답
        return Response(
            content=f"'{query}'에 대한 정보를 찾을 수 없습니다.",
            confidence=0.3,
            is_fallback=True
        )

    def _assess_confidence(self, query: str, context: Context, response: Response) -> float:
        """신뢰도 평가"""
        if self._confidence_assessor:
            assessment = self._confidence_assessor.assess(
                query=query,
                context=context,
                response=response
            )
            return assessment.score

        return response.confidence

    def _update_stats(self, confidence: float, processing_time: float) -> None:
        """통계 업데이트"""
        total = self._stats["total_queries"]

        # 이동 평균
        self._stats["avg_confidence"] = (
            (self._stats["avg_confidence"] * (total - 1) + confidence) / total
        )
        self._stats["avg_processing_time"] = (
            (self._stats["avg_processing_time"] * (total - 1) + processing_time) / total
        )

    # =========================================================================
    # Knowledge Graph 쿼리
    # =========================================================================

    async def query_knowledge_graph(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Knowledge Graph 직접 쿼리

        Args:
            subject: 주어 패턴
            predicate: 술어 패턴
            obj: 목적어 패턴

        Returns:
            트리플 목록
        """
        if not self._initialized:
            await self.initialize()

        if not self._knowledge_graph:
            return []

        triples = self._knowledge_graph.query(subject, predicate, obj)
        return [
            {"subject": s, "predicate": p, "object": o}
            for s, p, o in triples
        ]

    async def get_entity_relations(self, entity: str) -> Dict[str, Any]:
        """
        엔티티의 모든 관계 조회

        Args:
            entity: 엔티티 이름

        Returns:
            관계 정보
        """
        if not self._initialized:
            await self.initialize()

        if not self._knowledge_graph:
            return {"entity": entity, "relations": []}

        # 주어로서의 관계
        as_subject = self._knowledge_graph.query(subject=entity)

        # 목적어로서의 관계
        as_object = self._knowledge_graph.query(obj=entity)

        return {
            "entity": entity,
            "outgoing": [{"predicate": p, "object": o} for _, p, o in as_subject],
            "incoming": [{"subject": s, "predicate": p} for s, p, _ in as_object]
        }

    # =========================================================================
    # 추론 쿼리
    # =========================================================================

    async def query_with_reasoning(self, query: str) -> Dict[str, Any]:
        """
        추론을 포함한 쿼리

        Args:
            query: 사용자 질문

        Returns:
            추론 결과 포함 응답
        """
        # 기본 처리
        result = await self.process(query)
        return result.to_dict()

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        return {
            **self._stats,
            "cache_hit_rate": (
                self._stats["cache_hits"] / self._stats["total_queries"]
                if self._stats["total_queries"] > 0 else 0
            )
        }

    def get_kg_stats(self) -> Dict[str, Any]:
        """Knowledge Graph 통계"""
        if not self._knowledge_graph:
            return {"initialized": False}

        stats = self._knowledge_graph.get_stats()
        stats["initialized"] = True
        return stats

    def clear_cache(self) -> None:
        """캐시 초기화"""
        if self._cache:
            self._cache.clear()

    @property
    def knowledge_graph(self) -> Optional[KnowledgeGraph]:
        """Knowledge Graph 인스턴스 반환"""
        return self._knowledge_graph

    @property
    def reasoner(self) -> Optional[OntologyReasoner]:
        """Reasoner 인스턴스 반환"""
        return self._reasoner

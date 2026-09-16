"""
Hybrid Retriever
================
Ontology + RAG 하이브리드 검색기 (지식 그래프 + 문서 검색 통합)

## 아키텍처 다이어그램
```
                        ┌─────────────────────┐
                        │     User Query      │
                        │  "LANEIGE 경쟁력?"  │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │  Entity Extraction  │
                        │ brands: ["LANEIGE"] │
                        │ categories: ["lip"] │
                        └──────────┬──────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Knowledge Graph │     │    Reasoner     │     │  RAG Document   │
│                 │     │                 │     │   Retriever     │
│ - 브랜드 제품   │     │ - 비즈니스 규칙 │     │                 │
│ - 경쟁 관계     │     │ - SoS 분석      │     │ - 지표 정의     │
│ - 카테고리 계층 │     │ - 경쟁력 추론   │     │ - 해석 가이드   │
│ - 감성 데이터   │     │ - 인사이트 생성 │     │ - 전략 플레이북 │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                      ┌──────────▼──────────┐
                      │    Context Merge    │
                      │                     │
                      │ 1. Ontology Facts   │
                      │ 2. Inferences       │
                      │ 3. RAG Chunks       │
                      │ 4. Category Context │
                      └──────────┬──────────┘
                                 │
                      ┌──────────▼──────────┐
                      │   HybridContext     │
                      │  (LLM 프롬프트용)   │
                      └─────────────────────┘
```

## 핵심 컴포넌트
1. **KnowledgeGraph**: 구조화된 관계 데이터 (브랜드-제품-카테고리)
2. **OntologyReasoner**: 비즈니스 규칙 기반 인사이트 추론
3. **DocumentRetriever**: 가이드라인 문서 키워드 검색 (docs/guides/)
4. **EntityExtractor**: 쿼리에서 브랜드/카테고리/지표 엔티티 추출

## 사용 예
```python
retriever = HybridRetriever(kg, reasoner, doc_retriever)
await retriever.initialize()

context = await retriever.retrieve(
    query="LANEIGE Lip Care 경쟁력 분석",
    current_metrics=dashboard_data
)

# context.ontology_facts: KG에서 조회한 사실
# context.inferences: 추론된 인사이트
# context.rag_chunks: RAG 문서 청크
# context.combined_context: LLM용 통합 컨텍스트
```

## 기능
1. 온톨로지에서 구조화된 지식 추론
2. RAG에서 비구조화된 가이드라인 검색
3. 두 결과를 통합하여 풍부한 컨텍스트 생성
4. 카테고리 계층 정보 포함
5. 감성 분석 데이터 통합

## Flow
Query → Entity Extraction → [Ontology Reasoning + RAG Search] → Context Merge → LLM
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.domain.value_objects.retrieval_result import UnifiedRetrievalResult

from src.core.intent import classify_intent as _unified_classify
from src.domain.entities.relations import InferenceResult
from src.monitoring.rag_metrics import RAGMetricsCollector
from src.ontology.inference_context import normalize_sentiment_clusters
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.ontology.rules import register_all_rules

from . import context_render, kg_facts, selfrag_gate
from .fusion import compute_fusion_confidence as _compute_fusion_confidence
from .fusion import load_retrieval_weights, weighted_merge
from .fusion.hybrid_search import bm25_actually_available, hybrid_search
from .hybrid_context import EntityExtractor, HybridContext
from .legacy_intent import (
    INTENT_DOC_TYPE_PRIORITY,
    QueryIntent,
    classify_intent,
    get_doc_type_filter,
)
from .query_expansion import QueryEnhancer, expand_query, rewrite_for_relevance
from .relevance_grader import RelevanceGrader
from .retriever import DocumentRetriever

# 로거 설정
logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Ontology + RAG 하이브리드 검색기 (파사드)

    자신은 조립만 하고, 각 책임은 별도 모듈이 갖는다:

    ==========================  =======================================
    책임                         모듈
    ==========================  =======================================
    Self-RAG 게이트              :mod:`src.rag.selfrag_gate`
    KG 조회 · 추론 컨텍스트       :mod:`src.rag.kg_facts`
    엣지 정렬 · 상한             :mod:`src.rag.kg_edges`
    쿼리 확장 · 재작성            :mod:`src.rag.query_expansion`
    RRF · 가중 병합              :mod:`src.rag.fusion`
    프롬프트 렌더링              :mod:`src.rag.context_render`
    문서 검색                    :mod:`src.rag.retriever`
    ==========================  =======================================

    동작 방식:
    1. 쿼리에서 엔티티 추출
    2. 지식 그래프에서 관련 사실 조회
    3. 온톨로지 추론 실행
    4. RAG 문서 검색 (추론 결과로 쿼리 확장)
    5. 결과 통합

    사용 예:
        retriever = HybridRetriever(kg, reasoner, doc_retriever)
        context = await retriever.retrieve(query, current_metrics)
    """

    # Self-RAG gate patterns (see src.rag.selfrag_gate)
    SKIP_PATTERNS = selfrag_gate.SKIP_PATTERNS
    RETRIEVE_PATTERNS = selfrag_gate.RETRIEVE_PATTERNS

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph | None = None,
        reasoner: OntologyReasoner | None = None,
        doc_retriever: DocumentRetriever | None = None,
        auto_init_rules: bool = True,
        owl_strategy: Any | None = None,
    ):
        """
        Args:
            knowledge_graph: 지식 그래프
            reasoner: 온톨로지 추론기
            doc_retriever: RAG 문서 검색기
            auto_init_rules: 비즈니스 규칙 자동 등록
            owl_strategy: OWLRetrievalStrategy 인스턴스 (옵션).
                          설정되면 retrieve_unified()에서 OWL 파이프라인을 사용.
        """
        # 컴포넌트 초기화
        # fallback 인스턴스는 읽기 전용 (정식 기록자는 daily_crawl의 exporter)
        self.kg = knowledge_graph or KnowledgeGraph(auto_save=False)
        self.reasoner = reasoner or OntologyReasoner(self.kg)
        self.doc_retriever = doc_retriever or DocumentRetriever()

        # OWL retrieval strategy (optional)
        self.owl_strategy = owl_strategy

        # 엔티티 추출기
        self.entity_extractor = EntityExtractor()

        # 관련성 판정기
        self.relevance_grader = RelevanceGrader()

        # 쿼리 강화기
        self.query_enhancer = QueryEnhancer()

        # 비즈니스 규칙 자동 등록
        if auto_init_rules and not self.reasoner.rules:
            register_all_rules(self.reasoner)
            logger.info(f"Registered {len(self.reasoner.rules)} business rules")

        # 검색 가중치 설정
        self._retrieval_weights = self._load_retrieval_weights()

        # RAG 메트릭 수집기
        self.rag_metrics = RAGMetricsCollector()

        # 초기화 상태
        self._initialized = False

    async def initialize(self) -> None:
        """비동기 초기화"""
        if not self._initialized:
            await self.doc_retriever.initialize()

            # 카테고리 계층 구조 로드 (지식그래프 강화)
            try:
                hierarchy_added = self.kg.load_category_hierarchy()
                if hierarchy_added > 0:
                    logger.info(f"Loaded category hierarchy: {hierarchy_added} relations added")
            except Exception as e:
                logger.warning(f"Failed to load category hierarchy: {e}")

            self._initialized = True

    def should_retrieve(self, query: str) -> tuple[bool, str, float]:
        """
        Self-RAG gate: determine if retrieval is needed.

        Returns:
            (should_retrieve, reason, confidence)
            confidence: 1.0 for strong domain queries, 0.8 for default, 0.0 for skip
        """
        return selfrag_gate.should_retrieve(query, self.SKIP_PATTERNS, self.RETRIEVE_PATTERNS)

    async def retrieve(
        self,
        query: str,
        current_metrics: dict[str, Any] | None = None,
        include_explanations: bool = True,
    ) -> HybridContext:
        """
        하이브리드 검색 수행

        Args:
            query: 사용자 쿼리
            current_metrics: 현재 계산된 지표 데이터
            include_explanations: 추론 설명 포함 여부

        Returns:
            HybridContext
        """
        # 초기화 확인
        if not self._initialized:
            await self.initialize()

        # Self-RAG gate (3-tuple: should, reason, confidence)
        should, reason, selfrag_confidence = self.should_retrieve(query)
        if not should:
            logger.info(f"Self-RAG: skipping retrieval for query (reason: {reason})")
            return HybridContext(
                query=query,
                ontology_facts=[],
                inferences=[],
                rag_chunks=[],
                combined_context=f"[Retrieval skipped: {reason}]",
                entities={},
                metadata={
                    "self_rag_skip": True,
                    "skip_reason": reason,
                    "selfrag_confidence": selfrag_confidence,
                },
            )

        start_time = datetime.now()

        # 결과 객체 초기화
        context = HybridContext(query=query)

        try:
            # 0. 쿼리 의도 분류 + 인텐트 기반 전략 선택
            query_intent = classify_intent(query)
            unified_intent = _unified_classify(query)

            from src.rag.retrieval_strategy import get_intent_retrieval_config

            intent_config = get_intent_retrieval_config(unified_intent)
            doc_type_filter = intent_config.doc_type_filter
            intent_top_k = intent_config.top_k

            # Self-RAG: reduce top_k for low-confidence queries
            if selfrag_confidence < 0.5:
                intent_top_k = max(2, intent_top_k // 2)
                logger.info(
                    f"Self-RAG: reduced top_k to {intent_top_k} "
                    f"(confidence={selfrag_confidence:.1f})"
                )

            logger.debug(
                f"Query intent: {query_intent.value}, "
                f"strategy: {intent_config.description}, "
                f"weights: {intent_config.weights}, top_k: {intent_top_k}"
            )

            # 1. 엔티티 추출 (지식 그래프 전달로 제품 ASIN도 추출 가능)
            entities = self.entity_extractor.extract(query, knowledge_graph=self.kg)
            context.entities = entities
            logger.debug(f"Extracted entities: {entities}")

            # 1.5. 쿼리 사전 강화 (동의어 확장)
            enhanced = self.query_enhancer.enhance(query, entities)
            search_query = enhanced.search_query
            logger.debug(f"Enhanced query: {search_query}")

            from src.infrastructure.feature_flags import FeatureFlags

            flags = FeatureFlags.get_instance()

            # 2. 지식 그래프에서 사실 조회 (ablation no-kg: FF_ONTOLOGY_USE_ONTOLOGY_KG=false)
            if flags.use_ontology_kg():
                ontology_facts = self._query_knowledge_graph(entities)
            else:
                logger.info("KG query disabled by feature flag (use_ontology_kg=false)")
                ontology_facts = []
            context.ontology_facts = ontology_facts

            # 3. 추론 컨텍스트 구성
            inference_context = self._build_inference_context(entities, current_metrics or {})

            # 4. 온톨로지 추론 실행 (ablation no-ontology: reasoner 플래그 둘 다 false)
            if flags.use_unified_reasoner() or flags.use_owl_reasoner():
                inferences = self.reasoner.infer(inference_context)
            else:
                logger.info("Ontology inference disabled by feature flags")
                inferences = []
            context.inferences = inferences
            logger.debug(f"Generated {len(inferences)} inferences")

            # 5. RAG 문서 검색 (추론 결과로 쿼리 확장 + 의도 기반 필터링)
            #    Uses hybrid (dense + BM25 RRF) when BM25 is available
            expanded_query = self._expand_query(search_query, inferences, entities)
            rag_results, search_method = await self._hybrid_search(
                expanded_query, top_k=intent_top_k, doc_type_filter=doc_type_filter
            )

            # 필터링된 결과가 부족하면 전체 문서에서 추가 검색
            if len(rag_results) < 3 and doc_type_filter:
                additional_results, _fallback_method = await self._hybrid_search(
                    expanded_query,
                    top_k=intent_top_k - len(rag_results),
                    doc_type_filter=None,  # 전체 문서에서 검색
                )
                # 중복 제거하며 추가 (BM25 결과처럼 "id" 가 없는 항목도 안전하게 처리)
                existing_keys = {self._chunk_key(r) for r in rag_results}
                for result in additional_results:
                    key = self._chunk_key(result)
                    if key not in existing_keys:
                        rag_results.append(result)
                        existing_keys.add(key)

            context.rag_chunks = rag_results

            # 5.5. 관련성 검증 (Relevance Grading)
            try:
                from src.infrastructure.feature_flags import FeatureFlags

                if not FeatureFlags.get_instance().use_reranker():
                    logger.info("Reranker disabled by feature flag, skipping relevance grading")
                    raise RuntimeError("reranker disabled")  # jump to except → keep originals

                relevant_docs, irrelevant_docs = await self.relevance_grader.grade_documents(
                    query, rag_results
                )
                if self.relevance_grader.needs_rewrite(len(relevant_docs)):
                    # 관련 문서 부족 → 쿼리 재작성 후 재검색 (최대 1회)
                    logger.info(
                        f"Relevance grading: only {len(relevant_docs)} relevant docs, "
                        f"attempting query rewrite"
                    )
                    rewritten_query = self._rewrite_for_relevance(query, entities)
                    if rewritten_query != query:
                        additional_results = await self.doc_retriever.search(
                            rewritten_query,
                            top_k=intent_top_k,
                            doc_type_filter=doc_type_filter,
                        )
                        # 기존 관련 문서 + 새 검색 결과 병합
                        existing_ids = {r.get("id") for r in relevant_docs}
                        for result in additional_results:
                            if result.get("id") not in existing_ids:
                                relevant_docs.append(result)
                        logger.info(f"After rewrite: {len(relevant_docs)} relevant docs")

                context.rag_chunks = relevant_docs
            except Exception as e:
                logger.warning(f"Relevance grading skipped: {e}")
                # 실패 시 원본 결과 유지

            # 5.8. RAG 메트릭 기록
            try:
                retrieval_time = (datetime.now() - start_time).total_seconds() * 1000
                self.rag_metrics.record_retrieval(
                    query=query,
                    chunks=rag_results,
                    relevant_chunks=context.rag_chunks
                    if context.rag_chunks != rag_results
                    else None,
                    retrieval_time_ms=retrieval_time,
                )
            except Exception as e:
                logger.debug(f"RAG metrics recording failed: {e}")

            # 5.7. 가중치 기반 병합 (인텐트 전략 가중치 적용)
            context = self._weighted_merge(context, intent_weights=intent_config.weights)

            # 6. 통합 컨텍스트 생성
            context.combined_context = self._combine_contexts(context, include_explanations)

            # 메타데이터 (_weighted_merge 가 넣은 "weighted_scores"/"fusion" 은 보존)
            if not isinstance(context.metadata, dict):
                context.metadata = {}
            context.metadata.update(
                {
                    "retrieval_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                    "ontology_facts_count": len(ontology_facts),
                    "inferences_count": len(inferences),
                    "rag_chunks_count": len(rag_results),
                    "query_expanded": expanded_query != query,
                    "query_intent": query_intent.value,
                    "doc_type_filter": doc_type_filter,
                    "intent_strategy": intent_config.description,
                    "intent_weights": intent_config.weights,
                    "search_method": search_method,
                    "selfrag_confidence": selfrag_confidence,
                    "bm25_available": self._bm25_actually_available(),
                }
            )

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            context.metadata["error"] = str(e)

        return context

    async def retrieve_unified(
        self,
        query: str,
        current_metrics: dict[str, Any] | None = None,
        top_k: int = 5,
        **kwargs: Any,
    ) -> UnifiedRetrievalResult:
        """
        통합 검색 — 모든 백엔드에서 UnifiedRetrievalResult 반환.

        OWL strategy가 설정되어 있으면 OWL 파이프라인 사용,
        아니면 legacy retrieve() 결과를 변환.

        Args:
            query: 사용자 쿼리
            current_metrics: 현재 메트릭 데이터
            top_k: 반환할 최대 결과 수
            **kwargs: 추가 인자

        Returns:
            UnifiedRetrievalResult
        """
        from src.domain.value_objects.retrieval_result import UnifiedRetrievalResult

        # Self-RAG 게이트 — OWL 경로 포함 모든 unified 검색에 적용
        # (인사/도움말 등 검색 불필요 쿼리는 검색 자체를 생략)
        should, reason, selfrag_confidence = self.should_retrieve(query)
        if not should:
            logger.info(f"Self-RAG: skipping unified retrieval (reason: {reason})")
            return UnifiedRetrievalResult(
                query=query,
                entities={},
                ontology_facts=[],
                inferences=[],
                rag_chunks=[],
                combined_context=f"[Retrieval skipped: {reason}]",
                confidence=selfrag_confidence,
                entity_links=[],
                metadata={
                    "self_rag_skip": True,
                    "skip_reason": reason,
                    "selfrag_confidence": selfrag_confidence,
                },
                retriever_type="selfrag_skip",
            )

        # OWL strategy가 있으면 위임
        if self.owl_strategy is not None:
            return await self.owl_strategy.retrieve(
                query=query,
                current_metrics=current_metrics,
                top_k=top_k,
                **kwargs,
            )

        # Legacy path: retrieve() → HybridContext → UnifiedRetrievalResult 변환
        ctx = await self.retrieve(
            query=query,
            current_metrics=current_metrics,
            include_explanations=kwargs.get("include_explanations", True),
        )

        # InferenceResult → dict 변환
        inferences_dicts = []
        for inf in ctx.inferences:
            if hasattr(inf, "to_dict"):
                inferences_dicts.append(inf.to_dict())
            elif isinstance(inf, dict):
                inferences_dicts.append(inf)

        return UnifiedRetrievalResult(
            query=query,
            entities=ctx.entities,
            ontology_facts=ctx.ontology_facts,
            inferences=inferences_dicts,
            rag_chunks=ctx.rag_chunks,
            combined_context=ctx.combined_context,
            confidence=0.0,
            entity_links=[],
            metadata=ctx.metadata,
            retriever_type="legacy",
        )

    async def search(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """문서 검색 (RetrieverProtocol 호환).

        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 결과 수
            doc_filter: 문서 필터

        Returns:
            검색된 문서 목록
        """
        if self.owl_strategy is not None and hasattr(self.owl_strategy, "search"):
            return await self.owl_strategy.search(query=query, top_k=top_k, doc_filter=doc_filter)
        return await self.doc_retriever.search(query=query, top_k=top_k, doc_filter=doc_filter)

    def _bm25_actually_available(self) -> bool:
        """BM25 sparse 검색 가용 여부 (:mod:`src.rag.fusion.hybrid_search` 로 위임)"""
        return bm25_actually_available(self.doc_retriever)

    async def _hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        doc_type_filter: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """Dense + BM25 hybrid search with RRF fusion.

        (:mod:`src.rag.fusion.hybrid_search` 로 위임)

        Returns:
            (results, search_method) where search_method is
            "hybrid_rrf" or "dense_only"
        """
        return await hybrid_search(self.doc_retriever, query, top_k, doc_type_filter)

    def _query_knowledge_graph(self, entities: dict[str, list[str]]) -> list[dict[str, Any]]:
        """지식 그래프에서 관련 사실 조회 (:mod:`src.rag.kg_facts` 로 위임)."""
        return kg_facts.query_knowledge_graph(self.kg, entities)

    @staticmethod
    def _chunk_key(chunk: Any) -> tuple[str, Any]:
        """중복 제거용 청크 키: "id" 가 있으면 id, 없으면 content 로 대체."""
        if not isinstance(chunk, dict):
            return ("obj", id(chunk))
        chunk_id = chunk.get("id")
        if chunk_id is not None:
            return ("id", chunk_id)
        return ("content", chunk.get("content"))

    @staticmethod
    def _normalize_sentiment_clusters(clusters: Any) -> dict[str, Any]:
        """감성 클러스터 정규화 (``src.ontology.inference_context`` 로 위임)."""
        return normalize_sentiment_clusters(clusters)

    def _build_inference_context(
        self, entities: dict[str, list[str]], current_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """추론용 컨텍스트 구성 (:mod:`src.rag.kg_facts` 로 위임)."""
        return kg_facts.build_retrieval_context(self.kg, entities, current_metrics)

    def _expand_query(
        self, query: str, inferences: list[InferenceResult], entities: dict[str, list[str]]
    ) -> str:
        """추론 결과 기반 쿼리 확장 (:mod:`src.rag.query_expansion` 로 위임)."""
        return expand_query(query, inferences, entities)

    def _rewrite_for_relevance(self, query: str, entities: dict) -> str:
        """관련성 부족 시 쿼리 재작성 (:mod:`src.rag.query_expansion` 로 위임)."""
        return rewrite_for_relevance(query, entities)

    def _load_retrieval_weights(self) -> dict:
        """config/retrieval_weights.json에서 가중치 로드 (:mod:`src.rag.fusion` 로 위임)."""
        return load_retrieval_weights()

    def _weighted_merge(
        self,
        context: HybridContext,
        intent_weights: dict[str, float] | None = None,
    ) -> HybridContext:
        """가중치 기반 컨텍스트 병합 (:mod:`src.rag.fusion` 로 위임).

        가중치 우선순위:
        1. intent_weights (인텐트 기반 전략에서 전달)
        2. config/retrieval_weights.json (파일 설정)
        3. 기본값: kg=0.4, rag=0.4, inference=0.2
        """
        return weighted_merge(context, self._retrieval_weights, intent_weights)

    def _compute_fusion_confidence(
        self,
        context: HybridContext,
        intent_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """ConfidenceFusion 신뢰도 + 충돌 감지 (:mod:`src.rag.fusion` 로 위임)."""
        return _compute_fusion_confidence(context, intent_weights)

    def _combine_contexts(self, context: HybridContext, include_explanations: bool = True) -> str:
        """온톨로지 + RAG 컨텍스트 통합 (:mod:`src.rag.context_render` 로 위임)."""
        return context_render.combine_contexts(context, include_explanations)

    def update_knowledge_graph(
        self, crawl_data: dict[str, Any] | None = None, metrics_data: dict[str, Any] | None = None
    ) -> dict[str, int]:
        """
        지식 그래프 업데이트

        Args:
            crawl_data: 크롤링 데이터
            metrics_data: 메트릭 데이터

        Returns:
            업데이트 통계
        """
        stats = {"crawl_relations": 0, "metrics_relations": 0}

        if crawl_data:
            stats["crawl_relations"] = self.kg.load_from_crawl_data(crawl_data)

        if metrics_data:
            stats["metrics_relations"] = self.kg.load_from_metrics_data(metrics_data)

        logger.info(f"KG updated: {stats}")
        return stats

    def get_stats(self) -> dict[str, Any]:
        """검색기 통계"""
        return {
            "knowledge_graph": self.kg.get_stats(),
            "reasoner": self.reasoner.get_inference_stats(),
            "rules_count": len(self.reasoner.rules),
            "rag_metrics": self.rag_metrics.get_metrics(),
            "initialized": self._initialized,
        }


__all__ = [
    "INTENT_DOC_TYPE_PRIORITY",
    "EntityExtractor",
    "HybridContext",
    "HybridRetriever",
    "QueryIntent",
    "classify_intent",
    "get_doc_type_filter",
]

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
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.domain.value_objects.retrieval_result import UnifiedRetrievalResult

from src.domain.entities.relations import InferenceResult, InsightType, RelationType
from src.monitoring.rag_metrics import RAGMetricsCollector
from src.ontology.business_rules import register_all_rules
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner

from .query_enhancer import QueryEnhancer
from .relevance_grader import RelevanceGrader
from .retriever import DocumentRetriever

# 로거 설정
logger = logging.getLogger(__name__)


# ============================================================================
# Query Intent Classification
# Delegates to unified classifier (src/core/intent.py).
# QueryIntent enum and helpers are kept for backward compatibility.
# ============================================================================

from src.core.intent import classify_intent as _unified_classify
from src.core.intent import to_query_intent as _to_query_intent


class QueryIntent(Enum):
    """쿼리 의도 분류 (backward compat - delegates to UnifiedIntent)"""

    DIAGNOSIS = "diagnosis"  # 원인 분석 → Type A (플레이북) 우선
    TREND = "trend"  # 트렌드 → Type B (인텔리전스) 우선
    CRISIS = "crisis"  # 위기 대응 → Type C (대응 가이드) 우선
    METRIC = "metric"  # 지표 해석 → Type D (기존 가이드) 우선
    GENERAL = "general"  # 일반 → 모든 문서


# 의도별 우선 검색 문서 유형 매핑
INTENT_DOC_TYPE_PRIORITY = {
    QueryIntent.DIAGNOSIS: ["playbook", "metric_guide", "intelligence"],
    QueryIntent.TREND: ["intelligence", "knowledge_base", "response_guide"],
    QueryIntent.CRISIS: ["response_guide", "intelligence", "playbook"],
    QueryIntent.METRIC: ["metric_guide", "playbook"],
    QueryIntent.GENERAL: None,  # 모든 문서 검색
}


def classify_intent(query: str) -> QueryIntent:
    """
    쿼리 의도 분류 - delegates to unified classifier.

    Args:
        query: 사용자 쿼리

    Returns:
        QueryIntent enum 값

    Note:
        키워드 우선순위: TREND > CRISIS > DIAGNOSIS > METRIC > GENERAL
        트렌드/위기 키워드가 있으면 분석 키워드보다 우선
    """
    unified = _unified_classify(query)
    value = _to_query_intent(unified)
    try:
        return QueryIntent(value)
    except ValueError:
        return QueryIntent.GENERAL


def get_doc_type_filter(intent: QueryIntent) -> list[str] | None:
    """
    의도에 따른 문서 유형 필터 반환

    Args:
        intent: 쿼리 의도

    Returns:
        우선 검색할 문서 유형 리스트 (None이면 모든 문서)
    """
    return INTENT_DOC_TYPE_PRIORITY.get(intent)


@dataclass
class HybridContext:
    """
    하이브리드 검색 결과

    Attributes:
        query: 원본 쿼리
        entities: 추출된 엔티티
        ontology_facts: 지식 그래프에서 조회한 사실
        inferences: 온톨로지 추론 결과
        rag_chunks: RAG 검색 결과 청크
        combined_context: 통합된 컨텍스트 (LLM 프롬프트용)
        metadata: 추가 메타데이터
    """

    query: str
    entities: dict[str, list[str]] = field(default_factory=dict)
    ontology_facts: list[dict[str, Any]] = field(default_factory=list)
    inferences: list[InferenceResult] = field(default_factory=list)
    rag_chunks: list[dict[str, Any]] = field(default_factory=list)
    combined_context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "query": self.query,
            "entities": self.entities,
            "ontology_facts": self.ontology_facts,
            "inferences": [inf.to_dict() for inf in self.inferences],
            "rag_chunks": self.rag_chunks,
            "combined_context": self.combined_context,
            "metadata": self.metadata,
        }


class EntityExtractor:
    """
    쿼리에서 엔티티 추출 (thin wrapper around EntityLinker).

    All extraction is delegated to EntityLinker.extract_entities().
    """

    def __init__(self) -> None:
        from src.rag.entity_linker import EntityLinker

        self._linker = EntityLinker(use_spacy=False)

    @classmethod
    def get_known_brands(cls) -> list:
        """EntityLinker에서 브랜드 목록 가져오기 (이름 + 별칭 평탄화)"""
        from src.rag.entity_linker import EntityLinker

        return list(EntityLinker(use_spacy=False)._get_merged_brands().keys())

    @classmethod
    def get_brand_normalization_map(cls) -> dict:
        """EntityLinker에서 별칭 → 정규화된 브랜드명 매핑"""
        from src.rag.entity_linker import EntityLinker

        return EntityLinker(use_spacy=False)._get_merged_brands()

    def extract(self, query: str, knowledge_graph=None) -> dict[str, list[str]]:
        """
        쿼리에서 엔티티 추출. Delegates to EntityLinker.extract_entities().

        Args:
            query: 사용자 쿼리
            knowledge_graph: 지식 그래프 (제품 검색용, optional)

        Returns:
            {
                "brands": [...],
                "categories": [...],
                "indicators": [...],
                "time_range": [...],
                "products": [...],
                "sentiments": [...],
                "sentiment_clusters": [...]
            }
        """
        return self._linker.extract_entities(query, knowledge_graph=knowledge_graph)


class HybridRetriever:
    """
    Ontology + RAG 하이브리드 검색기

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

    # Self-RAG: patterns that indicate retrieval is NOT needed
    SKIP_PATTERNS = [
        # Greetings (no \b for Korean; Korean chars are not word-boundary friendly)
        r"^(안녕|하이|헬로)",
        r"^(hi|hello|hey)\b",
        # Thanks
        r"^(고마워|감사|thanks|thank you)",
        # System commands
        r"^(도움말|설정|help|config)",
    ]

    # Self-RAG: patterns that indicate retrieval IS needed
    RETRIEVE_PATTERNS = [
        # Brand names
        r"(?i)(laneige|cosrx|anua|tirtir|round\s*lab|innisfree|sulwhasoo)",
        # Metrics
        r"(?i)(sos|hhi|cpi|share\s*of\s*shelf|순위|rank|점유율)",
        # Analysis keywords
        r"(분석|비교|전략|경쟁|트렌드|시장|매출|성장)",
        # Question words
        r"(왜|어떻게|뭐|몇|어디|언제|무엇|how|what|why|which)",
    ]

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
        self.kg = knowledge_graph or KnowledgeGraph()
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
        import re

        if not query or len(query.strip()) <= 2:
            return False, "query_too_short", 0.0

        query_stripped = query.strip()

        # Check skip patterns first
        for pattern in self.SKIP_PATTERNS:
            if re.search(pattern, query_stripped, re.IGNORECASE):
                return False, "greeting_or_command", 0.0

        # Check retrieve patterns
        for pattern in self.RETRIEVE_PATTERNS:
            if re.search(pattern, query_stripped):
                return True, "domain_query_detected", 1.0

        # Default: retrieve (conservative)
        if len(query_stripped) > 5:
            return True, "default_retrieve", 0.8

        return False, "short_non_domain_query", 0.0

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

            # 2. 지식 그래프에서 사실 조회
            ontology_facts = self._query_knowledge_graph(entities)
            context.ontology_facts = ontology_facts

            # 3. 추론 컨텍스트 구성
            inference_context = self._build_inference_context(entities, current_metrics or {})

            # 4. 온톨로지 추론 실행
            inferences = self.reasoner.infer(inference_context)
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
                # 중복 제거하며 추가
                existing_ids = {r["id"] for r in rag_results}
                for result in additional_results:
                    if result["id"] not in existing_ids:
                        rag_results.append(result)

            context.rag_chunks = rag_results

            # 5.5. 관련성 검증 (Relevance Grading)
            try:
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

            # 메타데이터
            context.metadata = {
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
                "bm25_available": hasattr(self.doc_retriever, "search_bm25"),
            }

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

    async def _hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        doc_type_filter: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """
        Dense + BM25 hybrid search with RRF fusion.

        Args:
            query: Search query
            top_k: Number of results to return
            doc_type_filter: Optional document type filter

        Returns:
            (results, search_method) where search_method is
            "hybrid_rrf" or "dense_only"
        """
        # 1. Dense search via doc_retriever.search()
        dense_results = await self.doc_retriever.search(
            query, top_k=top_k, doc_type_filter=doc_type_filter
        )

        # 2. BM25 search (if available)
        bm25_results = []
        if hasattr(self.doc_retriever, "search_bm25"):
            try:
                bm25_results = self.doc_retriever.search_bm25(query, top_k=top_k)
            except Exception as e:
                logger.debug(f"BM25 search failed in _hybrid_search: {e}")

        # 3. RRF fusion
        if bm25_results:
            if hasattr(self.doc_retriever, "reciprocal_rank_fusion"):
                fused = self.doc_retriever.reciprocal_rank_fusion(
                    dense_results, bm25_results, k=60, top_k=top_k
                )
                return fused, "hybrid_rrf"
            # Fallback: try confidence_fusion.fuse_documents_rrf
            try:
                from src.rag.confidence_fusion import ConfidenceFusion

                fusion = ConfidenceFusion()
                fused = fusion.fuse_documents_rrf(
                    {"dense": dense_results, "bm25": bm25_results},
                    k=60,
                    top_n=top_k,
                )
                return fused, "hybrid_rrf"
            except (ImportError, Exception) as e:
                logger.debug(f"Confidence fusion RRF fallback failed: {e}")

        return dense_results, "dense_only"

    def _query_knowledge_graph(self, entities: dict[str, list[str]]) -> list[dict[str, Any]]:
        """
        지식 그래프에서 관련 사실 조회

        Args:
            entities: 추출된 엔티티

        Returns:
            사실 리스트
        """
        facts = []

        # 브랜드 관련 사실
        for brand in entities.get("brands", []):
            # 브랜드 메타데이터
            brand_meta = self.kg.get_entity_metadata(brand)
            if brand_meta:
                facts.append({"type": "brand_info", "entity": brand, "data": brand_meta})

            # 브랜드의 제품들
            products = self.kg.get_brand_products(brand)
            if products:
                facts.append(
                    {
                        "type": "brand_products",
                        "entity": brand,
                        "data": {
                            "product_count": len(products),
                            "products": products[:10],  # 상위 10개
                        },
                    }
                )

            # 경쟁사
            competitors = self.kg.get_competitors(brand)
            if competitors:
                facts.append(
                    {
                        "type": "competitors",
                        "entity": brand,
                        "data": competitors[:5],  # 상위 5개
                    }
                )

            # 경쟁사 네트워크 (직/간접 이웃)
            try:
                network = self.kg.get_neighbors(
                    brand,
                    direction="both",
                    predicate_filter=[
                        RelationType.COMPETES_WITH,
                        RelationType.DIRECT_COMPETITOR,
                        RelationType.INDIRECT_COMPETITOR,
                    ],
                )
                if network.get("outgoing") or network.get("incoming"):
                    facts.append(
                        {
                            "type": "competitor_network",
                            "entity": brand,
                            "data": {
                                "outgoing": network.get("outgoing", [])[:10],
                                "incoming": network.get("incoming", [])[:10],
                            },
                        }
                    )
            except Exception:
                logger.warning("Suppressed Exception", exc_info=True)

            # 트렌드 키워드 (브랜드 우선, 없으면 MARKET)
            trend_relations = self.kg.query(subject=brand, predicate=RelationType.HAS_TREND)
            if not trend_relations:
                trend_relations = self.kg.query(subject="MARKET", predicate=RelationType.HAS_TREND)
            if trend_relations:
                trend_keywords = [rel.object for rel in trend_relations[:10]]
                facts.append(
                    {
                        "type": "trend_keywords",
                        "entity": brand,
                        "data": {"keywords": trend_keywords, "count": len(trend_relations)},
                    }
                )

        # 카테고리 관련 사실
        for category in entities.get("categories", []):
            # 카테고리 브랜드 정보
            category_brands = self.kg.get_category_brands(category)
            if category_brands:
                facts.append(
                    {
                        "type": "category_brands",
                        "entity": category,
                        "data": {
                            "brand_count": len(category_brands),
                            "top_brands": category_brands[:5],
                        },
                    }
                )

            # 카테고리 계층 정보 (부모/자식 관계)
            try:
                hierarchy = self.kg.get_category_hierarchy(category)
                if hierarchy and not hierarchy.get("error"):
                    facts.append(
                        {
                            "type": "category_hierarchy",
                            "entity": category,
                            "data": {
                                "name": hierarchy.get("name", ""),
                                "level": hierarchy.get("level", 0),
                                "path": hierarchy.get("path", []),
                                "ancestors": hierarchy.get("ancestors", []),
                                "descendants": hierarchy.get("descendants", []),
                            },
                        }
                    )
            except Exception:
                logger.warning("Suppressed Exception", exc_info=True)

        # 감성 관련 사실 조회
        sentiment_clusters = entities.get("sentiment_clusters", [])
        if sentiment_clusters or entities.get("sentiments"):
            # 제품이 지정된 경우 해당 제품의 감성 조회
            for asin in entities.get("products", []):
                try:
                    product_sentiments = self.kg.get_product_sentiments(asin)
                    if product_sentiments.get("sentiment_tags") or product_sentiments.get(
                        "ai_summary"
                    ):
                        facts.append(
                            {
                                "type": "product_sentiment",
                                "entity": asin,
                                "data": product_sentiments,
                            }
                        )
                except Exception:
                    logger.warning("Suppressed Exception", exc_info=True)

            # 브랜드가 지정된 경우 브랜드 감성 프로필 조회
            for brand in entities.get("brands", []):
                try:
                    brand_sentiment = self.kg.get_brand_sentiment_profile(brand)
                    if brand_sentiment.get("all_tags"):
                        facts.append(
                            {"type": "brand_sentiment", "entity": brand, "data": brand_sentiment}
                        )
                except Exception:
                    logger.warning("Suppressed Exception", exc_info=True)

            # 특정 감성 클러스터로 제품 검색
            for cluster in sentiment_clusters:
                if cluster not in ["sentiment_general", "ai_summary"]:
                    try:
                        # 해당 감성을 가진 제품 찾기
                        from src.domain.entities.relations import SENTIMENT_CLUSTERS

                        cluster_tags = SENTIMENT_CLUSTERS.get(cluster, [])
                        for tag in cluster_tags[:2]:  # 상위 2개 태그만
                            products_with_sentiment = self.kg.find_products_by_sentiment(tag)
                            if products_with_sentiment:
                                facts.append(
                                    {
                                        "type": "sentiment_products",
                                        "entity": tag,
                                        "data": {
                                            "sentiment_tag": tag,
                                            "cluster": cluster,
                                            "product_count": len(products_with_sentiment),
                                            "products": products_with_sentiment[:5],
                                        },
                                    }
                                )
                                break
                    except Exception:
                        logger.warning("Suppressed Exception", exc_info=True)

        return facts

    def _build_inference_context(
        self, entities: dict[str, list[str]], current_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """
        추론용 컨텍스트 구성

        Args:
            entities: 추출된 엔티티
            current_metrics: 현재 지표 데이터

        Returns:
            추론 컨텍스트
        """
        context = {}

        # 엔티티 정보
        if entities.get("brands"):
            context["brand"] = entities["brands"][0]  # 첫 번째 브랜드
            context["is_target"] = entities["brands"][0].lower() == "laneige"

        if entities.get("categories"):
            context["category"] = entities["categories"][0]

        # 메트릭 정보 (summary에서)
        summary = current_metrics.get("summary", {})

        # 브랜드별 SoS
        sos_by_category = summary.get("laneige_sos_by_category", {})
        if entities.get("categories") and entities["categories"][0] in sos_by_category:
            context["sos"] = sos_by_category[entities["categories"][0]]
        elif sos_by_category:
            # 첫 번째 카테고리의 SoS
            context["sos"] = list(sos_by_category.values())[0] if sos_by_category else 0

        # 브랜드 메트릭에서 추가 정보
        brand_metrics = current_metrics.get("brand_metrics", [])
        for bm in brand_metrics:
            if (
                bm.get("is_laneige")
                or bm.get("brand_name", "").lower() == context.get("brand", "").lower()
            ):
                context["sos"] = bm.get("share_of_shelf", context.get("sos", 0))
                context["avg_rank"] = bm.get("avg_rank")
                context["product_count"] = bm.get("product_count", 0)
                break

        # 마켓 메트릭에서 HHI 등
        market_metrics = current_metrics.get("market_metrics", [])
        for mm in market_metrics:
            if not entities.get("categories") or mm.get("category_id") == entities["categories"][0]:
                context["hhi"] = mm.get("hhi", 0)
                context["cpi"] = mm.get("cpi", 100)
                context["churn_rate"] = mm.get("churn_rate_7d", 0)
                context["rating_gap"] = mm.get("avg_rating_gap", 0)
                break

        # 제품 메트릭에서
        product_metrics = current_metrics.get("product_metrics", [])
        if product_metrics:
            # 첫 번째 제품 또는 가장 좋은 순위 제품
            best_product = min(product_metrics, key=lambda p: p.get("current_rank", 100))
            context["current_rank"] = best_product.get("current_rank")
            context["rank_change_1d"] = best_product.get("rank_change_1d")
            context["rank_change_7d"] = best_product.get("rank_change_7d")
            context["rank_volatility"] = best_product.get("rank_volatility", 0)
            context["streak_days"] = best_product.get("streak_days", 0)
            context["asin"] = best_product.get("asin")

        # 알림 정보
        alerts = current_metrics.get("alerts", [])
        context["has_rank_shock"] = any(a.get("type") == "rank_shock" for a in alerts)
        context["alert_count"] = len(alerts)

        # 경쟁사 수 (지식 그래프에서)
        if context.get("brand"):
            competitors = self.kg.get_competitors(context["brand"])
            context["competitor_count"] = len(competitors)
            context["competitors"] = competitors

            # 트렌드 키워드 (브랜드 우선, 없으면 MARKET)
            trend_relations = self.kg.query(
                subject=context["brand"], predicate=RelationType.HAS_TREND
            )
            if not trend_relations:
                trend_relations = self.kg.query(subject="MARKET", predicate=RelationType.HAS_TREND)
            if trend_relations:
                context["trend_keywords"] = [rel.object for rel in trend_relations[:10]]

        # 감성 데이터 (지식 그래프에서)
        if entities.get("sentiments") or entities.get("sentiment_clusters"):
            # 자사 브랜드 감성 프로필
            if context.get("brand"):
                try:
                    brand_sentiment = self.kg.get_brand_sentiment_profile(context["brand"])
                    context["sentiment_tags"] = brand_sentiment.get("all_tags", [])
                    context["sentiment_clusters"] = brand_sentiment.get("clusters", {})
                    context["dominant_sentiment"] = brand_sentiment.get("dominant_sentiment")
                except Exception:
                    logger.warning("Suppressed Exception", exc_info=True)

            # 제품별 감성 데이터
            if context.get("asin"):
                try:
                    product_sentiment = self.kg.get_product_sentiments(context["asin"])
                    context["ai_summary"] = product_sentiment.get("ai_summary")
                    if not context.get("sentiment_tags"):
                        context["sentiment_tags"] = product_sentiment.get("sentiment_tags", [])
                        context["sentiment_clusters"] = product_sentiment.get(
                            "sentiment_clusters", {}
                        )
                except Exception:
                    logger.warning("Suppressed Exception", exc_info=True)

            # 경쟁사 감성 데이터 (비교용)
            if context.get("competitors"):
                competitor_tags = []
                competitor_clusters = {}
                for comp in context["competitors"][:3]:  # 상위 3개 경쟁사
                    comp_brand = comp.get("brand", comp) if isinstance(comp, dict) else comp
                    try:
                        comp_sentiment = self.kg.get_brand_sentiment_profile(comp_brand)
                        competitor_tags.extend(comp_sentiment.get("all_tags", []))
                        for cluster, count in comp_sentiment.get("clusters", {}).items():
                            competitor_clusters[cluster] = (
                                competitor_clusters.get(cluster, 0) + count
                            )
                    except Exception:
                        logger.warning("Suppressed Exception", exc_info=True)
                context["competitor_sentiment_tags"] = list(set(competitor_tags))
                context["competitor_sentiment_clusters"] = competitor_clusters

        return context

    def _expand_query(
        self, query: str, inferences: list[InferenceResult], entities: dict[str, list[str]]
    ) -> str:
        """
        추론 결과 기반 쿼리 확장

        Args:
            query: 원본 쿼리
            inferences: 추론 결과
            entities: 엔티티

        Returns:
            확장된 쿼리
        """
        expanded = query
        expansion_terms = []

        # 추론된 인사이트 유형에 따라 검색 키워드 추가
        insight_types = {inf.insight_type for inf in inferences}

        if (
            InsightType.MARKET_POSITION in insight_types
            or InsightType.MARKET_DOMINANCE in insight_types
        ):
            expansion_terms.append("시장 포지션 해석")

        if InsightType.RISK_ALERT in insight_types:
            expansion_terms.append("위험 신호 대응")

        if InsightType.COMPETITIVE_THREAT in insight_types:
            expansion_terms.append("경쟁 위협 분석")

        if (
            InsightType.GROWTH_OPPORTUNITY in insight_types
            or InsightType.GROWTH_MOMENTUM in insight_types
        ):
            expansion_terms.append("성장 기회 전략")

        if (
            InsightType.PRICE_QUALITY_GAP in insight_types
            or InsightType.PRICE_POSITION in insight_types
        ):
            expansion_terms.append("가격 전략 해석")

        # 지표 관련 확장
        for indicator in entities.get("indicators", []):
            if indicator == "sos":
                expansion_terms.append("SoS 점유율 해석")
            elif indicator == "hhi":
                expansion_terms.append("HHI 시장집중도 해석")
            elif indicator == "cpi":
                expansion_terms.append("CPI 가격지수 해석")

        if expansion_terms:
            expanded = f"{query} {' '.join(expansion_terms)}"

        return expanded

    def _rewrite_for_relevance(self, query: str, entities: dict) -> str:
        """
        관련성 부족 시 쿼리 재작성

        엔티티 정보를 활용하여 더 구체적인 검색 쿼리를 생성합니다.

        Args:
            query: 원본 쿼리
            entities: 추출된 엔티티

        Returns:
            재작성된 쿼리
        """
        parts = [query]

        # 브랜드 추가
        brands = entities.get("brands", [])
        if brands and brands[0].lower() not in query.lower():
            parts.append(brands[0])

        # 지표 추가
        indicators = entities.get("indicators", [])
        if indicators:
            indicator_names = {
                "sos": "Share of Shelf 점유율",
                "hhi": "HHI 시장집중도",
                "cpi": "CPI 가격지수",
            }
            for ind in indicators[:2]:
                full_name = indicator_names.get(ind, ind)
                if full_name.lower() not in query.lower():
                    parts.append(full_name)

        # 카테고리 추가
        categories = entities.get("categories", [])
        if categories:
            category_names = {
                "lip_care": "Lip Care 립케어",
                "lip_makeup": "Lip Makeup 립메이크업",
                "face_powder": "Face Powder 파우더",
            }
            for cat in categories[:1]:
                full_name = category_names.get(cat, cat)
                if full_name.lower() not in query.lower():
                    parts.append(full_name)

        rewritten = " ".join(parts)
        if rewritten != query:
            logger.info(f"Query rewritten for relevance: '{query}' → '{rewritten}'")
        return rewritten

    def _load_retrieval_weights(self) -> dict:
        """config/retrieval_weights.json에서 가중치 로드"""
        import json
        from pathlib import Path

        defaults = {
            "weights": {"kg": 0.4, "rag": 0.4, "inference": 0.2},
            "freshness": {"weekly": 1.0, "quarterly": 0.9, "static": 0.8},
            "max_context_items": {"ontology_facts": 5, "inferences": 5, "rag_chunks": 3},
        }

        config_path = Path(__file__).parent.parent.parent / "config" / "retrieval_weights.json"
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    loaded = json.load(f)
                    # Merge with defaults (loaded overrides)
                    for key in defaults:
                        if key in loaded:
                            defaults[key] = loaded[key]
                logger.info(f"Retrieval weights loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load retrieval weights: {e}, using defaults")

        return defaults

    def _weighted_merge(
        self,
        context: HybridContext,
        intent_weights: dict[str, float] | None = None,
    ) -> HybridContext:
        """
        가중치 기반 컨텍스트 병합

        KG facts, RAG chunks, Ontology inferences에 가중치를 부여하고
        최종 점수로 정렬하여 상위 항목만 유지합니다.

        가중치 우선순위:
        1. intent_weights (인텐트 기반 전략에서 전달)
        2. config/retrieval_weights.json (파일 설정)
        3. 기본값: kg=0.4, rag=0.4, inference=0.2

        Args:
            context: 병합 전 HybridContext
            intent_weights: 인텐트 기반 가중치 (optional override)

        Returns:
            가중치 적용된 HybridContext
        """
        weights = (
            intent_weights if intent_weights is not None else self._retrieval_weights["weights"]
        )
        freshness = self._retrieval_weights["freshness"]
        max_items = self._retrieval_weights["max_context_items"]

        weighted_scores = {}

        # 1. Ontology facts 점수 계산
        if context.ontology_facts:
            scored_facts = []
            for fact in context.ontology_facts:
                fact_type = fact.get("type", "")

                # 기본 점수 할당
                if fact_type in ["brand_info", "competitors", "competitor_network"]:
                    base_score = 1.0
                elif fact_type in ["category_brands", "category_hierarchy"]:
                    base_score = 0.8
                else:
                    base_score = 0.6

                weighted_score = weights["kg"] * base_score
                fact["_weighted_score"] = weighted_score
                scored_facts.append(fact)

            # 점수로 정렬 및 제한
            scored_facts.sort(key=lambda x: x.get("_weighted_score", 0), reverse=True)
            context.ontology_facts = scored_facts[: max_items["ontology_facts"]]
            weighted_scores["ontology_facts"] = [
                f.get("_weighted_score", 0) for f in context.ontology_facts
            ]

        # 2. RAG chunks 점수 계산
        if context.rag_chunks:
            scored_chunks = []
            for chunk in context.rag_chunks:
                similarity_score = chunk.get("score", 0.5)

                # Freshness factor 결정
                doc_type = chunk.get("metadata", {}).get("doc_type", "")
                if doc_type in ["intelligence", "response_guide"]:
                    freshness_factor = freshness["weekly"]
                elif doc_type in ["playbook", "knowledge_base"]:
                    freshness_factor = freshness["quarterly"]
                else:
                    freshness_factor = freshness["static"]

                weighted_score = weights["rag"] * similarity_score * freshness_factor
                chunk["_weighted_score"] = weighted_score
                scored_chunks.append(chunk)

            # 점수로 정렬 및 제한
            scored_chunks.sort(key=lambda x: x.get("_weighted_score", 0), reverse=True)
            context.rag_chunks = scored_chunks[: max_items["rag_chunks"]]
            weighted_scores["rag_chunks"] = [
                c.get("_weighted_score", 0) for c in context.rag_chunks
            ]

        # 3. Inferences 점수 계산
        if context.inferences:
            scored_inferences = []
            for inference in context.inferences:
                confidence = getattr(inference, "confidence", 0.5)
                weighted_score = weights["inference"] * confidence

                # Store score as attribute (not in dict)
                inference._weighted_score = weighted_score
                scored_inferences.append(inference)

            # 점수로 정렬 및 제한
            scored_inferences.sort(key=lambda x: getattr(x, "_weighted_score", 0), reverse=True)
            context.inferences = scored_inferences[: max_items["inferences"]]
            weighted_scores["inferences"] = [
                getattr(i, "_weighted_score", 0) for i in context.inferences
            ]

        # 메타데이터에 점수 저장
        if not context.metadata:
            context.metadata = {}
        context.metadata["weighted_scores"] = weighted_scores

        # ConfidenceFusion: 전체 신뢰도 계산 + 충돌 감지
        fusion_meta = self._compute_fusion_confidence(context, intent_weights)
        context.metadata["fusion"] = fusion_meta

        logger.info(
            f"Weighted merge applied: {len(context.ontology_facts)} facts, "
            f"{len(context.rag_chunks)} chunks, {len(context.inferences)} inferences"
            f" | fusion_confidence={fusion_meta.get('confidence', 0):.3f}"
            f" strategy={fusion_meta.get('strategy', 'n/a')}"
        )

        if fusion_meta.get("warnings"):
            for w in fusion_meta["warnings"]:
                logger.warning(f"Fusion conflict: {w}")

        return context

    def _compute_fusion_confidence(
        self,
        context: HybridContext,
        intent_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        ConfidenceFusion을 사용해 전체 신뢰도를 계산하고 소스 간 충돌을 감지합니다.

        기존 _weighted_merge의 per-item 점수 산정은 유지하고,
        이 메서드는 3개 소스의 aggregate 신뢰도 + 충돌 경고를 추가합니다.

        Args:
            context: 가중 병합 완료된 HybridContext
            intent_weights: 인텐트별 가중치 (kg/rag/inference)

        Returns:
            dict with confidence, strategy, warnings, source_scores, explanation
        """
        try:
            from src.rag.confidence_fusion import (
                ConfidenceFusion,
                FusionStrategy,
                LinkedEntity,
                ScoreNormalizationMethod,
                SearchResult,
            )
            from src.rag.confidence_fusion import (
                InferenceResult as FusionInferenceResult,
            )
        except ImportError:
            logger.debug("confidence_fusion module not available, skipping fusion scoring")
            return {"confidence": 0.0, "strategy": "unavailable", "warnings": []}

        # 인텐트 가중치 → ConfidenceFusion 가중치 매핑
        # ConfidenceFusion uses: vector(=rag), ontology(=inference), entity(=kg)
        w = intent_weights or {"kg": 0.4, "rag": 0.4, "inference": 0.2}
        fusion_weights = {
            "vector": w.get("rag", 0.4),
            "ontology": w.get("inference", 0.2),
            "entity": w.get("kg", 0.4),
        }

        # 인텐트 설정에서 fusion_strategy 결정
        fusion_strategy_name = "weighted_sum"
        try:
            from src.core.intent import classify_intent as _cl
            from src.rag.retrieval_strategy import get_intent_retrieval_config

            intent = _cl(context.query)
            config = get_intent_retrieval_config(intent)
            fusion_strategy_name = config.fusion_strategy
        except Exception:
            pass

        strategy_map = {
            "weighted_sum": FusionStrategy.WEIGHTED_SUM,
            "harmonic_mean": FusionStrategy.HARMONIC_MEAN,
            "geometric_mean": FusionStrategy.GEOMETRIC_MEAN,
            "max_score": FusionStrategy.MAX_SCORE,
            "rrf": FusionStrategy.RRF,
        }
        strategy = strategy_map.get(fusion_strategy_name, FusionStrategy.WEIGHTED_SUM)

        # harmonic/geometric mean은 0 점수에 취약 → 정규화 생략 (원점수가 이미 0-1)
        if strategy in (FusionStrategy.HARMONIC_MEAN, FusionStrategy.GEOMETRIC_MEAN):
            normalization = ScoreNormalizationMethod.NONE
        else:
            normalization = ScoreNormalizationMethod.MIN_MAX

        fusion = ConfidenceFusion(
            weights=fusion_weights,
            normalization=normalization,
            strategy=strategy,
            min_sources=1,
            conflict_threshold=0.3,
        )

        # HybridContext → ConfidenceFusion 입력 변환
        vector_results = []
        for chunk in context.rag_chunks or []:
            vector_results.append(
                SearchResult(
                    content=chunk.get("content", chunk.get("text", "")),
                    score=chunk.get("_weighted_score", chunk.get("score", 0.5)),
                    metadata=chunk.get("metadata", {}),
                    source="vector",
                )
            )

        ontology_results = []
        for inf in context.inferences or []:
            ontology_results.append(
                FusionInferenceResult(
                    insight=getattr(inf, "conclusion", str(inf)),
                    confidence=getattr(inf, "confidence", 0.5),
                    evidence=getattr(inf, "evidence", {}),
                    rule_name=getattr(inf, "rule_name", None),
                )
            )

        entity_links = []
        for fact in context.ontology_facts or []:
            entity_links.append(
                LinkedEntity(
                    entity_id=fact.get("type", "unknown"),
                    entity_name=fact.get("subject", fact.get("type", "")),
                    entity_type=fact.get("type", "KG_Fact"),
                    link_confidence=fact.get("_weighted_score", 0.6),
                    context=str(fact.get("data", "")),
                )
            )

        # Fusion 실행
        result = fusion.fuse(
            vector_results=vector_results or None,
            ontology_results=ontology_results or None,
            entity_links=entity_links or None,
            query=context.query,
        )

        return {
            "confidence": round(result.confidence, 4),
            "strategy": result.fusion_strategy,
            "warnings": result.warnings,
            "explanation": result.explanation,
            "source_scores": [
                {
                    "source": s.source_name,
                    "raw": round(s.raw_score, 3),
                    "normalized": round(s.normalized_score, 3),
                    "weight": round(s.weight, 3),
                    "contribution": round(s.contribution, 3),
                    "level": s.confidence_level,
                }
                for s in result.source_scores
            ],
        }

    def _combine_contexts(self, context: HybridContext, include_explanations: bool = True) -> str:
        """
        온톨로지 + RAG 컨텍스트 통합

        Args:
            context: HybridContext
            include_explanations: 추론 설명 포함

        Returns:
            통합된 컨텍스트 문자열
        """
        parts = []

        # 1. 온톨로지 추론 결과 (구조화된 인사이트)
        if context.inferences:
            parts.append("## 분석 결과 (Ontology Reasoning)\n")

            for i, inf in enumerate(context.inferences, 1):
                parts.append(
                    f"### 인사이트 {i}: {inf.insight_type.value.replace('_', ' ').title()}"
                )
                parts.append(f"- **결론**: {inf.insight}")

                if inf.recommendation:
                    parts.append(f"- **권장 액션**: {inf.recommendation}")

                parts.append(f"- **신뢰도**: {inf.confidence:.0%}")

                if include_explanations and inf.evidence:
                    conditions = inf.evidence.get("satisfied_conditions", [])
                    if conditions:
                        parts.append(f"- **근거 조건**: {', '.join(conditions)}")

                parts.append("")

        # 2. 지식 그래프 사실 (관련 정보)
        if context.ontology_facts:
            parts.append("## 관련 정보 (Knowledge Graph)\n")

            for fact in context.ontology_facts[:5]:  # 상위 5개
                fact_type = fact.get("type", "unknown")
                entity = fact.get("entity", "")
                data = fact.get("data", {})

                if fact_type == "brand_info":
                    sos = data.get("sos", 0)
                    if sos:
                        parts.append(f"- **{entity}** SoS: {sos * 100:.1f}%")
                    if data.get("avg_rank"):
                        parts.append(f"  - 평균 순위: {data['avg_rank']:.1f}")

                elif fact_type == "brand_products":
                    parts.append(f"- **{entity}** 제품 수: {data.get('product_count', 0)}개")

                elif fact_type == "competitors":
                    competitors = [c.get("brand", "") for c in data[:3]]
                    parts.append(f"- **{entity}** 주요 경쟁사: {', '.join(competitors)}")

                elif fact_type == "category_brands":
                    top_brands = [b.get("brand", "") for b in data.get("top_brands", [])[:3]]
                    parts.append(f"- **{entity}** Top 브랜드: {', '.join(top_brands)}")

                elif fact_type == "category_hierarchy":
                    level = data.get("level", 0)
                    path = data.get("path", [])
                    ancestors = data.get("ancestors", [])
                    name = data.get("name", entity)
                    if path:
                        path_str = " > ".join(
                            [
                                a.get("name", a.get("id", "")) if isinstance(a, dict) else a
                                for a in path
                            ]
                        )
                        parts.append(f"- **{name}** 계층: {path_str} (Level {level})")
                    if ancestors:
                        parent_names = [a.get("name", "") for a in ancestors[:2]]
                        parts.append(f"  - 상위 카테고리: {', '.join(parent_names)}")

            parts.append("")

        # 3. RAG 가이드라인 (비구조화 문서)
        if context.rag_chunks:
            parts.append("## 참고 가이드라인 (RAG)\n")

            for chunk in context.rag_chunks[:3]:  # 상위 3개
                title = chunk.get("metadata", {}).get("title", "")
                content = chunk.get("content", "")

                if title:
                    parts.append(f"### {title}")

                # 내용 축약 (500자)
                if len(content) > 500:
                    content = content[:500] + "..."

                parts.append(content)
                parts.append("")

        return "\n".join(parts)

    async def retrieve_for_entity(
        self, entity: str, entity_type: str = "brand", current_metrics: dict[str, Any] | None = None
    ) -> HybridContext:
        """
        특정 엔티티에 대한 하이브리드 검색

        Args:
            entity: 엔티티 ID
            entity_type: 엔티티 유형 (brand, product, category)
            current_metrics: 현재 지표

        Returns:
            HybridContext
        """
        # 엔티티 기반 쿼리 생성
        if entity_type == "brand":
            query = f"{entity} 브랜드 분석"
            entities = {"brands": [entity.lower()]}
        elif entity_type == "product":
            query = f"{entity} 제품 분석"
            entities = {"products": [entity]}
        elif entity_type == "category":
            query = f"{entity} 카테고리 분석"
            entities = {"categories": [entity]}
        else:
            query = f"{entity} 분석"
            entities = {}

        # 검색 수행
        context = await self.retrieve(query, current_metrics)
        context.entities.update(entities)

        return context

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

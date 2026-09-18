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

from src.domain.entities.evidence import Evidence
from src.domain.entities.relations import InferenceResult, InsightType, RelationType
from src.monitoring.rag_metrics import RAGMetricsCollector
from src.ontology.business_rules import register_all_rules
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.ontology.rule_contracts import evaluate_rules_on_cards

from .evidence_adapters import EvidenceAdapter
from .evidence_assembly import EvidenceBundle, assemble_evidence, build_rule_input_cards
from .evidence_renderer import render_for_prompt
from .ontology_context import MAX_EXPANDED_BRANDS, STATIC_BRAND_PREDICATES
from .ontology_context import apply_cap as apply_expansion_cap
from .ontology_context import plan_query as plan_ontology_query
from .ontology_context import static_fact as ontology_static_fact
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

from .entity_linker import product_name_slugs as _product_name_slugs

# 브랜드 정체성·지표 엣지의 정보량 우선순위 (12개 상한에서 살아남을 순서)
_REST_EDGE_PRIORITY = {
    "ownedBy": 0,
    "hasSoS": 1,
    "rankedIn": 2,
    "hasHHI": 3,
    "hasPosition": 4,
}

# 플래그 ontology.use_class_reasoning ON (트랙 O3): 정식 술어 이름과 정적 정의 술어를 쓴다
_REST_EDGE_PRIORITY_CANONICAL = {
    "ownedByGroup": 0,
    "hasSegment": 0,
    "originatesFrom": 0,
    "acquiredIn": 0,
    "ownsBrand": 0,
    "hasSoS": 1,
    "rankedIn": 2,
    "hasHHI": 3,
    "hasPricePosition": 4,
    "siblingBrand": 5,
}
_PRIORITY_PREDS = frozenset(
    {"hasSoS", "hasHHI", "rankedIn", "competesWith", "hasPosition", "ownedBy"}
)
# ON: 정적 정의 술어를 버리지 않는다 (검토 보고서 §3.1). hasPosition은 읽을 때 나뉜다.
_PRIORITY_PREDS_CANONICAL = frozenset(
    {
        "hasSoS",
        "hasHHI",
        "hasPricePosition",
        "rankedIn",
        "competesWith",
        "ownedByGroup",
        "ownsBrand",
        "siblingBrand",
        "hasSegment",
        "originatesFrom",
        "acquiredIn",
    }
)
# 온톨로지 카드를 프롬프트에 더할 때의 상한 (관계 종류 상한 20과 별도 — O3 카드 수 가드)
ONTOLOGY_PROMPT_MAX = 24


def _dedupe_edges(edges: list[dict]) -> list[dict]:
    """같은 (subject, predicate, object) 엣지를 대소문자 무시로 1회만 남긴다."""
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict] = []
    for edge in edges:
        key = (
            str(edge["subject"]).lower(),
            str(edge["predicate"]),
            str(edge["object"]).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    return unique


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
        evidence: 이번 질의에서 만든 전체 증거 카드 (중복 제거, 순서 결정적)
        prompt_evidence: 답변 프롬프트에 실제로 렌더링된 카드 (``select_cards`` 결과,
            ``combined_context``의 렌더 입력과 같은 목록)
        combined_context: 통합된 컨텍스트 (LLM 프롬프트용) = ``render_for_prompt(prompt_evidence)``
        metadata: 추가 메타데이터
    """

    query: str
    entities: dict[str, list[str]] = field(default_factory=dict)
    ontology_facts: list[dict[str, Any]] = field(default_factory=list)
    inferences: list[InferenceResult] = field(default_factory=list)
    rag_chunks: list[dict[str, Any]] = field(default_factory=list)
    # 크롤 DB 수치 사실 (src/rag/metric_facts.py). ontology_facts와 분리한다 — 섞으면
    # 평가 러너가 여기서 엔티티를 뽑아 L3 Hits@k가 KG와 무관하게 오른다.
    metric_facts: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    prompt_evidence: list[Evidence] = field(default_factory=list)
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
            "metric_facts": self.metric_facts,
            "evidence": [card.model_dump(mode="json") for card in self.evidence],
            "prompt_evidence": [card.model_dump(mode="json") for card in self.prompt_evidence],
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
                "sentiment_clusters": [...],
                "concepts": [...]
            }
        """
        entities = self._linker.extract_entities(query, knowledge_graph=knowledge_graph)
        try:
            entities["concepts"] = self._linker.extract_concepts(query)
        except Exception:
            entities["concepts"] = []
        return entities


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
        metric_facts_provider: Any | None = None,
    ):
        """
        Args:
            knowledge_graph: 지식 그래프
            reasoner: 온톨로지 추론기
            doc_retriever: RAG 문서 검색기
            auto_init_rules: 비즈니스 규칙 자동 등록
        """
        # 컴포넌트 초기화
        # fallback 인스턴스는 읽기 전용 (정식 기록자는 daily_crawl의 exporter)
        self.kg = knowledge_graph or KnowledgeGraph(auto_save=False)
        self.reasoner = reasoner or OntologyReasoner(self.kg)
        self.doc_retriever = doc_retriever or DocumentRetriever()

        # 크롤 DB 수치 사실 제공자 (SQLite 정본, 스냅샷 날짜 포함)
        from src.rag.metric_facts import MetricFactsProvider

        self.metric_facts_provider = metric_facts_provider or MetricFactsProvider()

        # 검색 결과 → 증거 카드 (단위·브랜드 정규화, KG 수치 엣지 제외)
        self.evidence_adapter = EvidenceAdapter()

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

    @staticmethod
    def _record_degraded(
        degraded: list[dict[str, Any]] | None, component: str, exc: Exception
    ) -> None:
        """선택 기능(비핵심) 실패를 degraded 목록에 기록한다.

        호출자가 소유한 요청-로컬 리스트에만 쓰며 ``self``에는 아무것도
        남기지 않는다 — 동시 요청 간 공유 상태로 트레이스가 오염됐던
        ``_last_hybrid_context`` 사례(F3 배경)를 반복하지 않기 위함이다.
        ``degraded``가 None이면(호출자가 추적하지 않는 경로) 조용히 무시한다.
        """
        if degraded is not None:
            degraded.append({"component": component, "error": f"{type(exc).__name__}: {exc}"})

    async def retrieve(
        self,
        query: str,
        current_metrics: dict[str, Any] | None = None,
        include_explanations: bool = True,
    ) -> HybridContext:
        """
        하이브리드 검색 수행

        단계: 엔티티 → KG 사실 → DB 수치 → 규칙 입력 카드(metric·relation) → 규칙 추론 →
        문서 검색(추론으로 질의 확장) → 관련성 판정 → 가중 병합 → 카드 조립·선별 → 렌더링.

        Args:
            query: 사용자 쿼리
            current_metrics: 시그니처 호환용. **규칙 추론 입력으로 쓰지 않는다** — 추론 입력은
                DB 수치·KG 관계 증거 카드뿐이다 (설계 E3, 결함 F7: 대시보드 JSON 키가 운영
                데이터에 없어 추론이 0건이었다).
            include_explanations: 하위 호환용 (출력에 영향 없음)

        Returns:
            HybridContext. 규칙을 판정했으면 ``metadata["rule_evaluation"]``에 조합·판정 수·
            발화 규칙·미발화 사유 상위가 남는다 (``RuleRun.summary``).
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

        # 선택 기능(비핵심 하위 조회) 실패 기록 — 요청 로컬 리스트라 동시 요청 간
        # 공유 상태가 아니다 (F3: _last_hybrid_context 경쟁 상태 교훈 적용)
        degraded: list[dict[str, Any]] = []

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

            # 1.8 온톨로지 질의 해석 (플래그 ontology.use_class_reasoning, 기본 OFF, 트랙 O3).
            #     OFF면 아래 모든 단계가 O3 이전과 같은 인자로 호출된다 (특성화 테스트로 고정).
            ontology_plan = None
            onto = None
            adapter = self.evidence_adapter
            if flags.use_class_reasoning():
                try:
                    onto, ontology_plan = await self._plan_ontology(entities)
                    adapter = self._canonical_adapter(onto)
                except Exception as e:
                    logger.warning("온톨로지 질의 해석 실패 — 기존 경로로 계속", exc_info=True)
                    self._record_degraded(degraded, "ontology_plan", e)
                    onto, ontology_plan, adapter = None, None, self.evidence_adapter

            # 2. 지식 그래프에서 사실 조회 (ablation no-kg: FF_KG_ENABLED=false, 옛 FF_ONTOLOGY_USE_ONTOLOGY_KG)
            if flags.kg_enabled():
                if ontology_plan is not None:
                    ontology_facts = self._query_knowledge_graph(
                        entities, degraded=degraded, ontology_plan=ontology_plan, ontology=onto
                    )
                else:
                    ontology_facts = self._query_knowledge_graph(entities, degraded=degraded)
            else:
                logger.info("KG query disabled by feature flag (kg.enabled=false)")
                ontology_facts = []
            if ontology_plan is not None and not flags.kg_enabled():
                # 정적 정의 사실은 온톨로지 원본에서 오므로 KG 조회 스위치와 무관하다
                static = self._ontology_static_fact(onto, ontology_plan, degraded)
                ontology_facts = [static] if static else []
            context.ontology_facts = ontology_facts

            # 2.5 크롤 DB 수치 사실 (ablation no-db-metrics: FF_RETRIEVER_USE_DB_METRIC_FACTS=false)
            # 수치 질문의 근거는 KG·문서가 아니라 SQLite 스냅샷이다 (사이클 10 §2).
            if flags.use_db_metric_facts():
                try:
                    if ontology_plan is not None:
                        context.metric_facts = await self._collect_metric_facts_expanded(
                            entities, ontology_plan, onto
                        )
                    else:
                        context.metric_facts = await self.metric_facts_provider.collect(entities)
                except Exception as e:
                    logger.warning("DB 지표 사실 조회 실패", exc_info=True)
                    context.metric_facts = []
                    self._record_degraded(degraded, "db_metric_facts", e)

            # 3. 규칙 입력 카드 (DB 수치 → metric, KG 사실 → relation) — 추론 전에 만든다 (E3)
            input_cards = build_rule_input_cards(
                metric_facts=context.metric_facts,
                ontology_facts=context.ontology_facts,
                adapter=adapter,
                degraded=degraded,
            )

            # 4. 규칙 추론: 카드 → 계약 입력 → 판정 (ablation no-ontology: reasoner 플래그
            #    둘 다 false면 규칙을 판정하지 않는다 — 이름과 달리 "규칙 추론 off" 스위치)
            rule_evaluation: dict[str, Any] | None = None
            if flags.use_unified_reasoner() or flags.reasoner_enabled():
                if ontology_plan is not None:
                    # 포함 확장(OE3): 하위 카테고리 조합도 판정한다 (카드는 자기 카테고리 그대로)
                    inferences, rule_evaluation = self._evaluate_rules(
                        entities,
                        input_cards,
                        adapter=adapter,
                        extra_categories=ontology_plan.descendant_categories,
                    )
                else:
                    inferences, rule_evaluation = self._evaluate_rules(entities, input_cards)
            else:
                logger.info("Ontology inference disabled by feature flags")
                inferences = []
            context.inferences = inferences
            logger.debug(f"Generated {len(inferences)} inferences")

            # 5. RAG 문서 검색 (추론 결과로 쿼리 확장 + 의도 기반 필터링)
            #    Uses hybrid (dense + BM25 RRF) when BM25 is available
            #    질의 엔티티(원 질의 기준)와 색인 태그가 겹치는 청크는 재정렬 보너스를 받는다
            #    — 필터가 아니라 순서만 바뀐다 (E10, F9)
            from .entity_tags import result_chunk_id

            expanded_query = self._expand_query(search_query, inferences, entities)
            entity_rerank: dict[str, Any] = {}
            # 포함 확장(OE3, ON): 문서 태그 가산점 대상 카테고리에 하위 카테고리를 더한다
            search_entities = (
                self._scope_entities(entities, ontology_plan)
                if ontology_plan is not None
                else entities
            )
            rag_results, search_method = await self._hybrid_search(
                expanded_query,
                top_k=intent_top_k,
                doc_type_filter=doc_type_filter,
                degraded=degraded,
                entities=search_entities,
                rerank_stats=entity_rerank,
            )

            # 필터링된 결과가 k개보다 적으면 전체 문서(필터 없음)에서 채운다
            if len(rag_results) < intent_top_k and doc_type_filter:
                additional_results, _fallback_method = await self._hybrid_search(
                    expanded_query,
                    top_k=intent_top_k,
                    doc_type_filter=None,  # 전체 문서에서 검색
                    degraded=degraded,
                    entities=search_entities,
                    rerank_stats=entity_rerank,
                )
                # 중복 제거하며 추가 (BM25 결과는 최상위 id가 없어 청크 id로 비교)
                existing_ids = {result_chunk_id(r) for r in rag_results}
                for result in additional_results:
                    if len(rag_results) >= intent_top_k:
                        break
                    rid = result_chunk_id(result)
                    if rid not in existing_ids:
                        rag_results.append(result)
                        existing_ids.add(rid)

            entity_rerank["bonus_chunks"] = sum(
                1 for r in rag_results if r.get("entity_bonus", 0) > 0
            )
            context.rag_chunks = rag_results

            # 5.5. 관련성 검증 (Relevance Grading)
            try:
                from src.infrastructure.feature_flags import FeatureFlags

                if not FeatureFlags.get_instance().use_reranker():
                    # reranker 비활성화는 실패가 아니라 정상 분기 — degraded에 넣지 않는다
                    logger.info("Reranker disabled by feature flag, skipping relevance grading")
                else:
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
                # 실패 시 원본 결과 유지 — 선택 기능 실패로 기록
                self._record_degraded(degraded, "relevance_grading", e)

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
                self._record_degraded(degraded, "rag_metrics_recording", e)

            # 5.7. 가중치 기반 병합 (인텐트 전략 가중치 적용)
            context = self._weighted_merge(
                context, intent_weights=intent_config.weights, degraded=degraded
            )

            # 5.9. 증거 카드 조립·선별 (최종 ontology_facts·inferences·rag_chunks + DB 수치).
            #      추론 근거 카드가 사실 상한으로 잘렸으면 입력 카드에서 되살린다
            if ontology_plan is not None:
                evidence_bundle = self._assemble_evidence(
                    context,
                    degraded=degraded,
                    input_cards=input_cards,
                    adapter=adapter,
                    ontology_plan=ontology_plan,
                    ontology=onto,
                )
            else:
                evidence_bundle = self._assemble_evidence(
                    context, degraded=degraded, input_cards=input_cards
                )

            # 6. 통합 컨텍스트 생성 (프롬프트 카드만 렌더링)
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
                # 엔티티 태그 재정렬: 질의 대상·후보/태그된 후보/보너스 후보 수·최종 보너스 청크 수
                "entity_rerank": entity_rerank,
                "selfrag_confidence": selfrag_confidence,
                "bm25_available": self._bm25_actually_available(),
                "evidence_count": len(context.evidence),
                "prompt_evidence_count": len(context.prompt_evidence),
                # 어댑터가 증거에서 뺀 KG 항목 수 (날짜 없는 수치 엣지·엔티티 메타데이터 등, E2)
                "evidence_excluded": len(evidence_bundle.excluded),
                "evidence_excluded_by_reason": evidence_bundle.excluded_by_reason,
                # 선택 기능(비핵심) 실패 목록 — 이 재할당이 위에서 누적된 degraded를
                # 지우지 않도록 여기서 함께 포함한다 (F3)
                "degraded": degraded,
            }
            if rule_evaluation is not None:  # 추론 off면 키가 없다 (판정하지 않았다)
                context.metadata["rule_evaluation"] = rule_evaluation
            if ontology_plan is not None:  # 플래그 OFF면 키가 없다 (O7 카드 수 비교용)
                context.metadata["ontology"] = self._ontology_trace(
                    context, ontology_plan, onto, evidence_bundle
                )

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            # 핵심 검색 실패 — 서비스 동작(빈 컨텍스트 반환)은 그대로 유지하고
            # 평가 하니스가 인프라 실패로 분류할 수 있도록 원인을 남긴다 (F3)
            context.metadata["retrieval_error"] = f"{type(e).__name__}: {e}"
            context.metadata["error"] = str(e)  # 기존 소비자 호환용 키 유지

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

        # Self-RAG 게이트 — 모든 unified 검색에 적용
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

        # retrieve() → HybridContext → UnifiedRetrievalResult 변환
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
            evidence=list(ctx.evidence),
            prompt_evidence=list(ctx.prompt_evidence),
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
        return await self.doc_retriever.search(query=query, top_k=top_k, doc_filter=doc_filter)

    def _bm25_actually_available(self) -> bool:
        """BM25 sparse 검색 가용 여부 — 메서드 존재가 아니라 rank_bm25 설치 여부까지 확인"""
        if not hasattr(self.doc_retriever, "search_bm25"):
            return False
        try:
            from src.rag.retriever import BM25_AVAILABLE

            return bool(BM25_AVAILABLE)
        except ImportError:
            return False

    async def _hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        doc_type_filter: list[str] | None = None,
        degraded: list[dict[str, Any]] | None = None,
        entities: dict[str, Any] | None = None,
        rerank_stats: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """
        Dense + BM25 hybrid search with RRF fusion, then entity-tag re-rank bonus.

        RRF 병합은 후보 전체(dense ∪ BM25)에 대해 하고, 엔티티 태그 보너스로 재정렬한 뒤
        top_k로 자른다. 보너스는 필터가 아니다 — 후보 수는 줄지 않고, 보너스가 하나도 없으면
        기존(top_k로 바로 자른 RRF) 결과와 같다 (설계 E10).

        Args:
            query: Search query
            top_k: Number of results to return
            doc_type_filter: Optional document type filter
            degraded: 선택 기능 실패 기록 대상 (호출자가 소유한 리스트, 공유 상태 아님)
            entities: 질의 엔티티 (``EntityExtractor.extract`` 형식). None이면 보너스 없음
            rerank_stats: 재정렬 통계 누적 대상 (호출자 소유 dict)

        Returns:
            (results, search_method) where search_method is
            "hybrid_rrf" or "dense_only"
        """
        # 1. Dense search via doc_retriever.search() — 핵심 검색 경로.
        # 여기서 발생하는 예외는 의도적으로 잡지 않고 retrieve()의 바깥
        # except로 전파시켜 retrieval_error로 분류한다 (F3)
        dense_results = await self.doc_retriever.search(
            query, top_k=top_k, doc_type_filter=doc_type_filter
        )

        # 2. BM25 search (if available) — 선택 기능, 실패해도 dense 결과로 계속
        bm25_results = []
        if hasattr(self.doc_retriever, "search_bm25"):
            try:
                bm25_results = self.doc_retriever.search_bm25(query, top_k=top_k)
            except Exception as e:
                logger.debug(f"BM25 search failed in _hybrid_search: {e}")
                self._record_degraded(degraded, "bm25_search", e)

        # 3. RRF fusion — 후보 전체를 병합해 둔다 (보너스 재정렬 후 top_k로 자른다)
        candidate_limit = len(dense_results) + len(bm25_results)
        if bm25_results:
            if hasattr(self.doc_retriever, "reciprocal_rank_fusion"):
                fused = self.doc_retriever.reciprocal_rank_fusion(
                    dense_results, bm25_results, k=60, top_k=candidate_limit
                )
                return self._rerank_by_entity_tags(
                    fused, top_k, entities, rerank_stats, degraded
                ), "hybrid_rrf"
            # Fallback: try confidence_fusion.fuse_documents_rrf
            try:
                from src.rag.confidence_fusion import ConfidenceFusion

                fusion = ConfidenceFusion()
                fused = fusion.fuse_documents_rrf(
                    {"dense": dense_results, "bm25": bm25_results},
                    k=60,
                    top_n=candidate_limit,
                )
                return self._rerank_by_entity_tags(
                    fused, top_k, entities, rerank_stats, degraded
                ), "hybrid_rrf"
            except (ImportError, Exception) as e:
                logger.debug(f"Confidence fusion RRF fallback failed: {e}")
                self._record_degraded(degraded, "rrf_fusion_fallback", e)

        return self._rerank_by_entity_tags(
            dense_results, top_k, entities, rerank_stats, degraded
        ), "dense_only"

    def _rerank_by_entity_tags(
        self,
        results: list[dict[str, Any]],
        top_k: int,
        entities: dict[str, Any] | None,
        rerank_stats: dict[str, Any] | None,
        degraded: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """질의 엔티티와 색인 태그가 겹치는 청크에 가산 보너스를 주고 top_k로 자른다.

        보너스 = Σ w_f · 1[질의 f ∩ 청크 태그 f ≠ ∅], f ∈ {brands, categories}.
        가중치는 ``config/retrieval_weights.json``의 ``entity_tag_bonus``. 태그가 없는 색인·
        엔티티 없는 질의·가중치 0이면 순서를 바꾸지 않는다. 태그 조회 실패는 선택 기능
        실패로 기록하고 보너스 없이 계속한다.
        """
        from .entity_tags import (
            apply_entity_bonus,
            query_tag_targets,
            resolve_bonus_weights,
            result_chunk_id,
        )

        targets = query_tag_targets(entities)
        weights_config = getattr(self, "_retrieval_weights", {}) or {}
        weights = resolve_bonus_weights(weights_config.get("entity_tag_bonus"))

        stats = {"candidates": len(results), "tagged_candidates": 0, "bonus_candidates": 0}
        reranked = results
        if targets and any(weights.get(field, 0) > 0 for field in targets) and results:
            tags_by_id: dict[str, Any] = {}
            get_tags = getattr(self.doc_retriever, "get_entity_tags", None)
            if callable(get_tags):
                try:
                    looked_up = get_tags([result_chunk_id(r) for r in results])
                    tags_by_id = looked_up if isinstance(looked_up, dict) else {}
                except Exception as e:
                    logger.warning("엔티티 태그 조회 실패 — 보너스 없이 진행", exc_info=True)
                    self._record_degraded(degraded, "entity_tag_lookup", e)
            reranked, stats = apply_entity_bonus(results, tags_by_id, targets, weights)

        if rerank_stats is not None:
            rerank_stats["query_targets"] = {f: sorted(v) for f, v in targets.items()}
            rerank_stats["weights"] = weights
            for key, value in stats.items():
                rerank_stats[key] = rerank_stats.get(key, 0) + value

        return reranked[:top_k]

    def _query_knowledge_graph(
        self,
        entities: dict[str, list[str]],
        degraded: list[dict[str, Any]] | None = None,
        ontology_plan: Any | None = None,
        ontology: Any | None = None,
    ) -> list[dict[str, Any]]:
        """
        지식 그래프에서 관련 사실 조회

        Args:
            entities: 추출된 엔티티
            degraded: 선택 기능 실패 기록 대상 (호출자가 소유한 리스트, 공유 상태 아님)
            ontology_plan: 플래그 ``ontology.use_class_reasoning`` ON일 때의 질의 해석
                (``ontology_context.OntologyPlan``). None이면 O3 이전 동작 그대로다. 주면
                (1) KG 술어를 읽을 때 정식화하고(``ownedBy`` → ``ownedByGroup``, ``hasPosition`` →
                ``hasSoS``·``hasHHI``·``hasPricePosition``), (2) 정적 정의 술어를 버리지 않되
                등록부 브랜드의 것은 등록부 카드가 대신하며, (3) 하위 카테고리도 조회 범위에
                넣고(OE3), (4) 끝에 정적 사실(``ontology_static``)을 붙인다.
            ontology: ``ontology_plan``을 만든 ``Ontology``.

        Returns:
            사실 리스트
        """
        facts = []
        canonical = ontology_plan is not None and ontology is not None
        scope_categories: set[str] = (
            {c.lower() for c in ontology_plan.scope_categories} if canonical else set()
        )
        expanded_names: set[str] = set()
        if canonical:
            for bid in [*ontology_plan.mentioned_brands, *ontology_plan.expanded_brands]:
                expanded_names.add(bid.lower())
                name = ontology.brand_name(bid)
                if name:
                    expanded_names.add(name.lower())

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
            except Exception as e:
                logger.warning("브랜드 관계 네트워크 조회 실패", exc_info=True)
                self._record_degraded(degraded, "kg_competitor_network", e)

            # 메트릭/관계 엣지 (kg_enricher가 저장한 hasSoS·rankedIn·competesWith 등)
            try:
                # 시드 온톨로지는 'LANEIGE'(대문자), kg_enricher는 'laneige'(소문자)로
                # 저장한다. 추출기가 내는 브랜드는 소문자이므로 대문자 변형을 조회하지
                # 않으면 시드 트리플(ownedByGroup 등)이 통째로 누락됐다.
                subject_variants: list[str] = []
                for variant in (brand, brand.lower(), brand.upper(), brand.title()):
                    if variant not in subject_variants:
                        subject_variants.append(variant)
                edge_relations = []
                for variant in subject_variants:
                    edge_relations += list(self.kg.query(subject=variant))
                priority_preds = _PRIORITY_PREDS_CANONICAL if canonical else _PRIORITY_PREDS
                query_categories = {c.lower() for c in entities.get("categories", [])}
                query_brands = {b.lower() for b in entities.get("brands", [])}
                if canonical:
                    query_categories |= scope_categories
                    query_brands |= expanded_names
                query_products = {p.lower() for p in entities.get("products", [])}
                relevant, competes, rest = [], [], []
                top_products: list[tuple[int, str, str]] = []  # (rank, title, category)
                for rel in edge_relations:
                    enum_pred = (
                        rel.predicate.value
                        if hasattr(rel.predicate, "value")
                        else str(rel.predicate)
                    )
                    orig = rel.properties.get("original_predicate")
                    # 골드/KG 표기는 camelCase — original_predicate는 hasSoS처럼
                    # 의미가 더 구체적인 camelCase일 때만 우선한다
                    pred = orig if orig and "_" not in orig and not orig.isupper() else enum_pred
                    if canonical:
                        # 읽을 때 정식화 (O3): 별칭 → 정식 이름, hasPosition은 원 술어로 분리
                        pred = ontology.resolve_kg_predicate(enum_pred, orig) or pred
                        if (
                            pred in STATIC_BRAND_PREDICATES
                            and ontology.normalize_brand(str(rel.subject)) is not None
                        ):
                            # 등록부 브랜드의 정적 사실(그룹·세그먼트·원산지·인수·자매)은
                            # 등록부 카드(ontology_static)가 한 번만 싣는다 — KG 사본과 중복 방지
                            continue
                    else:
                        # 시드 온톨로지 표기 정합화 (ownedByGroup → ownedBy)
                        pred = {"ownedByGroup": "ownedBy"}.get(pred, pred)
                    if pred == "hasProduct":
                        # 상위 랭크 제품은 제품명 슬러그 엣지로 방출 (ASIN은 조회 불가 표기)
                        title = rel.properties.get("title", "")
                        rank = rel.properties.get("rank")
                        if title and isinstance(rank, int) and rank <= 10:
                            top_products.append((rank, title, rel.properties.get("category", "")))
                        elif title and any(
                            slug in query_products for slug in _product_name_slugs(title, brand)
                        ):
                            # 질의가 직접 지목한 제품은 랭크와 무관하게 포함
                            top_products.append(
                                (
                                    rank if isinstance(rank, int) else 999,
                                    title,
                                    rel.properties.get("category", ""),
                                ),
                            )
                        continue
                    if pred not in priority_preds:
                        continue  # siblingBrand 등 시드 온톨로지는 엣지 노출에서 제외
                    edge = {"subject": rel.subject, "predicate": pred, "object": rel.object}
                    obj_lower = str(rel.object).lower()
                    # 쿼리에 등장한 카테고리/브랜드와 닿는 엣지를 우선
                    if obj_lower in query_categories or obj_lower in query_brands:
                        relevant.append(edge)
                    elif pred == "competesWith":
                        competes.append(edge)
                    else:
                        rest.append(edge)

                # 상위 랭크 제품(브랜드당 2개) → 제품명 슬러그 hasProduct/belongsToCategory 엣지
                product_edges = []
                for _rank, title, category in sorted(set(top_products))[:2]:
                    for slug in _product_name_slugs(title, brand):
                        product_edges.append(
                            {"subject": brand, "predicate": "hasProduct", "object": slug}
                        )
                    if category:
                        main_slugs = _product_name_slugs(title, brand)
                        if main_slugs:
                            product_edges.append(
                                {
                                    "subject": main_slugs[0],
                                    "predicate": "belongsToCategory",
                                    "object": category,
                                }
                            )
                # 선택 순서: 질의와 직접 닿는 엣지 → 제품 엣지 → 브랜드 정체성·지표
                # 엣지 → 경쟁 엣지. competesWith는 브랜드당 최대 8개로 가장 수가 많고
                # 질의 특정성이 낮아, 이전 순서(경쟁 우선)에서는 12개 상한이
                # ownedBy·rankedIn을 밀어냈다. rest 안에서도 정보량 순으로 정렬한다
                # (소유관계 > 점유율 > 랭킹 > 집중도 > 가격 포지션).
                # 상한 12개는 유지한다 — recall 게이트를 "엣지 전량 방출"로 우회하지
                # 않기 위한 정밀도 가드 (kg_edge_precision으로 감시).
                rest_priority = _REST_EDGE_PRIORITY_CANONICAL if canonical else _REST_EDGE_PRIORITY
                rest.sort(key=lambda e: rest_priority.get(e["predicate"], 9))
                metric_edges = _dedupe_edges(relevant + product_edges + rest + competes[:3])[:12]
                if metric_edges:
                    facts.append(
                        {"type": "metric_edges", "entity": brand, "data": {"edges": metric_edges}}
                    )
            except Exception as e:
                logger.debug("metric edge query failed", exc_info=True)
                self._record_degraded(degraded, "kg_metric_edges", e)

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
            except Exception as e:
                logger.warning("카테고리 계층 조회 실패", exc_info=True)
                self._record_degraded(degraded, "kg_category_hierarchy", e)

        # 카테고리 포함 확장 (OE3, ON): 하위 카테고리의 브랜드 순위 소속도 조회 범위에 넣는다.
        # 사실은 하위 카테고리 자신을 entity로 둔다 — 상위 카테고리로 옮기지 않는다.
        if canonical:
            for category in ontology_plan.descendant_categories:
                category_brands = self.kg.get_category_brands(category)
                if category_brands:
                    facts.append(
                        {
                            "type": "category_brands",
                            "entity": category,
                            "data": {
                                "brand_count": len(category_brands),
                                "top_brands": category_brands[:5],
                                "scope_of": list(ontology_plan.categories),
                            },
                        }
                    )

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
                except Exception as e:
                    logger.warning("제품 감성 조회 실패", exc_info=True)
                    self._record_degraded(degraded, "kg_product_sentiment", e)

            # 브랜드가 지정된 경우 브랜드 감성 프로필 조회
            for brand in entities.get("brands", []):
                try:
                    brand_sentiment = self.kg.get_brand_sentiment_profile(brand)
                    if brand_sentiment.get("all_tags"):
                        facts.append(
                            {"type": "brand_sentiment", "entity": brand, "data": brand_sentiment}
                        )
                except Exception as e:
                    logger.warning("브랜드 감성 프로필 조회 실패", exc_info=True)
                    self._record_degraded(degraded, "kg_brand_sentiment", e)

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
                    except Exception as e:
                        logger.warning("감성 클러스터 제품 조회 실패", exc_info=True)
                        self._record_degraded(degraded, "kg_sentiment_cluster_products", e)

        if canonical:
            static = self._ontology_static_fact(ontology, ontology_plan, degraded)
            if static:
                facts.append(static)
        return facts

    # ------------------------------------------------------------------
    # 온톨로지 질의 경로 (트랙 O3, 플래그 ontology.use_class_reasoning) [2026-09 사후]
    # ------------------------------------------------------------------

    async def _plan_ontology(self, entities: dict[str, Any]) -> tuple[Any, Any]:
        """엔티티 → (Ontology, OntologyPlan). 전개 상한 순위는 크롤 DB 등장 → KG 크롤 관계
        등장 → 이름 순이다 (결정적)."""
        from src.ontology.ontology import get_ontology

        onto = get_ontology()
        plan = plan_ontology_query(onto, entities)
        if not plan.candidates:
            return onto, apply_expansion_cap(plan)

        present: set[str] = set()
        present_fn = getattr(self.metric_facts_provider, "present_brands", None)
        if callable(present_fn):
            try:
                present = set(await present_fn(plan.scope_categories or None))
            except Exception:
                logger.debug("present_brands lookup failed", exc_info=True)

        def rank_key(bid: str) -> tuple[int, int, str]:
            name = onto.brand_name(bid) or bid
            in_db = name.lower() in present or bid in present
            in_kg = False
            for variant in dict.fromkeys((name, name.lower(), bid)):
                try:
                    if self.kg.get_brand_products(variant):
                        in_kg = True
                        break
                except Exception:
                    continue
            return (0 if in_db else 1, 0 if in_kg else 1, name.lower())

        return onto, apply_expansion_cap(plan, rank_key)

    def _canonical_adapter(self, onto: Any) -> EvidenceAdapter:
        """정식 술어 모드 어댑터 (질의마다 고른다 — 플래그가 질의마다 평가되므로)."""
        cached = getattr(self, "_canonical_adapter_cache", None)
        if cached is None or cached.ontology is not onto:
            cached = EvidenceAdapter(
                brand_normalizer=self.evidence_adapter._brand,
                category_normalizer=self.evidence_adapter._category,
                ontology=onto,
            )
            self._canonical_adapter_cache = cached
        return cached

    def _ontology_static_fact(
        self, onto: Any, plan: Any, degraded: list[dict[str, Any]] | None
    ) -> dict[str, Any] | None:
        try:
            return ontology_static_fact(onto, plan)
        except Exception as e:
            logger.warning("온톨로지 정적 사실 생성 실패", exc_info=True)
            self._record_degraded(degraded, "ontology_static_facts", e)
            return None

    @staticmethod
    def _metric_brands(entities: dict[str, Any], plan: Any, onto: Any) -> list[str]:
        """수치 조회 브랜드: 질의 브랜드(그룹 표기·가짜 브랜드 제외) + 전개 브랜드 이름."""
        brands: list[str] = []
        for raw in entities.get("brands") or []:
            if not raw:
                continue
            bid = onto.normalize_brand(raw)
            if bid is None and onto.normalize_group(raw) is not None:
                continue  # 그룹은 브랜드가 아니다 — 소속 브랜드로 전개했다
            if bid is not None and onto.is_placeholder(bid):
                continue
            brands.append(str(raw))
        for bid in plan.expanded_brands:
            brands.append(onto.brand_name(bid) or bid)
        return list(dict.fromkeys(brands))

    async def _collect_metric_facts_expanded(
        self, entities: dict[str, Any], plan: Any, onto: Any
    ) -> list[dict[str, Any]]:
        """ON: 그룹 전개 브랜드까지(상한 12) + 하위 카테고리 조회 범위로 DB 수치를 모은다.

        브랜드 3개 상한은 전개가 있을 때만 올린다. 수치 사실은 자기 카테고리 그대로다(OE3).
        """
        metric_entities = dict(entities)
        metric_entities["brands"] = self._metric_brands(entities, plan, onto)
        kwargs: dict[str, Any] = {}
        if plan.expanded_brands:
            kwargs["max_brands"] = min(MAX_EXPANDED_BRANDS, len(metric_entities["brands"]))
        if plan.descendant_categories:
            kwargs["scope_categories"] = list(plan.descendant_categories)
        return await self.metric_facts_provider.collect(metric_entities, **kwargs)

    @staticmethod
    def _scope_entities(entities: dict[str, Any], plan: Any) -> dict[str, Any]:
        """카테고리에 하위 카테고리를 더한 엔티티 사본 (문서 태그 가산점용, OE3)."""
        scoped = dict(entities)
        scoped["categories"] = list(
            dict.fromkeys([*(entities.get("categories") or []), *plan.descendant_categories])
        )
        return scoped

    def _ontology_trace(
        self, context: HybridContext, plan: Any, onto: Any, bundle: EvidenceBundle
    ) -> dict[str, Any]:
        """질의 1건의 온톨로지 추적 (O7이 프롬프트 증가를 잰다: 기준 ≈70장/질의)."""
        from .evidence_adapters import ONTOLOGY_SOURCE

        onto_cards = [c for c in context.evidence if c.source == ONTOLOGY_SOURCE]
        prompt_onto = [c for c in context.prompt_evidence if c.source == ONTOLOGY_SOURCE]
        by_predicate: dict[str, int] = {}
        for card in onto_cards:
            by_predicate[card.predicate] = by_predicate.get(card.predicate, 0) + 1
        return {
            "enabled": True,
            "version": onto.version,
            "as_of": onto.as_of,
            **plan.summary(),
            "ontology_cards": len(onto_cards),
            "prompt_ontology_cards": len(prompt_onto),
            "ontology_cards_by_predicate": by_predicate,
            "metric_facts_count": len(context.metric_facts),
            "evidence_count": len(context.evidence),
            "prompt_evidence_count": len(context.prompt_evidence),
        }

    def _evaluate_rules(
        self,
        entities: dict[str, list[str]],
        cards: list[Evidence],
        adapter: EvidenceAdapter | None = None,
        extra_categories: list[str] | None = None,
    ) -> tuple[list[InferenceResult], dict[str, Any]]:
        """증거 카드로 규칙을 판정한다 (설계 E3, 트랙 3-B).

        질의 엔티티의 (브랜드 또는 없음) × 카테고리 조합마다 ``build_rule_context``로 계약 입력을
        채우고 ``evaluate_rule``로 판정한다(``rule_contracts.evaluate_rules_on_cards``).
        입력은 카드뿐이다 — 대시보드 JSON(current_metrics)·KG 수치 엣지를 읽지 않고,
        ``OntologyReasoner._enrich_context``(호출자 값을 KG 조회로 덮어쓴다)도 거치지 않는다.

        ``adapter``·``extra_categories``는 플래그 ``ontology.use_class_reasoning`` ON에서만 온다
        (정식 술어 모드 정규화, 카테고리 포함 확장의 하위 카테고리 — OE3, 트랙 O3).

        Returns:
            (발화 결과 — ``evidence["derived_from"]``에 근거 카드 id, ``metadata["rule_evaluation"]``)
        """
        adapter = adapter or self.evidence_adapter
        brands = [adapter.normalize_brand(str(b)) for b in entities.get("brands") or [] if b]
        categories = [
            adapter.normalize_category(str(c)) for c in entities.get("categories") or [] if c
        ]
        if extra_categories:
            # 하위 카테고리는 DB 수치 카드가 있는 것만 — 조합 상한(MAX_RULE_CATEGORIES)을 빈
            # 카테고리(body_skincare 등 크롤 대상 아님)가 차지하지 않게
            metric_cards = [c for c in cards if c.kind.value == "metric"]
            carded = {c.object for c in metric_cards} | {c.subject for c in metric_cards}
            for category in extra_categories:
                normalized = adapter.normalize_category(str(category))
                if normalized not in categories and normalized in carded:
                    categories.append(normalized)
        run = evaluate_rules_on_cards(self.reasoner.rules_by_priority, cards, brands, categories)
        inferences = run.fired_results()
        summary = run.summary()
        self.reasoner.record_inference(
            {"combinations": summary["combinations"], "evaluated": summary["evaluated"]},
            inferences,
        )
        return inferences, summary

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

        # 코드 기본값은 config/retrieval_weights.json과 일치해야 한다.
        # rag_chunks가 3으로 남아 있어 설정 파일이 없는 배포 환경에서만
        # 사이클 2 버그 값으로 조용히 회귀했다 (§6.3).
        defaults = {
            "weights": {"kg": 0.4, "rag": 0.4, "inference": 0.2},
            "freshness": {"weekly": 1.0, "quarterly": 0.9, "static": 0.8},
            "max_context_items": {"ontology_facts": 5, "inferences": 5, "rag_chunks": 8},
            # 엔티티 태그 재정렬 보너스 (RRF 점수에 가산, src/rag/entity_tags.py)
            "entity_tag_bonus": {"brands": 0.001, "categories": 0.001},
        }

        config_path = Path(__file__).parent.parent.parent / "config" / "retrieval_weights.json"
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    loaded = json.load(f)
                # 딥 머지: 설정 파일이 일부 키만 담고 있어도 나머지 기본값이 살아남는다.
                # (기존 최상위 교체 방식은 부분 설정이 오면 키가 통째로 사라졌다.)
                for key, default_value in defaults.items():
                    loaded_value = loaded.get(key)
                    if isinstance(default_value, dict) and isinstance(loaded_value, dict):
                        defaults[key] = {**default_value, **loaded_value}
                    elif loaded_value is not None:
                        defaults[key] = loaded_value
                logger.info(f"Retrieval weights loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load retrieval weights: {e}, using defaults")

        return defaults

    def _weighted_merge(
        self,
        context: HybridContext,
        intent_weights: dict[str, float] | None = None,
        degraded: list[dict[str, Any]] | None = None,
    ) -> HybridContext:
        """
        가중치 기반 컨텍스트 병합

        KG facts, RAG chunks, Ontology inferences에 가중치를 부여하고
        최종 점수로 정렬하여 상위 항목만 유지합니다 (추론은 정렬만 하고 자르지 않는다 —
        ``max_context_items.inferences``는 추론에 적용하지 않는다).

        가중치 우선순위:
        1. intent_weights (인텐트 기반 전략에서 전달)
        2. config/retrieval_weights.json (파일 설정)
        3. 기본값: kg=0.4, rag=0.4, inference=0.2

        Args:
            context: 병합 전 HybridContext
            intent_weights: 인텐트 기반 가중치 (optional override)
            degraded: 선택 기능 실패 기록 대상 (호출자가 소유한 리스트, 공유 상태 아님)

        Returns:
            가중치 적용된 HybridContext
        """
        weights = (
            intent_weights if intent_weights is not None else self._retrieval_weights["weights"]
        )
        freshness = self._retrieval_weights["freshness"]
        max_items = self._retrieval_weights["max_context_items"]

        weighted_scores = {}

        # 1. Ontology facts 점수 계산. 온톨로지 정적 사실(``ontology_static``, 플래그 ON에서만)은
        #    KG 사실 상한에서 빼고 뒤에 붙인다 — 상한이 다른 KG 사실을 밀어내지 않고, 정적
        #    사실도 잘리지 않게 (트랙 O3). OFF면 이 목록이 비어 기존 동작과 같다.
        static_facts = [f for f in context.ontology_facts if f.get("type") == "ontology_static"]
        if static_facts:
            context.ontology_facts = [
                f for f in context.ontology_facts if f.get("type") != "ontology_static"
            ]
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
        if static_facts:
            context.ontology_facts = [*context.ontology_facts, *static_facts]

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

            # 점수로 정렬만 한다 — 자르지 않는다 (트랙 3-B). 발화한 규칙은 전부 inference 카드
            # (context.evidence)와 평가 applied_rules에 남아야 한다. 여기서 max_context_items로
            # 자르면 발화 6개 이상인 질의에서 질문한 규칙이 카드·applied_rules에서 사라졌다.
            # 프롬프트 상한은 카드 선별(PROMPT_MAX_PER_KIND[INFERENCE])이 맡는다.
            scored_inferences.sort(key=lambda x: getattr(x, "_weighted_score", 0), reverse=True)
            context.inferences = scored_inferences
            weighted_scores["inferences"] = [
                getattr(i, "_weighted_score", 0) for i in context.inferences
            ]

        # 메타데이터에 점수 저장
        if not context.metadata:
            context.metadata = {}
        context.metadata["weighted_scores"] = weighted_scores

        # ConfidenceFusion: 전체 신뢰도 계산 + 충돌 감지
        fusion_meta = self._compute_fusion_confidence(context, intent_weights, degraded=degraded)
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
        degraded: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        ConfidenceFusion을 사용해 전체 신뢰도를 계산하고 소스 간 충돌을 감지합니다.

        기존 _weighted_merge의 per-item 점수 산정은 유지하고,
        이 메서드는 3개 소스의 aggregate 신뢰도 + 충돌 경고를 추가합니다.

        Args:
            context: 가중 병합 완료된 HybridContext
            intent_weights: 인텐트별 가중치 (kg/rag/inference)
            degraded: 선택 기능 실패 기록 대상 (호출자가 소유한 리스트, 공유 상태 아님)

        Returns:
            dict with confidence, strategy, warnings, source_scores, explanation
        """
        from src.infrastructure.feature_flags import FeatureFlags

        if not FeatureFlags.get_instance().use_confidence_fusion():
            logger.info("Confidence fusion disabled by feature flag")
            return {"confidence": 0.0, "strategy": "disabled", "warnings": []}

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
        except Exception as e:
            # 무기록으로 weighted_sum 폴백하면 융합 전략이 바뀐 사실이 드러나지 않는다
            logger.warning(f"인텐트 분류 실패, fusion_strategy=weighted_sum 폴백: {e}")
            self._record_degraded(degraded, "fusion_strategy_selection", e)

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
                    # str이어야 한다(FusionInferenceResult.insight: str, RRF가 content를 hash).
                    # 예전 getattr(inf, "conclusion", ...)은 InferenceResult에 conclusion 속성이
                    # 없어 늘 str(inf)였다 — 결론 dict 필드(트랙 3-B)가 생겨 dict가 들어갔다
                    insight=str(inf),
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

    def _assemble_evidence(
        self,
        context: HybridContext,
        degraded: list[dict[str, Any]] | None = None,
        input_cards: list[Evidence] | None = None,
        adapter: EvidenceAdapter | None = None,
        ontology_plan: Any | None = None,
        ontology: Any | None = None,
    ) -> EvidenceBundle:
        """최종 검색 결과를 증거 카드로 바꿔 ``context.evidence``·``prompt_evidence``에 담는다.

        - metric_facts → metric 카드, ontology_facts → relation 카드(KG 수치 엣지 제외),
          inferences → inference 카드(``derived_from`` = 규칙 입력 카드 id),
          최종 rag_chunks → document 카드.
        - ``input_cards``: 추론 전에 만든 규칙 입력 카드. 추론 근거 카드가 최종 사실에서 다시
          만들어지지 않으면(사실 상한) 여기서 가져온다.
        - 어댑터 실패는 ``degraded``에 기록하고 나머지 종류는 계속 만든다 (0-B).
        - ``ontology_plan``(플래그 ON, 트랙 O3): 전개 브랜드도 질의 브랜드로 보고 수치 카드를
          정렬하며, 온톨로지 카드(``ontology:registry``)는 관계 종류 상한과 별도로
          ``ONTOLOGY_PROMPT_MAX``장까지 프롬프트에 싣는다 — 기존 관계 카드를 밀어내지 않는다.
        """
        entities = context.entities
        if ontology_plan is not None and ontology is not None:
            entities = dict(context.entities)
            entities["brands"] = list(
                dict.fromkeys(
                    [
                        *(context.entities.get("brands") or []),
                        *(ontology.brand_name(b) or b for b in ontology_plan.expanded_brands),
                    ]
                )
            )
        bundle = assemble_evidence(
            entities=entities,
            metric_facts=context.metric_facts,
            ontology_facts=context.ontology_facts,
            inferences=context.inferences,
            rag_chunks=context.rag_chunks,
            adapter=adapter or self.evidence_adapter,
            input_cards=input_cards or (),
        )
        if ontology_plan is not None:
            bundle.prompt_evidence = self._with_ontology_cards(
                bundle.evidence, bundle.prompt_evidence
            )
        context.evidence = bundle.evidence
        context.prompt_evidence = bundle.prompt_evidence
        if degraded is not None:
            degraded.extend(bundle.degraded)
        return bundle

    @staticmethod
    def _with_ontology_cards(evidence: list[Evidence], prompt: list[Evidence]) -> list[Evidence]:
        """프롬프트 카드에 빠진 온톨로지 카드를 관계 카드 뒤에 끼운다 (상한 별도, 순서 결정적)."""
        from src.domain.entities.evidence import EvidenceKind

        from .evidence_adapters import ONTOLOGY_SOURCE

        present = {card.id for card in prompt}
        already = sum(1 for card in prompt if card.source == ONTOLOGY_SOURCE)
        room = max(0, ONTOLOGY_PROMPT_MAX - already)
        missing = [
            card for card in evidence if card.source == ONTOLOGY_SOURCE and card.id not in present
        ][:room]
        if not missing:
            return prompt
        kinds_before = (EvidenceKind.METRIC, EvidenceKind.RELATION)
        insert_at = 0
        for index, card in enumerate(prompt):
            if card.kind in kinds_before:
                insert_at = index + 1
        return [*prompt[:insert_at], *missing, *prompt[insert_at:]]

    def _combine_contexts(self, context: HybridContext, include_explanations: bool = True) -> str:
        """답변 프롬프트용 컨텍스트 = 프롬프트 카드 렌더링 (설계 E1).

        추론·KG 사실·DB 수치·문서는 모두 카드로만 싣는다 — 같은 정보를 카드 밖 형식으로
        다시 렌더링하지 않는다. v1 ``ContextBuilder``도 같은 렌더러를 쓴다.

        Args:
            context: ``_assemble_evidence``를 거친 HybridContext
            include_explanations: 하위 호환용 인자. 추론의 근거는 카드의
                ``derived_from``으로 표시되므로 더 이상 출력에 영향을 주지 않는다.

        Returns:
            ``render_for_prompt(context.prompt_evidence)`` (카드가 없으면 빈 문자열)
        """
        return render_for_prompt(context.prompt_evidence)

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

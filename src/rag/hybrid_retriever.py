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

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

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
# ============================================================================


class QueryIntent(Enum):
    """쿼리 의도 분류"""

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
    쿼리 의도 분류

    Args:
        query: 사용자 쿼리

    Returns:
        QueryIntent enum 값

    Note:
        키워드 우선순위: TREND > CRISIS > DIAGNOSIS > METRIC > GENERAL
        트렌드/위기 키워드가 있으면 분석 키워드보다 우선
    """
    query_lower = query.lower()

    # 1순위: 트렌드 의도 (Type B 우선)
    # "트렌드 분석" 같은 쿼리는 TREND로 분류
    trend_keywords = [
        "트렌드",
        "요즘",
        "최근",
        "인기",
        "바이럴",
        "키워드",
        "성분",
        "펩타이드",
        "pdrn",
        "글래스스킨",
        "모닝쉐드",
    ]
    if any(kw in query_lower for kw in trend_keywords):
        return QueryIntent.TREND

    # 2순위: 위기 대응 의도 (Type C 우선)
    crisis_keywords = [
        "부정",
        "문제",
        "이슈",
        "대응",
        "어떻게 해",
        "위기",
        "리뷰",
        "불만",
        "인플루언서",
        "마케팅",
        "메시지",
    ]
    if any(kw in query_lower for kw in crisis_keywords):
        return QueryIntent.CRISIS

    # 3순위: 원인 분석 의도 (Type A 우선)
    diagnosis_keywords = [
        "왜",
        "원인",
        "갑자기",
        "급변",
        "떨어",
        "올라",
        "변동",
        "이유",
        "분석",
        "진단",
        "체크",
        "확인",
    ]
    if any(kw in query_lower for kw in diagnosis_keywords):
        return QueryIntent.DIAGNOSIS

    # 4순위: 지표 해석 의도 (Type D 우선)
    metric_keywords = [
        "sos",
        "hhi",
        "cpi",
        "지표",
        "점유율",
        "해석",
        "의미",
        "정의",
        "공식",
        "계산",
    ]
    if any(kw in query_lower for kw in metric_keywords):
        return QueryIntent.METRIC

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
    쿼리에서 엔티티 추출

    추출 대상:
    - 브랜드명 (LANEIGE, COSRX 등)
    - 카테고리 (Lip Care, Skin Care 등)
    - 지표명 (SoS, HHI, CPI 등)
    - 시간 범위 (오늘, 최근 7일 등)

    설정 파일:
    - config/entities.json에서 동적 로드
    - 브랜드 추가/수정 시 설정 파일만 수정하면 됨
    """

    # 설정 파일 경로
    CONFIG_PATH = "config/entities.json"

    # 캐시된 설정 (싱글톤 패턴)
    _config_cache = None
    _config_loaded_at = None

    @classmethod
    def _get_config_ttl_seconds(cls) -> int:
        """설정 파일에서 캐시 TTL 로드 (기본: 300초)"""
        import json
        from pathlib import Path

        thresholds_path = Path(__file__).parent.parent.parent / "config/thresholds.json"
        if thresholds_path.exists():
            try:
                with open(thresholds_path, encoding="utf-8") as f:
                    config = json.load(f)
                    return config.get("system", {}).get("rag", {}).get("ttl_seconds", 300)
            except Exception:
                logger.warning("Suppressed Exception", exc_info=True)
        return 300

    @classmethod
    def _load_config(cls) -> dict:
        """설정 파일에서 엔티티 매핑 로드 (캐싱 적용)"""
        import json
        from pathlib import Path

        # 캐시가 있으면 반환 (설정 파일 TTL 적용)
        if cls._config_cache is not None and cls._config_loaded_at is not None:
            from datetime import datetime, timedelta

            ttl_seconds = cls._get_config_ttl_seconds()
            if datetime.now() - cls._config_loaded_at < timedelta(seconds=ttl_seconds):
                return cls._config_cache

        # 설정 파일 경로 찾기
        config_path = Path(cls.CONFIG_PATH)
        if not config_path.exists():
            # 프로젝트 루트에서 찾기
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / cls.CONFIG_PATH

        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    cls._config_cache = json.load(f)
                    cls._config_loaded_at = datetime.now()
                    logger.info(f"EntityExtractor config loaded from {config_path}")
                    return cls._config_cache
            except Exception as e:
                logger.warning(f"Failed to load entity config: {e}, using defaults")

        # 기본값 반환
        return cls._get_default_config()

    @classmethod
    def _get_default_config(cls) -> dict:
        """기본 설정값 (설정 파일 없을 때 fallback)"""
        return {
            "known_brands": [
                {"name": "laneige", "aliases": ["라네즈"]},
                {"name": "cosrx", "aliases": ["코스알엑스"]},
                {"name": "tirtir", "aliases": ["티르티르"]},
                {"name": "rare beauty", "aliases": ["레어뷰티"]},
                {"name": "innisfree", "aliases": ["이니스프리"]},
                {"name": "etude", "aliases": ["에뛰드"]},
                {"name": "sulwhasoo", "aliases": ["설화수"]},
                {"name": "hera", "aliases": ["헤라"]},
            ],
            "category_map": {
                "lip care": "lip_care",
                "립케어": "lip_care",
                "lip makeup": "lip_makeup",
                "립메이크업": "lip_makeup",
                "skin care": "skin_care",
                "스킨케어": "skin_care",
                "face powder": "face_powder",
                "파우더": "face_powder",
                "beauty": "beauty",
                "뷰티": "beauty",
            },
            "indicator_map": {
                "sos": "sos",
                "점유율": "sos",
                "share of shelf": "sos",
                "hhi": "hhi",
                "시장집중도": "hhi",
                "허핀달": "hhi",
                "cpi": "cpi",
                "가격지수": "cpi",
                "churn": "churn_rate",
                "교체율": "churn_rate",
                "streak": "streak_days",
                "연속": "streak_days",
                "volatility": "rank_volatility",
                "변동성": "rank_volatility",
                "shock": "rank_shock",
                "급변": "rank_shock",
            },
            "time_range_map": {
                "오늘": "today",
                "today": "today",
                "어제": "yesterday",
                "yesterday": "yesterday",
                "이번 주": "week",
                "이번 달": "month",
                "최근 7일": "7days",
                "최근 30일": "30days",
                "3개월": "90days",
                "1개월": "30days",
            },
            "sentiment_map": {
                "moisturizing": "Hydration",
                "hydrating": "Hydration",
                "보습": "Hydration",
                "수분": "Hydration",
                "촉촉": "Hydration",
                "value for money": "Pricing",
                "가성비": "Pricing",
                "affordable": "Pricing",
                "저렴": "Pricing",
                "easy to use": "Usability",
                "사용감": "Usability",
                "편리": "Usability",
                "효과": "Effectiveness",
                "effective": "Effectiveness",
                "scent": "Sensory",
                "향": "Sensory",
                "texture": "Sensory",
                "packaging": "Packaging",
                "패키징": "Packaging",
                "gentle": "Skin_Compatibility",
                "순한": "Skin_Compatibility",
                "리뷰": "sentiment_general",
                "review": "sentiment_general",
            },
        }

    @classmethod
    def get_known_brands(cls) -> list:
        """설정에서 브랜드 목록 가져오기 (이름 + 별칭 평탄화)"""
        config = cls._load_config()
        brands = []
        for brand_info in config.get("known_brands", []):
            if isinstance(brand_info, dict):
                brands.append(brand_info["name"].lower())
                for alias in brand_info.get("aliases", []):
                    brands.append(alias.lower())
            else:
                brands.append(str(brand_info).lower())
        return brands

    @classmethod
    def get_brand_normalization_map(cls) -> dict:
        """별칭 → 정규화된 브랜드명 매핑"""
        config = cls._load_config()
        mapping = {}
        for brand_info in config.get("known_brands", []):
            if isinstance(brand_info, dict):
                name = brand_info["name"].lower()
                mapping[name] = name
                for alias in brand_info.get("aliases", []):
                    mapping[alias.lower()] = name
        return mapping

    # 프로퍼티로 동적 로드 (설정 파일 기반)
    @property
    def KNOWN_BRANDS(self) -> list:
        return self.get_known_brands()

    @property
    def CATEGORY_MAP(self) -> dict:
        return self._load_config().get("category_map", {})

    @property
    def INDICATOR_MAP(self) -> dict:
        return self._load_config().get("indicator_map", {})

    @property
    def TIME_RANGE_MAP(self) -> dict:
        return self._load_config().get("time_range_map", {})

    @property
    def SENTIMENT_MAP(self) -> dict:
        return self._load_config().get("sentiment_map", {})

    def extract(self, query: str, knowledge_graph=None) -> dict[str, list[str]]:
        """
        쿼리에서 엔티티 추출

        Args:
            query: 사용자 쿼리
            knowledge_graph: 지식 그래프 (제품 검색용, optional)

        Returns:
            {
                "brands": [...],
                "categories": [...],
                "indicators": [...],
                "time_range": [...],
                "products": [...]
            }
        """
        import re

        query_lower = query.lower()

        entities = {
            "brands": [],
            "categories": [],
            "indicators": [],
            "time_range": [],
            "products": [],
        }

        # 브랜드 추출 (설정 파일 기반 정규화)
        brand_norm_map = self.get_brand_normalization_map()
        for brand in self.KNOWN_BRANDS:
            if brand in query_lower:
                # 설정 파일의 매핑으로 정규화
                normalized = brand_norm_map.get(brand, brand)
                if normalized not in entities["brands"]:
                    entities["brands"].append(normalized)

        # 카테고리 추출
        for cat_name, cat_id in self.CATEGORY_MAP.items():
            if cat_name in query_lower:
                if cat_id not in entities["categories"]:
                    entities["categories"].append(cat_id)

        # 지표 추출
        for indicator_name, indicator_id in self.INDICATOR_MAP.items():
            if indicator_name in query_lower:
                if indicator_id not in entities["indicators"]:
                    entities["indicators"].append(indicator_id)

        # 시간 범위 추출
        for time_name, time_id in self.TIME_RANGE_MAP.items():
            if time_name in query_lower:
                if time_id not in entities["time_range"]:
                    entities["time_range"].append(time_id)

        # 제품 ASIN 추출 (B0로 시작하는 10자리 형식)
        asin_pattern = r"\bB0[A-Z0-9]{8}\b"
        asins = re.findall(asin_pattern, query)
        if asins:
            entities["products"].extend(asins)

        # 순위 기반 제품 추출 (지식 그래프 활용)
        if knowledge_graph:
            # "1위 제품", "top 1 product" 같은 패턴 감지
            rank_patterns = [
                (r"(\d+)위\s*제품", "ko"),
                (r"top\s*(\d+)\s*product", "en"),
                (r"(\d+)위", "ko"),
                (r"rank\s*(\d+)", "en"),
            ]

            for pattern, _lang in rank_patterns:
                matches = re.findall(pattern, query_lower)
                if matches and entities.get("categories"):
                    # 해당 카테고리의 특정 순위 제품 찾기
                    for rank_str in matches:
                        rank = int(rank_str)
                        for category in entities["categories"]:
                            # 해당 카테고리+순위의 제품 찾기
                            products = knowledge_graph.query(predicate=None, object_=category)
                            for rel in products:
                                if rel.properties.get("rank") == rank:
                                    asin = rel.subject
                                    if asin not in entities["products"]:
                                        entities["products"].append(asin)
                                    break

        # 감성 키워드 추출
        entities["sentiments"] = []
        entities["sentiment_clusters"] = []

        for keyword, cluster in self.SENTIMENT_MAP.items():
            if keyword in query_lower:
                if keyword not in entities["sentiments"]:
                    entities["sentiments"].append(keyword)
                if cluster not in entities["sentiment_clusters"]:
                    entities["sentiment_clusters"].append(cluster)

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

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph | None = None,
        reasoner: OntologyReasoner | None = None,
        doc_retriever: DocumentRetriever | None = None,
        auto_init_rules: bool = True,
    ):
        """
        Args:
            knowledge_graph: 지식 그래프
            reasoner: 온톨로지 추론기
            doc_retriever: RAG 문서 검색기
            auto_init_rules: 비즈니스 규칙 자동 등록
        """
        # 컴포넌트 초기화
        self.kg = knowledge_graph or KnowledgeGraph()
        self.reasoner = reasoner or OntologyReasoner(self.kg)
        self.doc_retriever = doc_retriever or DocumentRetriever()

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

        start_time = datetime.now()

        # 결과 객체 초기화
        context = HybridContext(query=query)

        try:
            # 0. 쿼리 의도 분류
            query_intent = classify_intent(query)
            doc_type_filter = get_doc_type_filter(query_intent)
            logger.debug(f"Query intent: {query_intent.value}, doc_type_filter: {doc_type_filter}")

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
            expanded_query = self._expand_query(search_query, inferences, entities)
            rag_results = await self.doc_retriever.search(
                expanded_query, top_k=5, doc_type_filter=doc_type_filter
            )

            # 필터링된 결과가 부족하면 전체 문서에서 추가 검색
            if len(rag_results) < 3 and doc_type_filter:
                additional_results = await self.doc_retriever.search(
                    expanded_query,
                    top_k=5 - len(rag_results),
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
                            rewritten_query, top_k=5, doc_type_filter=doc_type_filter
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

            # 5.7. 가중치 기반 병합
            context = self._weighted_merge(context)

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
            }

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            context.metadata["error"] = str(e)

        return context

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
    ) -> HybridContext:
        """
        가중치 기반 컨텍스트 병합

        KG facts, RAG chunks, Ontology inferences에 가중치를 부여하고
        최종 점수로 정렬하여 상위 항목만 유지합니다.

        가중치 설정: config/retrieval_weights.json
        기본값: kg=0.4, rag=0.4, inference=0.2

        Args:
            context: 병합 전 HybridContext

        Returns:
            가중치 적용된 HybridContext
        """
        weights = self._retrieval_weights["weights"]
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

        logger.info(
            f"Weighted merge applied: {len(context.ontology_facts)} facts, "
            f"{len(context.rag_chunks)} chunks, {len(context.inferences)} inferences"
        )

        return context

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

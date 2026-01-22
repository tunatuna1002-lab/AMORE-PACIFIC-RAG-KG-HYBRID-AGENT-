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

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.domain.entities.relations import InsightType, InferenceResult, RelationType
from src.ontology.business_rules import register_all_rules

from .retriever import DocumentRetriever


# 로거 설정
logger = logging.getLogger(__name__)


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
    entities: Dict[str, List[str]] = field(default_factory=dict)
    ontology_facts: List[Dict[str, Any]] = field(default_factory=list)
    inferences: List[InferenceResult] = field(default_factory=list)
    rag_chunks: List[Dict[str, Any]] = field(default_factory=list)
    combined_context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "query": self.query,
            "entities": self.entities,
            "ontology_facts": self.ontology_facts,
            "inferences": [inf.to_dict() for inf in self.inferences],
            "rag_chunks": self.rag_chunks,
            "combined_context": self.combined_context,
            "metadata": self.metadata
        }


class EntityExtractor:
    """
    쿼리에서 엔티티 추출

    추출 대상:
    - 브랜드명 (LANEIGE, COSRX 등)
    - 카테고리 (Lip Care, Skin Care 등)
    - 지표명 (SoS, HHI, CPI 등)
    - 시간 범위 (오늘, 최근 7일 등)
    """

    # 알려진 엔티티 매핑
    KNOWN_BRANDS = [
        "laneige", "라네즈",
        "cosrx", "코스알엑스",
        "tirtir", "티르티르",
        "rare beauty", "레어뷰티",
        "innisfree", "이니스프리",
        "etude", "에뛰드",
        "sulwhasoo", "설화수",
        "hera", "헤라"
    ]

    CATEGORY_MAP = {
        "lip care": "lip_care",
        "립케어": "lip_care",
        "lip makeup": "lip_makeup",
        "립메이크업": "lip_makeup",
        "skin care": "skin_care",
        "스킨케어": "skin_care",
        "face powder": "face_powder",
        "파우더": "face_powder",
        "beauty": "beauty",
        "뷰티": "beauty"
    }

    INDICATOR_MAP = {
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
        "급변": "rank_shock"
    }

    TIME_RANGE_MAP = {
        "오늘": "today",
        "today": "today",
        "어제": "yesterday",
        "yesterday": "yesterday",
        "이번 주": "week",
        "이번 달": "month",
        "최근 7일": "7days",
        "최근 30일": "30days",
        "3개월": "90days",
        "1개월": "30days"
    }

    # 감성 관련 키워드 (한/영)
    SENTIMENT_MAP = {
        # 영어 키워드 → 클러스터
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
        "works well": "Effectiveness",
        "scent": "Sensory",
        "향": "Sensory",
        "texture": "Sensory",
        "텍스처": "Sensory",
        "질감": "Sensory",
        "packaging": "Packaging",
        "패키징": "Packaging",
        "포장": "Packaging",
        "gentle": "Skin_Compatibility",
        "순한": "Skin_Compatibility",
        "민감": "Skin_Compatibility",
        "리뷰": "sentiment_general",
        "review": "sentiment_general",
        "고객 반응": "sentiment_general",
        "customer": "sentiment_general",
        "ai 요약": "ai_summary",
        "ai summary": "ai_summary",
        "customers say": "ai_summary",
    }

    def extract(self, query: str, knowledge_graph=None) -> Dict[str, List[str]]:
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
            "products": []
        }

        # 브랜드 추출
        for brand in self.KNOWN_BRANDS:
            if brand in query_lower:
                # 정규화 (영문 소문자)
                normalized = brand.replace("라네즈", "laneige").replace("코스알엑스", "cosrx")
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
        asin_pattern = r'\bB0[A-Z0-9]{8}\b'
        asins = re.findall(asin_pattern, query)
        if asins:
            entities["products"].extend(asins)

        # 순위 기반 제품 추출 (지식 그래프 활용)
        if knowledge_graph:
            # "1위 제품", "top 1 product" 같은 패턴 감지
            rank_patterns = [
                (r'(\d+)위\s*제품', 'ko'),
                (r'top\s*(\d+)\s*product', 'en'),
                (r'(\d+)위', 'ko'),
                (r'rank\s*(\d+)', 'en')
            ]

            for pattern, lang in rank_patterns:
                matches = re.findall(pattern, query_lower)
                if matches and entities.get("categories"):
                    # 해당 카테고리의 특정 순위 제품 찾기
                    for rank_str in matches:
                        rank = int(rank_str)
                        for category in entities["categories"]:
                            # 해당 카테고리+순위의 제품 찾기
                            products = knowledge_graph.query(
                                predicate=None,
                                object_=category
                            )
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


class QueryDecomposer:
    """
    복잡한 쿼리를 하위 쿼리로 분해

    분해 전략:
    1. 비교 쿼리 → 각 대상별 쿼리
    2. 복합 지표 쿼리 → 지표별 쿼리
    3. 시간 범위 쿼리 → 기간별 쿼리

    예시:
    "LANEIGE와 COSRX의 Lip Care 경쟁력 비교" →
    - "LANEIGE Lip Care 현재 성과"
    - "COSRX Lip Care 현재 성과"
    - "Lip Care 경쟁 전략"
    """

    # 비교 패턴
    COMPARISON_PATTERNS = [
        r'(.+)[와과]\s*(.+)[의를]\s*비교',  # "A와 B의 비교"
        r'(.+)\s*vs\.?\s*(.+)',  # "A vs B"
        r'(.+)[와과]\s*(.+)\s*비교',  # "A와 B 비교"
        r'compare\s+(.+)\s+(?:and|with)\s+(.+)',  # "compare A and B"
    ]

    # 복합 지표 패턴
    MULTI_INDICATOR_KEYWORDS = {
        "경쟁력": ["SoS", "순위", "경쟁사"],
        "시장 분석": ["HHI", "SoS", "브랜드 수"],
        "가격 전략": ["CPI", "가격", "프리미엄"],
        "성장 분석": ["순위 변화", "streak", "성장률"],
        "종합 분석": ["SoS", "HHI", "CPI", "순위"]
    }

    # 시간 범위 패턴
    TIME_COMPARISON_PATTERNS = [
        r'(\d+)일\s*(?:전|이전)',  # "7일 전"
        r'지난\s*(\d+)일',  # "지난 7일"
        r'(\d+)주\s*간',  # "2주 간"
        r'추이|변화|트렌드',  # 시계열 분석 암시
    ]

    @classmethod
    def should_decompose(cls, query: str) -> bool:
        """
        쿼리 분해 필요 여부 판단

        Args:
            query: 사용자 쿼리

        Returns:
            분해 필요 여부
        """
        import re
        query_lower = query.lower()

        # 비교 쿼리 감지
        for pattern in cls.COMPARISON_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True

        # 복합 지표 키워드 감지 (2개 이상)
        indicator_count = 0
        for keyword in ["sos", "hhi", "cpi", "순위", "점유율", "집중도", "가격"]:
            if keyword in query_lower:
                indicator_count += 1
        if indicator_count >= 2:
            return True

        # 시간 비교 쿼리 감지
        for pattern in cls.TIME_COMPARISON_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True

        # 길이 기반 (복잡한 쿼리일 가능성)
        if len(query) > 50 and ("분석" in query or "비교" in query or "전략" in query):
            return True

        return False

    @classmethod
    def decompose(
        cls,
        query: str,
        entities: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        쿼리 분해

        Args:
            query: 원본 쿼리
            entities: 추출된 엔티티

        Returns:
            하위 쿼리 리스트
            [{
                "query": str,
                "type": "kg" | "rag" | "both",
                "priority": int,
                "focus": str
            }]
        """
        import re
        sub_queries = []

        brands = entities.get("brands", [])
        categories = entities.get("categories", [])
        indicators = entities.get("indicators", [])

        # 1. 비교 쿼리 분해
        for pattern in cls.COMPARISON_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # 비교 대상 추출
                targets = [match.group(1).strip(), match.group(2).strip()]

                # 각 대상에 대한 쿼리
                for i, target in enumerate(targets):
                    category_str = categories[0] if categories else ""
                    sub_queries.append({
                        "query": f"{target} {category_str} 현재 성과 분석".strip(),
                        "type": "kg",  # KG 우선
                        "priority": 1,
                        "focus": f"target_{i}"
                    })

                # 경쟁/전략 가이드 쿼리
                sub_queries.append({
                    "query": f"{category_str} 경쟁 전략 가이드라인".strip() if category_str else "경쟁 전략 가이드라인",
                    "type": "rag",  # RAG 우선
                    "priority": 2,
                    "focus": "strategy"
                })

                return sub_queries

        # 2. 복합 지표 분해
        if len(indicators) >= 2:
            # 각 지표별 쿼리
            for indicator in indicators:
                brand_str = brands[0] if brands else "LANEIGE"
                indicator_name = {
                    "sos": "점유율(SoS)",
                    "hhi": "시장집중도(HHI)",
                    "cpi": "가격지수(CPI)"
                }.get(indicator, indicator)

                sub_queries.append({
                    "query": f"{brand_str} {indicator_name} 분석",
                    "type": "both",
                    "priority": 1,
                    "focus": indicator
                })

            # 종합 해석 쿼리
            sub_queries.append({
                "query": "지표 조합 해석 전략",
                "type": "rag",
                "priority": 2,
                "focus": "interpretation"
            })

            return sub_queries

        # 3. 시간 범위 분해
        for pattern in cls.TIME_COMPARISON_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                brand_str = brands[0] if brands else "LANEIGE"

                # 현재 상태 쿼리
                sub_queries.append({
                    "query": f"{brand_str} 현재 순위 성과",
                    "type": "kg",
                    "priority": 1,
                    "focus": "current"
                })

                # 변화 분석 쿼리
                sub_queries.append({
                    "query": f"{brand_str} 순위 변화 추이",
                    "type": "kg",
                    "priority": 1,
                    "focus": "trend"
                })

                # 해석 가이드
                sub_queries.append({
                    "query": "순위 변화 해석 가이드",
                    "type": "rag",
                    "priority": 2,
                    "focus": "guide"
                })

                return sub_queries

        # 4. 기본 분해 (긴 복합 쿼리)
        if len(query) > 50:
            brand_str = brands[0] if brands else ""
            category_str = categories[0] if categories else ""

            # 현황 쿼리
            if brand_str:
                sub_queries.append({
                    "query": f"{brand_str} {category_str} 현황".strip(),
                    "type": "kg",
                    "priority": 1,
                    "focus": "status"
                })

            # 전략 쿼리
            sub_queries.append({
                "query": query,  # 원본 쿼리를 RAG에 전달
                "type": "rag",
                "priority": 2,
                "focus": "strategy"
            })

            return sub_queries

        # 분해 불필요 - 원본 쿼리 반환
        return [{
            "query": query,
            "type": "both",
            "priority": 1,
            "focus": "original"
        }]


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
        knowledge_graph: Optional[KnowledgeGraph] = None,
        reasoner: Optional[OntologyReasoner] = None,
        doc_retriever: Optional[DocumentRetriever] = None,
        auto_init_rules: bool = True
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

        # 비즈니스 규칙 자동 등록
        if auto_init_rules and not self.reasoner.rules:
            register_all_rules(self.reasoner)
            logger.info(f"Registered {len(self.reasoner.rules)} business rules")

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
        current_metrics: Optional[Dict[str, Any]] = None,
        include_explanations: bool = True
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
            # 1. 엔티티 추출 (지식 그래프 전달로 제품 ASIN도 추출 가능)
            entities = self.entity_extractor.extract(query, knowledge_graph=self.kg)
            context.entities = entities
            logger.debug(f"Extracted entities: {entities}")

            # 2. 지식 그래프에서 사실 조회
            ontology_facts = self._query_knowledge_graph(entities)
            context.ontology_facts = ontology_facts

            # 3. 추론 컨텍스트 구성
            inference_context = self._build_inference_context(
                entities, current_metrics or {}
            )

            # 4. 온톨로지 추론 실행
            inferences = self.reasoner.infer(inference_context)
            context.inferences = inferences
            logger.debug(f"Generated {len(inferences)} inferences")

            # 5. RAG 문서 검색 (추론 결과로 쿼리 확장)
            expanded_query = self._expand_query(query, inferences, entities, ontology_facts)
            rag_results = await self.doc_retriever.search(expanded_query, top_k=5)
            context.rag_chunks = rag_results

            # 6. 통합 컨텍스트 생성
            context.combined_context = self._combine_contexts(
                context, include_explanations
            )

            # 메타데이터
            context.metadata = {
                "retrieval_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "ontology_facts_count": len(ontology_facts),
                "inferences_count": len(inferences),
                "rag_chunks_count": len(rag_results),
                "query_expanded": expanded_query != query
            }

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            context.metadata["error"] = str(e)

        return context

    async def retrieve_with_decomposition(
        self,
        query: str,
        current_metrics: Optional[Dict[str, Any]] = None,
        include_explanations: bool = True
    ) -> HybridContext:
        """
        쿼리 분해 기반 하이브리드 검색

        복잡한 쿼리를 분해하여 각각 처리 후 결과 합성

        Args:
            query: 사용자 쿼리
            current_metrics: 현재 지표
            include_explanations: 설명 포함

        Returns:
            HybridContext
        """
        # 초기화 확인
        if not self._initialized:
            await self.initialize()

        start_time = datetime.now()

        # 엔티티 추출
        entities = self.entity_extractor.extract(query, knowledge_graph=self.kg)

        # 분해 필요 여부 판단
        if not QueryDecomposer.should_decompose(query):
            # 단순 쿼리 - 기존 검색 사용
            return await self.retrieve(query, current_metrics, include_explanations)

        # 쿼리 분해
        sub_queries = QueryDecomposer.decompose(query, entities)
        logger.info(f"Query decomposed into {len(sub_queries)} sub-queries")

        # 결과 수집
        all_ontology_facts = []
        all_inferences = []
        all_rag_chunks = []

        # 각 하위 쿼리 처리
        for sq in sub_queries:
            sub_query = sq["query"]
            query_type = sq["type"]

            if query_type in ["kg", "both"]:
                # KG 조회
                sub_entities = self.entity_extractor.extract(sub_query, knowledge_graph=self.kg)
                facts = self._query_knowledge_graph(sub_entities)
                all_ontology_facts.extend(facts)

                # 추론
                if query_type == "kg" or query_type == "both":
                    inference_ctx = self._build_inference_context(sub_entities, current_metrics or {})
                    inferences = self.reasoner.infer(inference_ctx)
                    all_inferences.extend(inferences)

            if query_type in ["rag", "both"]:
                # RAG 검색
                rag_results = await self.doc_retriever.search(sub_query, top_k=3)
                all_rag_chunks.extend(rag_results)

        # 중복 제거
        seen_facts = set()
        unique_facts = []
        for fact in all_ontology_facts:
            key = (fact.get("type"), fact.get("entity"))
            if key not in seen_facts:
                seen_facts.add(key)
                unique_facts.append(fact)

        seen_rules = set()
        unique_inferences = []
        for inf in all_inferences:
            if inf.rule_name not in seen_rules:
                seen_rules.add(inf.rule_name)
                unique_inferences.append(inf)

        seen_chunks = set()
        unique_chunks = []
        for chunk in all_rag_chunks:
            if chunk["id"] not in seen_chunks:
                seen_chunks.add(chunk["id"])
                unique_chunks.append(chunk)

        # 결과 조합
        context = HybridContext(
            query=query,
            entities=entities,
            ontology_facts=unique_facts,
            inferences=unique_inferences,
            rag_chunks=sorted(unique_chunks, key=lambda x: x.get("score", 0), reverse=True)[:5]
        )

        # 통합 컨텍스트 생성
        context.combined_context = self._combine_contexts(context, include_explanations)

        # 메타데이터
        context.metadata = {
            "retrieval_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
            "decomposed": True,
            "sub_query_count": len(sub_queries),
            "sub_queries": [sq["query"] for sq in sub_queries],
            "ontology_facts_count": len(unique_facts),
            "inferences_count": len(unique_inferences),
            "rag_chunks_count": len(unique_chunks)
        }

        return context

    def _query_knowledge_graph(
        self,
        entities: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
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
                facts.append({
                    "type": "brand_info",
                    "entity": brand,
                    "data": brand_meta
                })

            # 브랜드의 제품들
            products = self.kg.get_brand_products(brand)
            if products:
                facts.append({
                    "type": "brand_products",
                    "entity": brand,
                    "data": {
                        "product_count": len(products),
                        "products": products[:10]  # 상위 10개
                    }
                })

            # 경쟁사
            competitors = self.kg.get_competitors(brand)
            if competitors:
                facts.append({
                    "type": "competitors",
                    "entity": brand,
                    "data": competitors[:5]  # 상위 5개
                })

        # 카테고리 관련 사실
        for category in entities.get("categories", []):
            # 카테고리 브랜드 정보
            category_brands = self.kg.get_category_brands(category)
            if category_brands:
                facts.append({
                    "type": "category_brands",
                    "entity": category,
                    "data": {
                        "brand_count": len(category_brands),
                        "top_brands": category_brands[:5]
                    }
                })

            # 카테고리 계층 정보 (부모/자식 관계)
            try:
                hierarchy = self.kg.get_category_hierarchy(category)
                if hierarchy and not hierarchy.get("error"):
                    facts.append({
                        "type": "category_hierarchy",
                        "entity": category,
                        "data": {
                            "name": hierarchy.get("name", ""),
                            "level": hierarchy.get("level", 0),
                            "path": hierarchy.get("path", []),
                            "ancestors": hierarchy.get("ancestors", []),
                            "descendants": hierarchy.get("descendants", [])
                        }
                    })
            except Exception:
                pass

        # 감성 관련 사실 조회
        sentiment_clusters = entities.get("sentiment_clusters", [])
        if sentiment_clusters or entities.get("sentiments"):
            # 제품이 지정된 경우 해당 제품의 감성 조회
            for asin in entities.get("products", []):
                try:
                    product_sentiments = self.kg.get_product_sentiments(asin)
                    if product_sentiments.get("sentiment_tags") or product_sentiments.get("ai_summary"):
                        facts.append({
                            "type": "product_sentiment",
                            "entity": asin,
                            "data": product_sentiments
                        })
                except Exception:
                    pass

            # 브랜드가 지정된 경우 브랜드 감성 프로필 조회
            for brand in entities.get("brands", []):
                try:
                    brand_sentiment = self.kg.get_brand_sentiment_profile(brand)
                    if brand_sentiment.get("all_tags"):
                        facts.append({
                            "type": "brand_sentiment",
                            "entity": brand,
                            "data": brand_sentiment
                        })
                except Exception:
                    pass

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
                                facts.append({
                                    "type": "sentiment_products",
                                    "entity": tag,
                                    "data": {
                                        "sentiment_tag": tag,
                                        "cluster": cluster,
                                        "product_count": len(products_with_sentiment),
                                        "products": products_with_sentiment[:5]
                                    }
                                })
                                break
                    except Exception:
                        pass

        return facts

    def _build_inference_context(
        self,
        entities: Dict[str, List[str]],
        current_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
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
            if bm.get("is_laneige") or bm.get("brand_name", "").lower() == context.get("brand", "").lower():
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
                    pass

            # 제품별 감성 데이터
            if context.get("asin"):
                try:
                    product_sentiment = self.kg.get_product_sentiments(context["asin"])
                    context["ai_summary"] = product_sentiment.get("ai_summary")
                    if not context.get("sentiment_tags"):
                        context["sentiment_tags"] = product_sentiment.get("sentiment_tags", [])
                        context["sentiment_clusters"] = product_sentiment.get("sentiment_clusters", {})
                except Exception:
                    pass

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
                            competitor_clusters[cluster] = competitor_clusters.get(cluster, 0) + count
                    except Exception:
                        pass
                context["competitor_sentiment_tags"] = list(set(competitor_tags))
                context["competitor_sentiment_clusters"] = competitor_clusters

        return context

    def _expand_query(
        self,
        query: str,
        inferences: List[InferenceResult],
        entities: Dict[str, List[str]],
        ontology_facts: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Graph-Guided Query Expansion

        Args:
            query: 원본 쿼리
            inferences: 추론 결과
            entities: 엔티티
            ontology_facts: KG에서 조회한 사실 (NEW)

        Returns:
            확장된 쿼리
        """
        expansion_terms = []

        # 1. KG 기반 확장 (경쟁사, 관련 제품명)
        if ontology_facts:
            for fact in ontology_facts:
                fact_type = fact.get("type", "")
                data = fact.get("data", {})

                # 경쟁사 이름 추가
                if fact_type == "competitors":
                    for comp in data[:3]:  # 상위 3개
                        brand = comp.get("brand", "") if isinstance(comp, dict) else str(comp)
                        if brand and brand not in expansion_terms:
                            expansion_terms.append(brand)

                # 카테고리 이름 추가
                elif fact_type == "category_brands":
                    top_brands = data.get("top_brands", [])
                    for brand_info in top_brands[:2]:
                        brand = brand_info.get("brand", "") if isinstance(brand_info, dict) else str(brand_info)
                        if brand and brand not in expansion_terms:
                            expansion_terms.append(brand)

                # 카테고리 계층에서 관련 카테고리 추가
                elif fact_type == "category_hierarchy":
                    cat_name = data.get("name", "")
                    if cat_name:
                        expansion_terms.append(cat_name)
                    # 상위 카테고리도 추가
                    for ancestor in data.get("ancestors", [])[:1]:
                        anc_name = ancestor.get("name", "") if isinstance(ancestor, dict) else str(ancestor)
                        if anc_name and anc_name not in expansion_terms:
                            expansion_terms.append(anc_name)

        # 2. 추론된 인사이트 유형에 따른 키워드 추가 (기존 로직)
        insight_types = set(inf.insight_type for inf in inferences)

        if InsightType.MARKET_POSITION in insight_types or InsightType.MARKET_DOMINANCE in insight_types:
            expansion_terms.append("시장 포지션 해석")

        if InsightType.RISK_ALERT in insight_types:
            expansion_terms.append("위험 신호 대응")

        if InsightType.COMPETITIVE_THREAT in insight_types:
            expansion_terms.append("경쟁 위협 분석")

        if InsightType.GROWTH_OPPORTUNITY in insight_types or InsightType.GROWTH_MOMENTUM in insight_types:
            expansion_terms.append("성장 기회 전략")

        if InsightType.PRICE_QUALITY_GAP in insight_types or InsightType.PRICE_POSITION in insight_types:
            expansion_terms.append("가격 전략 해석")

        # 3. 지표 관련 확장 (기존 로직)
        for indicator in entities.get("indicators", []):
            if indicator == "sos":
                expansion_terms.append("SoS 점유율 해석")
            elif indicator == "hhi":
                expansion_terms.append("HHI 시장집중도 해석")
            elif indicator == "cpi":
                expansion_terms.append("CPI 가격지수 해석")

        # 확장된 쿼리 생성
        if expansion_terms:
            return f"{query} {' '.join(expansion_terms)}"

        return query

    def _combine_contexts(
        self,
        context: HybridContext,
        include_explanations: bool = True
    ) -> str:
        """
        Ontology-Guided Context Ranking

        Priority Order:
        1. High-confidence Inferences (>= 0.8) - 가장 신뢰도 높은 인사이트
        2. Direct KG Facts (brand_info, competitors) - 직접 관계
        3. High-score RAG Chunks - 높은 유사도 문서
        4. Medium-confidence Inferences (0.5-0.8)
        5. Supporting Context - 보조 정보

        Args:
            context: HybridContext
            include_explanations: 추론 설명 포함

        Returns:
            통합된 컨텍스트 문자열 (우선순위 기반)
        """
        parts = []

        # 신뢰도 기준으로 인사이트 분류
        high_conf_inferences = [inf for inf in context.inferences if inf.confidence >= 0.8]
        medium_conf_inferences = [inf for inf in context.inferences if 0.5 <= inf.confidence < 0.8]
        low_conf_inferences = [inf for inf in context.inferences if inf.confidence < 0.5]

        # KG 사실을 유형별로 분류
        direct_facts = []  # brand_info, competitors, brand_products
        category_facts = []  # category_brands, category_hierarchy
        sentiment_facts = []  # product_sentiment, brand_sentiment

        for fact in context.ontology_facts:
            fact_type = fact.get("type", "")
            if fact_type in ["brand_info", "competitors", "brand_products"]:
                direct_facts.append(fact)
            elif fact_type in ["category_brands", "category_hierarchy"]:
                category_facts.append(fact)
            elif fact_type in ["product_sentiment", "brand_sentiment", "sentiment_products"]:
                sentiment_facts.append(fact)

        # RAG 청크를 점수순으로 정렬
        sorted_rag_chunks = sorted(
            context.rag_chunks,
            key=lambda x: x.get("rrf_score", x.get("score", 0)),
            reverse=True
        )
        high_score_chunks = sorted_rag_chunks[:2]  # 상위 2개
        remaining_chunks = sorted_rag_chunks[2:4]  # 나머지

        # ============================================================
        # 1. 핵심 인사이트 (High Confidence >= 0.8)
        # ============================================================
        if high_conf_inferences:
            parts.append("## 🎯 핵심 분석 결과\n")
            for i, inf in enumerate(high_conf_inferences, 1):
                parts.append(f"### {inf.insight_type.value.replace('_', ' ').title()}")
                parts.append(f"- **결론**: {inf.insight}")
                if inf.recommendation:
                    parts.append(f"- **권장 액션**: {inf.recommendation}")
                parts.append(f"- **신뢰도**: {inf.confidence:.0%} ⭐")
                if include_explanations and inf.evidence:
                    conditions = inf.evidence.get("satisfied_conditions", [])
                    if conditions:
                        parts.append(f"- **근거**: {', '.join(conditions[:3])}")
                parts.append("")

        # ============================================================
        # 2. 직접 관계 정보 (KG Direct Facts)
        # ============================================================
        if direct_facts:
            parts.append("## 📊 핵심 정보 (Knowledge Graph)\n")
            for fact in direct_facts[:4]:
                fact_type = fact.get("type", "")
                entity = fact.get("entity", "")
                data = fact.get("data", {})

                if fact_type == "brand_info":
                    sos = data.get("sos", 0)
                    if sos:
                        parts.append(f"- **{entity}** SoS: {sos*100:.1f}%")
                    if data.get("avg_rank"):
                        parts.append(f"  - 평균 순위: {data['avg_rank']:.1f}")
                    if data.get("product_count"):
                        parts.append(f"  - 제품 수: {data['product_count']}개")

                elif fact_type == "competitors":
                    comps = [c.get("brand", str(c)) if isinstance(c, dict) else str(c) for c in data[:3]]
                    if comps:
                        parts.append(f"- **{entity}** 주요 경쟁사: {', '.join(comps)}")

                elif fact_type == "brand_products":
                    parts.append(f"- **{entity}** 제품 수: {data.get('product_count', 0)}개")
            parts.append("")

        # ============================================================
        # 3. 핵심 참고 문서 (High Score RAG)
        # ============================================================
        if high_score_chunks:
            parts.append("## 📚 핵심 가이드라인\n")
            for chunk in high_score_chunks:
                title = chunk.get("metadata", {}).get("title", "")
                content = chunk.get("content", "")
                score = chunk.get("rrf_score", chunk.get("score", 0))

                if title:
                    parts.append(f"### {title}")
                # 내용 축약 (400자)
                if len(content) > 400:
                    content = content[:400] + "..."
                parts.append(content)
                parts.append("")

        # ============================================================
        # 4. 추가 분석 (Medium Confidence)
        # ============================================================
        if medium_conf_inferences:
            parts.append("## 📋 추가 분석\n")
            for inf in medium_conf_inferences[:3]:
                parts.append(f"- **{inf.insight_type.value.replace('_', ' ').title()}**: {inf.insight}")
                if inf.recommendation:
                    parts.append(f"  - 권장: {inf.recommendation}")
            parts.append("")

        # ============================================================
        # 5. 보조 정보
        # ============================================================
        supporting_info = []

        # 카테고리 정보
        if category_facts:
            for fact in category_facts[:2]:
                fact_type = fact.get("type", "")
                entity = fact.get("entity", "")
                data = fact.get("data", {})

                if fact_type == "category_brands":
                    top = [b.get("brand", "") for b in data.get("top_brands", [])[:3] if isinstance(b, dict)]
                    if top:
                        supporting_info.append(f"- {entity} Top 브랜드: {', '.join(top)}")

                elif fact_type == "category_hierarchy":
                    name = data.get("name", entity)
                    level = data.get("level", 0)
                    if name:
                        supporting_info.append(f"- {name} (Level {level})")

        # 감성 정보
        if sentiment_facts:
            for fact in sentiment_facts[:2]:
                entity = fact.get("entity", "")
                data = fact.get("data", {})
                tags = data.get("sentiment_tags", data.get("all_tags", []))[:3]
                if tags:
                    supporting_info.append(f"- {entity} 감성: {', '.join(tags)}")

        # 나머지 RAG 청크
        if remaining_chunks:
            for chunk in remaining_chunks[:1]:
                title = chunk.get("metadata", {}).get("title", "")
                if title:
                    supporting_info.append(f"- 참고: {title}")

        # Low confidence inferences
        if low_conf_inferences:
            for inf in low_conf_inferences[:2]:
                supporting_info.append(f"- (참고) {inf.insight}")

        if supporting_info:
            parts.append("## 💡 보조 정보\n")
            parts.extend(supporting_info)
            parts.append("")

        return "\n".join(parts)

    async def retrieve_for_entity(
        self,
        entity: str,
        entity_type: str = "brand",
        current_metrics: Optional[Dict[str, Any]] = None
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
        self,
        crawl_data: Optional[Dict[str, Any]] = None,
        metrics_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, int]:
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

    def get_stats(self) -> Dict[str, Any]:
        """검색기 통계"""
        return {
            "knowledge_graph": self.kg.get_stats(),
            "reasoner": self.reasoner.get_inference_stats(),
            "rules_count": len(self.reasoner.rules),
            "initialized": self._initialized
        }

"""
Hybrid Retriever
Ontology + RAG 하이브리드 검색기

기능:
1. 온톨로지에서 구조화된 지식 추론
2. RAG에서 비구조화된 가이드라인 검색
3. 두 결과를 통합하여 풍부한 컨텍스트 생성

Flow:
    Query → Entity Extraction → [Ontology Reasoning + RAG Search] → Context Merge → LLM
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.ontology.relations import InsightType, InferenceResult, RelationType
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
            expanded_query = self._expand_query(query, inferences, entities)
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

        return context

    def _expand_query(
        self,
        query: str,
        inferences: List[InferenceResult],
        entities: Dict[str, List[str]]
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

    def _combine_contexts(
        self,
        context: HybridContext,
        include_explanations: bool = True
    ) -> str:
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
                parts.append(f"### 인사이트 {i}: {inf.insight_type.value.replace('_', ' ').title()}")
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
                        parts.append(f"- **{entity}** SoS: {sos*100:.1f}%")
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

"""
KG Auto-Enricher
=================
크롤링 데이터에서 자동으로 엔티티·관계를 추출하여 지식 그래프를 보강합니다.

LangGraph LLMGraphTransformer 패턴의 경량 구현.

추출 관계:
- 브랜드-제품 (HAS_PRODUCT)
- 제품-카테고리 (BELONGS_TO_CATEGORY)
- 경쟁 관계 (COMPETES_WITH)
- 가격 위치 (PRICE_POSITION)
"""

import logging
from typing import Any

from src.domain.entities.relations import Relation, RelationType

logger = logging.getLogger(__name__)


class Triple:
    """추출된 트리플 (경량 버전)"""

    def __init__(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 1.0,
        properties: dict[str, Any] | None = None,
    ):
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.confidence = confidence
        self.properties = properties or {}


class KGEnricher:
    """
    지식 그래프 자동 보강기

    크롤링 데이터에서 구조적 관계를 추출하고 KG에 저장합니다.

    2단계:
    1. 규칙 기반 추출 (빠르고 확실한 관계) — LLM 불필요
    2. LLM 기반 추출 (복합 관계, 경쟁 분석) — 선택적

    Usage:
        enricher = KGEnricher(knowledge_graph)
        triples = enricher.enrich_from_crawl(crawl_data)
        enricher.store_triples(triples)
    """

    # 최소 신뢰도 임계값 (이 이상만 저장)
    MIN_CONFIDENCE = 0.7

    def __init__(self, knowledge_graph=None, min_confidence: float = 0.7):
        """
        Args:
            knowledge_graph: KnowledgeGraph 인스턴스
            min_confidence: 최소 신뢰도 임계값
        """
        self.kg = knowledge_graph
        self.min_confidence = min_confidence
        self._stats = {
            "total_enrichments": 0,
            "triples_extracted": 0,
            "triples_stored": 0,
            "triples_filtered": 0,
        }

    def enrich_from_crawl(self, crawl_data: dict[str, Any]) -> list[Triple]:
        """
        크롤링 데이터에서 트리플 추출

        Args:
            crawl_data: 크롤링 결과 데이터
                {
                    "category": "lip_care",
                    "products": [
                        {"asin": "B0...", "brand": "LANEIGE", "rank": 1, "price": 24.0, "title": "..."},
                        ...
                    ]
                }

        Returns:
            추출된 Triple 리스트
        """
        self._stats["total_enrichments"] += 1
        triples = []

        products = crawl_data.get("products", [])
        category = crawl_data.get("category", "unknown")

        if not products:
            return triples

        # 1. 브랜드-제품 관계
        triples.extend(self._extract_brand_product_relations(products, category))

        # 2. 경쟁 관계 (같은 카테고리 상위 브랜드끼리)
        triples.extend(self._extract_competitive_relations(products, category))

        # 3. 가격 포지셔닝
        triples.extend(self._extract_price_positions(products, category))

        # 4. 카테고리 점유 관계
        triples.extend(self._extract_category_dominance(products, category))

        self._stats["triples_extracted"] += len(triples)
        logger.info(
            f"KG enrichment: {len(triples)} triples extracted from {len(products)} products"
        )

        return triples

    def _extract_brand_product_relations(self, products: list[dict], category: str) -> list[Triple]:
        """브랜드-제품 관계 추출"""
        triples = []
        seen = set()

        for product in products:
            brand = product.get("brand", "").strip()
            asin = product.get("asin", "")

            if not brand or not asin:
                continue

            key = (brand.lower(), asin)
            if key in seen:
                continue
            seen.add(key)

            # 브랜드 → 제품
            triples.append(
                Triple(
                    subject=brand.lower(),
                    predicate="HAS_PRODUCT",
                    object=asin,
                    confidence=1.0,
                    properties={
                        "category": category,
                        "rank": product.get("rank"),
                        "title": product.get("title", "")[:100],
                    },
                )
            )

            # 제품 → 카테고리
            triples.append(
                Triple(
                    subject=asin,
                    predicate="BELONGS_TO_CATEGORY",
                    object=category,
                    confidence=1.0,
                    properties={"rank": product.get("rank")},
                )
            )

        return triples

    def _extract_competitive_relations(self, products: list[dict], category: str) -> list[Triple]:
        """경쟁 관계 추출 (같은 카테고리 상위 브랜드)"""
        triples = []

        # 브랜드별 제품 수 집계
        brand_products: dict[str, list] = {}
        for product in products:
            brand = product.get("brand", "").strip().lower()
            if brand:
                brand_products.setdefault(brand, []).append(product)

        # 상위 브랜드 추출 (제품 2개 이상)
        significant_brands = [b for b, prods in brand_products.items() if len(prods) >= 2]

        # 상위 브랜드끼리 경쟁 관계 생성
        seen_pairs = set()
        for i, brand_a in enumerate(significant_brands):
            for brand_b in significant_brands[i + 1 :]:
                pair = tuple(sorted([brand_a, brand_b]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    triples.append(
                        Triple(
                            subject=brand_a,
                            predicate="COMPETES_WITH",
                            object=brand_b,
                            confidence=0.8,
                            properties={"category": category},
                        )
                    )

        return triples

    def _extract_price_positions(self, products: list[dict], category: str) -> list[Triple]:
        """가격 포지셔닝 추출"""
        triples = []

        # 가격 데이터 수집
        prices = []
        for product in products:
            price = product.get("price")
            if price and isinstance(price, (int, float)) and price > 0:
                prices.append(price)

        if not prices:
            return triples

        avg_price = sum(prices) / len(prices)

        # 브랜드별 평균 가격 계산
        brand_prices: dict[str, list[float]] = {}
        for product in products:
            brand = product.get("brand", "").strip().lower()
            price = product.get("price")
            if brand and price and isinstance(price, (int, float)) and price > 0:
                brand_prices.setdefault(brand, []).append(price)

        for brand, brand_price_list in brand_prices.items():
            brand_avg = sum(brand_price_list) / len(brand_price_list)

            if brand_avg > avg_price * 1.3:
                position = "premium"
            elif brand_avg < avg_price * 0.7:
                position = "budget"
            else:
                position = "mid_range"

            triples.append(
                Triple(
                    subject=brand,
                    predicate="PRICE_POSITION",
                    object=position,
                    confidence=0.75,
                    properties={
                        "category": category,
                        "avg_price": round(brand_avg, 2),
                        "market_avg": round(avg_price, 2),
                    },
                )
            )

        return triples

    def _extract_category_dominance(self, products: list[dict], category: str) -> list[Triple]:
        """카테고리 지배력 추출"""
        triples = []

        # 브랜드별 제품 수
        brand_counts: dict[str, int] = {}
        for product in products:
            brand = product.get("brand", "").strip().lower()
            if brand:
                brand_counts[brand] = brand_counts.get(brand, 0) + 1

        total = len(products)
        if total == 0:
            return triples

        for brand, count in brand_counts.items():
            share = count / total
            if share >= 0.1:  # 10% 이상만
                triples.append(
                    Triple(
                        subject=brand,
                        predicate="DOMINATES_CATEGORY",
                        object=category,
                        confidence=min(0.5 + share, 1.0),
                        properties={
                            "product_count": count,
                            "share": round(share, 4),
                        },
                    )
                )

        return triples

    def store_triples(self, triples: list[Triple]) -> int:
        """
        트리플을 KG에 저장

        Args:
            triples: 저장할 Triple 리스트

        Returns:
            저장된 트리플 수
        """
        if not self.kg:
            logger.warning("No knowledge graph available for storing triples")
            return 0

        stored = 0
        for triple in triples:
            if triple.confidence < self.min_confidence:
                self._stats["triples_filtered"] += 1
                continue

            try:
                # Triple을 Relation으로 변환
                relation = self._triple_to_relation(triple)
                self.kg.add_relation(relation)
                stored += 1
            except Exception as e:
                logger.debug(f"Failed to store triple: {e}")

        self._stats["triples_stored"] += stored
        logger.info(f"Stored {stored}/{len(triples)} triples to KG")
        return stored

    def _triple_to_relation(self, triple: Triple) -> Relation:
        """Triple을 Relation 객체로 변환"""
        # predicate 문자열을 RelationType으로 매핑
        predicate_map = {
            "HAS_PRODUCT": RelationType.HAS_PRODUCT,
            "BELONGS_TO_CATEGORY": RelationType.BELONGS_TO_CATEGORY,
            "COMPETES_WITH": RelationType.COMPETES_WITH,
            "PRICE_POSITION": RelationType.HAS_POSITION,
            "DOMINATES_CATEGORY": RelationType.HAS_POSITION,  # 지배력도 포지션으로
        }

        predicate = predicate_map.get(triple.predicate, RelationType.HAS_PRODUCT)

        return Relation(
            subject=triple.subject,
            predicate=predicate,
            object=triple.object,
            properties=triple.properties,
            confidence=triple.confidence,
            source="kg_enricher",
        )

    def enrich_and_store(self, crawl_data: dict[str, Any]) -> dict[str, int]:
        """추출 + 저장 일괄 실행"""
        triples = self.enrich_from_crawl(crawl_data)
        stored = self.store_triples(triples)
        return {
            "extracted": len(triples),
            "stored": stored,
            "filtered": len(triples) - stored,
        }

    def get_stats(self) -> dict[str, Any]:
        """통계 반환"""
        return self._stats

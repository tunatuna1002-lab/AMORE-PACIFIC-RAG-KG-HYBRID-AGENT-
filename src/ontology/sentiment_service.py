"""
Sentiment Service
=================
KnowledgeGraph에서 분리된 감성 분석 서비스

책임:
- 제품 감성 데이터 로드 (AI Customers Say, Sentiment Tags)
- 제품/브랜드 감성 프로필 조회
- 감성 기반 제품 검색/비교
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.domain.entities import RelationType

if TYPE_CHECKING:
    from src.ontology.knowledge_graph import KnowledgeGraph


class SentimentService:
    """
    감성 분석 서비스

    KnowledgeGraph의 감성 관련 로직을 분리하여 SRP 준수.
    KnowledgeGraph를 composition으로 사용합니다.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph):
        """
        Args:
            knowledge_graph: 의존할 KnowledgeGraph 인스턴스
        """
        self._kg = knowledge_graph

    def load_from_data(self, sentiment_data: dict[str, Any]) -> int:
        """
        AI Customers Say 및 감성 태그 데이터에서 관계 로드

        Args:
            sentiment_data: AmazonProductScraper 결과
                {
                    "products": {
                        "B0BSHRYY1S": {
                            "asin": "B0BSHRYY1S",
                            "ai_customers_say": "Customers like the moisturizing...",
                            "sentiment_tags": ["Moisturizing", "Value for money"],
                            "success": True
                        },
                        ...
                    },
                    "collected_at": "2025-01-15"
                }

        Returns:
            추가된 관계 수
        """
        from .relations import (
            create_ai_summary_relation,
            create_sentiment_relation,
            get_cluster_for_sentiment,
        )

        added = 0
        collected_at = sentiment_data.get("collected_at", "")

        products = sentiment_data.get("products", {})
        for asin, product_data in products.items():
            if not product_data.get("success"):
                continue

            # AI Summary 관계 추가
            ai_summary = product_data.get("ai_customers_say")
            if ai_summary:
                rel = create_ai_summary_relation(
                    product_asin=asin, ai_summary=ai_summary, collected_at=collected_at
                )
                if self._kg.add_relation(rel):
                    added += 1

                # 제품 메타데이터 업데이트
                self._kg.set_entity_metadata(
                    asin,
                    {"ai_summary": ai_summary, "ai_summary_collected_at": collected_at},
                )

            # Sentiment Tag 관계 추가
            sentiment_tags = product_data.get("sentiment_tags", [])
            for tag in sentiment_tags:
                cluster = get_cluster_for_sentiment(tag)
                rel = create_sentiment_relation(
                    product_asin=asin, sentiment_tag=tag, sentiment_cluster=cluster
                )
                if self._kg.add_relation(rel):
                    added += 1

        return added

    def get_product_sentiments(self, product_asin: str) -> dict[str, Any]:
        """
        제품의 감성 정보 조회

        Args:
            product_asin: 제품 ASIN

        Returns:
            {
                "asin": str,
                "ai_summary": str,
                "sentiment_tags": List[str],
                "sentiment_clusters": Dict[str, List[str]]
            }
        """
        result = {
            "asin": product_asin,
            "ai_summary": None,
            "sentiment_tags": [],
            "sentiment_clusters": {},
        }

        # AI Summary 조회
        summary_rels = self._kg.query(subject=product_asin, predicate=RelationType.HAS_AI_SUMMARY)
        if summary_rels:
            result["ai_summary"] = summary_rels[0].properties.get("summary_text")

        # Sentiment Tags 조회
        sentiment_rels = self._kg.query(subject=product_asin, predicate=RelationType.HAS_SENTIMENT)
        for rel in sentiment_rels:
            tag = rel.object
            cluster = rel.properties.get("cluster")
            result["sentiment_tags"].append(tag)
            if cluster:
                if cluster not in result["sentiment_clusters"]:
                    result["sentiment_clusters"][cluster] = []
                result["sentiment_clusters"][cluster].append(tag)

        return result

    def get_brand_profile(self, brand: str) -> dict[str, Any]:
        """
        브랜드의 전체 감성 프로필 조회

        Args:
            brand: 브랜드명

        Returns:
            {
                "brand": str,
                "all_tags": List[str],
                "clusters": Dict[str, int],  # 클러스터별 언급 빈도
                "dominant_sentiment": str,
                "product_count": int
            }
        """
        result = {
            "brand": brand,
            "all_tags": [],
            "clusters": {},
            "dominant_sentiment": None,
            "product_count": 0,
        }

        # 브랜드의 모든 제품 조회
        products = self._kg.get_brand_products(brand)
        result["product_count"] = len(products)

        # 각 제품의 감성 태그 집계
        tag_counts = {}
        cluster_counts = {}

        for product in products:
            asin = product.get("asin")
            if not asin:
                continue

            sentiments = self.get_product_sentiments(asin)
            for tag in sentiments.get("sentiment_tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

            for cluster, tags in sentiments.get("sentiment_clusters", {}).items():
                cluster_counts[cluster] = cluster_counts.get(cluster, 0) + len(tags)

        result["all_tags"] = list(tag_counts.keys())
        result["clusters"] = cluster_counts

        # 가장 많이 언급된 태그
        if tag_counts:
            result["dominant_sentiment"] = max(tag_counts, key=tag_counts.get)

        return result

    def compare_products(self, asin1: str, asin2: str) -> dict[str, Any]:
        """
        두 제품의 감성 비교

        Args:
            asin1: 제품 1 ASIN
            asin2: 제품 2 ASIN

        Returns:
            {
                "product1": {...},
                "product2": {...},
                "common_tags": List[str],
                "unique_to_1": List[str],
                "unique_to_2": List[str],
                "cluster_comparison": {...}
            }
        """
        sent1 = self.get_product_sentiments(asin1)
        sent2 = self.get_product_sentiments(asin2)

        tags1 = set(sent1.get("sentiment_tags", []))
        tags2 = set(sent2.get("sentiment_tags", []))

        return {
            "product1": sent1,
            "product2": sent2,
            "common_tags": list(tags1 & tags2),
            "unique_to_1": list(tags1 - tags2),
            "unique_to_2": list(tags2 - tags1),
            "cluster_comparison": {
                "product1_clusters": sent1.get("sentiment_clusters", {}),
                "product2_clusters": sent2.get("sentiment_clusters", {}),
            },
        }

    def find_products_by_sentiment(
        self, sentiment_tag: str, brand_filter: str | None = None
    ) -> list[str]:
        """
        특정 감성 태그를 가진 제품 검색

        Args:
            sentiment_tag: 감성 태그
            brand_filter: 브랜드 필터 (선택)

        Returns:
            ASIN 리스트
        """
        rels = self._kg.query(predicate=RelationType.HAS_SENTIMENT, object_=sentiment_tag)

        asins = [rel.subject for rel in rels]

        # 브랜드 필터 적용
        if brand_filter:
            filtered = []
            for asin in asins:
                meta = self._kg.get_entity_metadata(asin)
                if meta.get("brand", "").lower() == brand_filter.lower():
                    filtered.append(asin)
            return filtered

        return asins

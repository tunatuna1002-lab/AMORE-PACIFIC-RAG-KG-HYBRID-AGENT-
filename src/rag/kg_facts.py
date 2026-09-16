"""Knowledge-graph fact lookup + inference-context assembly
=========================================================

Everything ``HybridRetriever`` asks the KnowledgeGraph for:

* :func:`query_knowledge_graph` — brand / category / sentiment facts for the
  extracted entities, in the order the prompt renderer expects.
* :func:`build_retrieval_context` — the reasoner's input dict: the shared
  metric normalisation from ``src.ontology.inference_context`` plus the KG
  lookups (competitors, trends, sentiment) that only the retriever does.

Edge ordering and the 12-edge cap live next door in :mod:`src.rag.kg_edges`.

Moved verbatim out of ``hybrid_retriever.py`` (F3 split).
"""

from __future__ import annotations

import logging
from typing import Any

from src.domain.entities.relations import RelationType
from src.ontology.inference_context import build_inference_context, normalize_sentiment_clusters

from .kg_edges import select_metric_edges

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fact lookup
# ---------------------------------------------------------------------------


def _brand_facts(knowledge_graph: Any, brand: str, entities: dict[str, list[str]]) -> list[dict]:
    """한 브랜드에 대한 사실들 (메타 · 제품 · 경쟁사 · 네트워크 · 엣지 · 트렌드)."""
    facts: list[dict[str, Any]] = []

    # 브랜드 메타데이터
    brand_meta = knowledge_graph.get_entity_metadata(brand)
    if brand_meta:
        facts.append({"type": "brand_info", "entity": brand, "data": brand_meta})

    # 브랜드의 제품들
    products = knowledge_graph.get_brand_products(brand)
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
    competitors = knowledge_graph.get_competitors(brand)
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
        network = knowledge_graph.get_neighbors(
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

    # 메트릭/관계 엣지 (kg_enricher가 저장한 hasSoS·rankedIn·competesWith 등)
    try:
        metric_edges = select_metric_edges(knowledge_graph, brand, entities)
        if metric_edges:
            facts.append({"type": "metric_edges", "entity": brand, "data": {"edges": metric_edges}})
    except Exception:
        logger.debug("metric edge query failed", exc_info=True)

    # 트렌드 키워드 (브랜드 우선, 없으면 MARKET)
    trend_relations = _trend_relations(knowledge_graph, brand)
    if trend_relations:
        facts.append(
            {
                "type": "trend_keywords",
                "entity": brand,
                "data": {
                    "keywords": [rel.object for rel in trend_relations[:10]],
                    "count": len(trend_relations),
                },
            }
        )

    return facts


def _trend_relations(knowledge_graph: Any, subject: str) -> list:
    """트렌드 키워드 관계 (브랜드 우선, 없으면 MARKET 폴백)."""
    relations = knowledge_graph.query(subject=subject, predicate=RelationType.HAS_TREND)
    if not relations:
        relations = knowledge_graph.query(subject="MARKET", predicate=RelationType.HAS_TREND)
    return relations


def _category_facts(knowledge_graph: Any, category: str) -> list[dict]:
    """카테고리 브랜드 구성 + 계층 정보."""
    facts: list[dict[str, Any]] = []

    category_brands = knowledge_graph.get_category_brands(category)
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

    try:
        hierarchy = knowledge_graph.get_category_hierarchy(category)
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

    return facts


def _sentiment_facts(knowledge_graph: Any, entities: dict[str, list[str]]) -> list[dict]:
    """제품 · 브랜드 감성 프로필 + 감성 클러스터로 찾은 제품."""
    facts: list[dict[str, Any]] = []
    sentiment_clusters = entities.get("sentiment_clusters", [])

    # 제품이 지정된 경우 해당 제품의 감성 조회
    for asin in entities.get("products", []):
        try:
            product_sentiments = knowledge_graph.get_product_sentiments(asin)
            if product_sentiments.get("sentiment_tags") or product_sentiments.get("ai_summary"):
                facts.append(
                    {"type": "product_sentiment", "entity": asin, "data": product_sentiments}
                )
        except Exception:
            logger.warning("Suppressed Exception", exc_info=True)

    # 브랜드가 지정된 경우 브랜드 감성 프로필 조회
    for brand in entities.get("brands", []):
        try:
            brand_sentiment = knowledge_graph.get_brand_sentiment_profile(brand)
            if brand_sentiment.get("all_tags"):
                facts.append({"type": "brand_sentiment", "entity": brand, "data": brand_sentiment})
        except Exception:
            logger.warning("Suppressed Exception", exc_info=True)

    # 특정 감성 클러스터로 제품 검색
    for cluster in sentiment_clusters:
        if cluster in ("sentiment_general", "ai_summary"):
            continue
        try:
            from src.domain.entities.relations import SENTIMENT_CLUSTERS

            cluster_tags = SENTIMENT_CLUSTERS.get(cluster, [])
            for tag in cluster_tags[:2]:  # 상위 2개 태그만
                products_with_sentiment = knowledge_graph.find_products_by_sentiment(tag)
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


def query_knowledge_graph(
    knowledge_graph: Any, entities: dict[str, list[str]]
) -> list[dict[str, Any]]:
    """지식 그래프에서 관련 사실 조회.

    Args:
        knowledge_graph: KnowledgeGraph 인스턴스
        entities: 추출된 엔티티

    Returns:
        사실 리스트 (브랜드 → 카테고리 → 감성 순서)
    """
    facts: list[dict[str, Any]] = []

    for brand in entities.get("brands", []):
        facts.extend(_brand_facts(knowledge_graph, brand, entities))

    for category in entities.get("categories", []):
        facts.extend(_category_facts(knowledge_graph, category))

    if entities.get("sentiment_clusters") or entities.get("sentiments"):
        facts.extend(_sentiment_facts(knowledge_graph, entities))

    return facts


# ---------------------------------------------------------------------------
# Inference context
# ---------------------------------------------------------------------------


def _add_competitor_sentiment(knowledge_graph: Any, context: dict[str, Any]) -> None:
    """상위 3개 경쟁사의 감성 태그·클러스터를 합산해 비교용으로 붙인다."""
    competitor_tags: list[str] = []
    competitor_clusters: dict[str, int] = {}
    for comp in context["competitors"][:3]:
        comp_brand = comp.get("brand", comp) if isinstance(comp, dict) else comp
        try:
            comp_sentiment = knowledge_graph.get_brand_sentiment_profile(comp_brand)
            competitor_tags.extend(comp_sentiment.get("all_tags", []))
            for cluster, count in comp_sentiment.get("clusters", {}).items():
                competitor_clusters[cluster] = competitor_clusters.get(cluster, 0) + count
        except Exception:
            logger.warning("Suppressed Exception", exc_info=True)
    context["competitor_sentiment_tags"] = list(set(competitor_tags))
    context["competitor_sentiment_clusters"] = competitor_clusters


def build_retrieval_context(
    knowledge_graph: Any,
    entities: dict[str, list[str]],
    current_metrics: dict[str, Any],
) -> dict[str, Any]:
    """추론용 컨텍스트 구성.

    지표 정규화(퍼센트→분수, 기본값 미조작)는 ``src.ontology.inference_context`` 의
    단일 빌더가 담당하고, 여기서는 KG 조회(경쟁사·트렌드·감성)만 덧붙인다.

    Args:
        knowledge_graph: KnowledgeGraph 인스턴스
        entities: 추출된 엔티티
        current_metrics: 현재 지표 데이터

    Returns:
        추론 컨텍스트
    """
    brand = entities["brands"][0] if entities.get("brands") else None
    category = entities["categories"][0] if entities.get("categories") else None
    context = build_inference_context(current_metrics or {}, brand, category=category)

    # 경쟁사 수 (지식 그래프에서)
    if context.get("brand"):
        competitors = knowledge_graph.get_competitors(context["brand"])
        context["competitor_count"] = len(competitors)
        context["competitors"] = competitors

        trend_relations = _trend_relations(knowledge_graph, context["brand"])
        if trend_relations:
            context["trend_keywords"] = [rel.object for rel in trend_relations[:10]]

    # 감성 데이터 (지식 그래프에서)
    if entities.get("sentiments") or entities.get("sentiment_clusters"):
        # 자사 브랜드 감성 프로필
        if context.get("brand"):
            try:
                brand_sentiment = knowledge_graph.get_brand_sentiment_profile(context["brand"])
                context["sentiment_tags"] = brand_sentiment.get("all_tags", [])
                context["sentiment_clusters"] = normalize_sentiment_clusters(
                    brand_sentiment.get("clusters", {})
                )
                context["dominant_sentiment"] = brand_sentiment.get("dominant_sentiment")
            except Exception:
                logger.warning("Suppressed Exception", exc_info=True)

        # 제품별 감성 데이터
        if context.get("asin"):
            try:
                product_sentiment = knowledge_graph.get_product_sentiments(context["asin"])
                context["ai_summary"] = product_sentiment.get("ai_summary")
                if not context.get("sentiment_tags"):
                    context["sentiment_tags"] = product_sentiment.get("sentiment_tags", [])
                    context["sentiment_clusters"] = normalize_sentiment_clusters(
                        product_sentiment.get("sentiment_clusters", {})
                    )
            except Exception:
                logger.warning("Suppressed Exception", exc_info=True)

        # 경쟁사 감성 데이터 (비교용)
        if context.get("competitors"):
            _add_competitor_sentiment(knowledge_graph, context)

    return context

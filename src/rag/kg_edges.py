"""KG metric-edge selection
=========================

Turns the raw relation list of one brand into the bounded, ordered edge list
that reaches the prompt. This is the "edge ordering/limits" half of the F3
``kg_facts`` responsibility; it is kept separate because the selection rules
(priority buckets, the 12-edge cap, the product-slug synthesis) are the part
with the delicate, measured behaviour.

Moved verbatim out of ``HybridRetriever._query_knowledge_graph`` (F3 split).
"""

from __future__ import annotations

import logging
from typing import Any

from .entity_linker import product_name_slugs as _product_name_slugs

logger = logging.getLogger(__name__)

# 브랜드 정체성·지표 엣지의 정보량 우선순위 (12개 상한에서 살아남을 순서)
_REST_EDGE_PRIORITY = {
    "ownedBy": 0,
    "hasSoS": 1,
    "rankedIn": 2,
    "hasHHI": 3,
    "hasPosition": 4,
}

# 엣지 노출 대상 술어. siblingBrand 등 시드 온톨로지는 제외한다.
_PRIORITY_PREDICATES = {
    "hasSoS",
    "hasHHI",
    "rankedIn",
    "competesWith",
    "hasPosition",
    "ownedBy",
}

# 방출 엣지 상한. recall 게이트를 "엣지 전량 방출"로 우회하지 않기 위한
# 정밀도 가드 (kg_edge_precision으로 감시).
MAX_METRIC_EDGES = 12


def dedupe_edges(edges: list[dict]) -> list[dict]:
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


def _normalize_predicate(rel: Any) -> str:
    """관계의 표시용 술어 이름을 고른다.

    골드/KG 표기는 camelCase — ``original_predicate`` 는 hasSoS처럼 의미가 더
    구체적인 camelCase일 때만 우선한다. 시드 온톨로지 표기는 정합화한다
    (ownedByGroup → ownedBy).
    """
    enum_pred = rel.predicate.value if hasattr(rel.predicate, "value") else str(rel.predicate)
    orig = rel.properties.get("original_predicate")
    pred = orig if orig and "_" not in orig and not orig.isupper() else enum_pred
    return {"ownedByGroup": "ownedBy"}.get(pred, pred)


def _subject_variants(brand: str) -> list[str]:
    """브랜드 표기 변형 목록.

    시드 온톨로지는 'LANEIGE'(대문자), kg_enricher는 'laneige'(소문자)로
    저장한다. 추출기가 내는 브랜드는 소문자이므로 대문자 변형을 조회하지
    않으면 시드 트리플(ownedByGroup 등)이 통째로 누락됐다.
    """
    variants: list[str] = []
    for variant in (brand, brand.lower(), brand.upper(), brand.title()):
        if variant not in variants:
            variants.append(variant)
    return variants


def select_metric_edges(
    knowledge_graph: Any, brand: str, entities: dict[str, list[str]]
) -> list[dict]:
    """브랜드의 지표·관계 엣지를 우선순위대로 골라 최대 12개 반환.

    선택 순서: 질의와 직접 닿는 엣지 → 제품 엣지 → 브랜드 정체성·지표 엣지 →
    경쟁 엣지. ``competesWith`` 는 브랜드당 최대 8개로 가장 수가 많고 질의
    특정성이 낮아, 이전 순서(경쟁 우선)에서는 12개 상한이 ownedBy·rankedIn을
    밀어냈다. ``rest`` 안에서도 정보량 순으로 정렬한다
    (소유관계 > 점유율 > 랭킹 > 집중도 > 가격 포지션).
    """
    edge_relations = []
    for variant in _subject_variants(brand):
        edge_relations += list(knowledge_graph.query(subject=variant))

    query_categories = {c.lower() for c in entities.get("categories", [])}
    query_brands = {b.lower() for b in entities.get("brands", [])}
    query_products = {p.lower() for p in entities.get("products", [])}

    relevant: list[dict] = []
    competes: list[dict] = []
    rest: list[dict] = []
    top_products: list[tuple[int, str, str]] = []  # (rank, title, category)

    for rel in edge_relations:
        pred = _normalize_predicate(rel)
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
        if pred not in _PRIORITY_PREDICATES:
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
    product_edges: list[dict] = []
    for _rank, title, category in sorted(set(top_products))[:2]:
        for slug in _product_name_slugs(title, brand):
            product_edges.append({"subject": brand, "predicate": "hasProduct", "object": slug})
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

    rest.sort(key=lambda e: _REST_EDGE_PRIORITY.get(e["predicate"], 9))
    return dedupe_edges(relevant + product_edges + rest + competes[:3])[:MAX_METRIC_EDGES]

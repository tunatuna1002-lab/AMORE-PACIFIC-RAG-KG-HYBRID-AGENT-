#!/usr/bin/env python3
"""④다단계 유형 문항을 KG 관계 + 2026-08-31 DB 수치 조합으로 **기계적으로** 생성한다.

배경 (docs/plans/evidence-react-ontology-kickoff-prompt-2026-09-17.md 1-B)
--------------------------------------------------------------------------
원본 골든셋의 다단계 문항은 시험지에 17개뿐이다(scripts/classify_golden_types.py). 질문에
이름이 없는 엔티티를 먼저 해석해야 하는 연쇄 질문을 만든다. 각 문항은 `metadata.hop_plan`에
필요한 단계(KG 관계 → DB 조회 → 비교)를 순서대로 남긴다.

정답 근거
--------
- 관계: KG 큐레이션 트리플(config/brands.json 출처)만. 크롤 파생 KG 엣지·수치 엣지는 쓰지 않는다.
- 수치: raw_data 독립 계산(scripts/golden_snapshot_facts.py). 지표 테이블 값으로 같은 연쇄를 다시
  풀어 **중간 결과(선택된 카테고리·브랜드)와 최종 수치가 모두 맞을 때만** 싣는다. 지표 테이블에
  없는 값(제품 순위·가격)은 raw_data 단일 출처이며 cross_check에 그렇게 적는다.
- 브랜드 매칭: KG 브랜드명과 raw_data.brand를 대소문자 무시로 맞춘다. 제품명에 브랜드명이 없는
  오귀속 행이 있는 브랜드(예: lip_care "Hera" = Vaseline 제품)가 연쇄에 끼는 카테고리는 쓰지 않는다.

사용법:
    python3 scripts/generate_multihop_questions.py --db <amore_data.db> --kg <knowledge_graph.json>
    python3 scripts/generate_multihop_questions.py --db ... --kg ... --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import golden_snapshot_facts as gf  # noqa: E402

GENERATOR = "scripts/generate_multihop_questions.py"
D = gf.AS_OF
L = gf.CATEGORY_LABEL
GROUP = "AMOREPACIFIC"


class Skip(Exception):
    """후보를 싣지 않는 이유."""


def _pct(v: float) -> str:
    return f"{v:g}%"


def _table_sos(tables: dict, cat: str, brand: str) -> float | None:
    hits = [r for name, r in tables[cat]["brands"].items() if name.lower() == brand.lower()]
    return hits[0]["sos"] if len(hits) == 1 else None


def _table_brand_names(tables: dict, cat: str) -> set[str]:
    return {name.lower() for name in tables[cat]["brands"]}


def _table_top(tables: dict, cat: str) -> dict[str, Any]:
    """지표 테이블의 SoS 1위 행 (동률이면 브랜드명 오름차순 첫 행)."""
    rows = sorted(tables[cat]["brands"].values(), key=lambda r: (-(r["sos"] or 0), r["brand"]))
    if not rows:
        raise Skip(f"{cat}: 지표 테이블 비어 있음")
    return rows[0]


def _group_brands_in_category(
    facts: gf.CategoryFacts, group_brands: list[str]
) -> list[gf.BrandFacts]:
    """KG 그룹 브랜드 중 이 카테고리에 있는 브랜드. 매칭된 브랜드에 오귀속이 있으면 Skip."""
    failures = facts.attribution_failures()
    matched = []
    for name in group_brands:
        brand = gf.find_brand(facts, name)
        if brand is None:
            continue
        if brand.brand in failures:
            raise Skip(
                f"{facts.category}: KG '{name}'와 매칭된 raw_data '{brand.brand}' 행이 "
                f"오귀속 {failures[brand.brand][:2]}"
            )
        matched.append(brand)
    return sorted(matched, key=lambda b: b.brand.lower())


def _step(n: int, kind: str, action: str, result: Any) -> dict[str, Any]:
    return {"step": n, "source": kind, "action": action, "result": result}


# ---------------------------------------------------------------------------
# 후보 (각 함수는 레코드 재료 dict를 돌려주거나 Skip을 던진다)
# ---------------------------------------------------------------------------


def group_brands_sos(ctx: dict, cat: str) -> dict:
    facts, tables, triples = ctx["facts"], ctx["tables"], ctx["triples"]
    group = gf.subjects(triples, "ownedByGroup", GROUP)
    matched = _group_brands_in_category(facts[cat], group)
    if not matched:
        raise Skip(f"{cat}: 그룹 브랜드 없음")
    for b in matched:
        if not gf.within_tolerance(b.sos, _table_sos(tables, cat, b.brand)):
            raise Skip(
                f"{cat}/{b.brand}: SoS raw={b.sos} 지표테이블={_table_sos(tables, cat, b.brand)}"
            )
    table_group = {n.lower() for n in group} & _table_brand_names(tables, cat)
    if table_group != {b.brand.lower() for b in matched}:
        raise Skip(
            f"{cat}: 그룹 브랜드 집합 불일치 raw={[b.brand for b in matched]} 표={table_group}"
        )
    listing = ", ".join(f"{b.brand} {_pct(b.sos)}" for b in matched)
    return {
        "question": (
            f"아모레퍼시픽 그룹 소속 브랜드 중 {D} {L[cat]} Top 100에 제품이 있는 브랜드와 "
            "각 브랜드의 SoS는?"
        ),
        "answer": (
            f"KG에서 AMOREPACIFIC 소속(ownedByGroup)으로 등록된 {len(group)}개 브랜드 중 {D} "
            f"{L[cat]} Top 100에 있는 브랜드는 {listing}입니다."
        ),
        "expected_values": {f"{gf.norm_id(b.brand)}_sos": b.sos for b in matched},
        "kg_entities": ["amorepacific", *[gf.norm_id(b.brand) for b in matched], cat],
        "kg_edges": [gf.edge(b.brand, "ownedByGroup", GROUP) for b in matched],
        "concepts": ["multi_hop", "brand_portfolio", "sos"],
        "hop_plan": [
            _step(1, "kg", f"?brand -ownedByGroup-> {GROUP}", f"{len(group)}개 브랜드"),
            _step(
                2,
                "db",
                f"{cat} {D} raw_data에서 ?brand 매칭(대소문자 무시)",
                [b.brand for b in matched],
            ),
            _step(3, "db", "브랜드별 SoS 계산", {b.brand: b.sos for b in matched}),
        ],
        "cross_check": {
            "metric_tables_sos": {b.brand: _table_sos(tables, cat, b.brand) for b in matched}
        },
    }


def laneige_most_concentrated_category(ctx: dict) -> dict:
    facts, tables = ctx["facts"], ctx["tables"]
    cats = [c for c in gf.CATEGORIES if "LANEIGE" in facts[c].brands]
    if not cats:
        raise Skip("LANEIGE 제품이 있는 카테고리 없음")
    raw_pick = max(cats, key=lambda c: facts[c].hhi)
    table_cats = [c for c in gf.CATEGORIES if "laneige" in _table_brand_names(tables, c)]
    if not table_cats:
        raise Skip("지표 테이블에 LANEIGE 카테고리 없음")
    table_pick = max(table_cats, key=lambda c: tables[c]["market"]["hhi"])
    if set(cats) != set(table_cats) or raw_pick != table_pick:
        raise Skip(f"선택 불일치 raw={raw_pick}{cats} 표={table_pick}{table_cats}")
    hhi = facts[raw_pick].hhi
    if not gf.within_tolerance(hhi, tables[raw_pick]["market"]["hhi"]):
        raise Skip("HHI 불일치")
    return {
        "question": (
            f"{D} 기준 LANEIGE 제품이 Top 100에 있는 카테고리 중 시장 집중도(HHI)가 가장 높은 "
            "카테고리와 그 HHI는?"
        ),
        "answer": (
            f"{D} LANEIGE가 진입한 카테고리는 "
            + ", ".join(f"{L[c]}(HHI {facts[c].hhi:.4f})" for c in cats)
            + f"이고, 이 중 HHI가 가장 높은 곳은 {L[raw_pick]}({hhi:.4f})입니다."
        ),
        "expected_values": {"hhi": hhi},
        "kg_entities": ["laneige", raw_pick],
        "kg_edges": [],
        "concepts": ["multi_hop", "hhi", "market_concentration"],
        "hop_plan": [
            _step(1, "db", f"{D} raw_data에서 LANEIGE 제품이 있는 카테고리", cats),
            _step(2, "db", "카테고리별 HHI 계산", {c: facts[c].hhi for c in cats}),
            _step(3, "compare", "HHI 최대 카테고리", raw_pick),
        ],
        "cross_check": {"metric_tables_hhi": {c: tables[c]["market"]["hhi"] for c in table_cats}},
    }


def laneige_best_avg_rank_category_leader(ctx: dict) -> dict:
    facts, tables = ctx["facts"], ctx["tables"]
    cats = [c for c in gf.CATEGORIES if "LANEIGE" in facts[c].brands]
    if not cats:
        raise Skip("LANEIGE 제품이 있는 카테고리 없음")
    raw_pick = min(cats, key=lambda c: facts[c].brands["LANEIGE"].avg_rank)
    table_pick = min(
        cats, key=lambda c: tables[c]["brands"].get("LANEIGE", {}).get("brand_avg_rank") or 999
    )
    if raw_pick != table_pick:
        raise Skip(f"선택 불일치 raw={raw_pick} 표={table_pick}")
    ranked = facts[raw_pick].ranked_brands()
    if len(ranked) < 2:
        raise Skip("브랜드 2개 미만")
    top = ranked[0]
    if ranked[1].sos == top.sos:
        raise Skip("1위 동률")
    if top.brand in facts[raw_pick].attribution_failures():
        raise Skip(f"1위 {top.brand} 오귀속")
    if not tables[raw_pick]["brands"]:
        raise Skip("지표 테이블 비어 있음")
    table_top = _table_top(tables, raw_pick)
    if table_top["brand"] != top.brand or not gf.within_tolerance(top.sos, table_top["sos"]):
        raise Skip(
            f"1위 불일치 raw={top.brand}:{top.sos} 표={table_top['brand']}:{table_top['sos']}"
        )
    avg = {c: facts[c].brands["LANEIGE"].avg_rank for c in cats}
    return {
        "question": (
            f"{D} 기준 LANEIGE 제품의 평균 순위가 가장 좋은 카테고리에서 SoS 1위 브랜드와 그 SoS는?"
        ),
        "answer": (
            f"{D} LANEIGE 평균 순위는 "
            + ", ".join(f"{L[c]} {avg[c]:g}위" for c in cats)
            + f"로 {L[raw_pick]}가 가장 좋고, {L[raw_pick]}의 SoS 1위 브랜드는 "
            f"{top.brand}({_pct(top.sos)})입니다."
        ),
        "expected_values": {"top_brand_sos": top.sos},
        "kg_entities": ["laneige", raw_pick, gf.norm_id(top.brand)],
        "kg_edges": [],
        "concepts": ["multi_hop", "category_leader", "sos"],
        "hop_plan": [
            _step(1, "db", "LANEIGE 제품이 있는 카테고리별 평균 순위", avg),
            _step(2, "compare", "평균 순위 최소 카테고리", raw_pick),
            _step(3, "db", f"{raw_pick} SoS 1위 브랜드", {top.brand: top.sos}),
        ],
        "cross_check": {
            "metric_tables_avg_rank": {
                c: tables[c]["brands"].get("LANEIGE", {}).get("brand_avg_rank") for c in cats
            },
            "metric_tables_top": {table_top["brand"]: table_top["sos"]},
        },
    }


def laneige_best_rank_category_brands_above(ctx: dict) -> dict:
    facts, tables = ctx["facts"], ctx["tables"]
    cats = [c for c in gf.CATEGORIES if "LANEIGE" in facts[c].brands]
    if not cats:
        raise Skip("LANEIGE 제품이 있는 카테고리 없음")
    best = {c: facts[c].brands["LANEIGE"].best_rank for c in cats}
    pick = min(cats, key=lambda c: best[c])
    if sorted(best.values())[:2].count(best[pick]) > 1:
        raise Skip("최고 순위 동률")
    lan = facts[pick].brands["LANEIGE"]
    above = sorted(b.brand for b in facts[pick].brands.values() if (b.sos or 0) > lan.sos)
    t_lan = tables[pick]["brands"].get("LANEIGE", {}).get("sos")
    t_above = sorted(n for n, r in tables[pick]["brands"].items() if (r["sos"] or 0) > (t_lan or 0))
    if len(above) != len(t_above):
        raise Skip(f"개수 불일치 raw={above} 표={t_above}")
    failures = facts[pick].attribution_failures()
    if any(b in failures for b in above):
        raise Skip(f"상위 브랜드 오귀속 {[b for b in above if b in failures]}")
    return {
        "question": (
            f"{D} 기준 LANEIGE의 최고 순위 제품이 속한 카테고리에서 LANEIGE보다 SoS가 높은 "
            "브랜드는 몇 개인가요?"
        ),
        "answer": (
            f"{D} LANEIGE 최고 순위 제품은 {L[pick]} {best[pick]}위이고, {L[pick]}에서 LANEIGE "
            f"SoS {_pct(lan.sos)}보다 SoS가 높은 브랜드는 {len(above)}개({', '.join(above)})입니다."
        ),
        "expected_values": {"brand_count": float(len(above))},
        "kg_entities": ["laneige", pick],
        "kg_edges": [],
        "concepts": ["multi_hop", "sos", "competitive_gap"],
        "hop_plan": [
            _step(1, "db", "LANEIGE 카테고리별 최고 순위", best),
            _step(2, "compare", "최고 순위 카테고리", pick),
            _step(3, "db", f"{pick}에서 SoS > LANEIGE({lan.sos})인 브랜드", above),
        ],
        "cross_check": {"metric_tables_brands_above": t_above, "metric_tables_laneige_sos": t_lan},
    }


def sibling_count_in_category(ctx: dict, anchor: str, cat: str) -> dict:
    facts, tables, triples = ctx["facts"], ctx["tables"], ctx["triples"]
    parents = gf.objects(triples, anchor, "ownedByGroup")
    if parents != [GROUP]:
        raise Skip(f"{anchor} 소속 그룹 {parents}")
    group = gf.subjects(triples, "ownedByGroup", GROUP)
    matched = _group_brands_in_category(facts[cat], group)
    table_group = {n.lower() for n in group} & _table_brand_names(tables, cat)
    if table_group != {b.brand.lower() for b in matched}:
        raise Skip(f"집합 불일치 raw={[b.brand for b in matched]} 표={sorted(table_group)}")
    names = [b.brand for b in matched]
    return {
        "question": (
            f"{anchor}와 같은 그룹에 속한 브랜드({anchor} 포함) 중 {D} {L[cat]} Top 100에 제품이 "
            "있는 브랜드는 몇 개인가요?"
        ),
        "answer": (
            f"{anchor}의 소속 그룹은 {GROUP}이고, 같은 그룹 브랜드 중 {D} {L[cat]} Top 100에 "
            f"있는 브랜드는 {len(names)}개({', '.join(names) or '없음'})입니다."
        ),
        "expected_values": {"brand_count": float(len(names))},
        "kg_entities": [gf.norm_id(anchor), "amorepacific", cat],
        "kg_edges": [gf.edge(anchor, "ownedByGroup", GROUP)]
        + [gf.edge(n, "ownedByGroup", GROUP) for n in names if n.lower() != anchor.lower()],
        "concepts": ["multi_hop", "corporate_structure", "brand_portfolio"],
        "hop_plan": [
            _step(1, "kg", f"{anchor} -ownedByGroup-> ?group", GROUP),
            _step(2, "kg", f"?brand -ownedByGroup-> {GROUP}", f"{len(group)}개 브랜드"),
            _step(3, "db", f"{cat} {D} raw_data에 제품이 있는 ?brand", names),
        ],
        "cross_check": {"metric_tables_group_brands": sorted(table_group)},
    }


def group_best_ranked_product(ctx: dict, cat: str) -> dict:
    facts, triples = ctx["facts"], ctx["triples"]
    group = gf.subjects(triples, "ownedByGroup", GROUP)
    matched = _group_brands_in_category(facts[cat], group)
    if len(matched) < 2:
        raise Skip(f"{cat}: 비교할 그룹 브랜드가 2개 미만")
    ranks = {b.brand: b.best_rank for b in matched}
    best = min(ranks, key=ranks.get)
    if list(ranks.values()).count(ranks[best]) > 1:
        raise Skip("순위 동률")
    mf = gf.metric_facts_provider_view(ctx["db_path"], [best], [cat])
    mf_rank = next((f["products"][0]["rank"] for f in mf if f["type"] == "brand_products"), None)
    if mf_rank != ranks[best]:
        raise Skip(f"MetricFactsProvider 순위 불일치 raw={ranks[best]} provider={mf_rank}")
    return {
        "question": (
            f"{D} 기준 {L[cat]} Top 100에 제품이 있는 아모레퍼시픽 그룹 브랜드 중 가장 높은 순위의 "
            "제품을 가진 브랜드와 그 순위는?"
        ),
        "answer": (
            f"{D} {L[cat]}의 아모레퍼시픽 그룹 브랜드 최고 순위는 "
            + ", ".join(f"{b} {r}위" for b, r in sorted(ranks.items(), key=lambda x: x[1]))
            + f"로, 가장 높은 브랜드는 {best}({ranks[best]}위)입니다."
        ),
        "expected_values": {"best_rank": float(ranks[best])},
        "kg_entities": ["amorepacific", *[gf.norm_id(b) for b in ranks], cat],
        "kg_edges": [gf.edge(b, "ownedByGroup", GROUP) for b in ranks],
        "concepts": ["multi_hop", "brand_portfolio", "product_ranking"],
        "hop_plan": [
            _step(1, "kg", f"?brand -ownedByGroup-> {GROUP}", f"{len(group)}개 브랜드"),
            _step(2, "db", f"{cat} {D} raw_data에 있는 ?brand", sorted(ranks)),
            _step(3, "db", "브랜드별 최고 순위", ranks),
            _step(4, "compare", "최고 순위 최소", best),
        ],
        "cross_check": {
            "source": "raw_data 단일 출처(지표 테이블에 제품 순위 없음)",
            "metric_facts_provider_best_rank": mf_rank,
        },
    }


def brand_category_leader(ctx: dict, brand_name: str) -> dict:
    facts, tables, triples = ctx["facts"], ctx["tables"], ctx["triples"]
    if not gf.objects(triples, brand_name, "ownedByGroup"):
        raise Skip(f"{brand_name}: KG 큐레이션 트리플에 없음")
    cats = [c for c in gf.CATEGORIES if gf.find_brand(facts[c], brand_name)]
    if len(cats) != 1:
        raise Skip(f"{brand_name}: 진입 카테고리가 하나가 아님 {cats}")
    cat = cats[0]
    b = gf.find_brand(facts[cat], brand_name)
    if b.brand in facts[cat].attribution_failures():
        raise Skip(f"{brand_name} 오귀속")
    ranked = facts[cat].ranked_brands()
    if len(ranked) < 2:
        raise Skip("브랜드 2개 미만")
    top = ranked[0]
    if ranked[1].sos == top.sos or top.brand in facts[cat].attribution_failures():
        raise Skip("1위 동률 또는 오귀속")
    if not tables[cat]["brands"]:
        raise Skip("지표 테이블 비어 있음")
    table_top = _table_top(tables, cat)
    table_cats = [c for c in gf.CATEGORIES if brand_name.lower() in _table_brand_names(tables, c)]
    if (
        table_cats != cats
        or table_top["brand"] != top.brand
        or not gf.within_tolerance(top.sos, table_top["sos"])
    ):
        raise Skip(f"불일치 표 카테고리={table_cats} 표 1위={table_top['brand']}")
    return {
        "question": (
            f"{D} 기준 {brand_name} 제품이 Top 100에 있는 카테고리에서 SoS 1위 브랜드와 그 SoS는?"
        ),
        "answer": (
            f"{D} {brand_name} 제품은 {L[cat]} Top 100에만 있고({b.best_rank}위), {L[cat]}의 "
            f"SoS 1위 브랜드는 {top.brand}({_pct(top.sos)})입니다."
        ),
        "expected_values": {"top_brand_sos": top.sos},
        "kg_entities": [gf.norm_id(brand_name), cat, gf.norm_id(top.brand)],
        "kg_edges": [],
        "concepts": ["multi_hop", "category_leader", "sos"],
        "hop_plan": [
            _step(1, "db", f"{brand_name} 제품이 있는 카테고리", cats),
            _step(2, "db", f"{cat} SoS 1위 브랜드", {top.brand: top.sos}),
        ],
        "cross_check": {
            "metric_tables_top": {table_top["brand"]: table_top["sos"]},
            "metric_tables_categories": table_cats,
        },
    }


def category_leader_top_product_price(ctx: dict, cat: str) -> dict:
    facts, tables = ctx["facts"], ctx["tables"]
    ranked = facts[cat].ranked_brands()
    if len(ranked) < 2:
        raise Skip("브랜드 2개 미만")
    top = ranked[0]
    if ranked[1].sos == top.sos or top.brand in facts[cat].attribution_failures():
        raise Skip("1위 동률 또는 오귀속")
    if not tables[cat]["brands"]:
        raise Skip("지표 테이블 비어 있음")
    table_top = _table_top(tables, cat)
    if table_top["brand"] != top.brand:
        raise Skip(f"1위 불일치 표={table_top['brand']}")
    product = next(p for p in facts[cat].products if p["rank"] == top.best_rank)
    if product["price"] is None:
        raise Skip("가격 없음")
    name = (product["product_name"] or "").split(",")[0].split(":")[0].strip()[:60]
    return {
        "question": (
            f"{D} 기준 {L[cat]} SoS 1위 브랜드의 최고 순위 제품은 무엇이고 가격은 얼마인가요?"
        ),
        "answer": (
            f"{D} {L[cat]} SoS 1위 브랜드는 {top.brand}({_pct(top.sos)})이고, 최고 순위 제품은 "
            f"{top.best_rank}위 '{name}'(ASIN {product['asin']}), 가격은 ${product['price']:.2f}입니다."
        ),
        "expected_values": {"price": product["price"], "best_rank": float(top.best_rank)},
        "kg_entities": [cat, gf.norm_id(top.brand)],
        "kg_edges": [],
        "concepts": ["multi_hop", "category_leader", "price_analysis"],
        "hop_plan": [
            _step(1, "db", f"{cat} SoS 1위 브랜드", {top.brand: top.sos}),
            _step(
                2,
                "db",
                "그 브랜드의 최고 순위 제품",
                {"rank": top.best_rank, "asin": product["asin"]},
            ),
            _step(3, "db", "제품 가격", product["price"]),
        ],
        "cross_check": {
            "metric_tables_top": {table_top["brand"]: table_top["sos"]},
            "price_source": "raw_data 단일 출처",
        },
    }


CANDIDATES = [
    ("group_brands_sos/face_powder", lambda ctx: group_brands_sos(ctx, "face_powder")),
    ("group_brands_sos/lip_makeup", lambda ctx: group_brands_sos(ctx, "lip_makeup")),
    ("group_brands_sos/lip_care", lambda ctx: group_brands_sos(ctx, "lip_care")),
    ("group_brands_sos/skin_care", lambda ctx: group_brands_sos(ctx, "skin_care")),
    ("laneige_most_concentrated_category", laneige_most_concentrated_category),
    ("laneige_best_avg_rank_category_leader", laneige_best_avg_rank_category_leader),
    ("laneige_best_rank_category_brands_above", laneige_best_rank_category_brands_above),
    (
        "sibling_count/COSRX/face_powder",  # pragma: allowlist secret
        lambda ctx: sibling_count_in_category(ctx, "COSRX", "face_powder"),
    ),
    (
        "sibling_count/Sulwhasoo/lip_makeup",
        lambda ctx: sibling_count_in_category(ctx, "Sulwhasoo", "lip_makeup"),
    ),
    (
        "group_best_ranked_product/face_powder",
        lambda ctx: group_best_ranked_product(ctx, "face_powder"),
    ),
    ("brand_category_leader/innisfree", lambda ctx: brand_category_leader(ctx, "innisfree")),
    (
        "category_leader_top_product_price/face_powder",
        lambda ctx: category_leader_top_product_price(ctx, "face_powder"),
    ),
    (
        "category_leader_top_product_price/skin_care",
        lambda ctx: category_leader_top_product_price(ctx, "skin_care"),
    ),
]


def build(db_path: Path, kg_path: Path) -> tuple[list[dict], dict]:
    conn = gf.connect_ro(db_path)
    ctx = {
        "db_path": db_path,
        "facts": {c: gf.category_facts(conn, c) for c in gf.CATEGORIES},
        "tables": {c: gf.metric_table_facts(conn, c) for c in gf.CATEGORIES},
        "triples": gf.load_curated_triples(kg_path),
    }
    records: list[dict] = []
    excluded: list[dict] = []
    for name, fn in CANDIDATES:
        try:
            item = fn(ctx)
        except Skip as e:
            excluded.append({"candidate": name, "reason": str(e)})
            continue
        uses_kg = any(s["source"] == "kg" for s in item["hop_plan"])
        records.append(
            gf.build_record(
                item_id=f"mh{len(records) + 1:03d}",
                question=item["question"],
                answer=item["answer"],
                question_type="multihop",
                generator=GENERATOR,
                domain="multi_hop",
                difficulty="hard" if len(item["hop_plan"]) >= 4 else "medium",
                requires_kg=uses_kg,
                gold_source="snapshot",
                expected_values=item["expected_values"],
                kg_entities=item["kg_entities"],
                kg_edges=item["kg_edges"],
                concepts=item["concepts"],
                extra_metadata={
                    "hop_plan": item["hop_plan"],
                    "multihop_gold": {
                        "candidate": name,
                        "as_of": gf.AS_OF,
                        "formula": "scripts/golden_snapshot_facts.py 모듈 docstring의 정본 공식",
                        "kg_sources": list(gf.CURATED_KG_SOURCES) if uses_kg else [],
                        "cross_check": item["cross_check"],
                    },
                },
            )
        )
    log = {
        "as_of": gf.AS_OF,
        "generator": GENERATOR,
        "generated": len(records),
        "excluded": excluded,
    }
    return records, log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", type=Path, default=gf.DEFAULT_DB)
    parser.add_argument("--kg", type=Path, default=gf.DEFAULT_KG)
    parser.add_argument("--out", type=Path, default=gf.TYPED_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    records, log = build(args.db, args.kg)
    print(f"생성 {len(records)}문항, 제외 {len(log['excluded'])}")
    for e in log["excluded"]:
        print(f"  제외 {e['candidate']}: {e['reason']}")
    return gf.write_or_check(
        {
            args.out / "generated_multihop.jsonl": gf.dumps_jsonl(records),
            args.out / "generation_log_multihop.json": gf.dumps_json(log),
        },
        args.check,
    )


if __name__ == "__main__":
    sys.exit(main())

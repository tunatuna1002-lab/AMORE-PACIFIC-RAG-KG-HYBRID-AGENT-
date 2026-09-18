#!/usr/bin/env python3
"""④다단계 유형 문항을 KG 관계 + 2026-08-31 DB 수치 조합으로 **기계적으로** 생성한다.

배경 (docs/plans/evidence-react-ontology-kickoff-prompt-2026-09-17.md 1-B)
--------------------------------------------------------------------------
원본 골든셋의 다단계 문항은 시험지에 17개뿐이다(scripts/classify_golden_types.py). 질문에
이름이 없는 엔티티를 먼저 해석해야 하는 연쇄 질문을 만든다. 각 문항은 `metadata.hop_plan`에
필요한 단계(KG 관계 → DB 조회 → 비교)를 순서대로 남긴다.

정답 근거 (리드 결정 2026-09-17: 지표 테이블 정본)
------------------------------------------------
- 관계: KG 큐레이션 트리플(config/brands.json 출처)만. 크롤 파생 KG 엣지·수치 엣지는 쓰지 않는다.
- 수치: **지표 테이블(brand_metrics·market_metrics)** 값. 기존 골든 snapshot 문항과 챗봇
  경로(MetricFactsProvider)가 읽는 값과 같게 한다. 테이블이 없는 값(브랜드 최고 순위·제품
  가격)은 raw_data 기록값.
- 교차 확인: raw_data 독립 재계산(scripts/golden_snapshot_facts.py)으로 같은 연쇄를 다시 푼다.
  **중간 결과(선택된 카테고리·브랜드·집합·개수)가 다르면 제외**, 수치 차이만 있으면 포함하고
  `multihop_gold.cross_check = {raw_recomputed, relative_diff, conclusion_agrees}`로 남긴다.
- 브랜드 매칭: KG 브랜드명과 DB 브랜드명을 대소문자 무시로 맞춘다. 제품명에 브랜드명이 없는
  오귀속 행이 있는 브랜드(예: lip_care "Hera" = Vaseline 제품)가 연쇄에 끼면 쓰지 않는다.

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


def _step(n: int, kind: str, action: str, result: Any) -> dict[str, Any]:
    return {"step": n, "source": kind, "action": action, "result": result}


# --- 지표 테이블(정답) -------------------------------------------------------


def _t_rows(tables: dict, cat: str) -> dict[str, dict[str, Any]]:
    return tables[cat]["brands"]


def _t_find(tables: dict, cat: str, name: str) -> dict[str, Any] | None:
    hits = [r for n, r in _t_rows(tables, cat).items() if n.lower() == name.lower()]
    return hits[0] if len(hits) == 1 else None


def _t_hhi(tables: dict, cat: str) -> float | None:
    return (tables[cat]["market"] or {}).get("hhi")


def _t_top(tables: dict, cat: str) -> dict[str, Any]:
    """지표 테이블 SoS 1위 행. 1위가 동률이면 Skip."""
    rows = sorted(
        (r for r in _t_rows(tables, cat).values() if r["sos"] is not None),
        key=lambda r: (-r["sos"], r["brand"]),
    )
    if len(rows) < 2:
        raise Skip(f"{cat}: 지표 테이블 브랜드 2개 미만")
    if rows[0]["sos"] == rows[1]["sos"]:
        raise Skip(f"{cat}: 지표 테이블 1위 동률")
    return rows[0]


def _t_group(tables: dict, cat: str, group: list[str]) -> list[dict[str, Any]]:
    names = {g.lower() for g in group}
    rows = [r for n, r in _t_rows(tables, cat).items() if n.lower() in names]
    return sorted(rows, key=lambda r: r["brand"].lower())


# --- raw_data(교차 확인) ----------------------------------------------------


def _r_top(facts: gf.CategoryFacts) -> gf.BrandFacts | None:
    """raw 재계산 SoS 1위. 동률이면 None(결론이 하나로 정해지지 않음)."""
    ranked = facts.ranked_brands()
    if len(ranked) < 2 or ranked[0].sos == ranked[1].sos:
        return None
    return ranked[0]


def _r_group(facts: gf.CategoryFacts, group: list[str]) -> list[gf.BrandFacts]:
    """KG 그룹 브랜드 중 raw에 있는 브랜드. 매칭된 브랜드에 오귀속이 있으면 Skip."""
    failures = facts.attribution_failures()
    matched = []
    for name in group:
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


def _require_attribution(facts: gf.CategoryFacts, brands: list[str]) -> None:
    failures = facts.attribution_failures()
    bad = [b for b in brands if b in failures]
    if bad:
        raise Skip(f"{facts.category}: 연쇄에 오귀속 브랜드 {bad}")


def _cross_check(
    gold: dict[str, float], raw: dict[str, float | None], **extra: Any
) -> dict[str, Any]:
    relative = {k: gf.relative_diff(v, raw.get(k)) for k, v in gold.items()}
    finite = [v for v in relative.values() if v is not None]
    return {
        "raw_recomputed": raw,
        "relative_diff": relative,
        "max_relative_diff": max(finite) if finite else None,
        "conclusion_agrees": True,  # 불일치 후보는 Skip으로 제외된다
        **extra,
    }


def _disagree(what: str, table: Any, raw: Any) -> Skip:
    return Skip(f"결론 불일치({what}) 지표테이블={table} raw재계산={raw}")


# ---------------------------------------------------------------------------
# 후보 (각 함수는 레코드 재료 dict를 돌려주거나 Skip을 던진다)
# ---------------------------------------------------------------------------


def group_brands_sos(ctx: dict, cat: str) -> dict:
    facts, tables, triples = ctx["facts"][cat], ctx["tables"], ctx["triples"]
    group = gf.subjects(triples, "ownedByGroup", GROUP)
    raw_matched = _r_group(facts, group)
    rows = _t_group(tables, cat, group)
    if not rows:
        raise Skip(f"{cat}: 지표 테이블에 그룹 브랜드 없음")
    t_names = sorted(r["brand"].lower() for r in rows)
    r_names = sorted(b.brand.lower() for b in raw_matched)
    if t_names != r_names:
        raise _disagree("그룹 브랜드 집합", t_names, r_names)
    if any(r["sos"] is None for r in rows):
        raise Skip(f"{cat}: 지표 테이블 SoS 없음")
    raw_by = {b.brand.lower(): b for b in raw_matched}
    gold = {f"{gf.norm_id(r['brand'])}_sos": r["sos"] for r in rows}
    raw = {f"{gf.norm_id(r['brand'])}_sos": raw_by[r["brand"].lower()].sos for r in rows}
    listing = ", ".join(f"{r['brand']} {_pct(r['sos'])}" for r in rows)
    return {
        "question": (
            f"아모레퍼시픽 그룹 소속 브랜드 중 {D} {L[cat]} Top 100에 제품이 있는 브랜드와 "
            "각 브랜드의 SoS는?"
        ),
        "answer": (
            f"KG에서 AMOREPACIFIC 소속(ownedByGroup)으로 등록된 {len(group)}개 브랜드 중 {D} "
            f"{L[cat]} Top 100에 있는 브랜드는 {listing}입니다."
        ),
        "expected_values": gold,
        "kg_entities": ["amorepacific", *[gf.norm_id(r["brand"]) for r in rows], cat],
        "kg_edges": [gf.edge(r["brand"], "ownedByGroup", GROUP) for r in rows],
        "concepts": ["multi_hop", "brand_portfolio", "sos"],
        "hop_plan": [
            _step(1, "kg", f"?brand -ownedByGroup-> {GROUP}", f"{len(group)}개 브랜드"),
            _step(
                2,
                "db",
                f"{cat} {D} brand_metrics에서 ?brand 매칭(대소문자 무시)",
                [r["brand"] for r in rows],
            ),
            _step(3, "db", "브랜드별 SoS", {r["brand"]: r["sos"] for r in rows}),
        ],
        "system_access_inputs": ["sos"],
        "cross_check": _cross_check(gold, raw),
    }


def laneige_most_concentrated_category(ctx: dict) -> dict:
    facts, tables = ctx["facts"], ctx["tables"]
    cats = [c for c in gf.CATEGORIES if _t_find(tables, c, "LANEIGE") and _t_hhi(tables, c)]
    if not cats:
        raise Skip("지표 테이블에 LANEIGE 카테고리 없음")
    pick = max(cats, key=lambda c: _t_hhi(tables, c))
    raw_cats = [c for c in gf.CATEGORIES if "LANEIGE" in facts[c].brands]
    raw_pick = max(raw_cats, key=lambda c: facts[c].hhi) if raw_cats else None
    if set(cats) != set(raw_cats) or pick != raw_pick:
        raise _disagree("카테고리 선택", (pick, cats), (raw_pick, raw_cats))
    hhi = _t_hhi(tables, pick)
    return {
        "question": (
            f"{D} 기준 LANEIGE 제품이 Top 100에 있는 카테고리 중 시장 집중도(HHI)가 가장 높은 "
            "카테고리와 그 HHI는?"
        ),
        "answer": (
            f"{D} LANEIGE가 진입한 카테고리는 "
            + ", ".join(f"{L[c]}(HHI {_t_hhi(tables, c):.4f})" for c in cats)
            + f"이고, 이 중 HHI가 가장 높은 곳은 {L[pick]}({hhi:.4f})입니다."
        ),
        "expected_values": {"hhi": hhi},
        "kg_entities": ["laneige", pick],
        "kg_edges": [],
        "concepts": ["multi_hop", "hhi", "market_concentration"],
        "hop_plan": [
            _step(1, "db", f"{D} brand_metrics에서 LANEIGE가 있는 카테고리", cats),
            _step(2, "db", "카테고리별 HHI(market_metrics)", {c: _t_hhi(tables, c) for c in cats}),
            _step(3, "compare", "HHI 최대 카테고리", pick),
        ],
        "system_access_inputs": ["hhi", "sos"],
        "cross_check": _cross_check(
            {"hhi": hhi},
            {"hhi": facts[pick].hhi},
            raw_hhi_by_category={c: facts[c].hhi for c in raw_cats},
        ),
    }


def laneige_best_avg_rank_category_leader(ctx: dict) -> dict:
    facts, tables = ctx["facts"], ctx["tables"]
    avg = {
        c: _t_find(tables, c, "LANEIGE")["brand_avg_rank"]
        for c in gf.CATEGORIES
        if _t_find(tables, c, "LANEIGE") and _t_find(tables, c, "LANEIGE")["brand_avg_rank"]
    }
    if not avg:
        raise Skip("지표 테이블에 LANEIGE 평균 순위 없음")
    pick = min(avg, key=avg.get)
    if list(avg.values()).count(avg[pick]) > 1:
        raise Skip("평균 순위 동률")
    raw_avg = {
        c: facts[c].brands["LANEIGE"].avg_rank
        for c in gf.CATEGORIES
        if "LANEIGE" in facts[c].brands
    }
    raw_pick = min(raw_avg, key=raw_avg.get) if raw_avg else None
    if pick != raw_pick:
        raise _disagree("평균 순위 최소 카테고리", pick, raw_pick)
    top = _t_top(tables, pick)
    raw_top = _r_top(facts[pick])
    if raw_top is None or raw_top.brand != top["brand"]:
        raise _disagree("1위 브랜드", top["brand"], raw_top and raw_top.brand)
    _require_attribution(facts[pick], [top["brand"]])
    return {
        "question": (
            f"{D} 기준 LANEIGE 제품의 평균 순위가 가장 좋은 카테고리에서 SoS 1위 브랜드와 그 SoS는?"
        ),
        "answer": (
            f"{D} LANEIGE 평균 순위는 "
            + ", ".join(f"{L[c]} {avg[c]:g}위" for c in avg)
            + f"로 {L[pick]}가 가장 좋고, {L[pick]}의 SoS 1위 브랜드는 "
            f"{top['brand']}({_pct(top['sos'])})입니다."
        ),
        "expected_values": {"top_brand_sos": top["sos"]},
        "kg_entities": ["laneige", pick, gf.norm_id(top["brand"])],
        "kg_edges": [],
        "concepts": ["multi_hop", "category_leader", "sos"],
        "hop_plan": [
            _step(1, "db", "LANEIGE 카테고리별 평균 순위(brand_metrics)", avg),
            _step(2, "compare", "평균 순위 최소 카테고리", pick),
            _step(3, "db", f"{pick} SoS 1위 브랜드", {top["brand"]: top["sos"]}),
        ],
        "system_access_inputs": ["avg_rank", "sos"],
        "cross_check": _cross_check(
            {"top_brand_sos": top["sos"]},
            {"top_brand_sos": raw_top.sos},
            raw_avg_rank_by_category=raw_avg,
        ),
    }


def laneige_best_rank_category_brands_above(ctx: dict) -> dict:
    facts, tables = ctx["facts"], ctx["tables"]
    cats = [
        c for c in gf.CATEGORIES if "LANEIGE" in facts[c].brands and _t_find(tables, c, "LANEIGE")
    ]
    if not cats:
        raise Skip("LANEIGE 카테고리 없음")
    best = {c: facts[c].brands["LANEIGE"].best_rank for c in cats}  # raw_data 기록값
    pick = min(cats, key=lambda c: best[c])
    if list(best.values()).count(best[pick]) > 1:
        raise Skip("최고 순위 동률")
    t_lan = _t_find(tables, pick, "LANEIGE")["sos"]
    above = sorted(n for n, r in _t_rows(tables, pick).items() if (r["sos"] or 0) > t_lan)
    lan = facts[pick].brands["LANEIGE"]
    raw_above = sorted(b.brand for b in facts[pick].brands.values() if (b.sos or 0) > lan.sos)
    if len(above) != len(raw_above):
        raise _disagree("상위 브랜드 수", above, raw_above)
    _require_attribution(facts[pick], above)
    return {
        "question": (
            f"{D} 기준 LANEIGE의 최고 순위 제품이 속한 카테고리에서 LANEIGE보다 SoS가 높은 "
            "브랜드는 몇 개인가요?"
        ),
        "answer": (
            f"{D} LANEIGE 최고 순위 제품은 {L[pick]} {best[pick]}위이고, {L[pick]}에서 LANEIGE "
            f"SoS {_pct(t_lan)}보다 SoS가 높은 브랜드는 {len(above)}개({', '.join(above)})입니다."
        ),
        "expected_values": {"brand_count": float(len(above))},
        "kg_entities": ["laneige", pick],
        "kg_edges": [],
        "concepts": ["multi_hop", "sos", "competitive_gap"],
        "hop_plan": [
            _step(1, "db", "LANEIGE 카테고리별 최고 순위(raw_data)", best),
            _step(2, "compare", "최고 순위 카테고리", pick),
            _step(3, "db", f"{pick}에서 SoS > LANEIGE({t_lan})인 브랜드", above),
        ],
        "system_access_inputs": ["current_rank", "sos"],
        "cross_check": _cross_check(
            {"brand_count": float(len(above))},
            {"brand_count": float(len(raw_above))},
            raw_brands_above=raw_above,
        ),
    }


def sibling_count_in_category(ctx: dict, anchor: str, cat: str) -> dict:
    facts, tables, triples = ctx["facts"][cat], ctx["tables"], ctx["triples"]
    parents = gf.objects(triples, anchor, "ownedByGroup")
    if parents != [GROUP]:
        raise Skip(f"{anchor} 소속 그룹 {parents}")
    group = gf.subjects(triples, "ownedByGroup", GROUP)
    raw_matched = _r_group(facts, group)
    rows = _t_group(tables, cat, group)
    names = [r["brand"] for r in rows]
    raw_names = [b.brand for b in raw_matched]
    if sorted(n.lower() for n in names) != sorted(n.lower() for n in raw_names):
        raise _disagree("그룹 브랜드 집합", names, raw_names)
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
            _step(3, "db", f"{cat} {D} brand_metrics에 있는 ?brand", names),
        ],
        "system_access_inputs": ["sos"],
        "cross_check": _cross_check(
            {"brand_count": float(len(names))},
            {"brand_count": float(len(raw_names))},
            raw_brands=raw_names,
        ),
    }


def group_best_ranked_product(ctx: dict, cat: str) -> dict:
    facts, tables, triples = ctx["facts"][cat], ctx["tables"], ctx["triples"]
    group = gf.subjects(triples, "ownedByGroup", GROUP)
    raw_matched = _r_group(facts, group)
    rows = _t_group(tables, cat, group)
    if sorted(r["brand"].lower() for r in rows) != sorted(b.brand.lower() for b in raw_matched):
        raise _disagree(
            "그룹 브랜드 집합", [r["brand"] for r in rows], [b.brand for b in raw_matched]
        )
    if len(raw_matched) < 2:
        raise Skip(f"{cat}: 비교할 그룹 브랜드가 2개 미만")
    ranks = {b.brand: b.best_rank for b in raw_matched}  # 제품 순위는 raw_data 기록값이 정본
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
            _step(2, "db", f"{cat} {D}에 있는 ?brand", sorted(ranks)),
            _step(3, "db", "브랜드별 최고 순위(raw_data)", ranks),
            _step(4, "compare", "최고 순위 최소", best),
        ],
        "system_access_inputs": ["current_rank"],
        "cross_check": _cross_check(
            {"best_rank": float(ranks[best])},
            {"best_rank": float(mf_rank)},
            source="제품 순위는 raw_data 기록값(지표 테이블 없음) — MetricFactsProvider 값과 대조",
        ),
    }


def brand_category_leader(ctx: dict, brand_name: str) -> dict:
    facts, tables, triples = ctx["facts"], ctx["tables"], ctx["triples"]
    if not gf.objects(triples, brand_name, "ownedByGroup"):
        raise Skip(f"{brand_name}: KG 큐레이션 트리플에 없음")
    cats = [c for c in gf.CATEGORIES if _t_find(tables, c, brand_name)]
    raw_cats = [c for c in gf.CATEGORIES if gf.find_brand(facts[c], brand_name)]
    if cats != raw_cats:
        raise _disagree("진입 카테고리", cats, raw_cats)
    if len(cats) != 1:
        raise Skip(f"{brand_name}: 진입 카테고리가 하나가 아님 {cats}")
    cat = cats[0]
    b = gf.find_brand(facts[cat], brand_name)
    top = _t_top(tables, cat)
    raw_top = _r_top(facts[cat])
    if raw_top is None or raw_top.brand != top["brand"]:
        raise _disagree("1위 브랜드", top["brand"], raw_top and raw_top.brand)
    _require_attribution(facts[cat], [b.brand, top["brand"]])
    return {
        "question": (
            f"{D} 기준 {brand_name} 제품이 Top 100에 있는 카테고리에서 SoS 1위 브랜드와 그 SoS는?"
        ),
        "answer": (
            f"{D} {brand_name} 제품은 {L[cat]} Top 100에만 있고({b.best_rank}위), {L[cat]}의 "
            f"SoS 1위 브랜드는 {top['brand']}({_pct(top['sos'])})입니다."
        ),
        "expected_values": {"top_brand_sos": top["sos"]},
        "kg_entities": [gf.norm_id(brand_name), cat, gf.norm_id(top["brand"])],
        "kg_edges": [],
        "concepts": ["multi_hop", "category_leader", "sos"],
        "hop_plan": [
            _step(1, "db", f"{brand_name} 제품이 있는 카테고리", cats),
            _step(2, "db", f"{cat} SoS 1위 브랜드", {top["brand"]: top["sos"]}),
        ],
        "system_access_inputs": ["sos"],
        "cross_check": _cross_check({"top_brand_sos": top["sos"]}, {"top_brand_sos": raw_top.sos}),
    }


def category_leader_top_product_price(ctx: dict, cat: str) -> dict:
    facts, tables = ctx["facts"][cat], ctx["tables"]
    top = _t_top(tables, cat)
    raw_top = _r_top(facts)
    if raw_top is None or raw_top.brand != top["brand"]:
        raise _disagree("1위 브랜드", top["brand"], raw_top and raw_top.brand)
    _require_attribution(facts, [top["brand"]])
    record = facts.brands[top["brand"]]
    product = next(p for p in facts.products if p["rank"] == record.best_rank)
    if product["price"] is None:
        raise Skip("가격 없음")
    name = (product["product_name"] or "").split(",")[0].split(":")[0].strip()[:60]
    return {
        "question": (
            f"{D} 기준 {L[cat]} SoS 1위 브랜드의 최고 순위 제품은 무엇이고 가격은 얼마인가요?"
        ),
        "answer": (
            f"{D} {L[cat]} SoS 1위 브랜드는 {top['brand']}({_pct(top['sos'])})이고, 최고 순위 "
            f"제품은 {record.best_rank}위 '{name}'(ASIN {product['asin']}), 가격은 "
            f"${product['price']:.2f}입니다."
        ),
        "expected_values": {"price": product["price"], "best_rank": float(record.best_rank)},
        "kg_entities": [cat, gf.norm_id(top["brand"])],
        "kg_edges": [],
        "concepts": ["multi_hop", "category_leader", "price_analysis"],
        "hop_plan": [
            _step(1, "db", f"{cat} SoS 1위 브랜드(brand_metrics)", {top["brand"]: top["sos"]}),
            _step(
                2,
                "db",
                "그 브랜드의 최고 순위 제품(raw_data)",
                {"rank": record.best_rank, "asin": product["asin"]},
            ),
            _step(3, "db", "제품 가격(raw_data)", product["price"]),
        ],
        "system_access_inputs": ["sos", "current_rank", "price"],
        "cross_check": _cross_check(
            {"top_brand_sos": top["sos"]},
            {"top_brand_sos": raw_top.sos},
            record_only=["price", "best_rank"],
        ),
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

# 입력 → MetricFactsProvider가 챗봇에 주는지 (generate_rule_questions.METRIC_FACTS_SERVES와 같은 기준)
METRIC_FACTS_SERVES = {"sos", "hhi", "current_rank", "price"}


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
        not_served = [k for k in item["system_access_inputs"] if k not in METRIC_FACTS_SERVES]
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
                    "system_access": "not_in_metric_facts" if not_served else "metric_facts",
                    "hop_plan": item["hop_plan"],
                    "multihop_gold": {
                        "candidate": name,
                        "as_of": gf.AS_OF,
                        "gold_source_detail": "지표 테이블(brand_metrics·market_metrics); "
                        "순위·가격은 raw_data 기록값",
                        "expected_values_units": "sos %(0~100), hhi 0~1, 순위 '위', 가격 USD",
                        "kg_sources": list(gf.CURATED_KG_SOURCES) if uses_kg else [],
                        "system_access_inputs": item["system_access_inputs"],
                        "cross_check": item["cross_check"],
                    },
                },
            )
        )
    diffs = [
        r["metadata"]["multihop_gold"]["cross_check"]["max_relative_diff"]
        for r in records
        if r["metadata"]["multihop_gold"]["cross_check"]["max_relative_diff"] is not None
    ]
    log = {
        "as_of": gf.AS_OF,
        "generator": GENERATOR,
        "generated": len(records),
        "max_relative_diff": gf.distribution(diffs),
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

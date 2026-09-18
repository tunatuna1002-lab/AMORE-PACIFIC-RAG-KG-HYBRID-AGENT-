#!/usr/bin/env python3
"""②관계 유형 문항을 KG 큐레이션 트리플에서 **기계적으로** 생성한다.

배경
----
유형 분류(scripts/classify_golden_types.py) 결과 원본 골든셋의 관계 문항은 8개로, 유형별
시험지 게이트(20문항 이상)에 못 미친다. 킥오프 지시서 1-B는 ③규칙·④다단계 보충만 적었지만
게이트가 네 유형 모두 20문항 이상을 요구하므로 같은 방식(코드·데이터에서 기계적으로, 근거
저장)으로 관계 문항을 보충한다 — 판단 근거는 이 docstring이 남긴다.

정답 근거
--------
`data/knowledge_graph.json` 중 `source == "config/brands.json"` 트리플만 쓴다(ownedByGroup·
ownsBrand·hasSegment·originatesFrom·acquiredIn·siblingBrand). 크롤에서 만든 competesWith·
hasProduct는 날짜 버전이 없고 매일 바뀌어 정답이 고정되지 않으므로 쓰지 않는다. 트리플이
config/brands.json 원본과 일치하는지도 확인한다(불일치면 싣지 않는다).

정답이 DB 스냅샷이 아니라 정적 설정에서 오므로 `gold_source="document"`, `as_of` 없음으로 둔다.
부정 문항("자매 브랜드인가?" → 아니요)은 config/brands.json이 해당 브랜드를 경쟁 브랜드로
따로 등록한 경우에만 만든다(KG에 트리플이 없다는 것만으로 부정하지 않는다).

사용법:
    python3 scripts/generate_relation_questions.py --kg <knowledge_graph.json>
    python3 scripts/generate_relation_questions.py --kg ... --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import golden_snapshot_facts as gf  # noqa: E402

GENERATOR = "scripts/generate_relation_questions.py"
GROUP = "AMOREPACIFIC"
BRANDS_CONFIG = gf.REPO_ROOT / "config" / "brands.json"


class Skip(Exception):
    """후보를 싣지 않는 이유."""


def _config() -> dict[str, Any]:
    return json.loads(BRANDS_CONFIG.read_text(encoding="utf-8"))


def _config_brand(cfg: dict, name: str) -> dict[str, Any] | None:
    hits = [b for b in cfg["amorepacific_brands"] if b["name"] == name]
    return hits[0] if len(hits) == 1 else None


def _one(triples: list[dict], subject: str, predicate: str) -> str:
    values = gf.objects(triples, subject, predicate)
    if len(values) != 1:
        raise Skip(f"{subject} -{predicate}-> 값이 {len(values)}개 {values}")
    return values[0]


def _item(
    question: str,
    answer: str,
    edges: list[tuple[str, str, str]],
    entities: list[str],
    concepts: list[str],
    config_check: dict[str, Any],
    expected_values: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "question": question,
        "answer": answer,
        "kg_edges": [gf.edge(s, p, o) for s, p, o in edges],
        "kg_entities": entities,
        "concepts": concepts,
        "expected_values": expected_values or {},
        "relation_gold": {
            "triples": [{"subject": s, "predicate": p, "object": o} for s, p, o in edges],
            "kg_source": "config/brands.json",
            "config_check": config_check,
        },
    }


def parent_group(triples: list[dict], cfg: dict, brand: str) -> dict:
    parent = _one(triples, brand, "ownedByGroup")
    if _config_brand(cfg, brand) is None or parent != GROUP:
        raise Skip(f"{brand}: config amorepacific_brands에 없음 또는 그룹 {parent}")
    return _item(
        question=f"{brand}의 모회사(소속 그룹)는 어디인가요?",
        answer=f"{brand}는 {parent}(아모레퍼시픽) 그룹 소속 브랜드입니다.",
        edges=[(brand, "ownedByGroup", parent)],
        entities=[gf.norm_id(brand), "amorepacific"],
        concepts=["corporate_structure", "brand_portfolio"],
        config_check={"amorepacific_brands": True},
    )


def segment(triples: list[dict], cfg: dict, brand: str) -> dict:
    seg = _one(triples, brand, "hasSegment")
    conf = _config_brand(cfg, brand)
    if conf is None or conf.get("segment") != seg:
        raise Skip(f"{brand}: config segment {conf and conf.get('segment')} != KG {seg}")
    return _item(
        question=f"아모레퍼시픽 브랜드 포트폴리오에서 {brand}의 세그먼트는 무엇인가요?",
        answer=f"아모레퍼시픽 포트폴리오에서 {brand}의 세그먼트는 {seg}입니다.",
        edges=[(brand, "hasSegment", seg)],
        entities=[gf.norm_id(brand), "amorepacific"],
        concepts=["brand_portfolio", "brand_positioning"],
        config_check={"segment": conf.get("segment")},
    )


def origin(triples: list[dict], cfg: dict, brand: str) -> dict:
    country = _one(triples, brand, "originatesFrom")
    ownership = cfg.get("brand_ownership", {}).get(brand, {})
    conf_country = ownership.get("country_of_origin") or (_config_brand(cfg, brand) or {}).get(
        "country"
    )
    if conf_country != country:
        raise Skip(f"{brand}: config 원산지 {conf_country} != KG {country}")
    return _item(
        question=f"아모레퍼시픽 그룹 브랜드 {brand}의 원산지(출신 국가)는 어디인가요?",
        answer=f"{brand}의 원산지는 {country}이며 {GROUP} 그룹 소속입니다.",
        edges=[(brand, "originatesFrom", country), (brand, "ownedByGroup", GROUP)],
        entities=[gf.norm_id(brand), gf.norm_id(country), "amorepacific"],
        concepts=["brand_portfolio", "k_beauty"],
        config_check={"country": conf_country},
    )


def acquired(triples: list[dict], cfg: dict, brand: str) -> dict:
    year = _one(triples, brand, "acquiredIn")
    if not year.isdigit():
        raise Skip(f"{brand}: acquiredIn 값이 연도가 아님 '{year}'")
    conf = cfg.get("brand_ownership", {}).get(brand, {})
    if conf.get("acquired_date") != year:
        raise Skip(f"{brand}: config acquired_date {conf.get('acquired_date')} != KG {year}")
    return _item(
        question=f"아모레퍼시픽이 {brand}를 인수한 연도는?",
        answer=f"{brand}는 {year}년에 아모레퍼시픽({GROUP})에 인수됐습니다.",
        edges=[(brand, "acquiredIn", year), (brand, "ownedByGroup", GROUP)],
        entities=[gf.norm_id(brand), "amorepacific"],
        concepts=["corporate_structure", "brand_portfolio"],
        config_check={"acquired_date": conf.get("acquired_date")},
        expected_values={"acquired_year": float(year)},
    )


def sibling(triples: list[dict], cfg: dict, a: str, b: str) -> dict:
    siblings = gf.objects(triples, a, "siblingBrand")
    competitors = {c["name"] for c in cfg.get("competitor_brands", [])}
    if b in siblings:
        return _item(
            question=f"{a}와 {b}는 같은 그룹에 속한 자매 브랜드인가요?",
            answer=f"네. {a}와 {b}는 모두 {GROUP} 그룹 소속인 자매 브랜드입니다.",
            edges=[(a, "siblingBrand", b), (a, "ownedByGroup", GROUP), (b, "ownedByGroup", GROUP)],
            entities=[gf.norm_id(a), gf.norm_id(b), "amorepacific"],
            concepts=["corporate_structure", "brand_portfolio"],
            config_check={
                "both_in_amorepacific_brands": bool(_config_brand(cfg, a) and _config_brand(cfg, b))
            },
        )
    if b not in competitors:
        raise Skip(f"{a}-{b}: siblingBrand 없음이지만 부정 근거(config competitor_brands)도 없음")
    return _item(
        question=f"{a}와 {b}는 같은 그룹에 속한 자매 브랜드인가요?",
        answer=(
            f"아니요. {a}는 {GROUP} 그룹 소속이지만 {b}는 아모레퍼시픽 브랜드 설정에서 경쟁 "
            "브랜드로 등록돼 있어 자매 브랜드가 아닙니다."
        ),
        edges=[(a, "ownedByGroup", GROUP)],
        entities=[gf.norm_id(a), gf.norm_id(b), "amorepacific"],
        concepts=["corporate_structure", "competitor_analysis"],
        config_check={"b_in_competitor_brands": True},
    )


def group_brand_count(triples: list[dict], cfg: dict) -> dict:
    owned = gf.objects(triples, GROUP, "ownsBrand")
    inverse = gf.subjects(triples, "ownedByGroup", GROUP)
    conf = sorted(b["name"] for b in cfg["amorepacific_brands"])
    if owned != inverse or owned != conf:
        raise Skip(
            f"ownsBrand {len(owned)} / ownedByGroup {len(inverse)} / config {len(conf)} 불일치"
        )
    return _item(
        question="아모레퍼시픽 그룹이 소유한 브랜드로 등록된 브랜드는 몇 개인가요?",
        answer=f"아모레퍼시픽({GROUP})이 소유한 브랜드로 등록된 브랜드는 {len(owned)}개입니다.",
        edges=[],  # 31개 엣지를 모두 골드로 두면 엣지 recall이 개수 질문을 과하게 벌한다
        entities=["amorepacific"],
        concepts=["brand_portfolio", "corporate_structure"],
        config_check={"amorepacific_brands_count": len(conf)},
        expected_values={"brand_count": float(len(owned))},
    )


def same_segment_brands(triples: list[dict], cfg: dict, brand: str) -> dict:
    seg = _one(triples, brand, "hasSegment")
    peers = sorted(
        t["subject"]
        for t in triples
        if t["predicate"] == "hasSegment" and t["object"] == seg and t["subject"] != brand
    )
    conf_peers = sorted(
        b["name"]
        for b in cfg["amorepacific_brands"]
        if b.get("segment") == seg and b["name"] != brand
    )
    if peers != conf_peers or not peers:
        raise Skip(f"{brand}: 세그먼트 동료 KG {peers} != config {conf_peers}")
    return _item(
        question=f"아모레퍼시픽 포트폴리오에서 {brand}와 같은 세그먼트에 속한 브랜드는?",
        answer=f"{brand}의 세그먼트는 {seg}이며, 같은 세그먼트 브랜드는 {', '.join(peers)}입니다.",
        edges=[(brand, "hasSegment", seg)] + [(p, "hasSegment", seg) for p in peers],
        entities=[gf.norm_id(brand), *[gf.norm_id(p) for p in peers]],
        concepts=["brand_portfolio", "brand_positioning"],
        config_check={"segment_peers": conf_peers},
    )


CANDIDATES = [
    ("parent_group/COSRX", lambda t, c: parent_group(t, c, "COSRX")),
    ("parent_group/Sulwhasoo", lambda t, c: parent_group(t, c, "Sulwhasoo")),
    ("parent_group/HERA", lambda t, c: parent_group(t, c, "HERA")),
    ("parent_group/ETUDE", lambda t, c: parent_group(t, c, "ETUDE")),
    ("parent_group/AESTURA", lambda t, c: parent_group(t, c, "AESTURA")),
    ("segment/Sulwhasoo", lambda t, c: segment(t, c, "Sulwhasoo")),
    ("segment/COSRX", lambda t, c: segment(t, c, "COSRX")),
    ("segment/ETUDE", lambda t, c: segment(t, c, "ETUDE")),
    ("segment/AESTURA", lambda t, c: segment(t, c, "AESTURA")),
    ("origin/TATA HARPER", lambda t, c: origin(t, c, "TATA HARPER")),
    ("origin/COSRX", lambda t, c: origin(t, c, "COSRX")),
    ("acquired/COSRX", lambda t, c: acquired(t, c, "COSRX")),
    ("acquired/TATA HARPER", lambda t, c: acquired(t, c, "TATA HARPER")),
    ("sibling/LANEIGE-Sulwhasoo", lambda t, c: sibling(t, c, "LANEIGE", "Sulwhasoo")),
    ("sibling/LANEIGE-innisfree", lambda t, c: sibling(t, c, "LANEIGE", "innisfree")),
    ("sibling/LANEIGE-TIRTIR", lambda t, c: sibling(t, c, "LANEIGE", "TIRTIR")),
    ("sibling/LANEIGE-Beauty of Joseon", lambda t, c: sibling(t, c, "LANEIGE", "Beauty of Joseon")),
    ("group_brand_count", group_brand_count),
    ("same_segment/LANEIGE", lambda t, c: same_segment_brands(t, c, "LANEIGE")),
    ("same_segment/Sulwhasoo", lambda t, c: same_segment_brands(t, c, "Sulwhasoo")),
]


def build(kg_path: Path) -> tuple[list[dict], dict]:
    triples = gf.load_curated_triples(kg_path)
    cfg = _config()
    records: list[dict] = []
    excluded: list[dict] = []
    for name, fn in CANDIDATES:
        try:
            item = fn(triples, cfg)
        except Skip as e:
            excluded.append({"candidate": name, "reason": str(e)})
            continue
        records.append(
            gf.build_record(
                item_id=f"rl{len(records) + 1:03d}",
                question=item["question"],
                answer=item["answer"],
                question_type="relation",
                generator=GENERATOR,
                domain="brand",
                difficulty="easy" if len(item["kg_edges"]) <= 1 else "medium",
                requires_kg=True,
                gold_source="document",
                expected_values=item["expected_values"],
                kg_entities=item["kg_entities"],
                kg_edges=item["kg_edges"],
                concepts=item["concepts"],
                as_of=None,
                extra_metadata={"relation_gold": {"candidate": name, **item["relation_gold"]}},
            )
        )
    log = {"generator": GENERATOR, "generated": len(records), "excluded": excluded}
    return records, log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--kg", type=Path, default=gf.DEFAULT_KG)
    parser.add_argument("--out", type=Path, default=gf.TYPED_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    records, log = build(args.kg)
    print(f"생성 {len(records)}문항, 제외 {len(log['excluded'])}")
    for e in log["excluded"]:
        print(f"  제외 {e['candidate']}: {e['reason']}")
    return gf.write_or_check(
        {
            args.out / "generated_relation.jsonl": gf.dumps_jsonl(records),
            args.out / "generation_log_relation.json": gf.dumps_json(log),
        },
        args.check,
    )


if __name__ == "__main__":
    sys.exit(main())

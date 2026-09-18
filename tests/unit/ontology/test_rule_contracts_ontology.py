"""규칙 입력 ↔ 온톨로지 정식 술어·정적 사실 카드 (트랙 O4) [2026-09 사후].

- 플래그 ``ontology.use_class_reasoning`` OFF 카드(기존 어댑터)로 만든 규칙 판정은 O4 이전과
  같다 — 특성화 스냅샷(``fixtures/o4_rule_off_snapshot.json``, O4 변경 전 코드 ``e0ffac7``에서 생성).
- ON 카드(정식 술어 ``ownedByGroup``, 출처 ``ontology:registry``의 정적 사실)로도 소유 검증 규칙이
  입력을 채운다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.domain.entities.evidence import Evidence
from src.ontology.rule_contracts import build_rule_context, evaluate_rules_on_cards
from src.ontology.rules import ALL_BUSINESS_RULES
from src.rag.evidence_adapters import EvidenceAdapter

AS_OF = "2026-08-31"
SNAPSHOT = Path(__file__).parent / "fixtures" / "o4_rule_off_snapshot.json"
RULES_BY_PRIORITY = sorted(ALL_BUSINESS_RULES, key=lambda r: -r.priority)


def _share(brand: str, category: str, sos: float, **fields: Any) -> dict[str, Any]:
    return {
        "type": "brand_share",
        "brand": brand,
        "category": category,
        "snapshot_date": AS_OF,
        "present": True,
        "sos": sos,
        "product_count": 3,
        "brand_rank": 1,
        **fields,
    }


def _off_input_facts() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """OFF 경로가 받을 수 있는 모든 모양의 정적 사실을 담은 입력 (실제 KG 표기 그대로).

    - ``metric_edges``: OFF 검색기가 내는 ``ownedBy``와, 시드 표기 ``ownedByGroup``·정적 술어
      (``hasSegment``·``originatesFrom``·``acquiredIn`` — OFF 검색기는 거르지만 어댑터 입력으로는
      올 수 있다)
    - ``brand_info``: 엔티티 메타데이터의 정적 키(OFF 어댑터는 제외한다)
    """
    metric_facts = [
        {"type": "category_market", "category": "lip_care", "snapshot_date": AS_OF, "hhi": 0.08},
        _share("LANEIGE", "lip_care", 6.0, cpi=160.0, avg_rating_gap=0.2, brand_avg_rank=12.0),
        _share("COSRX", "skin_care", 3.0, cpi=85.0, avg_rating_gap=0.1),
    ]
    kg_facts = [
        {
            "type": "metric_edges",
            "entity": "cosrx",
            "data": {
                "edges": [
                    {"subject": "COSRX", "predicate": "ownedByGroup", "object": "AMOREPACIFIC"},
                    {"subject": "COSRX", "predicate": "hasSegment", "object": "K-Beauty"},
                    {"subject": "COSRX", "predicate": "originatesFrom", "object": "Korea"},
                    {"subject": "COSRX", "predicate": "acquiredIn", "object": "2024"},
                ]
            },
        },
        {
            "type": "metric_edges",
            "entity": "laneige",
            "data": {
                "edges": [{"subject": "LANEIGE", "predicate": "ownedBy", "object": "AMOREPACIFIC"}]
            },
        },
        {
            "type": "brand_info",
            "entity": "tirtir",
            "data": {"parent_group": "goodai", "segment": "Mid", "country_of_origin": "Korea"},
        },
    ]
    return metric_facts, kg_facts


def _fingerprint(cards: list[Evidence], brands: list[str], categories: list[str]) -> dict:
    run = evaluate_rules_on_cards(RULES_BY_PRIORITY, cards, brands, categories)
    evaluations = []
    for e in run.evaluations:
        ev = e.evaluation
        result = e.result()
        evaluations.append(
            {
                "combination": list(e.combination),
                "rule": ev.rule_name,
                "fired": ev.fired,
                "reason": ev.non_fire_reason.labels() if ev.non_fire_reason else [],
                "inputs": {k: repr(v) for k, v in sorted(ev.inputs.items())},
                "derived_from": list(e.derived_from),
                "scope": list(e.scope),
                "result": None
                if result is None
                else {
                    "insight": result.insight,
                    "metadata": repr(result.metadata),
                    "evidence": repr(sorted(result.evidence.items())),
                },
            }
        )
    return {"summary": run.summary(), "evaluations": evaluations}


def off_fingerprint() -> dict:
    metric_facts, kg_facts = _off_input_facts()
    adapter = EvidenceAdapter()  # OFF: 온톨로지 없음
    cards = [*adapter.from_metric_facts(metric_facts), *adapter.from_kg_facts(kg_facts).cards]
    return {
        "predicates": sorted({c.predicate for c in cards}),
        "brand_only": _fingerprint(cards, ["cosrx", "laneige", "tirtir"], []),
        "with_category": _fingerprint(cards, ["laneige", "cosrx"], ["lip_care", "skin_care"]),
    }


class TestFlagOffCharacterization:
    def test_off_cards_never_carry_canonical_group_predicate(self):
        # OFF 어댑터는 시드 표기 ownedByGroup을 ownedBy로 바꾼다 — 규칙이 ownedByGroup을
        # 받아도 OFF 카드에는 그 술어가 없으므로 no-op이다
        metric_facts, kg_facts = _off_input_facts()
        cards = EvidenceAdapter().from_kg_facts(kg_facts).cards

        predicates = {c.predicate for c in cards}
        assert "ownedByGroup" not in predicates
        assert "ownedBy" in predicates
        assert all(c.source != "ontology:registry" for c in cards)

    def test_off_rule_run_matches_pre_o4_snapshot(self):
        expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

        assert off_fingerprint() == expected


# ---------------------------------------------------------------------------
# 플래그 ON 카드: 정식 술어 + 등록부 정적 사실
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def onto():
    from src.ontology.ontology import get_ontology

    return get_ontology()


@pytest.fixture(scope="module")
def on_adapter(onto) -> EvidenceAdapter:
    return EvidenceAdapter(ontology=onto)


def _registry_cards(onto, adapter: EvidenceAdapter, brands: list[str]) -> list[Evidence]:
    from src.rag.ontology_context import plan_query, static_fact

    fact = static_fact(onto, plan_query(onto, {"brands": brands}))
    assert fact is not None
    return adapter.from_kg_facts([fact]).cards


class TestParentGroupAcceptsCanonicalPredicate:
    def test_on_kg_edge_is_read_as_owned_by_group(self, on_adapter):
        cards = on_adapter.from_kg_facts(
            [
                {
                    "type": "metric_edges",
                    "entity": "laneige",
                    "data": {
                        "edges": [
                            {"subject": "LANEIGE", "predicate": "ownedBy", "object": "AMOREPACIFIC"}
                        ]
                    },
                }
            ]
        ).cards
        assert [c.predicate for c in cards] == ["ownedByGroup"]  # ON 어댑터의 정식화

        context, card_ids = build_rule_context(cards, "laneige", None)

        assert context["parent_group"] == "amorepacific"
        assert card_ids["parent_group"] == (cards[0].id,)

    def test_registry_group_card_fills_parent_group(self, onto, on_adapter):
        cards = _registry_cards(onto, on_adapter, ["COSRX"])
        group = [c for c in cards if c.predicate == "ownedByGroup"]
        assert group and group[0].source == "ontology:registry"

        context, card_ids = build_rule_context(cards, "cosrx", None)

        assert context["parent_group"] == "amorepacific"
        assert card_ids["parent_group"] == (group[0].id,)


class TestOwnershipProfileInputs:
    def test_registry_profile_fills_origin_segment_acquired(self, onto, on_adapter):
        cards = _registry_cards(onto, on_adapter, ["COSRX"])
        by_pred = {c.predicate: c for c in cards}

        context, card_ids = build_rule_context(cards, "cosrx", None)

        assert context["country_of_origin"] == "South Korea"  # 등록부 라벨 (id south_korea)
        assert context["segment"] == "K-Beauty"
        assert context["acquired"] == "2024"
        assert card_ids["country_of_origin"] == (by_pred["originatesFrom"].id,)
        assert card_ids["segment"] == (by_pred["hasSegment"].id,)
        assert card_ids["acquired"] == (by_pred["acquiredIn"].id,)

    def test_ownership_rule_reports_registry_profile(self, onto, on_adapter):
        cards = _registry_cards(onto, on_adapter, ["COSRX"])

        run = evaluate_rules_on_cards(RULES_BY_PRIORITY, cards, ["cosrx"], [])
        (result,) = [
            r for r in run.fired_results() if r.rule_name == "brand_ownership_verification"
        ]

        assert result.metadata["country_of_origin"] == "South Korea"
        assert result.metadata["segment"] == "K-Beauty"
        assert result.metadata["acquired"] == "2024"
        assert "인수 연도: 2024" in result.insight
        assert sorted(result.evidence["derived_from"]) == sorted(
            c.id
            for c in cards
            if c.predicate in {"ownedByGroup", "originatesFrom", "hasSegment", "acquiredIn"}
        )

    def test_unknown_origin_stays_missing(self, onto, on_adapter):
        # IOPE: 등록부에 원산지가 없다(KG의 "Korea"는 kg_updater 기본값 — 결정 OA-5)
        cards = _registry_cards(onto, on_adapter, ["IOPE"])

        context, _ = build_rule_context(cards, "iope", None)

        assert context["parent_group"] == "amorepacific"
        assert "country_of_origin" not in context
        assert context["segment"] == "Luxury"

    def test_kg_profile_edges_are_not_profile_inputs(self, on_adapter):
        # KG 정적 트리플(출처 kg)은 원산지·세그먼트·인수 입력이 아니다 — 등록부 카드만 읽는다
        cards = on_adapter.from_kg_facts(
            [
                {
                    "type": "metric_edges",
                    "entity": "somebrand",
                    "data": {
                        "edges": [
                            {"subject": "somebrand", "predicate": "originatesFrom", "object": "X"},
                            {"subject": "somebrand", "predicate": "hasSegment", "object": "Y"},
                            {"subject": "somebrand", "predicate": "acquiredIn", "object": "1999"},
                        ]
                    },
                }
            ]
        ).cards
        assert len(cards) == 3

        context, _ = build_rule_context(cards, "somebrand", None)

        assert not {"country_of_origin", "segment", "acquired"} & set(context)

    def test_registry_brand_without_group_does_not_fire(self, onto, on_adapter):
        # rg032 (TIRTIR): 등록부에 그룹이 없다 → parent_group 결측, 프로필만으로 발화하지 않는다
        cards = _registry_cards(onto, on_adapter, ["TIRTIR"])

        run = evaluate_rules_on_cards(RULES_BY_PRIORITY, cards, ["tirtir"], [])
        (evaluation,) = [
            e.evaluation
            for e in run.evaluations
            if e.evaluation.rule_name == "brand_ownership_verification"
        ]

        assert not evaluation.fired
        assert evaluation.non_fire_reason.labels() == ["missing_input:parent_group"]


class TestPriceRulesDoNotReadSegment:
    @pytest.mark.parametrize("rule_name", ["value_position", "premium_price_position"])
    def test_price_rule_inputs_unchanged(self, rule_name):
        # 두 가격 규칙의 조건·결론은 cpi·rating_gap·brand·asin만 읽는다 — 세그먼트·티어를 쓰는
        # 로직이 없어 입력으로 연결하지 않는다 (의미를 만들지 않는다)
        from src.ontology.rule_contracts import RULE_CONTRACTS

        assert "segment" not in RULE_CONTRACTS[rule_name].input_names


def test_registry_source_matches_adapter():
    from src.ontology.rule_contracts import REGISTRY_SOURCE
    from src.rag.evidence_adapters import ONTOLOGY_SOURCE

    assert REGISTRY_SOURCE == ONTOLOGY_SOURCE


if __name__ == "__main__":  # 스냅샷 재생성 (O4 변경 전 코드에서만 실행할 것)
    SNAPSHOT.parent.mkdir(exist_ok=True)
    SNAPSHOT.write_text(
        json.dumps(off_fingerprint(), ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pytest.main([__file__, "-q"])

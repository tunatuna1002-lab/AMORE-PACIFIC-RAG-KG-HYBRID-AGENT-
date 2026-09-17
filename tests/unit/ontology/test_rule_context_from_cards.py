"""증거 카드 → 규칙 추론 컨텍스트 (트랙 3-B, 설계 E3)

배경(F7): 규칙 추론 입력이 대시보드 JSON 키를 읽었고 그 키가 운영 데이터에 없어서 v4
기준선 233문항 × 3회에서 추론이 0건이었다. 이제 입력은 증거 카드이고, 계약
(``InputSpec.binding``)이 어떤 카드에서 값을 읽는지 정한다.

규칙·계약·추론 엔진·어댑터·KG는 모두 실제 객체다 (mock 없음).
"""

from __future__ import annotations

from typing import Any

import pytest

from src.domain.entities.evidence import Evidence
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.ontology.relations import Relation, RelationType
from src.ontology.rule_contracts import (
    MAX_RULE_BRANDS,
    MAX_RULE_CATEGORIES,
    RULE_CONTRACTS,
    build_rule_context,
    evaluate_rule,
    evaluate_rules_on_cards,
)
from src.ontology.rules import ALL_BUSINESS_RULES, register_all_rules
from src.rag.evidence_adapters import EvidenceAdapter
from src.rag.metric_facts import MAX_BRANDS, MAX_CATEGORIES

AS_OF = "2026-08-31"
RULES = {rule.name: rule for rule in ALL_BUSINESS_RULES}


@pytest.fixture(scope="module")
def adapter() -> EvidenceAdapter:
    return EvidenceAdapter()


@pytest.fixture
def reasoner() -> OntologyReasoner:
    reasoner = OntologyReasoner()
    register_all_rules(reasoner)
    return reasoner


def _market(category: str, **fields: Any) -> dict[str, Any]:
    return {"type": "category_market", "category": category, "snapshot_date": AS_OF, **fields}


def _share(brand: str, category: str, sos: float | None = None, **fields: Any) -> dict[str, Any]:
    if sos is None:
        return {
            "type": "brand_share",
            "brand": brand,
            "category": category,
            "snapshot_date": AS_OF,
            "present": False,
        }
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


def _products(brand: str, category: str, products: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "brand_products",
        "brand": brand,
        "category": category,
        "snapshot_date": AS_OF,
        "products": [{"brand": brand, **p} for p in products],
    }


def _card(cards: list[Evidence], predicate: str, subject: str, obj: str | None = None) -> Evidence:
    (card,) = [
        c for c in cards if c.predicate == predicate and c.subject == subject and c.object == obj
    ]
    return card


# ---------------------------------------------------------------------------
# build_rule_context: 카드 → 컨텍스트 + 입력별 근거 카드 id
# ---------------------------------------------------------------------------


class TestBuildRuleContext:
    def test_sos_and_hhi_come_from_cards_with_their_ids(self, adapter):
        cards = adapter.from_metric_facts(
            [_market("lip_care", hhi=0.10), _share("LANEIGE", "lip_care", 18.0)]
        )

        context, card_ids = build_rule_context(cards, "laneige", "lip_care")

        sos_card = _card(cards, "sos", "laneige", "lip_care")
        hhi_card = _card(cards, "hhi", "lip_care")
        assert context["sos"] == pytest.approx(0.18)  # 카드 정본 0~1
        assert context["hhi"] == pytest.approx(0.10)
        assert card_ids["sos"] == (sos_card.id,)
        assert card_ids["hhi"] == (hhi_card.id,)
        # 파생 입력: 질의 엔티티에서 (카드 id 없음)
        assert context["brand"] == "laneige"
        assert context["category"] == "lip_care"
        assert context["is_target"] is True
        assert "brand" not in card_ids and "is_target" not in card_ids

    def test_other_brand_is_not_target(self, adapter):
        cards = adapter.from_metric_facts([_share("eos", "lip_care", 9.0)])

        context, _ = build_rule_context(cards, "eos", "lip_care")

        assert context["is_target"] is False

    def test_absent_brand_leaves_sos_missing_not_zero(self, adapter):
        cards = adapter.from_metric_facts(
            [_market("lip_care", hhi=0.10), _share("LANEIGE", "lip_care", None)]
        )

        context, card_ids = build_rule_context(cards, "laneige", "lip_care")

        assert "sos" not in context  # present_in_top100=False 카드를 0%로 읽지 않는다
        assert "sos" not in card_ids
        assert context["hhi"] == pytest.approx(0.10)

    def test_card_of_another_category_or_brand_is_not_bound(self, adapter):
        cards = adapter.from_metric_facts(
            [
                _market("skin_care", hhi=0.30),
                _share("LANEIGE", "skin_care", 18.0),
                _share("eos", "lip_care", 9.0),
            ]
        )

        context, card_ids = build_rule_context(cards, "laneige", "lip_care")

        assert "sos" not in context and "hhi" not in context
        assert card_ids == {}

    def test_no_brand_means_no_brand_inputs(self, adapter):
        cards = adapter.from_metric_facts(
            [_market("lip_care", hhi=0.10), _share("LANEIGE", "lip_care", 18.0)]
        )

        context, _ = build_rule_context(cards, None, "lip_care")

        assert context == {"category": "lip_care", "hhi": pytest.approx(0.10)}

    def test_cpi_rating_gap_and_avg_rank_cards(self, adapter):
        cards = adapter.from_metric_facts(
            [
                _share(
                    "LANEIGE",
                    "face_powder",
                    1.0,
                    cpi=226.7,
                    avg_rating_gap=-0.281,
                    brand_avg_rank=94.0,
                )
            ]
        )

        context, card_ids = build_rule_context(cards, "laneige", "face_powder")

        assert context["cpi"] == pytest.approx(226.7)
        assert context["rating_gap"] == pytest.approx(-0.281)
        assert context["avg_rank"] == pytest.approx(94.0)
        assert card_ids["cpi"] == (_card(cards, "cpi", "laneige", "face_powder").id,)
        assert card_ids["rating_gap"] == (
            _card(cards, "avg_rating_gap", "laneige", "face_powder").id,
        )
        assert card_ids["avg_rank"] == (
            _card(cards, "brand_avg_rank", "laneige", "face_powder").id,
        )

    def test_product_inputs_come_from_one_best_ranked_product(self, adapter):
        cards = adapter.from_metric_facts(
            [
                _market("face_powder", category_avg_price=17.2),
                _products(
                    "LANEIGE",
                    "face_powder",
                    [
                        {"rank": 9, "name": "LANEIGE Neo Blurring Powder", "price": 25.0},
                        {"rank": 40, "name": "LANEIGE Neo Essential Powder", "price": 30.0},
                    ],
                ),
            ]
        )

        context, card_ids = build_rule_context(cards, "laneige", "face_powder")

        best = "LANEIGE Neo Blurring Powder"
        assert context["current_rank"] == 9
        assert context["rank"] == 9
        assert context["price"] == pytest.approx(25.0)  # 순위와 같은 제품의 가격
        assert context["category_avg_price"] == pytest.approx(17.2)
        assert card_ids["rank"] == (_card(cards, "bsr_rank", best, "face_powder").id,)
        assert card_ids["price"] == (_card(cards, "price", best, "face_powder").id,)

    def test_relation_inputs_from_kg_cards(self, adapter, tmp_path):
        kg = KnowledgeGraph(
            persist_path=str(tmp_path / "kg.json"), auto_load=False, auto_save=False
        )
        for competitor in ("blistex", "chapstick", "aquaphor", "burt's bees", "nivea"):
            kg.add_relation(
                Relation(
                    "laneige",
                    RelationType.COMPETES_WITH,
                    competitor,
                    properties={"category": "lip_care"},
                )
            )
        facts = [
            {"type": "competitors", "entity": "laneige", "data": kg.get_competitors("laneige")},
            {
                "type": "metric_edges",
                "entity": "laneige",
                "data": {
                    "edges": [
                        {"subject": "LANEIGE", "predicate": "ownedBy", "object": "AMOREPACIFIC"}
                    ]
                },
            },
        ]
        cards = adapter.from_kg_facts(facts).cards

        context, card_ids = build_rule_context(cards, "laneige", "lip_care")

        competitor_cards = [c for c in cards if c.predicate == "competesWith"]
        assert context["competitor_count"] == 5
        assert sorted(item["brand"] for item in context["competitors"]) == sorted(
            c.object for c in competitor_cards
        )
        assert sorted(card_ids["competitor_count"]) == sorted(c.id for c in competitor_cards)
        assert context["parent_group"] == "amorepacific"  # canonical id
        assert card_ids["parent_group"] == (_card(cards, "ownedBy", "laneige", "amorepacific").id,)


# ---------------------------------------------------------------------------
# 결함: 대소문자 비교 (brand_ownership_verification)
# ---------------------------------------------------------------------------


class TestBrandOwnershipCase:
    @pytest.mark.parametrize("parent_group", ["amorepacific", "AMOREPACIFIC", "AmorePacific"])
    def test_group_identity_is_case_insensitive(self, parent_group):
        evaluation = evaluate_rule(
            RULES["brand_ownership_verification"], {"brand": "cosrx", "parent_group": parent_group}
        )

        assert evaluation.fired, evaluation.non_fire_reason

    def test_other_group_does_not_fire(self):
        evaluation = evaluate_rule(
            RULES["brand_ownership_verification"], {"brand": "tirtir", "parent_group": "goodai"}
        )

        assert not evaluation.fired


# ---------------------------------------------------------------------------
# 결함: reasoner.apply가 결론 필드(position 등)를 버린다
# ---------------------------------------------------------------------------


class TestConclusionFieldsPreserved:
    def test_position_is_kept_on_result(self):
        result = RULES["market_dominance_fragmented"].apply(
            {"brand": "laneige", "sos": 0.2, "hhi": 0.1}
        )

        assert result is not None
        assert result.conclusion["position"] == "dominant_in_fragmented"
        assert "insight" not in result.conclusion  # 이미 필드로 옮긴 키는 중복 저장하지 않는다
        assert "metadata" not in result.conclusion
        assert result.to_dict()["conclusion"] == {"position": "dominant_in_fragmented"}
        # 기존 metadata는 그대로
        assert result.metadata["market_type"] == "fragmented"

    def test_market_structure_is_kept(self):
        competitors = [{"brand": f"b{i}"} for i in range(6)]
        result = RULES["fragmented_market_competition"].apply(
            {"hhi": 0.1, "competitor_count": 6, "competitors": competitors}
        )

        assert result.conclusion["market_structure"] == "fragmented_competitive"

    def test_inference_card_value_is_the_conclusion(self, adapter):
        result = RULES["market_dominance_fragmented"].apply(
            {"brand": "laneige", "category": "lip_care", "sos": 0.2, "hhi": 0.1}
        )

        (card,) = adapter.from_inferences([result])

        assert card.value == "dominant_in_fragmented"


# ---------------------------------------------------------------------------
# 조합 평가·중복 제거·derived_from
# ---------------------------------------------------------------------------


class TestEvaluateRulesOnCards:
    def test_caps_follow_metric_facts_provider(self):
        # 제공자가 조회하지 않는 조합에는 카드가 없다 — 상한을 같게 둔다
        assert MAX_RULE_BRANDS == MAX_BRANDS
        assert MAX_RULE_CATEGORIES == MAX_CATEGORIES

    def test_market_dominance_fires_with_sorted_card_basis(self, adapter, reasoner):
        cards = adapter.from_metric_facts(
            [_market("lip_care", hhi=0.10), _share("LANEIGE", "lip_care", 18.0)]
        )

        run = evaluate_rules_on_cards(reasoner.rules_by_priority, cards, ["laneige"], ["lip_care"])
        results = run.fired_results()

        assert run.combinations == [("laneige", "lip_care")]
        (result,) = [r for r in results if r.rule_name == "market_dominance_fragmented"]
        expected = sorted(
            [_card(cards, "sos", "laneige", "lip_care").id, _card(cards, "hhi", "lip_care").id]
        )
        assert result.evidence["derived_from"] == expected
        snapshot = result.evidence["context_snapshot"]
        assert snapshot["brand"] == "laneige"
        assert snapshot["category"] == "lip_care"
        assert snapshot["as_of"] == AS_OF
        (card,) = adapter.from_inferences([result])
        assert card.derived_from == tuple(expected)
        assert (card.subject, card.object, card.as_of) == ("laneige", "lip_care", AS_OF)

    def test_brand_only_query_uses_categories_of_brand_cards(self, adapter, reasoner):
        cards = adapter.from_metric_facts(
            [
                _market("lip_care", hhi=0.07),
                _share("LANEIGE", "lip_care", 2.0),
                _market("skin_care", hhi=0.07),
                _share("LANEIGE", "skin_care", 1.0),
                _share("eos", "lip_makeup", 9.0),
            ]
        )

        run = evaluate_rules_on_cards(reasoner.rules_by_priority, cards, ["laneige"], [])

        assert run.combinations == [("laneige", "lip_care"), ("laneige", "skin_care")]
        fired = [r for r in run.fired_results() if r.rule_name == "category_entry_opportunity"]
        # 카테고리마다 입력 카드가 다르다 → 별개 발화
        assert sorted(r.evidence["context_snapshot"]["category"] for r in fired) == [
            "lip_care",
            "skin_care",
        ]

    def test_absence_card_is_not_an_entered_category(self, adapter, reasoner):
        # 브랜드만 링크된 질의 "COSRX는 아모레퍼시픽 소속?" → 제공자는 cosrx가 진입한 skin_care에서
        # 함께 링크된 amorepacific의 부재 카드도 싣는다. 부재는 진입이 아니다.
        cards = adapter.from_metric_facts(
            [_share("COSRX", "skin_care", 1.04), _share("AMOREPACIFIC", "skin_care", None)]
        )

        run = evaluate_rules_on_cards(
            reasoner.rules_by_priority, cards, ["cosrx", "amorepacific"], []
        )

        assert run.combinations == [("cosrx", "skin_care"), ("amorepacific", None)]

    def test_combinations_are_capped(self, adapter, reasoner):
        brands = [f"brand{i}" for i in range(5)]
        categories = [f"cat{i}" for i in range(5)]

        run = evaluate_rules_on_cards(reasoner.rules_by_priority, [], brands, categories)

        assert len(run.combinations) == MAX_RULE_BRANDS * MAX_RULE_CATEGORIES

    def test_category_only_query(self, adapter, reasoner):
        cards = adapter.from_metric_facts([_market("lip_care", hhi=0.07)])

        run = evaluate_rules_on_cards(reasoner.rules_by_priority, cards, [], ["lip_care"])

        assert run.combinations == [(None, "lip_care")]
        assert run.fired_results() == []

    def test_same_inputs_across_combinations_fire_once(self, adapter, reasoner, tmp_path):
        metric = adapter.from_metric_facts(
            [_share("COSRX", "skin_care", 5.0), _share("COSRX", "lip_care", 1.0)]
        )
        relation = adapter.from_kg_facts(
            [
                {
                    "type": "metric_edges",
                    "entity": "cosrx",
                    "data": {
                        "edges": [
                            {"subject": "COSRX", "predicate": "ownedBy", "object": "AMOREPACIFIC"}
                        ]
                    },
                }
            ]
        ).cards

        run = evaluate_rules_on_cards(
            reasoner.rules_by_priority, [*metric, *relation], ["cosrx"], []
        )

        assert len(run.combinations) == 2
        ownership = [
            r for r in run.fired_results() if r.rule_name == "brand_ownership_verification"
        ]
        assert len(ownership) == 1  # 카테고리를 읽지 않는 규칙은 조합마다 중복 발화하지 않는다
        (card,) = adapter.from_inferences(ownership)
        assert card.subject == "cosrx"
        assert card.object is None  # 카테고리 범위가 아니다
        assert card.derived_from == (_card(relation, "ownedBy", "cosrx", "amorepacific").id,)
        assert run.summary()["fired"].count("brand_ownership_verification") == 1

    def test_summary_shape_and_missing_sos_reason(self, adapter, reasoner):
        cards = adapter.from_metric_facts(
            [_market("lip_care", hhi=0.10), _share("LANEIGE", "lip_care", None)]
        )

        run = evaluate_rules_on_cards(reasoner.rules_by_priority, cards, ["laneige"], ["lip_care"])
        summary = run.summary()

        assert set(summary) == {
            "combinations",
            "evaluated",
            "fired",
            "non_fire_top",
            "non_fire_counts_by_kind",
        }
        assert summary["combinations"] == [["laneige", "lip_care"]]
        assert summary["evaluated"] == len(RULE_CONTRACTS)
        assert "market_dominance_fragmented" not in summary["fired"]
        labels = dict(summary["non_fire_top"])
        assert labels["missing_input:sos"] >= 1
        assert len(summary["non_fire_top"]) <= 10
        assert all(isinstance(pair, list) and len(pair) == 2 for pair in summary["non_fire_top"])
        assert summary["non_fire_counts_by_kind"]["missing_input"] >= 1
        assert all(isinstance(k, str) for k in summary["non_fire_counts_by_kind"])
        assert (
            sum(summary["non_fire_counts_by_kind"].values()) + len(summary["fired"])
            == (summary["evaluated"])
        )

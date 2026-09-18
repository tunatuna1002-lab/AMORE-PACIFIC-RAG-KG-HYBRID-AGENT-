"""[2026-09 사후] O0-A: L4 새 지표 — 발화 규칙 제약 위반·기대 타입 일관성."""

import json

import pytest

from eval.metrics.l4_ontology import L4OntologyMetrics, aggregate_l4_extended
from eval.schemas import GoldEvidence, KGQueryTrace, OntologyReasoningTrace
from eval.validators.ontology_validator import EntityTypeRegistry


@pytest.fixture
def registry(tmp_path):
    (tmp_path / "brands.json").write_text(
        json.dumps(
            {
                "target_brand": {"name": "LANEIGE", "parent": "AMOREPACIFIC", "aliases": []},
                "amorepacific_brands": [{"name": "COSRX"}],
                "competitor_brands": [{"name": "TIRTIR", "tier": "mid"}],
                "segments": {"Premium": ["LANEIGE"]},
            }
        )
    )
    (tmp_path / "category_hierarchy.json").write_text(
        json.dumps({"categories": {"lip_care": {"amazon_node_id": "3761351"}, "beauty": {}}})
    )
    return EntityTypeRegistry.from_config(tmp_path)


def _inference(rule: str, snapshot: dict, related: list[str] | None = None) -> dict:
    return {
        "rule_name": rule,
        "insight_type": "market_position",
        "insight": "x",
        "confidence": 0.9,
        "evidence": {"context_snapshot": snapshot},
        "related_entities": related if related is not None else ["laneige"],
    }


GOOD = {"brand": "laneige", "category": "lip_care", "sos": 0.05, "as_of": "2026-08-31"}


class TestRuleConstraintViolation:
    def test_no_inference_is_none(self, registry):
        m = L4OntologyMetrics(registry=registry).compute(
            OntologyReasoningTrace(), KGQueryTrace(), GoldEvidence(), rule_evaluation=None
        )
        assert m.rule_constraint_violation_rate is None
        assert m.rule_checked_inferences == 0
        # 레거시 필드는 바뀌지 않는다
        assert m.constraint_violation_rate == 0.0
        assert m.type_consistency_rate == 1.0

    def test_clean_inference_is_zero(self, registry):
        trace = OntologyReasoningTrace(inferences=[_inference("r1", GOOD)])
        m = L4OntologyMetrics(registry=registry).compute(
            trace, KGQueryTrace(), GoldEvidence(), rule_evaluation={"fired": ["r1"]}
        )
        assert m.rule_constraint_violation_rate == 0.0
        assert m.rule_checked_inferences == 1

    @pytest.mark.parametrize(
        ("snapshot", "related", "kind"),
        [
            ({**GOOD, "brand": "unknown"}, None, "subject_type"),
            ({**GOOD, "brand": "amorepacific"}, None, "subject_type"),
            ({**GOOD, "brand": ""}, None, "subject_type"),
            ({**GOOD, "category": "laneige"}, None, "category_type"),
            (GOOD, ["", "laneige"], "related_entity_invalid"),
            (GOOD, ["fresh"], "related_entity_invalid"),
            ({**GOOD, "sos": 5.2}, None, "value_range"),
            ({**GOOD, "hhi": -0.1}, None, "value_range"),
            ({**GOOD, "avg_rank": 0}, None, "value_range"),
            ({**GOOD, "price": "25"}, None, "value_range"),
            ({k: v for k, v in GOOD.items() if k != "as_of"}, None, "missing_as_of"),
            ({"brand": "tirtir", "parent_group": "amorepacific"}, None, None),  # 그룹 모름
            ({"brand": "cosrx", "parent_group": "LVMH"}, None, "ownership_mismatch"),
        ],
    )
    def test_violation_kinds(self, registry, snapshot, related, kind):
        trace = OntologyReasoningTrace(inferences=[_inference("r1", snapshot, related)])
        m = L4OntologyMetrics(registry=registry).compute(
            trace, KGQueryTrace(), GoldEvidence(), rule_evaluation={"fired": ["r1"]}
        )
        if kind is None:
            assert m.rule_constraint_violation_rate == 0.0
        else:
            assert m.rule_constraint_violation_rate == 1.0
            assert kind in m.rule_violation_kinds

    def test_schema_violation_uses_legacy_check(self, registry):
        bad = _inference("r1", GOOD)
        bad["confidence"] = 1.5
        m = L4OntologyMetrics(registry=registry).compute(
            OntologyReasoningTrace(inferences=[bad]), KGQueryTrace(), GoldEvidence()
        )
        assert m.rule_violation_kinds == {"schema": 1}

    def test_only_fired_rules_are_checked(self, registry):
        trace = OntologyReasoningTrace(
            inferences=[_inference("r1", GOOD), _inference("r2", {**GOOD, "sos": 9})]
        )
        m = L4OntologyMetrics(registry=registry).compute(
            trace, KGQueryTrace(), GoldEvidence(), rule_evaluation={"fired": ["r1", "r3"]}
        )
        assert m.rule_checked_inferences == 1
        assert m.rule_constraint_violation_rate == 0.0
        assert m.rule_fired_unchecked == 1  # r3는 추론 본문이 없다

    def test_rate_is_per_inference(self, registry):
        trace = OntologyReasoningTrace(
            inferences=[
                _inference("r1", GOOD),
                _inference("r2", {**GOOD, "sos": 9}),
                _inference("r3", GOOD),
                _inference("r4", {**GOOD, "brand": "unknown", "hhi": 2}),
            ]
        )
        m = L4OntologyMetrics(registry=registry).compute(trace, KGQueryTrace(), GoldEvidence())
        assert m.rule_constraint_violation_rate == 0.5
        assert m.rule_violation_kinds == {"value_range": 2, "subject_type": 1}


class TestTypedConsistency:
    def test_edge_signature_and_registry(self, registry):
        kg = KGQueryTrace(
            kg_edges_found=[
                "laneige -competesWith-> tirtir",  # 2 checks, ok
                "laneige -ownedBy-> amorepacific",  # 별칭 → ownedByGroup, ok
                "lip_care -competesWith-> laneige",  # subject category → 위반
                "laneige -hasPosition-> premium",  # 범위 리터럴 → subject만
                "mystery -competesWith-> laneige",  # mystery는 모름 → untyped
                "laneige -likes-> tirtir",  # 시그니처 없음
            ]
        )
        m = L4OntologyMetrics(registry=registry).compute(
            OntologyReasoningTrace(), kg, GoldEvidence()
        )
        assert m.type_checks == 8
        assert m.type_violations == 1
        assert m.type_untyped == 1
        assert m.typed_consistency_rate == 7 / 8
        assert m.type_violation_kinds == {"edge:competesWith:subject=category": 1}
        assert m.type_source == "registry"
        assert m.types_registry_derived is True

    def test_gold_edge_types_override_and_mark_source(self, registry):
        # 골드 엣지가 lip_sleeping_mask를 product로 정한다(등록부는 모름)
        gold = GoldEvidence(kg_edges=["laneige -hasProduct-> lip_sleeping_mask"])
        kg = KGQueryTrace(kg_edges_found=["lip_sleeping_mask -belongsToCategory-> lip_care"])
        m = L4OntologyMetrics(registry=registry).compute(OntologyReasoningTrace(), kg, gold)
        assert m.type_checks == 2
        assert m.typed_consistency_rate == 1.0
        assert m.type_source == "gold+registry"

    def test_explicit_gold_types_clear_registry_flag(self, registry):
        gold = GoldEvidence(kg_entity_types={"laneige": "category"})
        kg = KGQueryTrace(kg_edges_found=["laneige -hasSoS-> lip_care"])
        m = L4OntologyMetrics(registry=registry).compute(OntologyReasoningTrace(), kg, gold)
        assert m.types_registry_derived is False
        assert m.type_violations == 1

    def test_ontology_facts_placeholder_brand(self, registry):
        facts = [
            {
                "type": "category_brands",
                "entity": "lip_care",
                "data": {"top_brands": [{"brand": "unknown"}, {"brand": "laneige"}]},
            },
            {
                "type": "brand_products",
                "entity": "laneige",
                "data": {"products": [{"asin": "B000000001", "category": "lip_care"}]},
            },
        ]
        kg = KGQueryTrace(ontology_facts=facts)
        m = L4OntologyMetrics(registry=registry).compute(
            OntologyReasoningTrace(), kg, GoldEvidence()
        )
        # lip_care(entity) · unknown · laneige · laneige(entity, 같은 label 아님) · asin · category
        assert m.type_violation_kinds == {"fact:category_brands:brand=placeholder": 1}
        assert m.type_checks == 6
        assert "pattern" in m.type_source

    def test_no_checks_is_none(self, registry):
        m = L4OntologyMetrics(registry=registry).compute(
            OntologyReasoningTrace(), KGQueryTrace(), GoldEvidence()
        )
        assert m.typed_consistency_rate is None
        assert m.type_source == "none"


class TestAggregate:
    def test_macro_skips_none(self, registry):
        calc = L4OntologyMetrics(registry=registry)
        items = [
            calc.compute(OntologyReasoningTrace(), KGQueryTrace(), GoldEvidence()),
            calc.compute(
                OntologyReasoningTrace(
                    inferences=[_inference("r1", GOOD), _inference("r2", {**GOOD, "sos": 3})]
                ),
                KGQueryTrace(kg_edges_found=["lip_care -competesWith-> laneige"]),
                GoldEvidence(),
            ),
        ]
        agg = aggregate_l4_extended(items)
        assert agg["rule_checked_items"] == 1
        assert agg["rule_constraint_violation_rate"] == 0.5
        assert agg["rule_violation_micro"] == 0.5
        assert agg["type_checked_items"] == 1
        assert agg["typed_consistency_rate"] == 0.5
        assert agg["constraint_violation_rate_legacy"] == 0.0
        assert agg["types_registry_derived_items"] == 2
        assert agg["type_source_counts"] == {"none": 1, "registry": 1}

"""규칙 입력 계약 (트랙 3-A, 설계 E3)

배경: v4 평가 130문항 × 10실행에서 규칙 추론이 0건 발화했다. 추론 입력이 대시보드 JSON
키를 읽는데 그 키가 없어서, 결측이면 발화하지 않도록 고친 규칙들이 조용히 전부 침묵했다.
계약은 규칙마다 필요한 입력(이름·타입·단위·범위)과 그 입력을 채울 증거 카드를 선언하고,
래퍼는 발화하지 않은 이유를 남긴다.

규칙·추론 엔진은 실제 객체를 쓴다 (mock 없음).
"""

from __future__ import annotations

from typing import Any

import pytest

from src.domain.entities.evidence import KNOWN_UNITS, EvidenceKind
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import InferenceRule, OntologyReasoner, RuleCondition
from src.ontology.relations import InsightType, Relation, RelationType
from src.ontology.rule_contracts import (
    RULE_CONTRACTS,
    InputSpec,
    InputType,
    NonFireKind,
    NonFireReason,
    Reduce,
    RuleContract,
    RuleEvaluation,
    count_non_fire_kinds,
    evaluate_all,
    evaluate_rule,
    top_non_fire_reasons,
)
from src.ontology.rules import (
    ALERT_RULES,
    ALL_BUSINESS_RULES,
    GROWTH_RULES,
    IR_CROSS_ANALYSIS_RULES,
    MARKET_RULES,
    PRICE_RULES,
    SENTIMENT_RULES,
    register_all_rules,
)

RULES = {rule.name: rule for rule in ALL_BUSINESS_RULES}
ASIN = "B00LANEIGE"
COMPETITORS = [{"brand": f"brand_{i}"} for i in range(6)]

# 규칙마다 발화하는 정상 입력 (카드 정본 단위: SoS·HHI 0~1, CPI 100 기준, 평점 격차는 점수 차)
FIRING_CONTEXTS: dict[str, dict[str, Any]] = {
    "market_dominance_fragmented": {"brand": "laneige", "sos": 0.2, "hhi": 0.1},
    "market_dominance_concentrated": {"brand": "laneige", "sos": 0.3, "hhi": 0.3},
    "challenger_position": {"brand": "laneige", "sos": 0.1, "hhi": 0.3},
    "fragmented_market_competition": {
        "hhi": 0.1,
        "competitor_count": 6,
        "competitors": COMPETITORS,
    },
    "strong_avg_rank": {"brand": "laneige", "is_target": True, "avg_rank": 12.5},
    "competitive_pressure": {
        "sos_change": -0.03,
        "competitor_count": 4,
        "competitors": COMPETITORS,
    },
    "price_quality_mismatch": {"brand": "laneige", "asin": ASIN, "cpi": 130, "rating_gap": -0.2},
    "market_disruption": {"has_rank_shock": True, "churn_rate": 0.25, "products": [ASIN]},
    "rank_decline_alert": {"asin": ASIN, "rank_change_7d": 8, "rank_volatility": 6.5},
    "stable_growth": {"asin": ASIN, "brand": "laneige", "streak_days": 45, "rank_change_7d": -3},
    "trend_alignment_opportunity": {
        "brand": "laneige",
        "is_target": True,
        "trend_keywords": ["lip mask", "glass skin"],
    },
    "top10_stability": {
        "asin": ASIN,
        "brand": "laneige",
        "current_rank": 4,
        "streak_days": 20,
        "rank_volatility": 1.2,
    },
    "category_entry_opportunity": {
        "brand": "laneige",
        "category": "skin_care",
        "is_target": True,
        "hhi": 0.07,
        "sos": 0.01,
    },
    "rating_momentum_positive": {"asin": ASIN, "rating_trend": 0.08, "review_count": 250},
    "top3_achievement": {
        "brand": "laneige",
        "is_target": True,
        "category": "lip_care",
        "asin": ASIN,
        "current_rank": 2,
    },
    "strong_rating_position": {"brand": "laneige", "is_target": True, "rating_gap": 0.2},
    "value_position": {"brand": "laneige", "asin": ASIN, "cpi": 80, "rating_gap": 0.1},
    "premium_price_position": {"brand": "laneige", "cpi": 170, "rating_gap": 0.0},
    "discount_dependent": {
        "asin": ASIN,
        "brand": "laneige",
        "discount_periods": [{"start": "2026-07-01", "end": "2026-07-07"}],
        "rank_improvements": [{"start": "2026-07-02", "end": "2026-07-05"}],
    },
    "viral_effect": {"asin": ASIN, "brand": "laneige", "price_stable": True, "rank_change_7d": -5},
    "bestseller_badge_effect": {
        "asin": ASIN,
        "brand": "laneige",
        "badge": "Best Seller",
        "rank_change_7d": 2,
    },
    "high_discount_dependency_score": {
        "asin": ASIN,
        "brand": "laneige",
        "product_history": [
            {"rank": 20, "discount_percent": 0, "date": "2026-07-01"},
            {"rank": 10, "discount_percent": 20, "date": "2026-07-02"},
        ],
    },
    "premium_defense_success": {
        "asin": ASIN,
        "brand": "laneige",
        "price": 30.0,
        "category_avg_price": 20.0,
        "rank": 5,
    },
    "sentiment_strength_hydration": {
        "asin": ASIN,
        "brand": "laneige",
        "sentiment_clusters": {"Hydration": ["moisturizing", "hydrating"]},
    },
    "sentiment_value_advantage": {
        "asin": ASIN,
        "brand": "laneige",
        "sentiment_tags": ["Value for money"],
        "competitor_sentiment_tags": ["Long lasting"],
    },
    "sentiment_weakness_packaging": {
        "asin": ASIN,
        "brand": "laneige",
        "sentiment_clusters": {"Hydration": ["hydrating"]},
        "competitor_sentiment_clusters": {"Packaging": 3},
    },
    "sentiment_usability_strength": {
        "asin": ASIN,
        "brand": "laneige",
        "sentiment_clusters": {"Usability": ["easy to use"]},
    },
    "sentiment_effectiveness_strong": {
        "asin": ASIN,
        "brand": "laneige",
        "sentiment_clusters": {"Effectiveness": ["works overnight"]},
    },
    "sentiment_gap_sensory": {
        "asin": ASIN,
        "brand": "laneige",
        "sentiment_clusters": {"Hydration": ["hydrating"]},
        "competitor_sentiment_clusters": {"Sensory": ["nice scent"]},
    },
    "customer_perception_positive": {
        "asin": ASIN,
        "brand": "laneige",
        "ai_summary": "Customers love the texture.",
    },
    "customer_perception_mixed": {
        "asin": ASIN,
        "brand": "laneige",
        "ai_summary": "Customers like the scent but wish it lasted longer.",
    },
    "ir_prime_day_impact": {
        "brand": "laneige",
        "category": "lip_care",
        "ir_mentions_prime_day": True,
        "rank_change_during_event": -15,
    },
    "ir_americas_revenue_correlation": {
        "brand": "laneige",
        "is_target": True,
        "ir_americas_yoy": 6.9,
        "sos_change": 0.01,
    },
    "ir_growth_momentum": {"brand": "laneige", "ir_consecutive_growth_quarters": 3},
    "ir_growth_slowdown_warning": {
        "brand": "laneige",
        "ir_prev_qtr_growth": 10.0,
        "ir_current_qtr_growth": 3.0,
    },
    "ir_brand_campaign_effect": {
        "brand": "laneige",
        "asin": ASIN,
        "ir_campaign_mentioned": True,
        "rank_change_7d": -4,
        "campaign_name": "Lip Sleeping Mask",
        "ir_source": "AP_3Q25_EN.md",
    },
    "brand_ownership_verification": {
        "brand": "COSRX",
        "parent_group": "AMOREPACIFIC",
        "country_of_origin": "Korea",
        "acquired": "2024",
        "segment": "K-Beauty",
        "evidence": ["config/brands.json"],
    },
}


class RecordingDict(dict):
    """규칙 조건·결론이 실제로 읽는 키를 기록하는 컨텍스트."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.accessed: set[str] = set()

    def get(self, key: Any, default: Any = None) -> Any:
        self.accessed.add(key)
        return super().get(key, default)

    def __getitem__(self, key: Any) -> Any:
        self.accessed.add(key)
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self.accessed.add(key)  # type: ignore[arg-type]
        return super().__contains__(key)


def _condition_keys(rule_name: str) -> set[str]:
    """빈 컨텍스트와 발화 컨텍스트 두 경로에서 조건이 읽는 키."""
    rule = RULES[rule_name]
    accessed: set[str] = set()
    for probe in ({}, FIRING_CONTEXTS[rule_name]):
        recorder = RecordingDict(probe)
        for condition in rule.conditions:
            condition.evaluate(recorder)
        accessed |= recorder.accessed
    return accessed


def _conclusion_keys(rule_name: str) -> set[str]:
    recorder = RecordingDict(FIRING_CONTEXTS[rule_name])
    RULES[rule_name].conclusion(recorder)
    return recorder.accessed


@pytest.fixture()
def reasoner() -> OntologyReasoner:
    engine = OntologyReasoner()
    register_all_rules(engine)
    return engine


def _evaluation(evaluations: list[RuleEvaluation], rule_name: str) -> RuleEvaluation:
    return next(e for e in evaluations if e.rule_name == rule_name)


# ---------------------------------------------------------------------------
# 계약 ↔ 규칙 코드 정합성
# ---------------------------------------------------------------------------


class TestContractCoverage:
    def test_rule_count_is_37(self):
        assert len(ALL_BUSINESS_RULES) == 37
        assert len(RULES) == 37  # 이름 중복 없음

    def test_every_rule_has_exactly_one_contract(self):
        assert set(RULE_CONTRACTS) == set(RULES)

    def test_fixture_covers_every_rule(self):
        assert set(FIRING_CONTEXTS) == set(RULES)

    @pytest.mark.parametrize(
        ("family", "rules"),
        [
            ("market", MARKET_RULES),
            ("alert", ALERT_RULES),
            ("growth", GROWTH_RULES),
            ("price", PRICE_RULES),
            ("sentiment", SENTIMENT_RULES),
            ("ir", IR_CROSS_ANALYSIS_RULES),
        ],
    )
    def test_family_matches_rule_module(self, family, rules):
        for rule in rules:
            assert RULE_CONTRACTS[rule.name].family == family

    @pytest.mark.parametrize("rule_name", sorted(FIRING_CONTEXTS))
    def test_declared_inputs_cover_keys_read_by_conditions(self, rule_name):
        undeclared = _condition_keys(rule_name) - RULE_CONTRACTS[rule_name].input_names
        assert not undeclared, f"{rule_name} 조건이 계약에 없는 키를 읽는다: {undeclared}"

    @pytest.mark.parametrize("rule_name", sorted(FIRING_CONTEXTS))
    def test_declared_inputs_cover_keys_read_by_conclusion(self, rule_name):
        undeclared = _conclusion_keys(rule_name) - RULE_CONTRACTS[rule_name].input_names
        assert not undeclared, f"{rule_name} 결론이 계약에 없는 키를 읽는다: {undeclared}"

    @pytest.mark.parametrize("rule_name", sorted(FIRING_CONTEXTS))
    def test_required_inputs_are_read_by_conditions(self, rule_name):
        # 조건이 읽지 않는 입력을 필수로 두면 발화할 규칙을 계약이 막는다
        extra = RULE_CONTRACTS[rule_name].required_names - _condition_keys(rule_name)
        assert not extra, f"{rule_name} 필수 입력인데 조건이 읽지 않는다: {extra}"

    def test_every_input_has_a_source_or_a_gap(self):
        for contract in RULE_CONTRACTS.values():
            for spec in contract.inputs:
                if spec.binding is None and spec.derivation is None:
                    assert spec.gap, f"{contract.rule_name}.{spec.name}: 공급원도 사유도 없다"

    def test_binding_units_are_card_units(self):
        for contract in RULE_CONTRACTS.values():
            for spec in contract.inputs:
                if spec.binding is not None and spec.binding.unit is not None:
                    assert spec.binding.unit in KNOWN_UNITS

    def test_same_input_name_has_one_meaning_across_rules(self):
        seen: dict[str, tuple] = {}
        for contract in RULE_CONTRACTS.values():
            for spec in contract.inputs:
                shape = (spec.type, spec.unit, spec.min, spec.max, spec.binding)
                assert seen.setdefault(spec.name, shape) == shape, spec.name

    def test_spec_unit_equals_card_unit_for_value_bindings(self):
        for contract in RULE_CONTRACTS.values():
            for spec in contract.inputs:
                binding = spec.binding
                if binding is not None and binding.reduce in (Reduce.VALUE, Reduce.MIN_VALUE):
                    assert spec.unit == binding.unit, f"{contract.rule_name}.{spec.name}"

    def test_numeric_ranges_only_on_numeric_inputs(self):
        for contract in RULE_CONTRACTS.values():
            for spec in contract.inputs:
                if spec.min is not None or spec.max is not None:
                    assert spec.type in (InputType.NUMBER, InputType.INTEGER), spec.name


# ---------------------------------------------------------------------------
# 판정: 계약 검사 → 조건 → 결론
# ---------------------------------------------------------------------------


class TestEvaluateRule:
    @pytest.mark.parametrize("rule_name", sorted(FIRING_CONTEXTS))
    def test_every_rule_fires_on_valid_inputs(self, rule_name):
        evaluation = evaluate_rule(RULES[rule_name], FIRING_CONTEXTS[rule_name])

        assert evaluation.fired, evaluation.non_fire_reason
        assert evaluation.non_fire_reason is None
        assert evaluation.result is not None
        assert evaluation.result.rule_name == rule_name

    @pytest.mark.parametrize("rule_name", sorted(FIRING_CONTEXTS))
    def test_removing_any_required_input_reports_it_missing(self, rule_name):
        contract = RULE_CONTRACTS[rule_name]
        for name in sorted(contract.required_names):
            context = {k: v for k, v in FIRING_CONTEXTS[rule_name].items() if k != name}
            evaluation = evaluate_rule(RULES[rule_name], context)
            assert not evaluation.fired
            assert evaluation.non_fire_reason == NonFireReason.missing_input([name])

    def test_market_position_fires_with_real_reasoner(self, reasoner):
        evaluations = evaluate_all(reasoner.rules_by_priority, {"sos": 0.2, "hhi": 0.1})

        evaluation = _evaluation(evaluations, "market_dominance_fragmented")
        assert evaluation.fired
        assert "20.0%" in evaluation.result.insight
        assert evaluation.inputs == {"sos": 0.2, "hhi": 0.1}

    def test_missing_sos_is_reported_not_read_as_zero(self, reasoner):
        evaluations = evaluate_all(reasoner.rules_by_priority, {"hhi": 0.1})

        evaluation = _evaluation(evaluations, "market_dominance_fragmented")
        assert not evaluation.fired
        assert evaluation.result is None
        assert evaluation.non_fire_reason == NonFireReason.missing_input(["sos"])

    def test_none_counts_as_missing(self):
        evaluation = evaluate_rule(RULES["market_dominance_fragmented"], {"sos": None, "hhi": 0.1})

        assert evaluation.non_fire_reason == NonFireReason.missing_input(["sos"])

    def test_all_missing_inputs_are_listed_in_declaration_order(self):
        evaluation = evaluate_rule(RULES["market_dominance_fragmented"], {})

        assert evaluation.non_fire_reason == NonFireReason.missing_input(["sos", "hhi"])

    def test_percent_sos_is_out_of_range(self, reasoner):
        evaluations = evaluate_all(reasoner.rules_by_priority, {"sos": 13.5, "hhi": 0.1})

        evaluation = _evaluation(evaluations, "market_dominance_fragmented")
        assert not evaluation.fired
        assert evaluation.non_fire_reason == NonFireReason.out_of_range("sos", 13.5, (0.0, 1.0))

    def test_hhi_on_10000_scale_is_out_of_range(self):
        evaluation = evaluate_rule(RULES["market_dominance_fragmented"], {"sos": 0.2, "hhi": 681})

        assert evaluation.non_fire_reason == NonFireReason.out_of_range("hhi", 681, (0.01, 1.0))

    def test_hhi_zero_placeholder_is_out_of_range(self):
        # 운영 DB market_metrics.hhi=0 19행(2025-12)은 brand_metrics가 없는 미계산 행이다.
        # Top100에서 HHI는 1/100 이상이므로 0은 관측값이 아니라 결측의 자리표시자다.
        evaluation = evaluate_rule(RULES["market_dominance_fragmented"], {"sos": 0.2, "hhi": 0.0})

        assert evaluation.non_fire_reason == NonFireReason.out_of_range("hhi", 0.0, (0.01, 1.0))

    def test_conditions_not_met_lists_every_failed_condition(self, reasoner):
        evaluations = evaluate_all(reasoner.rules_by_priority, {"sos": 0.05, "hhi": 0.3})

        evaluation = _evaluation(evaluations, "market_dominance_fragmented")
        assert not evaluation.fired
        assert evaluation.non_fire_reason == NonFireReason.conditions_not_met(
            ["sos_above_0.15", "hhi_below_0.15"]
        )

    def test_type_mismatch(self):
        evaluation = evaluate_rule(RULES["market_dominance_fragmented"], {"sos": "20%", "hhi": 0.1})

        assert evaluation.non_fire_reason == NonFireReason.type_mismatch("sos", "20%")

    def test_bool_is_not_a_number(self):
        evaluation = evaluate_rule(RULES["market_dominance_fragmented"], {"sos": True, "hhi": 0.1})

        assert evaluation.non_fire_reason == NonFireReason.type_mismatch("sos", True)

    def test_brand_sentiment_cluster_counts_are_a_type_mismatch(self):
        # KG get_brand_sentiment_profile의 clusters는 {cluster: 빈도}다. 규칙은 태그 목록을
        # 기대해 len(int)에서 조건 예외 → False로 조용히 침묵했다.
        context = {"sentiment_clusters": {"Hydration": 4}}

        evaluation = evaluate_rule(RULES["sentiment_strength_hydration"], context)

        assert evaluation.non_fire_reason == NonFireReason.type_mismatch(
            "sentiment_clusters", {"Hydration": 4}
        )

    @pytest.mark.parametrize(
        ("context", "kind"),
        [
            ({}, NonFireKind.MISSING_INPUT),
            ({"x": "1"}, NonFireKind.TYPE_MISMATCH),
            ({"x": 2.0}, NonFireKind.OUT_OF_RANGE),
        ],
    )
    def test_contract_violation_skips_condition_evaluation(self, context, kind):
        # 결측을 0으로 읽는 조건(ctx.get("x", 0))이 있어도 계약 위반이면 조건을 부르지 않는다
        called: list[dict] = []

        def reads_missing_as_zero(ctx: dict) -> bool:
            called.append(ctx)
            return ctx.get("x", 0) < 0.5

        rule = InferenceRule(
            name="spy_rule",
            description="조건 호출 감시",
            conditions=[RuleCondition("x_low", reads_missing_as_zero, "x < 0.5")],
            conclusion=lambda ctx: {"insight": "x가 낮다"},
            insight_type=InsightType.MARKET_POSITION,
        )
        contracts = {
            "spy_rule": RuleContract(
                rule_name="spy_rule",
                family="test",
                inputs=(InputSpec("x", InputType.NUMBER, None, min=0.0, max=1.0, gap="테스트"),),
            )
        }

        evaluation = evaluate_rule(rule, context, contracts=contracts)

        assert not evaluation.fired
        assert evaluation.non_fire_reason.kind is kind
        assert called == []

    def test_rule_without_contract(self):
        rule = InferenceRule(
            name="unregistered_rule",
            description="계약 없는 규칙",
            conditions=[RuleCondition("always", lambda ctx: True, "항상")],
            conclusion=lambda ctx: {"insight": "x"},
            insight_type=InsightType.MARKET_POSITION,
        )

        evaluation = evaluate_rule(rule, {})

        assert not evaluation.fired
        assert evaluation.non_fire_reason == NonFireReason.no_contract()

    def test_conclusion_failure_is_distinguished_from_conditions(self):
        rule = InferenceRule(
            name="broken_conclusion",
            description="결론이 예외",
            conditions=[RuleCondition("has_x", lambda ctx: ctx.get("x") is not None, "x")],
            conclusion=lambda ctx: {"insight": 1 / 0},
            insight_type=InsightType.MARKET_POSITION,
        )
        contracts = {
            "broken_conclusion": RuleContract(
                rule_name="broken_conclusion",
                family="test",
                inputs=(InputSpec("x", InputType.NUMBER, None, gap="테스트"),),
            )
        }

        evaluation = evaluate_rule(rule, {"x": 1}, contracts=contracts)

        assert not evaluation.fired
        assert evaluation.non_fire_reason == NonFireReason.conclusion_failed()


# ---------------------------------------------------------------------------
# KG에서 채우는 입력
# ---------------------------------------------------------------------------


class TestKnowledgeGraphInput:
    @staticmethod
    def _kg(tmp_path) -> KnowledgeGraph:
        kg = KnowledgeGraph(
            persist_path=str(tmp_path / "kg.json"), auto_load=False, auto_save=False
        )
        for competitor in ("blistex", "chapstick", "aquaphor", "burt's bees", "nivea"):
            kg.add_relation(
                Relation(
                    subject="laneige",
                    predicate=RelationType.COMPETES_WITH,
                    object=competitor,
                    properties={"category": "lip_care"},
                )
            )
        return kg

    def test_competitor_count_missing_without_kg_enrichment(self):
        evaluation = evaluate_rule(
            RULES["fragmented_market_competition"], {"brand": "laneige", "hhi": 0.1}
        )

        assert evaluation.non_fire_reason == NonFireReason.missing_input(["competitor_count"])

    def test_competitor_count_from_real_kg_fires_rule(self, tmp_path):
        kg = self._kg(tmp_path)
        competitors = kg.get_competitors("laneige")
        context = {
            "brand": "laneige",
            "hhi": 0.1,
            "competitor_count": len(competitors),
            "competitors": competitors,
        }

        evaluation = evaluate_rule(RULES["fragmented_market_competition"], context)

        assert evaluation.fired, evaluation.non_fire_reason
        assert "5개의 경쟁 브랜드" in evaluation.result.insight

    def test_competitor_binding_matches_real_relation_cards(self, tmp_path):
        from src.rag.evidence_adapters import EvidenceAdapter

        kg = self._kg(tmp_path)
        adapter = EvidenceAdapter(brand_normalizer=str.lower, category_normalizer=str.lower)
        cards = adapter.from_kg_facts(
            [{"type": "competitors", "entity": "laneige", "data": kg.get_competitors("laneige")}]
        ).cards
        spec = RULE_CONTRACTS["fragmented_market_competition"].spec("competitor_count")

        matched = [card for card in cards if spec.binding.matches(card)]

        assert spec.binding.kind is EvidenceKind.RELATION
        assert len(matched) == 5
        assert {card.subject for card in matched} == {"laneige"}


class TestMetricBindingsAgainstAdapter:
    def test_sos_and_hhi_cards_fill_market_position_rule(self, reasoner):
        from src.rag.evidence_adapters import EvidenceAdapter

        adapter = EvidenceAdapter(brand_normalizer=str.lower, category_normalizer=str.lower)
        cards = adapter.from_metric_facts(
            [
                {
                    "type": "category_market",
                    "category": "lip_care",
                    "snapshot_date": "2026-09-01",
                    "hhi": 0.1,
                },
                {
                    "type": "brand_share",
                    "brand": "LANEIGE",
                    "category": "lip_care",
                    "snapshot_date": "2026-09-01",
                    "present": True,
                    "sos": 20.0,  # DB 퍼센트 → 카드 0.2
                    "product_count": 5,
                    "brand_rank": 1,
                },
            ]
        )
        contract = RULE_CONTRACTS["market_dominance_fragmented"]
        context: dict[str, Any] = {}
        for name in ("sos", "hhi"):
            binding = contract.spec(name).binding
            (card,) = [card for card in cards if binding.matches(card)]
            if binding.subject_role == "brand":
                assert (card.subject, card.object) == ("laneige", "lip_care")
            else:
                assert (card.subject, card.object) == ("lip_care", None)
            context[name] = card.value

        evaluations = evaluate_all(reasoner.rules_by_priority, context)

        assert _evaluation(evaluations, "market_dominance_fragmented").fired


# ---------------------------------------------------------------------------
# 미발화 사유 집계
# ---------------------------------------------------------------------------


class TestNonFireAggregation:
    def test_top_reasons_and_kind_counts(self, reasoner):
        evaluations = evaluate_all(reasoner.rules_by_priority, {"sos": 0.05, "hhi": 0.3})

        top = dict(top_non_fire_reasons(evaluations))
        kinds = count_non_fire_kinds(evaluations)

        fired = [e.rule_name for e in evaluations if e.fired]
        assert fired == ["challenger_position"]  # 집중 시장(HHI 0.3) + SoS 5% = 도전자
        assert sum(kinds.values()) == 36
        assert top["missing_input:competitor_count"] == 2
        assert top["conditions_not_met:sos_above_0.15"] == 1
        assert kinds[NonFireKind.MISSING_INPUT] > kinds[NonFireKind.CONDITIONS_NOT_MET]

    def test_top_reasons_are_sorted_and_limited(self, reasoner):
        evaluations = evaluate_all(reasoner.rules_by_priority, {})

        top = top_non_fire_reasons(evaluations, limit=3)

        assert len(top) == 3
        assert [count for _, count in top] == sorted((c for _, c in top), reverse=True)
        assert all(label.startswith("missing_input:") for label, _ in top)

    def test_fired_rules_have_no_reason(self, reasoner):
        evaluations = evaluate_all(reasoner.rules_by_priority, {"sos": 0.2, "hhi": 0.1})

        assert all(e.non_fire_reason is None for e in evaluations if e.fired)
        assert all(e.non_fire_reason is not None for e in evaluations if not e.fired)

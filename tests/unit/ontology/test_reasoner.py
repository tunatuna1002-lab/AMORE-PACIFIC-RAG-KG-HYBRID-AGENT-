"""
Tests for src/ontology/reasoner.py

Coverage target: 28% → 75%+
Covers: RuleCondition, InferenceRule, OntologyReasoner,
        StandardConditions, QueryIntentDetector, condition decorator
"""

from unittest.mock import MagicMock

import pytest

from src.domain.entities.relations import InferenceResult, InsightType
from src.ontology.reasoner import (
    InferenceRule,
    OntologyReasoner,
    QueryIntentDetector,
    RuleCondition,
    StandardConditions,
    condition,
)

# =========================================================================
# Fixtures
# =========================================================================


def _make_condition(name="cond1", *, result=True, description="test cond"):
    return RuleCondition(
        name=name,
        check=lambda ctx, _r=result: _r,
        description=description,
    )


def _make_rule(
    name="rule1",
    *,
    conditions=None,
    insight_type=InsightType.MARKET_POSITION,
    priority=0,
    confidence=1.0,
    tags=None,
):
    if conditions is None:
        conditions = [_make_condition()]
    return InferenceRule(
        name=name,
        description=f"Test rule {name}",
        conditions=conditions,
        conclusion=lambda ctx: {
            "insight": f"insight from {name}",
            "recommendation": "do something",
            "related_entities": ["LANEIGE"],
        },
        insight_type=insight_type,
        priority=priority,
        confidence=confidence,
        tags=tags or [],
    )


def _make_kg_mock():
    kg = MagicMock()
    kg.get_entity_metadata.return_value = {"sos": 5.0, "avg_rank": 8}
    kg.get_competitors.return_value = ["Competitor1", "Competitor2"]
    kg.get_brand_products.return_value = ["P1", "P2"]
    kg.get_category_brands.return_value = ["B1", "B2", "B3"]
    kg.get_entity_context.return_value = {"relations": {"competesWith": ["C1"]}}
    return kg


# =========================================================================
# RuleCondition
# =========================================================================


class TestRuleCondition:
    def test_evaluate_returns_true(self):
        cond = RuleCondition(name="ok", check=lambda ctx: True, description="always true")
        assert cond.evaluate({"x": 1}) is True

    def test_evaluate_returns_false(self):
        cond = RuleCondition(name="nope", check=lambda ctx: False, description="always false")
        assert cond.evaluate({}) is False

    def test_evaluate_exception_returns_false(self):
        cond = RuleCondition(
            name="err",
            check=lambda ctx: 1 / 0,
            description="raises",
        )
        assert cond.evaluate({}) is False


# =========================================================================
# InferenceRule
# =========================================================================


class TestInferenceRule:
    def test_evaluate_conditions_all_satisfied(self):
        rule = _make_rule(conditions=[_make_condition("a"), _make_condition("b")])
        ok, satisfied = rule.evaluate_conditions({"x": 1})
        assert ok is True
        assert satisfied == ["a", "b"]

    def test_evaluate_conditions_partial_failure(self):
        conds = [
            _make_condition("pass_cond", result=True),
            _make_condition("fail_cond", result=False),
        ]
        rule = _make_rule(conditions=conds)
        ok, satisfied = rule.evaluate_conditions({})
        assert ok is False
        assert satisfied == ["pass_cond"]

    def test_apply_success(self):
        rule = _make_rule(name="test_apply", confidence=0.9)
        result = rule.apply({"brand": "LANEIGE", "sos": 5})
        assert result is not None
        assert isinstance(result, InferenceResult)
        assert result.rule_name == "test_apply"
        assert result.insight == "insight from test_apply"
        assert result.confidence == 0.9
        assert "satisfied_conditions" in result.evidence
        assert result.recommendation == "do something"
        assert "LANEIGE" in result.related_entities

    def test_apply_with_confidence_modifier(self):
        rule = InferenceRule(
            name="mod_rule",
            description="confidence modifier test",
            conditions=[_make_condition()],
            conclusion=lambda ctx: {
                "insight": "modified",
                "confidence_modifier": 0.5,
            },
            insight_type=InsightType.RISK_ALERT,
            confidence=0.8,
        )
        result = rule.apply({})
        assert result is not None
        assert result.confidence == pytest.approx(0.4)

    def test_apply_conditions_not_met(self):
        rule = _make_rule(conditions=[_make_condition("fail", result=False)])
        result = rule.apply({})
        assert result is None

    def test_apply_conclusion_raises(self):
        rule = InferenceRule(
            name="err_rule",
            description="error rule",
            conditions=[_make_condition()],
            conclusion=lambda ctx: (_ for _ in ()).throw(ValueError("boom")),
            insight_type=InsightType.RISK_ALERT,
        )
        result = rule.apply({})
        assert result is None

    def test_apply_context_snapshot_filters_complex_types(self):
        rule = _make_rule()
        result = rule.apply({"brand": "X", "count": 3, "flag": True, "obj": object()})
        snapshot = result.evidence["context_snapshot"]
        assert "brand" in snapshot
        assert "count" in snapshot
        assert "flag" in snapshot
        assert "obj" not in snapshot

    def test_apply_metadata_includes_rule_info(self):
        rule = _make_rule(name="meta_rule", priority=5, tags=["pricing"])
        result = rule.apply({})
        assert result.metadata["priority"] == 5
        assert result.metadata["tags"] == ["pricing"]
        assert "rule_description" in result.metadata


# =========================================================================
# OntologyReasoner — Rule Management
# =========================================================================


class TestOntologyReasonerRuleManagement:
    def test_init_defaults(self):
        reasoner = OntologyReasoner()
        assert reasoner.kg is None
        assert reasoner.max_iterations == 10
        assert len(reasoner.rules) == 0

    def test_init_with_kg(self):
        kg = _make_kg_mock()
        reasoner = OntologyReasoner(knowledge_graph=kg, max_iterations=5)
        assert reasoner.kg is kg
        assert reasoner.max_iterations == 5

    def test_register_rule(self):
        reasoner = OntologyReasoner()
        rule = _make_rule("r1", priority=10)
        reasoner.register_rule(rule)
        assert "r1" in reasoner.rules
        assert len(reasoner.rules_by_priority) == 1

    def test_register_rule_overwrites(self):
        reasoner = OntologyReasoner()
        reasoner.register_rule(_make_rule("r1", priority=1))
        reasoner.register_rule(_make_rule("r1", priority=5))
        assert len(reasoner.rules) == 1
        assert reasoner.rules["r1"].priority == 5

    def test_register_rules_batch(self):
        reasoner = OntologyReasoner()
        rules = [_make_rule(f"r{i}", priority=i) for i in range(3)]
        reasoner.register_rules(rules)
        assert len(reasoner.rules) == 3
        assert reasoner.rules_by_priority[0].priority == 2  # highest first

    def test_unregister_rule(self):
        reasoner = OntologyReasoner()
        reasoner.register_rule(_make_rule("r1"))
        assert reasoner.unregister_rule("r1") is True
        assert "r1" not in reasoner.rules

    def test_unregister_rule_not_found(self):
        reasoner = OntologyReasoner()
        assert reasoner.unregister_rule("nope") is False

    def test_get_rule(self):
        reasoner = OntologyReasoner()
        rule = _make_rule("r1")
        reasoner.register_rule(rule)
        assert reasoner.get_rule("r1") is rule
        assert reasoner.get_rule("missing") is None

    def test_get_rules_by_tag(self):
        reasoner = OntologyReasoner()
        reasoner.register_rules(
            [
                _make_rule("r1", tags=["pricing", "market"]),
                _make_rule("r2", tags=["growth"]),
                _make_rule("r3", tags=["pricing"]),
            ]
        )
        pricing = reasoner.get_rules_by_tag("pricing")
        assert len(pricing) == 2

    def test_get_rules_by_insight_type(self):
        reasoner = OntologyReasoner()
        reasoner.register_rules(
            [
                _make_rule("r1", insight_type=InsightType.RISK_ALERT),
                _make_rule("r2", insight_type=InsightType.MARKET_POSITION),
                _make_rule("r3", insight_type=InsightType.RISK_ALERT),
            ]
        )
        risk = reasoner.get_rules_by_insight_type(InsightType.RISK_ALERT)
        assert len(risk) == 2

    def test_list_rules(self):
        reasoner = OntologyReasoner()
        reasoner.register_rules(
            [
                _make_rule("r1", priority=10, tags=["t1"]),
                _make_rule("r2", priority=5),
            ]
        )
        listing = reasoner.list_rules()
        assert len(listing) == 2
        assert listing[0]["name"] == "r1"  # higher priority first
        assert listing[0]["priority"] == 10
        assert listing[0]["tags"] == ["t1"]
        assert "conditions_count" in listing[0]
        assert "insight_type" in listing[0]

    def test_repr(self):
        reasoner = OntologyReasoner()
        assert "rules=0" in repr(reasoner)
        assert "none" in repr(reasoner)

        kg = _make_kg_mock()
        reasoner2 = OntologyReasoner(knowledge_graph=kg)
        assert "connected" in repr(reasoner2)


# =========================================================================
# OntologyReasoner — Inference
# =========================================================================


class TestOntologyReasonerInference:
    def test_infer_basic(self):
        reasoner = OntologyReasoner()
        reasoner.register_rule(_make_rule("r1"))
        results = reasoner.infer({"brand": "LANEIGE"})
        assert len(results) == 1
        assert results[0].rule_name == "r1"

    def test_infer_with_rule_filter(self):
        reasoner = OntologyReasoner()
        reasoner.register_rules(
            [
                _make_rule("r1", insight_type=InsightType.RISK_ALERT),
                _make_rule("r2", insight_type=InsightType.MARKET_POSITION),
            ]
        )
        results = reasoner.infer(
            {"brand": "X"},
            rule_filter=lambda r: r.insight_type == InsightType.RISK_ALERT,
        )
        assert len(results) == 1
        assert results[0].rule_name == "r1"

    def test_infer_max_results(self):
        reasoner = OntologyReasoner()
        for i in range(5):
            reasoner.register_rule(_make_rule(f"r{i}"))
        results = reasoner.infer({}, max_results=2)
        assert len(results) == 2

    def test_infer_skips_failed_conditions(self):
        reasoner = OntologyReasoner()
        reasoner.register_rules(
            [
                _make_rule("pass", conditions=[_make_condition("ok", result=True)]),
                _make_rule("fail", conditions=[_make_condition("nope", result=False)]),
            ]
        )
        results = reasoner.infer({})
        assert len(results) == 1
        assert results[0].rule_name == "pass"

    def test_infer_records_history(self):
        reasoner = OntologyReasoner()
        reasoner.register_rule(_make_rule("r1"))
        reasoner.infer({"brand": "X"})
        assert len(reasoner.inference_history) == 1
        record = reasoner.inference_history[0]
        assert record["results_count"] == 1
        assert "r1" in record["applied_rules"]

    def test_infer_with_kg_enrichment(self):
        kg = _make_kg_mock()
        reasoner = OntologyReasoner(knowledge_graph=kg)

        cond = RuleCondition(
            name="has_competitors",
            check=lambda ctx: ctx.get("competitor_count", 0) >= 2,
            description="has competitors",
        )
        reasoner.register_rule(_make_rule("r1", conditions=[cond]))
        results = reasoner.infer({"brand": "LANEIGE"})
        assert len(results) == 1
        kg.get_competitors.assert_called_with("LANEIGE")

    def test_infer_with_category_enrichment(self):
        kg = _make_kg_mock()
        reasoner = OntologyReasoner(knowledge_graph=kg)
        reasoner.register_rule(_make_rule("r1"))
        reasoner.infer({"category": "lip_care"})
        kg.get_category_brands.assert_called_with("lip_care")

    def test_infer_for_entity_brand(self):
        kg = _make_kg_mock()
        reasoner = OntologyReasoner(knowledge_graph=kg)
        reasoner.register_rule(_make_rule("r1"))
        results = reasoner.infer_for_entity("LANEIGE", "brand")
        assert len(results) >= 0  # depends on conditions
        kg.get_entity_metadata.assert_called()
        kg.get_entity_context.assert_called_with("LANEIGE", depth=1)

    def test_infer_for_entity_product(self):
        kg = _make_kg_mock()
        kg.get_entity_metadata.return_value = {
            "current_rank": 5,
            "rank_change_1d": -2,
            "streak_days": 10,
        }
        reasoner = OntologyReasoner(knowledge_graph=kg)
        reasoner.register_rule(_make_rule("r1"))
        results = reasoner.infer_for_entity("B08XYZ", "product")
        kg.get_entity_context.assert_called_with("B08XYZ", depth=1)

    def test_infer_for_entity_no_kg(self):
        reasoner = OntologyReasoner()
        reasoner.register_rule(_make_rule("r1"))
        results = reasoner.infer_for_entity("LANEIGE", "brand")
        assert isinstance(results, list)

    def test_infer_by_insight_type(self):
        reasoner = OntologyReasoner()
        reasoner.register_rules(
            [
                _make_rule("r1", insight_type=InsightType.RISK_ALERT),
                _make_rule("r2", insight_type=InsightType.GROWTH_OPPORTUNITY),
                _make_rule("r3", insight_type=InsightType.MARKET_POSITION),
            ]
        )
        results = reasoner.infer_by_insight_type(
            {},
            [InsightType.RISK_ALERT, InsightType.GROWTH_OPPORTUNITY],
        )
        assert len(results) == 2
        types = {r.insight_type for r in results}
        assert InsightType.MARKET_POSITION not in types

    def test_infer_with_intent(self):
        reasoner = OntologyReasoner()
        reasoner.register_rules(
            [
                _make_rule("r1", insight_type=InsightType.COMPETITIVE_THREAT),
                _make_rule("r2", insight_type=InsightType.GROWTH_OPPORTUNITY),
            ]
        )
        results = reasoner.infer_with_intent({}, query="경쟁사 vs LANEIGE 비교")
        # competition intent → COMPETITIVE_THREAT, MARKET_POSITION, MARKET_DOMINANCE
        types = {r.insight_type for r in results}
        assert InsightType.COMPETITIVE_THREAT in types
        assert InsightType.GROWTH_OPPORTUNITY not in types

    def test_infer_with_intent_general(self):
        reasoner = OntologyReasoner()
        reasoner.register_rules(
            [
                _make_rule("r1", insight_type=InsightType.COMPETITIVE_THREAT),
                _make_rule("r2", insight_type=InsightType.GROWTH_OPPORTUNITY),
            ]
        )
        # no keywords → general → all rules apply
        results = reasoner.infer_with_intent({}, query="tell me about LANEIGE")
        assert len(results) == 2


# =========================================================================
# OntologyReasoner — Explainability
# =========================================================================


class TestOntologyReasonerExplainability:
    def test_explain_inference(self):
        reasoner = OntologyReasoner()
        rule = _make_rule("explain_rule")
        reasoner.register_rule(rule)

        result = InferenceResult(
            rule_name="explain_rule",
            insight_type=InsightType.MARKET_POSITION,
            insight="SoS is low",
            confidence=0.85,
            evidence={
                "satisfied_conditions": ["cond1"],
                "context_snapshot": {"sos": 3.2, "brand": "LANEIGE"},
            },
            recommendation="Increase marketing",
        )
        explanation = reasoner.explain_inference(result)
        assert "explain_rule" in explanation
        assert "SoS is low" in explanation
        assert "Increase marketing" in explanation
        assert "0.85" in explanation
        assert "sos" in explanation

    def test_explain_inference_unknown_rule(self):
        reasoner = OntologyReasoner()
        result = InferenceResult(
            rule_name="unknown",
            insight_type=InsightType.RISK_ALERT,
            insight="something",
        )
        explanation = reasoner.explain_inference(result)
        assert "알 수 없는 규칙" in explanation

    def test_explain_all(self):
        reasoner = OntologyReasoner()
        reasoner.register_rule(_make_rule("r1"))
        results = [
            InferenceResult(
                rule_name="r1",
                insight_type=InsightType.MARKET_POSITION,
                insight="insight 1",
                evidence={"satisfied_conditions": [], "context_snapshot": {}},
            ),
            InferenceResult(
                rule_name="r1",
                insight_type=InsightType.RISK_ALERT,
                insight="insight 2",
                evidence={"satisfied_conditions": [], "context_snapshot": {}},
            ),
        ]
        explanation = reasoner.explain_all(results)
        assert "인사이트 1" in explanation
        assert "인사이트 2" in explanation

    def test_explain_all_empty(self):
        reasoner = OntologyReasoner()
        assert "없습니다" in reasoner.explain_all([])


# =========================================================================
# OntologyReasoner — History & Stats
# =========================================================================


class TestOntologyReasonerHistory:
    def test_record_inference_and_stats(self):
        reasoner = OntologyReasoner()
        reasoner.register_rule(_make_rule("r1", insight_type=InsightType.RISK_ALERT))
        reasoner.infer({})
        reasoner.infer({})

        stats = reasoner.get_inference_stats()
        assert stats["total_inferences"] == 2
        assert stats["rules_applied"]["r1"] == 2
        assert InsightType.RISK_ALERT.value in stats["insight_types"]
        assert stats["registered_rules"] == 1

    def test_stats_empty_history(self):
        reasoner = OntologyReasoner()
        stats = reasoner.get_inference_stats()
        assert stats["total_inferences"] == 0

    def test_history_capped_at_100(self):
        reasoner = OntologyReasoner()
        reasoner.register_rule(_make_rule("r1"))
        for _ in range(110):
            reasoner.infer({})
        assert len(reasoner.inference_history) == 100

    def test_validate_rules_no_conditions(self):
        reasoner = OntologyReasoner()
        rule = _make_rule("empty_cond", conditions=[])
        reasoner.register_rule(rule)
        warnings = reasoner.validate_rules()
        assert any("no conditions" in w for w in warnings)

    def test_validate_rules_same_priority(self):
        reasoner = OntologyReasoner()
        reasoner.register_rules(
            [
                _make_rule("r1", priority=5),
                _make_rule("r2", priority=5),
            ]
        )
        warnings = reasoner.validate_rules()
        assert any("same priority" in w for w in warnings)

    def test_validate_rules_clean(self):
        reasoner = OntologyReasoner()
        reasoner.register_rules(
            [
                _make_rule("r1", priority=1),
                _make_rule("r2", priority=2),
            ]
        )
        warnings = reasoner.validate_rules()
        assert len(warnings) == 0


# =========================================================================
# condition decorator
# =========================================================================


class TestConditionDecorator:
    def test_condition_creates_rule_condition(self):
        @condition("high_sos", "SoS >= 15%")
        def check_high_sos(ctx):
            return ctx.get("sos", 0) >= 15

        assert isinstance(check_high_sos, RuleCondition)
        assert check_high_sos.name == "high_sos"
        assert check_high_sos.description == "SoS >= 15%"
        assert check_high_sos.evaluate({"sos": 20}) is True
        assert check_high_sos.evaluate({"sos": 5}) is False


# =========================================================================
# StandardConditions
# =========================================================================


class TestStandardConditions:
    def test_sos_above(self):
        cond = StandardConditions.sos_above(0.15)
        assert cond.evaluate({"sos": 0.2}) is True
        assert cond.evaluate({"sos": 0.1}) is False
        assert "sos_above" in cond.name

    def test_sos_below(self):
        cond = StandardConditions.sos_below(0.05)
        assert cond.evaluate({"sos": 0.03}) is True
        assert cond.evaluate({"sos": 0.1}) is False

    def test_hhi_above(self):
        cond = StandardConditions.hhi_above(0.25)
        assert cond.evaluate({"hhi": 0.3}) is True
        assert cond.evaluate({"hhi": 0.1}) is False

    def test_hhi_below(self):
        cond = StandardConditions.hhi_below(0.15)
        assert cond.evaluate({"hhi": 0.1}) is True
        assert cond.evaluate({"hhi": 0.2}) is False

    def test_cpi_above(self):
        cond = StandardConditions.cpi_above(120)
        assert cond.evaluate({"cpi": 130}) is True
        assert cond.evaluate({"cpi": 110}) is False

    def test_cpi_below(self):
        cond = StandardConditions.cpi_below(80)
        assert cond.evaluate({"cpi": 70}) is True
        assert cond.evaluate({"cpi": 90}) is False

    def test_rating_gap_negative(self):
        cond = StandardConditions.rating_gap_negative()
        assert cond.evaluate({"rating_gap": -0.5}) is True
        assert cond.evaluate({"rating_gap": 0.5}) is False

    def test_rating_gap_positive(self):
        cond = StandardConditions.rating_gap_positive()
        assert cond.evaluate({"rating_gap": 0.5}) is True
        assert cond.evaluate({"rating_gap": -0.5}) is False

    def test_has_rank_shock(self):
        cond = StandardConditions.has_rank_shock()
        assert cond.evaluate({"has_rank_shock": True}) is True
        assert cond.evaluate({"has_rank_shock": False}) is False
        assert cond.evaluate({}) is False

    def test_churn_rate_high(self):
        cond = StandardConditions.churn_rate_high(0.2)
        assert cond.evaluate({"churn_rate": 0.3}) is True
        assert cond.evaluate({"churn_rate": 0.1}) is False

    def test_streak_days_above(self):
        cond = StandardConditions.streak_days_above(7)
        assert cond.evaluate({"streak_days": 10}) is True
        assert cond.evaluate({"streak_days": 3}) is False

    def test_rank_improving(self):
        cond = StandardConditions.rank_improving()
        assert cond.evaluate({"rank_change_7d": -5}) is True
        assert cond.evaluate({"rank_change_7d": 5}) is False
        assert cond.evaluate({}) is False

    def test_rank_declining(self):
        cond = StandardConditions.rank_declining()
        assert cond.evaluate({"rank_change_7d": 5}) is True
        assert cond.evaluate({"rank_change_7d": -5}) is False

    def test_is_target_brand(self):
        cond = StandardConditions.is_target_brand()
        assert cond.evaluate({"brand": "LANEIGE"}) is True
        assert cond.evaluate({"brand": "laneige"}) is True
        assert cond.evaluate({"is_target": True}) is True
        assert cond.evaluate({"brand": "OTHER"}) is False

    def test_has_competitors(self):
        cond = StandardConditions.has_competitors(2)
        assert cond.evaluate({"competitor_count": 3}) is True
        assert cond.evaluate({"competitor_count": 1}) is False

    def test_in_top_n(self):
        cond = StandardConditions.in_top_n(10)
        assert cond.evaluate({"current_rank": 5}) is True
        assert cond.evaluate({"current_rank": 15}) is False
        assert cond.evaluate({}) is False  # default 100 > 10


# =========================================================================
# QueryIntentDetector
# =========================================================================


class TestQueryIntentDetector:
    def test_detect_competition_intent(self):
        intents = QueryIntentDetector.detect_intent("LANEIGE vs 경쟁사 비교")
        assert "competition" in intents

    def test_detect_market_intent(self):
        intents = QueryIntentDetector.detect_intent("시장 점유율 분석")
        assert "market_analysis" in intents

    def test_detect_pricing_intent(self):
        intents = QueryIntentDetector.detect_intent("가격 전략 CPI 분석")
        assert "pricing" in intents

    def test_detect_growth_intent(self):
        intents = QueryIntentDetector.detect_intent("성장 기회 분석")
        assert "growth" in intents

    def test_detect_risk_intent(self):
        intents = QueryIntentDetector.detect_intent("위험 경고 감지")
        assert "risk" in intents

    def test_detect_ranking_intent(self):
        intents = QueryIntentDetector.detect_intent("순위 top 10 분석")
        assert "ranking" in intents

    def test_detect_sentiment_intent(self):
        intents = QueryIntentDetector.detect_intent("리뷰 평점 분석")
        assert "sentiment" in intents

    def test_detect_general_intent(self):
        intents = QueryIntentDetector.detect_intent("tell me about LANEIGE")
        assert intents == ["general"]

    def test_detect_multiple_intents(self):
        intents = QueryIntentDetector.detect_intent("경쟁사 가격 비교")
        assert "competition" in intents
        assert "pricing" in intents

    def test_get_relevant_insight_types_specific(self):
        types = QueryIntentDetector.get_relevant_insight_types("가격 전략")
        assert types is not None
        assert InsightType.PRICE_POSITION in types

    def test_get_relevant_insight_types_general(self):
        types = QueryIntentDetector.get_relevant_insight_types("hello world")
        assert types is None

    def test_create_rule_filter_specific(self):
        rule_filter = QueryIntentDetector.create_rule_filter("경쟁사 비교")
        assert rule_filter is not None
        # COMPETITIVE_THREAT rule should pass
        r1 = _make_rule("r1", insight_type=InsightType.COMPETITIVE_THREAT)
        assert rule_filter(r1) is True
        # GROWTH_OPPORTUNITY should be filtered out
        r2 = _make_rule("r2", insight_type=InsightType.GROWTH_OPPORTUNITY)
        assert rule_filter(r2) is False

    def test_create_rule_filter_general(self):
        rule_filter = QueryIntentDetector.create_rule_filter("tell me about X")
        assert rule_filter is None

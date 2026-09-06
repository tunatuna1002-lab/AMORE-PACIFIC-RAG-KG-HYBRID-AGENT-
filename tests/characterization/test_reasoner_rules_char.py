"""Characterization tests: OntologyReasoner + business rules.

Pins the exact frozenset of fired rule names per context with all rules registered.
"""

import pytest

from src.ontology.reasoner import OntologyReasoner
from src.ontology.rules import ALL_BUSINESS_RULES, register_all_rules
from src.shared.units import percent_to_fraction


@pytest.fixture
def reasoner() -> OntologyReasoner:
    r = OntologyReasoner()  # no KG -> no context enrichment
    register_all_rules(r)
    return r


def fired(reasoner: OntologyReasoner, ctx: dict) -> frozenset[str]:
    return frozenset(res.rule_name for res in reasoner.infer(ctx))


def test_register_all_rules_count(reasoner):
    assert register_all_rules(OntologyReasoner()) == 37
    assert len(ALL_BUSINESS_RULES) == 37
    assert len(reasoner.rules) == 37


def test_empty_context_fires_nothing(reasoner):
    assert fired(reasoner, {}) == frozenset()


# --- market rules --------------------------------------------------------------


def test_market_dominance_fragmented(reasoner):
    ctx = {"brand": "LANEIGE", "is_target": True, "sos": 0.18, "hhi": 0.10, "competitor_count": 6}
    assert fired(reasoner, ctx) == frozenset(
        {"market_dominance_fragmented", "fragmented_market_competition"}
    )


def test_challenger_position(reasoner):
    assert fired(reasoner, {"brand": "LANEIGE", "sos": 0.08, "hhi": 0.30}) == frozenset(
        {"challenger_position"}
    )


def test_category_entry_opportunity(reasoner):
    ctx = {"brand": "LANEIGE", "is_target": True, "sos": 0.02, "hhi": 0.10}
    assert fired(reasoner, ctx) == frozenset({"category_entry_opportunity"})


def test_reasoner_contract_is_fraction_sos(reasoner):
    # FIXED (bug D2): the reasoner's contract is a 0-1 SoS fraction. MetricCalculator still
    # emits percent; callers (hybrid_retriever / kg_updater / dashboard_exporter) convert at
    # the boundary through src.shared.units, so a 2% share reaches the rules as 0.02 and is
    # an entry opportunity, never "200%" dominance.
    ctx = {"brand": "LANEIGE", "is_target": True, "sos": percent_to_fraction(2.0), "hhi": 0.10}
    assert fired(reasoner, ctx) == frozenset({"category_entry_opportunity"})


# --- price rules ---------------------------------------------------------------


@pytest.mark.parametrize(
    "ctx, expected",
    [
        ({"cpi": 160, "rating_gap": 0.0}, {"premium_price_position"}),
        ({"cpi": 115, "rating_gap": -0.1}, {"price_quality_mismatch"}),
        ({"cpi": 85, "rating_gap": 0.2}, {"value_position"}),
    ],
)
def test_price_rules(reasoner, ctx, expected):
    assert fired(reasoner, ctx) == frozenset(expected)


# --- growth / rank rules -------------------------------------------------------


@pytest.mark.parametrize(
    "ctx, expected",
    [
        ({"rank_change_7d": 4, "rank_volatility": 6.0}, {"rank_decline_alert"}),
        ({"rank_change_7d": -3, "streak_days": 31}, {"stable_growth"}),
        ({"current_rank": 2, "brand": "LANEIGE"}, {"top3_achievement"}),
    ],
)
def test_rank_rules(reasoner, ctx, expected):
    assert fired(reasoner, ctx) == frozenset(expected)


# --- alert rules ---------------------------------------------------------------


def test_rank_shock_with_none_churn_does_not_raise(reasoner):
    # FIXED (bug D17): StandardConditions are None-safe (None reads as the default, 0), so
    # churn_rate=None is simply "not high" - market_disruption does not fire and no
    # warning-swallowed exception is involved.
    assert fired(reasoner, {"has_rank_shock": True, "churn_rate": None}) == frozenset()


# --- sentiment rules -----------------------------------------------------------


def test_sentiment_clusters_list_form_fires_hydration(reasoner):
    ctx = {"sentiment_clusters": {"Hydration": ["Moisturizing", "Hydrating"]}}
    assert fired(reasoner, ctx) == frozenset({"sentiment_strength_hydration"})


def test_sentiment_clusters_int_form_fires_hydration(reasoner):
    # FIXED (bug D17): cluster_size() accepts the KG brand-profile count form as well as the
    # product tag-list form, so {"Hydration": 2} fires exactly like the list form.
    assert fired(reasoner, {"sentiment_clusters": {"Hydration": 2}}) == frozenset(
        {"sentiment_strength_hydration"}
    )
    assert fired(reasoner, {"sentiment_clusters": {"Hydration": 1}}) == frozenset()


def test_inference_result_shape(reasoner):
    results = reasoner.infer({"cpi": 160, "rating_gap": 0.0})
    assert len(results) == 1
    d = results[0].to_dict()
    assert set(d.keys()) == {
        "rule_name",
        "insight_type",
        "insight",
        "confidence",
        "evidence",
        "recommendation",
        "related_entities",
        "metadata",
    }
    assert d["rule_name"] == "premium_price_position"

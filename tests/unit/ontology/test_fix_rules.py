"""Bug D17: rule inputs - None-safe conditions and sentiment cluster shapes."""

import logging

import pytest

from src.ontology.reasoner import OntologyReasoner, RuleCondition, StandardConditions, cluster_size
from src.ontology.rules import register_all_rules


@pytest.fixture
def reasoner() -> OntologyReasoner:
    r = OntologyReasoner()
    register_all_rules(r)
    return r


def fired(reasoner: OntologyReasoner, ctx: dict) -> frozenset[str]:
    return frozenset(res.rule_name for res in reasoner.infer(ctx))


# --- D17(a): None-safe conditions ------------------------------------------------


def test_churn_rate_none_is_false_without_warning(caplog):
    cond = StandardConditions.churn_rate_high(0.2)
    with caplog.at_level(logging.WARNING, logger="src.ontology.reasoner"):
        assert cond.evaluate({"churn_rate": None}) is False
    assert not [r for r in caplog.records if "evaluation failed" in r.getMessage()]


@pytest.mark.parametrize(
    "cond",
    [
        StandardConditions.sos_above(0.15),
        StandardConditions.sos_below(0.03),
        StandardConditions.hhi_above(0.25),
        StandardConditions.hhi_below(0.15),
        StandardConditions.cpi_above(150),
        StandardConditions.cpi_below(90),
        StandardConditions.rating_gap_negative(),
        StandardConditions.rating_gap_positive(),
        StandardConditions.streak_days_above(7),
    ],
)
def test_standard_conditions_do_not_raise_on_none(cond, caplog):
    ctx = {k: None for k in ("sos", "hhi", "cpi", "rating_gap", "streak_days")}
    with caplog.at_level(logging.WARNING, logger="src.ontology.reasoner"):
        cond.evaluate(ctx)
    assert not [r for r in caplog.records if "evaluation failed" in r.getMessage()]


def test_rank_shock_with_none_churn_does_not_fire_or_warn(reasoner, caplog):
    with caplog.at_level(logging.WARNING, logger="src.ontology.reasoner"):
        assert fired(reasoner, {"has_rank_shock": True, "churn_rate": None}) == frozenset()
    assert not [r for r in caplog.records if "evaluation failed" in r.getMessage()]


def test_failing_condition_warns_once(caplog):
    cond = RuleCondition(name="boom", check=lambda ctx: ctx["missing"] > 1, description="x")
    with caplog.at_level(logging.WARNING, logger="src.ontology.reasoner"):
        assert cond.evaluate({}) is False
        assert cond.evaluate({}) is False
        assert cond.evaluate({}) is False
    warnings = [r for r in caplog.records if "boom" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING


# --- D17(b): sentiment cluster shapes -------------------------------------------


def test_cluster_size_accepts_list_or_int():
    assert cluster_size(["a", "b"]) == 2
    assert cluster_size(2) == 2
    assert cluster_size(None) == 0
    assert cluster_size([]) == 0


def test_sentiment_clusters_int_form_fires_hydration(reasoner):
    assert fired(reasoner, {"sentiment_clusters": {"Hydration": 2}}) == frozenset(
        {"sentiment_strength_hydration"}
    )


def test_sentiment_clusters_list_form_still_fires_hydration(reasoner):
    ctx = {"sentiment_clusters": {"Hydration": ["Moisturizing", "Hydrating"]}}
    assert fired(reasoner, ctx) == frozenset({"sentiment_strength_hydration"})


def test_sentiment_clusters_int_below_threshold_does_not_fire(reasoner):
    assert fired(reasoner, {"sentiment_clusters": {"Hydration": 1}}) == frozenset()


def test_sentiment_effectiveness_int_form_fires(reasoner):
    assert fired(reasoner, {"sentiment_clusters": {"Effectiveness": 1}}) == frozenset(
        {"sentiment_effectiveness_strong"}
    )

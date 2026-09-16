"""F4-2: single inference-context builder (percent→fraction, no fabricated defaults)."""

from __future__ import annotations

import pytest

from src.ontology.inference_context import (
    build_inference_context,
    normalize_sentiment_clusters,
)


def test_sos_from_brand_metrics_is_fraction() -> None:
    metrics = {
        "brand_metrics": [
            {"brand_name": "LANEIGE", "share_of_shelf": 8.0, "avg_rank": 15.5, "product_count": 3}
        ]
    }
    ctx = build_inference_context(metrics, "laneige")
    assert ctx["brand"] == "laneige"
    assert ctx["is_target"] is True
    assert ctx["sos"] == pytest.approx(0.08)
    assert ctx["avg_rank"] == 15.5
    assert ctx["product_count"] == 3


def test_sos_from_summary_by_category() -> None:
    metrics = {"summary": {"laneige_sos_by_category": {"lip_care": 8.0, "face_powder": 5.0}}}
    ctx = build_inference_context(metrics, "LANEIGE", category="face_powder")
    assert ctx["sos"] == pytest.approx(0.05)
    assert ctx["category"] == "face_powder"
    ctx = build_inference_context(metrics, "LANEIGE")
    assert ctx["sos"] == pytest.approx(0.08)


def test_market_metrics_no_fabricated_defaults() -> None:
    metrics = {"market_metrics": [{"category_id": "lip_care", "hhi": 0.15, "cpi": 105}]}
    ctx = build_inference_context(metrics, "LANEIGE")
    assert ctx["hhi"] == 0.15
    assert ctx["cpi"] == 105
    # churn_rate_7d / avg_rating_gap absent -> keys absent (not 0)
    assert "churn_rate" not in ctx
    assert "rating_gap" not in ctx


def test_empty_metrics_has_no_metric_keys() -> None:
    ctx = build_inference_context({}, None)
    assert "sos" not in ctx and "hhi" not in ctx and "cpi" not in ctx
    assert ctx["has_rank_shock"] is False
    assert ctx["alert_count"] == 0


def test_products_override_selects_best_rank() -> None:
    metrics = {"product_metrics": [{"asin": "B1", "current_rank": 30}]}
    products = [{"asin": "B2", "current_rank": 4, "rank_change_7d": -2}]
    ctx = build_inference_context(metrics, "LANEIGE", products=products)
    assert ctx["asin"] == "B2"
    assert ctx["current_rank"] == 4
    assert ctx["rank_change_7d"] == -2
    assert "streak_days" not in ctx


def test_alerts_rank_shock() -> None:
    ctx = build_inference_context({"alerts": [{"type": "rank_shock"}, {"type": "x"}]}, None)
    assert ctx["has_rank_shock"] is True
    assert ctx["alert_count"] == 2


def test_percent_scale_sos_never_guessed() -> None:
    metrics = {"brand_metrics": [{"brand_name": "LANEIGE", "share_of_shelf": 0.5}]}
    ctx = build_inference_context(metrics, "LANEIGE")
    assert ctx["sos"] == pytest.approx(0.005)


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, {}),
        ({"Hydration": 2}, {"Hydration": 2}),
        (["Hydration", "Hydration", "Sensory"], {"Hydration": 2, "Sensory": 1}),
        ("junk", {}),
    ],
)
def test_normalize_sentiment_clusters(raw, expected) -> None:
    assert normalize_sentiment_clusters(raw) == expected

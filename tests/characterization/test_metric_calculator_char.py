"""Characterization tests: MetricCalculator (src/tools/calculators/metric_calculator.py).

Pins CURRENT outputs on a fixed 100-record lip_care snapshot. Public API only.
"""

from datetime import date

import pytest

from src.domain.entities import BrandMetrics, MarketMetrics
from src.tools.calculators.metric_calculator import MetricCalculator

from ._fixtures import build_lip_care_snapshot


@pytest.fixture
def records() -> list[dict]:
    return build_lip_care_snapshot()


@pytest.fixture
def calc() -> MetricCalculator:
    # dict config -> no file access
    return MetricCalculator(config={})


def test_snapshot_shape(records):
    assert len(records) == 100
    assert sum(1 for r in records if r["brand"] == "LANEIGE") == 3
    assert sum(1 for r in records if r["brand"] == "COSRX") == 10
    assert len({r["brand"] for r in records}) == 10


def test_calculate_sos_returns_percent(calc, records):
    # PINS CURRENT BEHAVIOR (bug D2 unit mismatch): calculate_sos returns PERCENT (3.0),
    # while reasoner rules / HHI treat SoS as a 0-1 ratio (0.03). Expected to change when fixed.
    assert calc.calculate_sos(records, "LANEIGE") == 3.0
    assert calc.calculate_sos(records, "COSRX") == 10.0
    # case-insensitive brand match
    assert calc.calculate_sos(records, "laneige") == 3.0
    # unknown brand -> 0.0
    assert calc.calculate_sos(records, "NOPE") == 0.0


def test_calculate_sos_top_n_window(calc, records):
    # LANEIGE at ranks 1, 5, 20 -> 2 of top 10 -> 20.0 %
    assert calc.calculate_sos(records, "LANEIGE", top_n=10) == 20.0
    assert calc.calculate_sos([], "LANEIGE") == 0.0


def test_calculate_hhi_exact(calc, records):
    # (3^2 + 10^2 + 7*11^2 + 10^2) / 100^2 = (9 + 100 + 847 + 100) / 10000 = 0.1056
    assert calc.calculate_hhi(records) == pytest.approx(0.1056)
    assert calc.calculate_hhi([]) == 0.0


def test_calculate_hhi_excludes_unknown_brand(calc, records):
    # Unknown brands are dropped from the numerator but the denominator still counts them.
    recs = [dict(r) for r in records]
    for r in recs:
        if r["brand"] == "COSRX":
            r["brand"] = "Unknown"
    # (9 + 847 + 100) / 10000 = 0.0956
    assert calc.calculate_hhi(recs) == pytest.approx(0.0956)


def test_calculate_cpi_exact(calc, records):
    # category avg price = 26.0; LANEIGE prices: 25.0 (rank1), 24.0 (rank5), 24.0 (rank20)
    # -> avg 24.3333 -> 93.59
    assert calc.calculate_cpi(records, "LANEIGE") == pytest.approx(93.59)
    assert calc.calculate_cpi(records, "NOPE") is None
    assert calc.calculate_cpi([], "LANEIGE") is None


def test_calculate_brand_avg_rank_and_rating_gap(calc, records):
    assert calc.calculate_brand_avg_rank(records, "LANEIGE") == pytest.approx(8.67)
    assert calc.calculate_brand_avg_rank(records, "NOPE") is None
    # category avg rating 4.45; LANEIGE ratings 4.1, 4.5, 4.0 -> avg 4.2 -> gap -0.25
    assert calc.calculate_avg_rating_gap(records, "LANEIGE") == pytest.approx(-0.25)


def test_calculate_brand_metrics_composite(calc, records):
    bm = calc.calculate_brand_metrics(records, "LANEIGE", "lip_care")
    assert isinstance(bm, BrandMetrics)
    data = bm.model_dump()
    assert set(data.keys()) == {
        "brand",
        "category_id",
        "sos",
        "brand_avg_rank",
        "product_count",
        "cpi",
        "avg_rating_gap",
        "calculated_at",
    }
    assert data["brand"] == "LANEIGE"
    assert data["category_id"] == "lip_care"
    assert data["sos"] == 3.0  # PINS CURRENT BEHAVIOR (bug D2): percent scale
    assert data["brand_avg_rank"] == pytest.approx(8.67)
    assert data["product_count"] == 3
    assert data["cpi"] == pytest.approx(93.59)
    assert data["avg_rating_gap"] == pytest.approx(-0.25)


def test_calculate_market_metrics_composite(calc, records):
    mm = calc.calculate_market_metrics(records, records, "lip_care", date(2026, 9, 1))
    assert isinstance(mm, MarketMetrics)
    data = mm.model_dump()
    assert set(data.keys()) == {
        "category_id",
        "snapshot_date",
        "hhi",
        "churn_rate",
        "category_avg_price",
        "category_avg_rating",
        "calculated_at",
    }
    assert data["hhi"] == pytest.approx(0.1056)
    assert data["churn_rate"] == 0.0  # identical today/yesterday -> no churn
    assert data["category_avg_price"] == pytest.approx(26.0)
    assert data["category_avg_rating"] == pytest.approx(4.45)


def test_calculate_churn_rate(calc, records):
    yesterday = [dict(r) for r in records]
    # replace 10 ASINs -> 10 new entries + 10 exits over 2*100
    for r in yesterday[:10]:
        r["asin"] = r["asin"] + "X"
    assert calc.calculate_churn_rate(records, yesterday) == pytest.approx(0.1)
    assert calc.calculate_churn_rate(records, []) == 0.0


def test_level3_product_helpers(calc):
    assert calc.calculate_rank_volatility([1, 2, 3, 2, 1]) == pytest.approx(0.75)
    assert calc.calculate_rank_volatility([1]) is None
    # default threshold 5 when config is empty
    assert calc.calculate_rank_shock(10, 5) is True
    assert calc.calculate_rank_shock(10, 6) is False
    assert calc.calculate_rank_change(10, 5) == 5
    hist = [{"asin": "A", "rank": 3}, {"asin": "A", "rank": 9}, {"asin": "A", "rank": 12}]
    assert calc.calculate_streak_days(hist, "A") == 2
    assert calc.calculate_rating_trend([4.5, 4.4, 4.3]) == pytest.approx(0.1)
    assert calc.calculate_best_rank([5, 3, 9]) == 3
    assert calc.calculate_days_in_top_n([1, 4, 8, 15, 60]) == {
        3: 1,
        5: 2,
        10: 3,
        20: 4,
        50: 4,
        100: 5,
    }


def test_calculate_product_metrics_composite(calc):
    history = [
        {"asin": "A", "rank": 3, "rating": 4.5},
        {"asin": "A", "rank": 9, "rating": 4.4},
        {"asin": "B", "rank": 1, "rating": 4.9},
    ]
    pm = calc.calculate_product_metrics(history, "A", "lip_care").model_dump()
    assert pm["asin"] == "A"
    assert pm["rank_change"] == -6
    assert pm["rank_shock"] is True
    assert pm["rank_volatility"] == pytest.approx(3.0)
    assert pm["streak_days"] == 2
    assert pm["best_rank"] == 3
    assert pm["days_in_top_n"] == {3: 1, 5: 1, 10: 2, 20: 2, 50: 2, 100: 2}
    empty = calc.calculate_product_metrics(history, "ZZZ", "lip_care").model_dump()
    assert empty["rank_change"] is None
    assert empty["streak_days"] == 0

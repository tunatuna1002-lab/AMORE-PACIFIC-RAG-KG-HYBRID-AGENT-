"""Characterization tests: AlertAgent.process_metrics / send_pending_alerts and
MetricsAgent.execute (history lookup key)."""

import json

import pytest

from src.agents.alert_agent import AlertAgent, AlertPriority
from src.agents.metrics_agent import MetricsAgent
from src.core.state_manager import StateManager
from src.monitoring.logger import AgentLogger

from ._fixtures import build_lip_care_snapshot


@pytest.fixture
def alert_agent(tmp_path) -> AlertAgent:
    # StateManager persists to tmp_path; no email subscriptions -> no recipients.
    # EmailSender reads env (isolated by conftest) -> disabled; nothing is ever sent.
    sm = StateManager(persist_dir=tmp_path / "state")
    return AlertAgent(sm)


PRODUCTS = [
    {"name": "A", "brand": "LANEIGE", "rank_change": 10, "previous_rank": 5, "current_rank": 15},
    {"name": "B", "brand": "LANEIGE", "rank_change": -10, "previous_rank": 25, "current_rank": 15},
    {"name": "C", "brand": "LANEIGE", "current_rank": 9, "previous_rank": 12},
]


@pytest.mark.asyncio
async def test_process_metrics_alert_count_types_priorities(alert_agent):
    alerts = await alert_agent.process_metrics({"products": PRODUCTS})

    assert len(alerts) == 3
    assert [a.type for a in alerts] == ["rank_change", "rank_change", "important_insight"]
    assert [a.priority for a in alerts] == [
        AlertPriority.HIGH,  # drop of >= 10
        AlertPriority.NORMAL,  # rise of <= -10
        AlertPriority.HIGH,  # new top-10 entry
    ]
    assert [a.title for a in alerts] == ["A 순위 급락", "B 순위 급등", "C Top 10 진입!"]
    assert alerts[0].data == {
        "product_name": "A",
        "brand": "LANEIGE",
        "previous_rank": 5,
        "current_rank": 15,
        "change": 10,
    }
    assert alerts[2].data == {"product_name": "C", "brand": "LANEIGE", "current_rank": 9}
    assert all(a.sent is False for a in alerts)

    assert alert_agent.get_pending_count() == 3
    assert alert_agent.get_stats() == {
        "total_alerts": 3,
        "emails_sent": 0,
        "emails_failed": 0,
        "pending": 3,
        "email_enabled": False,
    }


@pytest.mark.asyncio
async def test_process_metrics_thresholds_are_inclusive(alert_agent):
    products = [
        {"name": "X", "rank_change": 9, "previous_rank": 20, "current_rank": 29},
        {"name": "Y", "rank_change": -9, "previous_rank": 29, "current_rank": 20},
        {"name": "Z", "current_rank": 10, "previous_rank": 11},
        {"name": "W", "current_rank": 10, "previous_rank": 10},
    ]
    alerts = await alert_agent.process_metrics({"products": products})
    assert [(a.type, a.title) for a in alerts] == [("important_insight", "Z Top 10 진입!")]


@pytest.mark.asyncio
async def test_send_pending_alerts_result_keys(alert_agent):
    await alert_agent.process_metrics({"products": PRODUCTS})
    result = await alert_agent.send_pending_alerts()

    # PINS CURRENT BEHAVIOR (bug D6): the result dict uses key "sent", not "sent_count",
    # which is what some callers read. Expected to change when fixed.
    assert set(result.keys()) == {"processed", "sent", "failed", "skipped", "details"}
    assert "sent" in result
    assert "sent_count" not in result

    # no consented recipients -> every alert is skipped, none processed/sent
    assert result == {"processed": 0, "sent": 0, "failed": 0, "skipped": 3, "details": []}
    assert alert_agent.get_pending_count() == 0
    # skipped alerts stay in history
    assert len(alert_agent.get_alerts()) == 3


@pytest.mark.asyncio
async def test_metrics_agent_execute_ignores_history_keyed_by_asin(tmp_path):
    # PINS CURRENT BEHAVIOR (bug D5): MetricsAgent.execute looks up history with
    # product.get("product_asin") (no fallback to "asin"), so a crawl snapshot that uses the
    # "asin" key never finds its history -> rank_change_1d None, streak 0, no rank_shock alert.
    # Expected to change when fixed.
    config_path = tmp_path / "thresholds.json"
    config_path.write_text(
        json.dumps(
            {
                "ranking": {
                    "top_n_tiers": [3, 5, 10, 20, 50, 100],
                    "significant_drop": 5,
                    "significant_rise": 10,
                },
                "thresholds": {"significant_rank_drop": 5},
            }
        ),
        encoding="utf-8",
    )
    agent = MetricsAgent(
        config_path=str(config_path),
        logger=AgentLogger("metrics", log_dir=str(tmp_path / "logs")),
    )

    records = build_lip_care_snapshot()
    assert "product_asin" not in records[0] and records[0]["asin"] == "B000000001"
    # yesterday rank 15 -> today rank 1: would be rank_change_1d=-14 and a rank shock if found
    history = {"B000000001": [{"asin": "B000000001", "rank": 15, "date": "2026-08-31"}]}

    result = await agent.execute(
        {"categories": {"lip_care": {"rank_records": records}}}, historical_data=history
    )

    assert result["status"] == "completed"
    assert set(result.keys()) == {
        "status",
        "calculated_at",
        "brand_metrics",
        "product_metrics",
        "market_metrics",
        "alerts",
        "summary",
    }

    pms = result["product_metrics"]
    assert [p["asin"] for p in pms] == ["B000000001", "B000000005", "B000000020"]
    first = pms[0]
    assert first["current_rank"] == 1
    assert first["rank_change_1d"] is None
    assert first["rank_change_7d"] is None
    assert first["rank_volatility"] == 0
    assert first["streak_days"] == 0
    assert result["alerts"] == []

    laneige = [b for b in result["brand_metrics"] if b["is_laneige"]]
    assert laneige == [
        {
            "brand_name": "LANEIGE",
            "category_id": "lip_care",
            "share_of_shelf": 3.0,  # PINS CURRENT BEHAVIOR (bug D2): percent
            "avg_rank": 8.67,
            "product_count": 3,
            "top10_count": 2,
            "top20_count": 3,
            "is_laneige": True,
        }
    ]
    assert result["market_metrics"] == [
        {
            "category_id": "lip_care",
            "hhi": 0.1056,
            "cpi": 100.35,
            "churn_rate_7d": None,
            "avg_rating_gap": 0.9,
            "top_brand": "Burt's Bees",
            "top_brand_sos": 0.11,  # note: ratio here, percent in brand_metrics (bug D2)
            "total_products": 100,
        }
    ]
    assert result["summary"] == {
        "laneige_products_tracked": 3,
        "laneige_sos_by_category": {"lip_care": 3.0},
        "best_ranking_product": {
            "asin": "B000000001",
            "title": "LANEIGE product 1",
            "rank": 1,
            "category": "lip_care",
        },
        "alert_count": 0,
        "critical_alerts": 0,
        "warning_alerts": 0,
    }
    assert agent.get_results() is result

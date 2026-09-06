"""Regression tests for MetricsAgent defects D5 and D11.

D5:  ``execute`` looked up history with ``product.get("product_asin")`` only, so crawl
     records that carry ``"asin"`` never found their history.
D11: ``_check_alerts`` read ``config["thresholds"]["significant_rank_drop"]`` which does
     not exist in config/thresholds.json (real key: ``config["ranking"]["significant_drop"]``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.metrics_agent import MetricsAgent
from src.monitoring.logger import AgentLogger


def _write_config(tmp_path: Path, ranking: dict, legacy: dict | None = None) -> Path:
    config = {"ranking": {"top_n_tiers": [3, 5, 10, 20, 50, 100], **ranking}}
    if legacy is not None:
        config["thresholds"] = legacy
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _agent(tmp_path: Path, config_path: Path) -> MetricsAgent:
    return MetricsAgent(
        config_path=str(config_path),
        logger=AgentLogger("metrics", log_dir=str(tmp_path / "logs")),
    )


@pytest.mark.asyncio
async def test_execute_finds_history_keyed_by_asin(tmp_path: Path) -> None:
    """D5: records with "asin" (no "product_asin") must resolve their history."""
    agent = _agent(tmp_path, _write_config(tmp_path, {"significant_drop": 5}))
    records = [
        {"asin": "B0LANE1", "rank": 1, "brand": "LANEIGE", "title": "LANEIGE Lip Mask"},
        {"asin": "B0OTHER", "rank": 2, "brand": "COSRX", "title": "COSRX Lip"},
    ]
    history = {"B0LANE1": [{"asin": "B0LANE1", "rank": 15, "date": "2026-08-31"}]}

    result = await agent.execute(
        {"categories": {"lip_care": {"rank_records": records}}}, historical_data=history
    )

    metric = result["product_metrics"][0]
    assert metric["asin"] == "B0LANE1"
    assert metric["rank_change_1d"] == -14
    alert_types = {a["type"] for a in result["alerts"]}
    assert "rank_shock" in alert_types
    assert "top10_entry" in alert_types


@pytest.mark.asyncio
async def test_execute_prefers_product_asin_over_asin(tmp_path: Path) -> None:
    """D5: "product_asin" still wins when both keys are present."""
    agent = _agent(tmp_path, _write_config(tmp_path, {"significant_drop": 5}))
    records = [
        {
            "product_asin": "B0REAL",
            "asin": "B0WRONG",
            "rank": 3,
            "brand": "LANEIGE",
            "title": "LANEIGE Lip Mask",
        }
    ]
    history = {"B0REAL": [{"asin": "B0REAL", "rank": 4}], "B0WRONG": [{"rank": 50}]}

    result = await agent.execute(
        {"categories": {"lip_care": {"rank_records": records}}}, historical_data=history
    )

    assert result["product_metrics"][0]["rank_change_1d"] == -1


def _drop_alerts(agent: MetricsAgent, change: int) -> list[dict]:
    metric = {
        "asin": "B0LANE1",
        "product_title": "LANEIGE Lip Mask",
        "category_id": "lip_care",
        "current_rank": 20 + change,
        "rank_change_1d": change,
    }
    return [a for a in agent._check_alerts(metric, {}, []) if a["type"] == "rank_drop"]


def test_check_alerts_reads_ranking_significant_drop(tmp_path: Path) -> None:
    """D11: the real config key ranking.significant_drop drives the rank_drop alert."""
    agent = _agent(tmp_path, _write_config(tmp_path, {"significant_drop": 7}))

    assert len(_drop_alerts(agent, 8)) == 1
    assert _drop_alerts(agent, 6) == []


def test_check_alerts_falls_back_to_legacy_thresholds_key(tmp_path: Path) -> None:
    """D11: the old thresholds.significant_rank_drop key still works as a fallback."""
    config_path = _write_config(tmp_path, {}, legacy={"significant_rank_drop": 3})
    agent = _agent(tmp_path, config_path)

    assert len(_drop_alerts(agent, 3)) == 1
    assert _drop_alerts(agent, 2) == []


def test_check_alerts_ranking_key_wins_over_legacy(tmp_path: Path) -> None:
    """D11: when both keys exist, ranking.significant_drop is authoritative."""
    config_path = _write_config(
        tmp_path, {"significant_drop": 7}, legacy={"significant_rank_drop": 2}
    )
    agent = _agent(tmp_path, config_path)

    assert _drop_alerts(agent, 6) == []
    assert len(_drop_alerts(agent, 7)) == 1

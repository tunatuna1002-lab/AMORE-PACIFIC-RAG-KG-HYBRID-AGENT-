"""F4-1: AlertAgent.process_metrics / MetricsAgent._check_alerts read ranking.* thresholds."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.alert_agent import AlertAgent
from src.agents.metrics_agent import MetricsAgent
from src.ontology.thresholds import Thresholds, load_thresholds, reset_thresholds, set_thresholds


@pytest.fixture(autouse=True)
def _restore():
    yield
    reset_thresholds()


def _alert_agent() -> AlertAgent:
    sm = MagicMock()
    sm.record_alert = MagicMock()
    return AlertAgent(state_manager=sm, email_sender=AsyncMock())


@pytest.mark.asyncio
async def test_alert_agent_default_threshold_is_ten() -> None:
    agent = _alert_agent()
    alerts = await agent.process_metrics(
        {"products": [{"name": "A", "rank_change": 9, "previous_rank": 20, "current_rank": 29}]}
    )
    assert alerts == []
    alerts = await agent.process_metrics(
        {"products": [{"name": "A", "rank_change": 10, "previous_rank": 20, "current_rank": 30}]}
    )
    assert [a.type for a in alerts] == ["rank_change"]


@pytest.mark.asyncio
async def test_alert_agent_rank_drop_seven_fires_eight_not_six(tmp_path: Path) -> None:
    cfg = {"ranking": {"significant_drop": 7, "alert_rank_change": 7}}
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    set_thresholds(load_thresholds(path))

    agent = _alert_agent()
    eight = await agent.process_metrics(
        {"products": [{"name": "A", "rank_change": 8, "previous_rank": 20, "current_rank": 28}]}
    )
    six = await agent.process_metrics(
        {"products": [{"name": "A", "rank_change": 6, "previous_rank": 20, "current_rank": 26}]}
    )
    assert [a.type for a in eight] == ["rank_change"]
    assert six == []
    # rise side uses the same alert threshold
    rise = await agent.process_metrics(
        {"products": [{"name": "A", "rank_change": -8, "previous_rank": 28, "current_rank": 20}]}
    )
    assert [a.type for a in rise] == ["rank_change"]


def test_metrics_agent_rank_drop_seven_fires_eight_not_six(tmp_path: Path) -> None:
    set_thresholds(Thresholds(rank_drop=7))
    agent = MetricsAgent(config_path=str(Path(__file__).resolve().parents[3] / "config" / "thresholds.json"))

    def metric(change: int) -> dict:
        return {
            "asin": "B0",
            "product_title": "P",
            "category_id": "lip_care",
            "current_rank": 20 + change,
            "rank_change_1d": change,
        }

    eight = agent._check_alerts(metric(8), {}, [])
    six = agent._check_alerts(metric(6), {}, [])
    assert [a["type"] for a in eight] == ["rank_drop"]
    assert six == []

"""Regression tests for defect D6.

AlertAgent.send_pending_alerts returns ``{"sent": n, ...}`` while AlertWorkflow read
``"sent_count"``, so ``alerts_sent`` was always 0. The workflow now reads ``"sent"``
(falling back to ``"sent_count"``) and the agent returns both keys.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.agents.alert_agent import AlertAgent
from src.application.workflows.alert_workflow import AlertWorkflow
from src.core.state_manager import StateManager


class FakeAlertAgent:
    def __init__(self, send_result: dict[str, Any]):
        self.send_result = send_result
        self.calls = 0

    async def process_metrics(self, metrics_data: dict[str, Any]) -> list[Any]:
        return []

    async def send_pending_alerts(self) -> dict[str, Any]:
        self.calls += 1
        return self.send_result


@pytest.mark.asyncio
async def test_workflow_reads_sent_key_from_agent_result() -> None:
    agent = FakeAlertAgent({"processed": 3, "sent": 2, "failed": 1, "skipped": 0, "details": []})
    workflow = AlertWorkflow(alert_agent=agent)

    result = await workflow.send_pending_alerts()

    assert result.success is True
    assert result.alerts_sent == 2
    assert agent.calls == 1


@pytest.mark.asyncio
async def test_workflow_falls_back_to_sent_count_key() -> None:
    agent = FakeAlertAgent({"sent_count": 4})
    workflow = AlertWorkflow(alert_agent=agent)

    result = await workflow.send_pending_alerts()

    assert result.alerts_sent == 4


@pytest.mark.asyncio
async def test_alert_agent_returns_both_sent_and_sent_count(tmp_path) -> None:
    agent = AlertAgent(StateManager(persist_dir=tmp_path / "state"))
    await agent.process_metrics(
        {
            "products": [
                {
                    "name": "A",
                    "brand": "LANEIGE",
                    "rank_change": 10,
                    "previous_rank": 5,
                    "current_rank": 15,
                }
            ]
        }
    )

    result = await agent.send_pending_alerts()

    assert "sent" in result and "sent_count" in result
    assert result["sent"] == result["sent_count"]

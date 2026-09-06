"""
D1: Insight report email calls a non-existent HybridInsightAgent method
=======================================================================
``UnifiedBrain._send_insight_report_email`` called
``insight_agent.generate_insight(...)`` and read ``result["insight"]``.
``HybridInsightAgent`` only exposes ``execute(metrics_data, crawl_data, crawl_summary)``
returning ``daily_insight`` / ``action_items`` / ``highlights``.  The AttributeError was
swallowed, so every insight email carried the placeholder text.

Fix contract:
- call ``execute(...)`` with a metrics/crawl payload built from the products
- map ``daily_insight`` into the email's ``insight_content`` HTML
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.core.brain import UnifiedBrain, reset_brain

FIXED_INSIGHT = (
    "LANEIGE Lip Sleeping Mask가 Lip Care 1위를 유지했습니다.\n\n경쟁 강도는 완만합니다."
)


class FakeInsightAgent:
    """Stands in for HybridInsightAgent; records the execute() call."""

    calls: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def execute(
        self,
        metrics_data: dict[str, Any],
        crawl_data: dict[str, Any] | None = None,
        crawl_summary: dict | None = None,
    ) -> dict[str, Any]:
        FakeInsightAgent.calls.append(
            {"metrics_data": metrics_data, "crawl_data": crawl_data, "crawl_summary": crawl_summary}
        )
        return {
            "status": "completed",
            "daily_insight": FIXED_INSIGHT,
            "action_items": [],
            "highlights": [],
        }


@dataclass
class SendResultStub:
    success: bool = True
    sent_to: list[str] = field(default_factory=lambda: ["a@b.c"])
    message: str = ""


class FakeSender:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    async def send_insight_report(self, **kwargs: Any) -> SendResultStub:
        self.kwargs = kwargs
        return SendResultStub()


@pytest.fixture(autouse=True)
def _reset():
    FakeInsightAgent.calls = []
    yield
    reset_brain()


def _products() -> list[dict[str, Any]]:
    return [
        {"asin": "B1", "title": "LANEIGE Lip Sleeping Mask", "brand": "LANEIGE", "rank": 1},
        {"asin": "B2", "title": "Other Balm", "brand": "Burt's Bees", "rank": 2},
        {"asin": "B3", "title": "Glow Serum", "brand": "COSRX", "rank": 3},
    ]


@pytest.mark.asyncio
async def test_insight_email_contains_generated_daily_insight(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("src.agents.hybrid_insight_agent.HybridInsightAgent", FakeInsightAgent)
    brain = UnifiedBrain(
        context_gatherer=MagicMock(), tool_executor=MagicMock(), response_pipeline=MagicMock()
    )
    sender = FakeSender()

    await brain._send_insight_report_email(_products(), ["a@b.c"], sender)

    assert sender.kwargs is not None, "send_insight_report was not called"
    html = sender.kwargs["insight_content"]
    assert "LANEIGE Lip Sleeping Mask가 Lip Care 1위를 유지했습니다." in html
    assert "경쟁 강도는 완만합니다." in html
    assert "현재 생성된 인사이트가 없습니다" not in html
    # paragraph/line-break mapping preserved
    assert "</p><p>" in html


@pytest.mark.asyncio
async def test_insight_agent_execute_receives_products(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("src.agents.hybrid_insight_agent.HybridInsightAgent", FakeInsightAgent)
    brain = UnifiedBrain(
        context_gatherer=MagicMock(), tool_executor=MagicMock(), response_pipeline=MagicMock()
    )

    await brain._send_insight_report_email(_products(), ["a@b.c"], FakeSender())

    assert len(FakeInsightAgent.calls) == 1
    call = FakeInsightAgent.calls[0]
    assert isinstance(call["metrics_data"], dict)
    assert call["crawl_data"] is not None
    # the products must reach the agent via crawl_data (categories -> rank_records)
    records = [
        rec
        for cat in call["crawl_data"].get("categories", {}).values()
        for rec in cat.get("rank_records", [])
    ]
    assert len(records) == 3


@pytest.mark.asyncio
async def test_insight_email_falls_back_when_agent_fails(monkeypatch: pytest.MonkeyPatch):
    class BrokenAgent(FakeInsightAgent):
        async def execute(self, *a: Any, **k: Any) -> dict[str, Any]:
            raise RuntimeError("boom")

    monkeypatch.setattr("src.agents.hybrid_insight_agent.HybridInsightAgent", BrokenAgent)
    brain = UnifiedBrain(
        context_gatherer=MagicMock(), tool_executor=MagicMock(), response_pipeline=MagicMock()
    )
    sender = FakeSender()

    await brain._send_insight_report_email(_products(), ["a@b.c"], sender)

    assert sender.kwargs is not None
    assert "현재 생성된 인사이트가 없습니다" in sender.kwargs["insight_content"]

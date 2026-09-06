"""Regression tests for defect D27.

``BatchWorkflow.run_daily_workflow`` reported ``status == "completed"`` even when the
crawl payload said ``status="failed"`` (loop broke after the crawl step) or when an
agent raised (step recorded as failed). Final status is now:

- ``"failed"``  if the crawl step failed or a critical step (crawl, store) raised
- ``"partial"`` if only non-critical steps (update_kg, calculate, insight, export) failed
- ``"completed"`` otherwise
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.application.workflows.batch_workflow import BatchWorkflow
from src.ontology.knowledge_graph import KnowledgeGraph
from tests.characterization.conftest import PROJECT_ROOT
from tests.characterization.test_batch_workflow_char import (
    CRAWL_RESULT,
    INSIGHT_RESULT,
    METRICS_RESULT,
    STORE_RESULT,
    FakeExporter,
    RecordingAgent,
    RecordingChatbot,
)

CONFIG_PATH = str(PROJECT_ROOT / "config" / "thresholds.json")


class Boom:
    async def execute(self, *a: Any, **k: Any) -> Any:
        raise RuntimeError("agent down")


@pytest.fixture
def workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BatchWorkflow:
    monkeypatch.chdir(tmp_path)
    kg_path = tmp_path / "kg.json"
    wf = BatchWorkflow(config_path=CONFIG_PATH, kg_persist_path=str(kg_path))
    wf._knowledge_graph = KnowledgeGraph(
        persist_path=str(kg_path), auto_load=False, auto_save=False
    )
    wf._crawler = RecordingAgent(CRAWL_RESULT)
    wf._storage = RecordingAgent(STORE_RESULT)
    wf._metrics_agent = RecordingAgent(METRICS_RESULT)
    wf._hybrid_insight = RecordingAgent(INSIGHT_RESULT)
    wf._hybrid_chatbot = RecordingChatbot()
    wf._dashboard_exporter = FakeExporter()
    return wf


@pytest.mark.asyncio
async def test_happy_path_is_completed(workflow: BatchWorkflow) -> None:
    result = await workflow.run_daily_workflow(categories=["lip_care"])
    assert result["status"] == "completed"
    assert "error" not in result


@pytest.mark.asyncio
async def test_failed_crawl_payload_reports_failed(workflow: BatchWorkflow) -> None:
    failed_crawl = {**CRAWL_RESULT, "status": "failed", "categories": {}, "total_products": 0}
    workflow._crawler = RecordingAgent(failed_crawl)

    result = await workflow.run_daily_workflow(categories=["lip_care"])

    assert result["status"] == "failed"
    assert result["error"]
    assert list(result["steps"]) == ["crawl"]
    assert result["metrics"]["status"] == "failed"


@pytest.mark.asyncio
async def test_raising_crawler_reports_failed(workflow: BatchWorkflow) -> None:
    workflow._crawler = Boom()

    result = await workflow.run_daily_workflow(categories=["lip_care"])

    assert result["status"] == "failed"
    assert "agent down" in result["error"]
    assert result["steps"]["crawl"] == {"status": "failed", "error": "agent down"}


@pytest.mark.asyncio
async def test_raising_storage_reports_failed(workflow: BatchWorkflow) -> None:
    workflow._storage = Boom()

    result = await workflow.run_daily_workflow(categories=["lip_care"])

    assert result["status"] == "failed"
    assert "agent down" in result["error"]
    assert result["steps"]["store"] == {"status": "failed", "error": "agent down"}
    # later steps still ran (observe advances regardless)
    assert list(result["steps"]) == ["crawl", "store", "update_kg", "calculate", "insight", "export"]


@pytest.mark.asyncio
async def test_raising_insight_reports_partial(workflow: BatchWorkflow) -> None:
    workflow._hybrid_insight = Boom()

    result = await workflow.run_daily_workflow(categories=["lip_care"])

    assert result["status"] == "partial"
    assert result["steps"]["insight"]["status"] == "failed"
    assert result["steps"]["export"]["status"] == "completed"
    assert result["metrics"]["status"] == "partial"


@pytest.mark.asyncio
async def test_raising_exporter_reports_partial(workflow: BatchWorkflow) -> None:
    class BoomExporter(FakeExporter):
        async def export_dashboard_data(self, path: str) -> dict[str, Any]:
            raise RuntimeError("export down")

    workflow._dashboard_exporter = BoomExporter()

    result = await workflow.run_daily_workflow(categories=["lip_care"])

    assert result["status"] == "partial"
    assert result["steps"]["export"] == {"status": "failed", "error": "export down"}

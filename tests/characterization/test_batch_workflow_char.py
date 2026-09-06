"""
Characterization: src.application.workflows.batch_workflow.BatchWorkflow.run_daily_workflow

Injection notes
---------------
BatchWorkflow has no constructor parameters for its agents; they are created
lazily by properties (``crawler``, ``storage``, ``metrics_agent``,
``hybrid_insight``, ``hybrid_chatbot``, ``dashboard_exporter``,
``knowledge_graph``) from private ``_xxx`` slots. We pre-populate those slots
on the *instance* with fakes (no module-path patching).

The mocks in tests/unit/application/conftest.py (MockScraper, MockStorage,
MockMetricCalculator, MockInsightAgent, ...) model the domain *protocols*
(scrape_category / append_rank_records / calculate_sos ...), but BatchWorkflow
actually drives its agents through a single ``execute(...)`` coroutine, so
those mocks cannot be used here; the recording fakes below mirror the real
call shapes.

Filesystem side effects (./data/latest_crawl_result.json, ./data/history,
./logs) are relative to CWD, so every test chdirs into tmp_path. The workflow
also tries Google Sheets -> SQLite sync; with no ./config/google_credentials.json
in CWD it fails fast and offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.application.workflows.batch_workflow import BatchWorkflow
from src.ontology.knowledge_graph import KnowledgeGraph
from tests.characterization.conftest import PROJECT_ROOT

CONFIG_PATH = str(PROJECT_ROOT / "config" / "thresholds.json")

CRAWL_RESULT: dict[str, Any] = {
    "categories": {
        "lip_care": {
            # "products" intentionally empty: any Unknown brand here would
            # trigger the LLM brand resolver inside _observe.
            "products": [],
            "rank_records": [
                {
                    "brand": "LANEIGE",
                    "asin": "B0LANE1",
                    "product_name": "LANEIGE Lip Sleeping Mask",
                    "rank": 1,
                    "rating": 4.5,
                    "price": 24.0,
                },
                {
                    "brand": "COSRX",
                    "asin": "B0COSRX1",
                    "product_name": "COSRX Lip",
                    "rank": 2,
                    "rating": 4.3,
                    "price": 15.0,
                },
            ],
            "laneige_count": 1,
        }
    },
    "total_products": 2,
    "laneige_count": 1,
    "laneige_products": [{"asin": "B0LANE1"}],
    "all_products": [],
    "status": "completed",
    "summary": "2 products",
}

STORE_RESULT = {"raw_records": 2, "products_upserted": 2, "status": "completed"}

METRICS_RESULT: dict[str, Any] = {
    "brand_metrics": [
        {
            "brand_name": "LANEIGE",
            "is_laneige": True,
            "share_of_shelf": 0.5,
            "avg_rank": 1.0,
            "product_count": 1,
            "category_id": "lip_care",
        }
    ],
    "product_metrics": [],
    "market_metrics": [{"category_id": "lip_care", "hhi": 0.5}],
    "alerts": [{"type": "rank_change"}],
    "summary": {},
}

INSIGHT_RESULT = {
    "daily_insight": "LANEIGE leads",
    "action_items": ["a"],
    "highlights": ["h"],
    "inferences": [{"x": 1}],
    "explanations": [],
}


class RecordingAgent:
    """Fake for crawler/storage/metrics/insight agents: ``execute(*a, **k)``."""

    def __init__(self, result: Any):
        self.result = result
        self.calls: list[tuple[tuple, dict]] = []

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self.result


class RecordingChatbot:
    def __init__(self):
        self.data_contexts: list[dict] = []

    def set_data_context(self, data: dict) -> None:
        self.data_contexts.append(data)


class FakeExporter:
    def __init__(self):
        self.calls: list[Any] = []

    async def initialize(self) -> bool:
        self.calls.append("initialize")
        return True

    async def export_dashboard_data(self, path: str) -> dict[str, Any]:
        self.calls.append(("export", path))
        return {"metadata": {"total_products": 2, "laneige_products": 1}}


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


async def test_run_daily_workflow_happy_path(workflow: BatchWorkflow, tmp_path: Path) -> None:
    result = await workflow.run_daily_workflow(categories=["lip_care"])

    assert result["status"] == "completed"
    assert result["hybrid_mode"] is True
    assert sorted(result) == [
        "completed_at",
        "hybrid_mode",
        "metrics",
        "session_id",
        "started_at",
        "status",
        "steps",
        "summary",
        "trace",
    ]

    # Step order and per-step payloads
    assert list(result["steps"]) == [
        "crawl",
        "store",
        "update_kg",
        "calculate",
        "insight",
        "export",
    ]
    assert all(step["status"] == "completed" for step in result["steps"].values())
    assert result["steps"]["crawl"]["result"] is CRAWL_RESULT
    assert result["steps"]["store"]["result"] == STORE_RESULT
    assert result["steps"]["update_kg"]["result"] == {
        "relations_added": 6,
        "total_triples": 6,
        "unique_subjects": 4,
        "unique_objects": 5,
    }
    assert result["steps"]["calculate"]["result"] is METRICS_RESULT
    assert result["steps"]["insight"]["result"] is INSIGHT_RESULT
    assert result["steps"]["export"]["result"] == {
        "exported": True,
        "path": "./data/dashboard_data.json",
        "products": 2,
        "laneige_count": 1,
        "brands_verified": 0,
        "sqlite_synced": 0,
    }

    # Summary
    assert result["summary"] == {
        "products_crawled": 2,
        "laneige_tracked": 1,
        "categories": ["lip_care"],
        "alerts": 1,
        "action_items": 1,
        "daily_insight": "LANEIGE leads...",  # always suffixed with "..."
        "dashboard_exported": True,
        "dashboard_path": "./data/dashboard_data.json",
        "hybrid": {"kg_triples": 6, "inferences": 1, "explanations": 0},
    }

    # Metrics / trace summaries (QualityMetrics + ExecutionTracer)
    assert set(result["metrics"]) == {"duration_seconds", "status", "agents"}
    assert result["metrics"]["status"] == "completed"
    assert result["metrics"]["agents"] == {}  # fakes never record agent metrics
    assert set(result["trace"]) == {
        "completed",
        "failed",
        "session_id",
        "total_duration_ms",
        "total_spans",
        "trace_id",
    }
    assert result["trace"]["session_id"] == result["session_id"]


async def test_run_daily_workflow_agent_call_shapes(workflow: BatchWorkflow) -> None:
    await workflow.run_daily_workflow(categories=["lip_care"])

    assert workflow._crawler.calls == [((["lip_care"],), {})]
    assert workflow._storage.calls == [((CRAWL_RESULT,), {})]
    # historical_data is never populated in _state -> always None
    assert workflow._metrics_agent.calls == [((CRAWL_RESULT, None), {})]
    assert workflow._hybrid_insight.calls == [
        (
            (),
            {
                "metrics_data": METRICS_RESULT,
                "crawl_data": CRAWL_RESULT,
                "crawl_summary": "2 products",
            },
        )
    ]
    assert workflow._hybrid_chatbot.data_contexts == [METRICS_RESULT]
    assert workflow._dashboard_exporter.calls == [
        "initialize",
        ("export", "./data/dashboard_data.json"),
    ]


async def test_run_daily_workflow_kg_and_filesystem_effects(
    workflow: BatchWorkflow, tmp_path: Path
) -> None:
    await workflow.run_daily_workflow(categories=["lip_care"])

    kg = workflow._knowledge_graph
    assert kg.get_stats() == {
        "total_triples": 6,
        "unique_subjects": 4,
        "unique_objects": 5,
        "relations_by_type": {"hasProduct": 2, "belongsToCategory": 2, "directCompetitor": 2},
    }
    # load_from_metrics_data stores brand metadata keyed by the original-case brand name
    assert kg.get_entity_metadata("LANEIGE") == {
        "type": "brand",
        "sos": 0.5,
        "avg_rank": 1.0,
        "product_count": 1,
        "is_target": True,
        "category": "lip_care",
    }
    # PINS CURRENT BEHAVIOR: with auto_save=False the graph is never marked dirty,
    # so the workflow's explicit knowledge_graph.save() is a silent no-op
    # (it logs "Knowledge Graph saved" but writes nothing).
    assert not (tmp_path / "kg.json").exists()

    # Raw crawl result is dumped relative to CWD
    crawl_json = tmp_path / "data" / "latest_crawl_result.json"
    assert crawl_json.exists()
    assert json.loads(crawl_json.read_text(encoding="utf-8")) == CRAWL_RESULT
    assert (tmp_path / "data" / "history").is_dir()


async def test_get_status_after_run(workflow: BatchWorkflow) -> None:
    result = await workflow.run_daily_workflow(categories=["lip_care"])
    status = workflow.get_status()

    assert status["session_id"] == result["session_id"]
    assert status["current_step"] == "complete"
    assert status["hybrid_mode"] is True
    assert status["data"]["categories"] == ["lip_care"]
    assert status["data"]["total_products"] == 2
    assert status["data"]["laneige_count"] == 1
    assert status["data"]["metrics_ready"] is True
    assert status["data"]["insights_ready"] is True
    # PINS CURRENT BEHAVIOR: ContextManager.start_workflow seeds current_step with
    # the first step, and advance_workflow appends current_step before popping the
    # next one, so "crawl" is recorded twice, "export" is never marked completed
    # and progress stalls at 6/7 even though the workflow finished.
    assert status["workflow"]["completed_steps"] == [
        "crawl",
        "crawl",
        "store",
        "update_kg",
        "calculate",
        "insight",
    ]
    assert status["workflow"]["current_step"] == "export"
    assert status["workflow"]["pending_steps"] == []
    assert status["workflow"]["progress"] == "6/7"
    assert status["workflow"]["has_errors"] is False


async def test_failed_crawl_stops_after_crawl_but_reports_completed(
    workflow: BatchWorkflow,
) -> None:
    failed_crawl = {**CRAWL_RESULT, "status": "failed", "categories": {}, "total_products": 0}
    workflow._crawler = RecordingAgent(failed_crawl)

    result = await workflow.run_daily_workflow(categories=["lip_care"])

    # FIXED (D27): a crawl whose payload says status="failed" makes the STORE
    # think-step return should_continue=False; the loop breaks and the workflow now
    # reports status "failed" (crawl is a critical step) with an explanatory error.
    assert result["status"] == "failed"
    assert result["error"].startswith("critical step(s) failed: crawl:")
    assert list(result["steps"]) == ["crawl"]
    assert result["steps"]["crawl"]["status"] == "completed"
    assert workflow._storage.calls == []
    assert workflow._metrics_agent.calls == []
    assert workflow._hybrid_insight.calls == []
    assert workflow._dashboard_exporter.calls == []
    assert result["summary"]["products_crawled"] == 0
    assert result["summary"]["dashboard_exported"] is False
    assert result["summary"]["hybrid"] == {"kg_triples": 0, "inferences": 0, "explanations": 0}


async def test_agent_exception_marks_step_failed_and_continues(workflow: BatchWorkflow) -> None:
    class Boom:
        async def execute(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("sheets down")

    workflow._storage = Boom()
    result = await workflow.run_daily_workflow(categories=["lip_care"])

    # FIXED (D27): a failing store step is recorded and the observe phase still
    # advances (store -> update_kg), but store is a critical step so the final
    # status is "failed" (a failing insight/alert step would give "partial").
    assert result["status"] == "failed"
    assert result["error"] == "critical step(s) failed: store: sheets down"
    assert result["steps"]["store"] == {"status": "failed", "error": "sheets down"}
    assert list(result["steps"]) == [
        "crawl",
        "store",
        "update_kg",
        "calculate",
        "insight",
        "export",
    ]
    assert workflow.get_status()["workflow"]["has_errors"] is True
    assert workflow.get_status()["workflow"]["error_count"] == 1

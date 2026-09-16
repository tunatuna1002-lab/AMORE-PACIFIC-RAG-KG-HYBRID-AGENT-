"""F1 (single batch pipeline) tests for BatchWorkflow.

- ``WorkflowDependencies`` is wired for real: injected collaborators are used and no
  Container/agent construction happens.
- The pipeline includes the alert step (AlertAgent.process_metrics +
  send_pending_alerts) and reads the D6 ``"sent"`` key.
- ``run_daily_workflow`` reports ``errors`` (D12: partial crawl payload / storage
  errors -> ``"partial"``), supports ``crawl_only`` and a ``progress_callback``.
- F7: the workflow records crawl / metrics / KG progress in ``StateManager``.
- Layering: batch_workflow.py has no top-level import of agents/tools/infrastructure.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import src.application.workflows.batch_workflow as batch_workflow_module
from src.application.workflows.batch_workflow import (
    BatchWorkflow,
    WorkflowDependencies,
    WorkflowStep,
)
from src.core.state_manager import DataFreshness, StateManager
from src.ontology.knowledge_graph import KnowledgeGraph
from tests.characterization.conftest import PROJECT_ROOT
from tests.characterization.test_batch_workflow_char import (
    CRAWL_RESULT,
    INSIGHT_RESULT,
    METRICS_RESULT,
    STORE_RESULT,
    FakeExporter,
    RecordingAgent,
    RecordingAlertAgent,
    RecordingChatbot,
)

CONFIG_PATH = str(PROJECT_ROOT / "config" / "thresholds.json")


def _deps(**overrides: Any) -> WorkflowDependencies:
    deps = WorkflowDependencies(
        crawler=RecordingAgent(CRAWL_RESULT),
        storage=RecordingAgent(STORE_RESULT),
        metrics=RecordingAgent(METRICS_RESULT),
        insight=RecordingAgent(INSIGHT_RESULT),
        alert=RecordingAlertAgent(),
        exporter=FakeExporter(),
        chatbot=RecordingChatbot(),
    )
    for name, value in overrides.items():
        setattr(deps, name, value)
    return deps


@pytest.fixture
def state_manager(tmp_path: Path) -> StateManager:
    return StateManager(persist_dir=tmp_path / "state")


@pytest.fixture
def make_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state_manager: StateManager):
    monkeypatch.chdir(tmp_path)

    def _make(deps: WorkflowDependencies | None = None, **kwargs: Any) -> BatchWorkflow:
        kg_path = tmp_path / "kg.json"
        deps = deps or _deps()
        deps.knowledge_graph = KnowledgeGraph(
            persist_path=str(kg_path), auto_load=False, auto_save=False
        )
        return BatchWorkflow(
            config_path=CONFIG_PATH,
            kg_persist_path=str(kg_path),
            deps=deps,
            state_manager=state_manager,
            **kwargs,
        )

    return _make


# ---------------------------------------------------------------------------
# Layering / WorkflowDependencies
# ---------------------------------------------------------------------------


def test_batch_workflow_has_no_top_level_agent_tool_infra_imports() -> None:
    """application layer must not import agents/tools/infrastructure at import time."""
    source = Path(batch_workflow_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = ("src.agents", "src.tools", "src.infrastructure")
    offenders: list[str] = []
    for node in tree.body:  # top-level statements only (lazy imports inside defs are fine)
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(forbidden):
                offenders.append(node.module)
        elif isinstance(node, ast.Import):
            offenders.extend(a.name for a in node.names if a.name.startswith(forbidden))
    assert offenders == []


def test_workflow_dependencies_defaults_are_none() -> None:
    deps = WorkflowDependencies()
    for name in WorkflowDependencies.COMPONENTS:
        assert getattr(deps, name) is None
    assert deps.knowledge_graph is None
    assert deps.categories == []


def test_from_container_delegates_to_infrastructure_container() -> None:
    sentinel = WorkflowDependencies(crawler=MagicMock())
    with patch(
        "src.infrastructure.container.Container.build_workflow_dependencies",
        return_value=sentinel,
    ) as build:
        deps = WorkflowDependencies.from_container(only=("crawler",), config_path="x.json")

    assert deps is sentinel
    build.assert_called_once_with(only=("crawler",), config_path="x.json")


def test_injected_dependencies_are_used_without_container(make_workflow) -> None:
    deps = _deps()
    wf = make_workflow(deps)

    assert wf.crawler is deps.crawler
    assert wf.storage is deps.storage
    assert wf.metrics_agent is deps.metrics
    assert wf.hybrid_insight is deps.insight
    assert wf.alert_agent is deps.alert
    assert wf.dashboard_exporter is deps.exporter
    assert wf.hybrid_chatbot is deps.chatbot
    assert wf.knowledge_graph is deps.knowledge_graph


def test_missing_dependency_is_resolved_lazily_one_at_a_time(make_workflow) -> None:
    deps = _deps()
    deps.crawler = None
    wf = make_workflow(deps)
    fake_crawler = RecordingAgent(CRAWL_RESULT)

    with patch.object(
        WorkflowDependencies,
        "from_container",
        return_value=WorkflowDependencies(crawler=fake_crawler),
    ) as from_container:
        assert wf.crawler is fake_crawler
        assert wf.crawler is fake_crawler  # cached

    from_container.assert_called_once()
    assert from_container.call_args.kwargs["only"] == ("crawler",)
    # other injected collaborators were never re-resolved
    assert wf.storage is deps.storage


# ---------------------------------------------------------------------------
# Alert step (folded from the deleted AlertWorkflow) + D6
# ---------------------------------------------------------------------------


async def test_pipeline_runs_alert_step_between_insight_and_export(make_workflow) -> None:
    deps = _deps(alert=RecordingAlertAgent(created=2, sent=2))
    wf = make_workflow(deps)

    result = await wf.run_daily_workflow(categories=["lip_care"])

    assert result["status"] == "completed"
    steps = list(result["steps"])
    assert steps.index("insight") < steps.index("alert") < steps.index("export")
    assert deps.alert.processed == [METRICS_RESULT]
    assert deps.alert.send_calls == 1
    # D6: AlertAgent returns "sent" (not "sent_count"); the step must read it
    assert result["steps"]["alert"]["result"]["alerts_sent"] == 2
    assert result["steps"]["alert"]["result"]["alerts_created"] == 2
    assert result["summary"]["alerts_sent"] == 2


async def test_alert_step_reads_legacy_sent_count_key(make_workflow) -> None:
    class LegacyAlertAgent(RecordingAlertAgent):
        async def send_pending_alerts(self) -> dict[str, Any]:
            return {"sent_count": 3, "failed_count": 0}

    wf = make_workflow(_deps(alert=LegacyAlertAgent()))
    result = await wf.run_daily_workflow(categories=["lip_care"])
    assert result["steps"]["alert"]["result"]["alerts_sent"] == 3


async def test_alert_failure_is_partial_and_export_still_runs(make_workflow) -> None:
    class BoomAlert:
        async def process_metrics(self, metrics_data: dict) -> list:
            raise RuntimeError("smtp down")

        async def send_pending_alerts(self) -> dict:
            return {"sent": 0}

    deps = _deps(alert=BoomAlert())
    wf = make_workflow(deps)
    result = await wf.run_daily_workflow(categories=["lip_care"])

    assert result["status"] == "partial"
    assert result["steps"]["alert"] == {"status": "failed", "error": "smtp down"}
    assert result["steps"]["export"]["status"] == "completed"
    assert "alert: smtp down" in result["errors"]


async def test_think_act_observe_alert_step(make_workflow) -> None:
    deps = _deps()
    wf = make_workflow(deps)
    wf._current_step = WorkflowStep.ALERT
    wf._state = {"metrics_result": METRICS_RESULT}

    think = await wf._think()
    assert think.next_action == "alert"
    assert think.parameters["metrics_data"] is METRICS_RESULT

    act = await wf._act(think)
    assert act.success is True
    assert act.result["alerts_created"] == 1

    observe = await wf._observe(act)
    assert observe.next_step == WorkflowStep.EXPORT
    assert observe.state_updates["alert_result"] == act.result


# ---------------------------------------------------------------------------
# D12: errors list / partial status
# ---------------------------------------------------------------------------


async def test_partial_crawl_payload_reports_partial_with_errors(make_workflow) -> None:
    partial = {**CRAWL_RESULT, "status": "partial", "errors": ["lip_makeup: timeout"]}
    deps = _deps(crawler=RecordingAgent(partial))
    wf = make_workflow(deps)

    result = await wf.run_daily_workflow(categories=["lip_care"])

    assert result["status"] == "partial"
    assert result["errors"] == ["crawl: lip_makeup: timeout"]
    assert "lip_makeup: timeout" in result["error"]
    # the rest of the pipeline still ran on the partial data
    assert result["steps"]["export"]["status"] == "completed"


async def test_partial_crawl_without_error_list_still_partial(make_workflow) -> None:
    partial = {**CRAWL_RESULT, "status": "partial"}
    wf = make_workflow(_deps(crawler=RecordingAgent(partial)))
    result = await wf.run_daily_workflow(categories=["lip_care"])
    assert result["status"] == "partial"
    assert result["errors"] == ["crawl: crawler reported partial result"]


async def test_storage_errors_report_partial(make_workflow) -> None:
    store = {"raw_records": 0, "errors": ["Sheets quota exceeded"]}
    wf = make_workflow(_deps(storage=RecordingAgent(store)))
    result = await wf.run_daily_workflow(categories=["lip_care"])
    assert result["status"] == "partial"
    assert result["errors"] == ["store: Sheets quota exceeded"]


async def test_clean_run_has_no_errors(make_workflow) -> None:
    result = await make_workflow().run_daily_workflow(categories=["lip_care"])
    assert result["status"] == "completed"
    assert result["errors"] == []
    assert "error" not in result


# ---------------------------------------------------------------------------
# crawl_only / progress callback
# ---------------------------------------------------------------------------


async def test_crawl_only_stops_after_crawl(make_workflow) -> None:
    deps = _deps()
    wf = make_workflow(deps)
    result = await wf.run_daily_workflow(categories=["lip_care"], crawl_only=True)

    assert result["status"] == "completed"
    assert list(result["steps"]) == ["crawl"]
    assert deps.storage.calls == []
    assert deps.metrics.calls == []
    assert deps.alert.processed == []
    assert deps.exporter.calls == []


async def test_progress_callback_receives_each_step_in_order(make_workflow) -> None:
    wf = make_workflow()
    seen: list[tuple[str, str]] = []

    def on_step(step: str, payload: dict[str, Any]) -> None:
        seen.append((step, payload["status"]))

    await wf.run_daily_workflow(categories=["lip_care"], progress_callback=on_step)

    assert seen == [
        ("crawl", "completed"),
        ("store", "completed"),
        ("update_kg", "completed"),
        ("calculate", "completed"),
        ("insight", "completed"),
        ("alert", "completed"),
        ("export", "completed"),
    ]


async def test_progress_callback_errors_do_not_break_workflow(make_workflow) -> None:
    def bad_callback(step: str, payload: dict[str, Any]) -> None:
        raise RuntimeError("ui gone")

    result = await make_workflow().run_daily_workflow(
        categories=["lip_care"], progress_callback=bad_callback
    )
    assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# F7: StateManager is written by the pipeline
# ---------------------------------------------------------------------------


async def test_run_marks_state_manager(make_workflow, state_manager: StateManager) -> None:
    assert state_manager.data_freshness is DataFreshness.UNKNOWN
    assert state_manager.last_metrics_time is None
    assert state_manager.kg_initialized is False

    await make_workflow().run_daily_workflow(categories=["lip_care"])

    assert state_manager.data_freshness is DataFreshness.FRESH
    assert state_manager.last_crawl_success is True
    assert state_manager.last_crawl_count == 2
    assert state_manager.is_crawl_needed() is False
    assert state_manager.last_metrics_time is not None
    assert state_manager.kg_initialized is True
    # CHANGED (F9): update_kg now writes the ontology-built KG (category hierarchy,
    # brand→group ownership, materialized inferences), not just the crawl relations.
    assert state_manager.kg_triple_count == 107


async def test_failed_crawl_does_not_mark_state_fresh(
    make_workflow, state_manager: StateManager
) -> None:
    failed = {**CRAWL_RESULT, "status": "failed", "categories": {}, "total_products": 0}
    await make_workflow(_deps(crawler=RecordingAgent(failed))).run_daily_workflow()

    assert state_manager.data_freshness is DataFreshness.UNKNOWN
    assert state_manager.last_crawl_time is None


async def test_kg_stats_update_on_second_run(make_workflow, state_manager: StateManager) -> None:
    wf = make_workflow()
    await wf.run_daily_workflow(categories=["lip_care"])
    first_update = state_manager.kg_last_update
    await wf.run_daily_workflow(categories=["lip_care"])
    assert state_manager.kg_initialized is True
    assert state_manager.kg_last_update >= first_update


# ---------------------------------------------------------------------------
# Single crawl snapshot dump (moved here from CrawlManager)
# ---------------------------------------------------------------------------


async def test_crawl_snapshot_written_once_by_workflow(make_workflow, tmp_path: Path) -> None:
    crawl = {
        **CRAWL_RESULT,
        "snapshot_date": "2026-09-06",
        "categories": {
            "lip_care": {
                "products": [{"asin": "B0LANE1", "brand": "LANEIGE", "rank": 1}],
                "rank_records": [],
            }
        },
    }
    await make_workflow(_deps(crawler=RecordingAgent(crawl))).run_daily_workflow(
        categories=["lip_care"]
    )

    assert (tmp_path / "data" / "latest_crawl_result.json").exists()
    history = tmp_path / "data" / "raw_products" / "2026-09-06.json"
    assert history.exists()
    import json

    rows = json.loads(history.read_text(encoding="utf-8"))
    assert rows == [{"asin": "B0LANE1", "brand": "LANEIGE", "rank": 1, "category_id": "lip_care"}]
    # the original payload is not mutated by the dump
    assert "category_id" not in crawl["categories"]["lip_care"]["products"][0]

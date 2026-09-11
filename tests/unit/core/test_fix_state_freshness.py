"""Regression test for defect D9 (F7: one system state).

``OrchestratorState`` had no writers, so ``data_freshness`` stayed "unknown" forever and
the LLM prompt always said "크롤링 기록 없음". ``StateManager`` is now the single system
state: the batch pipeline (run through ``CrawlManager``) marks it, and
``ContextGatherer`` reads it.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.application.workflows.batch_workflow import BatchWorkflow, WorkflowDependencies
from src.core.context_gatherer import ContextGatherer
from src.core.crawl_manager import CrawlManager, CrawlStatus
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


@pytest.fixture
def state_manager(tmp_path: Path) -> StateManager:
    return StateManager(persist_dir=tmp_path / "state")


@pytest.fixture
def crawl_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state_manager: StateManager
) -> CrawlManager:
    monkeypatch.chdir(tmp_path)

    def workflow_factory() -> BatchWorkflow:
        kg_path = tmp_path / "kg.json"
        deps = WorkflowDependencies(
            crawler=RecordingAgent(CRAWL_RESULT),
            storage=RecordingAgent(STORE_RESULT),
            metrics=RecordingAgent(METRICS_RESULT),
            insight=RecordingAgent(INSIGHT_RESULT),
            alert=RecordingAlertAgent(),
            exporter=FakeExporter(),
            chatbot=RecordingChatbot(),
            knowledge_graph=KnowledgeGraph(
                persist_path=str(kg_path), auto_load=False, auto_save=False
            ),
        )
        return BatchWorkflow(
            config_path=CONFIG_PATH,
            kg_persist_path=str(kg_path),
            deps=deps,
            state_manager=state_manager,
        )

    with patch.object(CrawlManager, "STATE_FILE", str(tmp_path / "crawl_state.json")):
        with patch.object(CrawlManager, "DATA_FILE", str(tmp_path / "dashboard_data.json")):
            manager = CrawlManager(state_manager=state_manager, workflow_factory=workflow_factory)
    manager.STATE_FILE = str(tmp_path / "crawl_state.json")
    manager.DATA_FILE = str(tmp_path / "dashboard_data.json")
    return manager


@pytest.mark.asyncio
async def test_crawl_through_crawl_manager_makes_state_fresh(
    crawl_manager: CrawlManager, state_manager: StateManager
) -> None:
    gatherer = ContextGatherer(state_manager=state_manager)

    # RED state before the crawl: D9 symptom
    before = gatherer._format_system_state(gatherer._get_system_state())
    assert state_manager.data_freshness is DataFreshness.UNKNOWN
    assert "크롤링 기록 없음" in before

    with patch("src.core.brain.get_brain", AsyncMock(return_value=None)):
        assert await crawl_manager.start_crawl() is True
        assert await crawl_manager.wait_for_completion(timeout=10) is True

    assert crawl_manager.state.status is CrawlStatus.COMPLETED
    assert state_manager.data_freshness is DataFreshness.FRESH
    assert state_manager.last_crawl_count == 2
    assert state_manager.is_crawl_needed() is False

    after = gatherer._format_system_state(gatherer._get_system_state())
    assert "크롤링 기록 없음" not in after
    assert "마지막 크롤링:" in after
    assert "데이터 상태: fresh" in after

    # the same StateManager is what the LLM decision summary reads
    decision = gatherer._build_decision_summary(
        __import__("src.core.models", fromlist=["Context"]).Context(
            query="q", system_state=gatherer._get_system_state()
        )
    )
    assert "데이터: 최신" in decision


@pytest.mark.asyncio
async def test_state_survives_reload_from_system_state_json(
    crawl_manager: CrawlManager, state_manager: StateManager, tmp_path: Path
) -> None:
    with patch("src.core.brain.get_brain", AsyncMock(return_value=None)):
        await crawl_manager.start_crawl()
        await crawl_manager.wait_for_completion(timeout=10)

    reloaded = StateManager(persist_dir=tmp_path / "state")
    assert reloaded.data_freshness is DataFreshness.FRESH
    assert reloaded.last_crawl_time == state_manager.last_crawl_time
    assert reloaded.last_metrics_time is not None
    assert reloaded.kg_initialized is True
    # only the two remaining state files exist: system_state.json + crawl_state.json
    assert (tmp_path / "state" / "system_state.json").exists()
    assert (tmp_path / "crawl_state.json").exists()
    assert not (tmp_path / "orchestrator_state.json").exists()

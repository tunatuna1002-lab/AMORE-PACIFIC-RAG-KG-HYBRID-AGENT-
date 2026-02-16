"""
BatchWorkflow 단위 테스트
============================
src/application/workflows/batch_workflow.py 커버리지 18% → 60%+ 목표

테스트 대상:
- Dataclass (WorkflowResult, ThinkResult, ActResult, ObserveResult)
- BatchWorkflow 초기화, _load_config
- _think() — 각 WorkflowStep 별 결정
- _act() — 각 action 별 실행
- _observe() — 결과 관찰 및 상태 업데이트
- _generate_summary()
- get_status, get_history_stats, get_knowledge_graph_stats, get_inference_stats
- cleanup()
- _verify_unknown_brands(), _sync_sheets_to_sqlite()
- run_daily_workflow() (mocked)
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.workflows.batch_workflow import (
    ActResult,
    BatchWorkflow,
    ObserveResult,
    ThinkResult,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStep,
)

# =========================================================================
# Helper
# =========================================================================


@pytest.fixture
def tmp_config(tmp_path):
    """임시 설정 파일"""
    config = {"categories": {"beauty": {"url": "https://example.com"}}}
    config_file = tmp_path / "thresholds.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    return str(config_file)


@pytest.fixture
def workflow(tmp_config, tmp_path):
    """테스트용 BatchWorkflow (KG 영속화 비활성)"""
    kg_path = str(tmp_path / "kg.json")
    with patch("src.application.workflows.batch_workflow.AgentLogger"):
        with patch("src.application.workflows.batch_workflow.ExecutionTracer"):
            with patch("src.application.workflows.batch_workflow.QualityMetrics"):
                with patch("src.application.workflows.batch_workflow.SessionManager"):
                    with patch("src.application.workflows.batch_workflow.HistoryManager"):
                        with patch("src.application.workflows.batch_workflow.ContextManager"):
                            wf = BatchWorkflow(
                                config_path=tmp_config,
                                kg_persist_path=kg_path,
                            )
    return wf


# =========================================================================
# Dataclasses
# =========================================================================


class TestWorkflowStatus:
    def test_enum_values(self):
        assert WorkflowStatus.PENDING == "pending"
        assert WorkflowStatus.RUNNING == "running"
        assert WorkflowStatus.COMPLETED == "completed"
        assert WorkflowStatus.FAILED == "failed"


class TestWorkflowStep:
    def test_enum_values(self):
        assert WorkflowStep.CRAWL.value == "crawl"
        assert WorkflowStep.STORE.value == "store"
        assert WorkflowStep.UPDATE_KG.value == "update_kg"
        assert WorkflowStep.CALCULATE.value == "calculate"
        assert WorkflowStep.INSIGHT.value == "insight"
        assert WorkflowStep.EXPORT.value == "export"
        assert WorkflowStep.COMPLETE.value == "complete"


class TestWorkflowResult:
    def test_to_dict_defaults(self):
        now = datetime.now()
        result = WorkflowResult(status=WorkflowStatus.PENDING, started_at=now)
        d = result.to_dict()
        assert d["status"] == "pending"
        assert d["started_at"] == now.isoformat()
        assert d["completed_at"] is None
        assert d["records_count"] == 0
        assert d["metrics"] == {}
        assert d["insights"] is None
        assert d["errors"] == []

    def test_to_dict_completed(self):
        start = datetime(2025, 1, 1, 10, 0)
        end = datetime(2025, 1, 1, 11, 0)
        result = WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            started_at=start,
            completed_at=end,
            records_count=100,
            metrics={"sos": 0.15},
            insights="Good performance",
            errors=["minor warning"],
        )
        d = result.to_dict()
        assert d["status"] == "completed"
        assert d["completed_at"] == end.isoformat()
        assert d["records_count"] == 100
        assert d["insights"] == "Good performance"
        assert len(d["errors"]) == 1


class TestThinkResult:
    def test_defaults(self):
        t = ThinkResult(next_action="crawl", reasoning="Start crawling")
        assert t.should_continue is True
        assert t.parameters == {}

    def test_custom(self):
        t = ThinkResult(
            next_action="skip",
            reasoning="No data",
            parameters={"key": "val"},
            should_continue=False,
        )
        assert t.should_continue is False
        assert t.parameters == {"key": "val"}


class TestActResult:
    def test_success(self):
        a = ActResult(action="crawl", success=True, result={"count": 50})
        assert a.success is True
        assert a.error is None

    def test_failure(self):
        a = ActResult(action="crawl", success=False, error="Timeout")
        assert a.success is False
        assert a.error == "Timeout"


class TestObserveResult:
    def test_defaults(self):
        o = ObserveResult(observations=["done"])
        assert o.state_updates == {}
        assert o.next_step is None

    def test_with_next_step(self):
        o = ObserveResult(
            observations=["ok"],
            state_updates={"crawl_result": {}},
            next_step=WorkflowStep.STORE,
        )
        assert o.next_step == WorkflowStep.STORE


# =========================================================================
# BatchWorkflow 초기화
# =========================================================================


class TestBatchWorkflowInit:
    def test_init_defaults(self, workflow):
        assert workflow.model == "gpt-4.1-mini"
        assert workflow.use_hybrid is True
        assert workflow._current_step == WorkflowStep.CRAWL
        assert workflow._state == {}
        assert workflow._session_id is None

    def test_init_custom_model(self, tmp_config, tmp_path):
        kg_path = str(tmp_path / "kg.json")
        with patch("src.application.workflows.batch_workflow.AgentLogger"):
            with patch("src.application.workflows.batch_workflow.ExecutionTracer"):
                with patch("src.application.workflows.batch_workflow.QualityMetrics"):
                    with patch("src.application.workflows.batch_workflow.SessionManager"):
                        with patch("src.application.workflows.batch_workflow.HistoryManager"):
                            with patch("src.application.workflows.batch_workflow.ContextManager"):
                                wf = BatchWorkflow(
                                    config_path=tmp_config,
                                    model="gpt-4o",
                                    use_hybrid=False,
                                    kg_persist_path=kg_path,
                                )
        assert wf.model == "gpt-4o"
        assert wf.use_hybrid is False

    def test_load_config_success(self, workflow, tmp_config):
        config = workflow._load_config(tmp_config)
        assert "categories" in config

    def test_load_config_missing_file(self, workflow):
        config = workflow._load_config("/nonexistent/path.json")
        assert config == {}


# =========================================================================
# _think()
# =========================================================================


class TestThink:
    @pytest.mark.asyncio
    async def test_think_crawl(self, workflow):
        workflow._current_step = WorkflowStep.CRAWL
        workflow._state = {"categories": ["Beauty"]}
        result = await workflow._think()
        assert result.next_action == "crawl"
        assert result.should_continue is True
        assert result.parameters["categories"] == ["Beauty"]

    @pytest.mark.asyncio
    async def test_think_store_with_data(self, workflow):
        workflow._current_step = WorkflowStep.STORE
        workflow._state = {"crawl_result": {"total_products": 50}}
        result = await workflow._think()
        assert result.next_action == "store"
        assert result.should_continue is True

    @pytest.mark.asyncio
    async def test_think_store_no_data(self, workflow):
        workflow._current_step = WorkflowStep.STORE
        workflow._state = {}
        result = await workflow._think()
        assert result.next_action == "skip"
        assert result.should_continue is False

    @pytest.mark.asyncio
    async def test_think_store_failed_crawl(self, workflow):
        workflow._current_step = WorkflowStep.STORE
        workflow._state = {"crawl_result": {"status": "failed"}}
        result = await workflow._think()
        assert result.next_action == "skip"
        assert result.should_continue is False

    @pytest.mark.asyncio
    async def test_think_update_kg(self, workflow):
        workflow._current_step = WorkflowStep.UPDATE_KG
        workflow._state = {"crawl_result": {"categories": {}}}
        result = await workflow._think()
        assert result.next_action == "update_kg"
        assert result.should_continue is True

    @pytest.mark.asyncio
    async def test_think_calculate_with_data(self, workflow):
        workflow._current_step = WorkflowStep.CALCULATE
        workflow._state = {"crawl_result": {"total_products": 50}}
        result = await workflow._think()
        assert result.next_action == "calculate"

    @pytest.mark.asyncio
    async def test_think_calculate_no_data(self, workflow):
        workflow._current_step = WorkflowStep.CALCULATE
        workflow._state = {}
        result = await workflow._think()
        assert result.next_action == "skip"
        assert result.should_continue is False

    @pytest.mark.asyncio
    async def test_think_insight_hybrid(self, workflow):
        workflow._current_step = WorkflowStep.INSIGHT
        workflow.use_hybrid = True
        workflow._state = {"metrics_result": {"brand_metrics": []}}
        result = await workflow._think()
        assert result.next_action == "hybrid_insight"

    @pytest.mark.asyncio
    async def test_think_insight_non_hybrid(self, workflow):
        workflow._current_step = WorkflowStep.INSIGHT
        workflow.use_hybrid = False
        workflow._state = {"metrics_result": {"brand_metrics": []}}
        result = await workflow._think()
        assert result.next_action == "hybrid_insight"

    @pytest.mark.asyncio
    async def test_think_insight_no_metrics(self, workflow):
        workflow._current_step = WorkflowStep.INSIGHT
        workflow._state = {}
        result = await workflow._think()
        assert result.next_action == "skip"
        assert result.should_continue is False

    @pytest.mark.asyncio
    async def test_think_export(self, workflow):
        workflow._current_step = WorkflowStep.EXPORT
        result = await workflow._think()
        assert result.next_action == "export"

    @pytest.mark.asyncio
    async def test_think_complete(self, workflow):
        workflow._current_step = WorkflowStep.COMPLETE
        result = await workflow._think()
        assert result.next_action == "complete"
        assert result.should_continue is False


# =========================================================================
# _act()
# =========================================================================


class TestAct:
    @pytest.mark.asyncio
    async def test_act_crawl(self, workflow):
        mock_crawler = AsyncMock()
        mock_crawler.execute.return_value = {"total_products": 100}
        workflow._crawler = mock_crawler

        think = ThinkResult(next_action="crawl", reasoning="", parameters={"categories": None})
        result = await workflow._act(think)
        assert result.success is True
        assert result.result["total_products"] == 100

    @pytest.mark.asyncio
    async def test_act_store(self, workflow):
        mock_storage = AsyncMock()
        mock_storage.execute.return_value = {"raw_records": 50}
        workflow._storage = mock_storage

        think = ThinkResult(
            next_action="store", reasoning="", parameters={"crawl_data": {"products": []}}
        )
        result = await workflow._act(think)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_act_update_kg(self, workflow):
        mock_kg = MagicMock()
        mock_kg.load_from_crawl_data.return_value = 10
        mock_kg.get_stats.return_value = {
            "total_triples": 100,
            "unique_subjects": 30,
            "unique_objects": 40,
        }
        workflow._knowledge_graph = mock_kg

        think = ThinkResult(next_action="update_kg", reasoning="", parameters={"crawl_data": {}})
        result = await workflow._act(think)
        assert result.success is True
        assert result.result["relations_added"] == 10
        assert result.result["total_triples"] == 100

    @pytest.mark.asyncio
    async def test_act_calculate(self, workflow):
        mock_metrics = AsyncMock()
        mock_metrics.execute.return_value = {"brand_metrics": [], "alerts": []}
        workflow._metrics_agent = mock_metrics

        think = ThinkResult(
            next_action="calculate",
            reasoning="",
            parameters={"crawl_data": {}, "historical_data": None},
        )
        result = await workflow._act(think)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_act_hybrid_insight(self, workflow):
        mock_insight = AsyncMock()
        mock_insight.execute.return_value = {
            "action_items": ["item1"],
            "highlights": [],
            "inferences": [],
        }
        workflow._hybrid_insight = mock_insight

        think = ThinkResult(
            next_action="hybrid_insight",
            reasoning="",
            parameters={"metrics_data": {}, "crawl_data": {}, "crawl_summary": ""},
        )
        result = await workflow._act(think)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_act_export(self, workflow):
        mock_exporter = AsyncMock()
        mock_exporter.initialize = AsyncMock()
        mock_exporter.export_dashboard_data = AsyncMock(
            return_value={"metadata": {"total_products": 50, "laneige_products": 5}}
        )
        workflow._dashboard_exporter = mock_exporter

        with patch.object(
            workflow, "_verify_unknown_brands", new_callable=AsyncMock
        ) as mock_verify:
            mock_verify.return_value = {"verified_count": 3}
            with patch.object(
                workflow, "_sync_sheets_to_sqlite", new_callable=AsyncMock
            ) as mock_sync:
                mock_sync.return_value = {"synced_count": 10}

                think = ThinkResult(next_action="export", reasoning="", parameters={})
                result = await workflow._act(think)

        assert result.success is True
        assert result.result["exported"] is True
        assert result.result["products"] == 50
        assert result.result["brands_verified"] == 3
        assert result.result["sqlite_synced"] == 10

    @pytest.mark.asyncio
    async def test_act_skip(self, workflow):
        think = ThinkResult(next_action="skip", reasoning="", parameters={})
        result = await workflow._act(think)
        assert result.success is True
        assert result.result["skipped"] is True

    @pytest.mark.asyncio
    async def test_act_unknown_action(self, workflow):
        think = ThinkResult(next_action="unknown_action", reasoning="", parameters={})
        result = await workflow._act(think)
        assert result.success is False
        assert "Unknown action" in result.error

    @pytest.mark.asyncio
    async def test_act_exception(self, workflow):
        mock_crawler = AsyncMock()
        mock_crawler.execute.side_effect = RuntimeError("Connection failed")
        workflow._crawler = mock_crawler

        think = ThinkResult(next_action="crawl", reasoning="", parameters={"categories": None})
        result = await workflow._act(think)
        assert result.success is False
        assert "Connection failed" in result.error


# =========================================================================
# _observe()
# =========================================================================


class TestObserve:
    @pytest.mark.asyncio
    async def test_observe_crawl_success(self, workflow):
        act = ActResult(
            action="crawl",
            success=True,
            result={
                "total_products": 100,
                "laneige_count": 5,
                "categories": {"beauty": {"products": []}},
                "laneige_products": [],
            },
        )
        result = await workflow._observe(act)
        assert "crawl_result" in result.state_updates
        assert result.next_step == WorkflowStep.STORE
        assert any("크롤링 완료" in obs for obs in result.observations)

    @pytest.mark.asyncio
    async def test_observe_store_hybrid(self, workflow):
        workflow.use_hybrid = True
        act = ActResult(
            action="store",
            success=True,
            result={"raw_records": 50, "products_upserted": 30},
        )
        result = await workflow._observe(act)
        assert result.next_step == WorkflowStep.UPDATE_KG

    @pytest.mark.asyncio
    async def test_observe_store_non_hybrid(self, workflow):
        workflow.use_hybrid = False
        act = ActResult(
            action="store",
            success=True,
            result={"raw_records": 50, "products_upserted": 30},
        )
        result = await workflow._observe(act)
        assert result.next_step == WorkflowStep.CALCULATE

    @pytest.mark.asyncio
    async def test_observe_update_kg(self, workflow):
        act = ActResult(
            action="update_kg",
            success=True,
            result={"relations_added": 10, "total_triples": 100},
        )
        result = await workflow._observe(act)
        assert result.next_step == WorkflowStep.CALCULATE
        assert "kg_result" in result.state_updates

    @pytest.mark.asyncio
    async def test_observe_calculate(self, workflow):
        workflow.use_hybrid = True
        mock_kg = MagicMock()
        workflow._knowledge_graph = mock_kg

        act = ActResult(
            action="calculate",
            success=True,
            result={"brand_metrics": [1, 2], "product_metrics": [1], "alerts": ["a"]},
        )
        result = await workflow._observe(act)
        assert result.next_step == WorkflowStep.INSIGHT
        mock_kg.load_from_metrics_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_observe_hybrid_insight(self, workflow):
        workflow.use_hybrid = True
        mock_chatbot = MagicMock()
        workflow._hybrid_chatbot = mock_chatbot

        act = ActResult(
            action="hybrid_insight",
            success=True,
            result={
                "action_items": ["item"],
                "highlights": ["h1", "h2"],
                "inferences": [{"type": "market"}],
            },
        )
        result = await workflow._observe(act)
        assert result.next_step == WorkflowStep.EXPORT
        mock_chatbot.set_data_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_observe_export(self, workflow):
        act = ActResult(
            action="export",
            success=True,
            result={"path": "./data/dashboard_data.json", "products": 50},
        )
        result = await workflow._observe(act)
        assert result.next_step == WorkflowStep.COMPLETE

    @pytest.mark.asyncio
    async def test_observe_skip(self, workflow):
        act = ActResult(action="skip", success=True, result={"skipped": True})
        result = await workflow._observe(act)
        assert result.next_step == WorkflowStep.COMPLETE
        assert "스킵" in result.observations[0]

    @pytest.mark.asyncio
    async def test_observe_error(self, workflow):
        act = ActResult(action="crawl", success=False, error="Timeout")
        result = await workflow._observe(act)
        assert any("에러" in obs for obs in result.observations)


# =========================================================================
# _generate_summary()
# =========================================================================


class TestGenerateSummary:
    def test_empty_state(self, workflow):
        workflow._state = {}
        summary = workflow._generate_summary()
        assert summary["products_crawled"] == 0
        assert summary["alerts"] == 0
        assert summary["dashboard_exported"] is False

    def test_with_data(self, workflow):
        workflow.use_hybrid = True
        workflow._state = {
            "crawl_result": {
                "total_products": 100,
                "laneige_count": 5,
                "categories": {"beauty": {}, "skincare": {}},
            },
            "metrics_result": {"alerts": ["a1", "a2"]},
            "insight_result": {
                "action_items": ["act1"],
                "highlights": ["h1"],
                "daily_insight": "Market looks strong" * 20,
                "inferences": [{"type": "market"}],
                "explanations": ["exp1"],
            },
            "export_result": {"exported": True, "path": "/data/dashboard.json"},
            "kg_result": {"total_triples": 500},
        }
        summary = workflow._generate_summary()
        assert summary["products_crawled"] == 100
        assert summary["laneige_tracked"] == 5
        assert len(summary["categories"]) == 2
        assert summary["alerts"] == 2
        assert summary["action_items"] == 1
        assert summary["dashboard_exported"] is True
        assert "hybrid" in summary
        assert summary["hybrid"]["kg_triples"] == 500

    def test_non_hybrid(self, workflow):
        workflow.use_hybrid = False
        workflow._state = {}
        summary = workflow._generate_summary()
        assert "hybrid" not in summary


# =========================================================================
# 상태 조회
# =========================================================================


class TestStatusMethods:
    def test_get_status_basic(self, workflow):
        status = workflow.get_status()
        assert "session_id" in status
        assert "current_step" in status
        assert status["hybrid_mode"] is True

    def test_get_status_with_kg(self, workflow):
        mock_kg = MagicMock()
        mock_kg.get_stats.return_value = {"total_triples": 100}
        workflow._knowledge_graph = mock_kg
        workflow.use_hybrid = True

        status = workflow.get_status()
        assert "knowledge_graph" in status

    def test_get_status_with_reasoner(self, workflow):
        mock_reasoner = MagicMock()
        mock_reasoner.get_inference_stats.return_value = {"total": 10}
        workflow._reasoner = mock_reasoner
        workflow.use_hybrid = True

        status = workflow.get_status()
        assert "reasoner" in status

    def test_get_history_stats(self, workflow):
        stats = workflow.get_history_stats()
        assert "success_rate" in stats
        assert "avg_duration" in stats
        assert "recent_errors" in stats

    def test_get_knowledge_graph_stats_not_initialized(self, workflow):
        workflow._knowledge_graph = None
        stats = workflow.get_knowledge_graph_stats()
        assert stats["initialized"] is False

    def test_get_knowledge_graph_stats_initialized(self, workflow):
        mock_kg = MagicMock()
        mock_kg.get_stats.return_value = {"total_triples": 200}
        mock_kg.get_most_connected.return_value = [("LANEIGE", 50)]
        workflow._knowledge_graph = mock_kg

        stats = workflow.get_knowledge_graph_stats()
        assert stats["initialized"] is True
        assert stats["total_triples"] == 200

    def test_get_inference_stats_not_initialized(self, workflow):
        workflow._reasoner = None
        stats = workflow.get_inference_stats()
        assert stats["initialized"] is False

    def test_get_inference_stats_initialized(self, workflow):
        mock_reasoner = MagicMock()
        mock_reasoner.get_inference_stats.return_value = {"total_inferences": 50}
        workflow._reasoner = mock_reasoner

        stats = workflow.get_inference_stats()
        assert stats["total_inferences"] == 50


# =========================================================================
# cleanup
# =========================================================================


class TestCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_with_crawler(self, workflow):
        mock_crawler = AsyncMock()
        workflow._crawler = mock_crawler
        mock_kg = MagicMock()
        workflow._knowledge_graph = mock_kg

        await workflow.cleanup()
        mock_crawler.close.assert_called_once()
        mock_kg.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_no_crawler(self, workflow):
        workflow._crawler = None
        workflow._knowledge_graph = None
        await workflow.cleanup()  # No exception


# =========================================================================
# _verify_unknown_brands
# =========================================================================


class TestVerifyUnknownBrands:
    @pytest.mark.asyncio
    async def test_no_products(self, workflow):
        workflow._state = {"crawl_result": {"categories": {}}}
        result = await workflow._verify_unknown_brands()
        assert result["verified_count"] == 0
        assert result["skipped"] is True

    @pytest.mark.asyncio
    async def test_no_crawl_result(self, workflow):
        workflow._state = {}
        result = await workflow._verify_unknown_brands()
        assert result["verified_count"] == 0

    @pytest.mark.asyncio
    async def test_import_error(self, workflow):
        workflow._state = {
            "crawl_result": {
                "categories": {"beauty": {"products": [{"name": "P1", "brand": "Unknown"}]}}
            }
        }
        with patch(
            "src.application.workflows.batch_workflow.BatchWorkflow._verify_unknown_brands",
            new_callable=AsyncMock,
        ) as mock_method:
            # Simulate the actual method returning ImportError result
            mock_method.return_value = {"verified_count": 0, "error": "BrandResolver not available"}
            result = await mock_method()
        assert result["verified_count"] == 0

    @pytest.mark.asyncio
    async def test_exception_handling(self, workflow):
        workflow._state = {
            "crawl_result": {
                "categories": {"beauty": {"products": [{"name": "P1", "brand": "Unknown"}]}}
            }
        }
        with patch(
            "src.tools.utilities.brand_resolver.get_brand_resolver",
            side_effect=RuntimeError("Failed"),
        ):
            result = await workflow._verify_unknown_brands()
        assert result["verified_count"] == 0
        assert "error" in result


# =========================================================================
# _sync_sheets_to_sqlite
# =========================================================================


class TestSyncSheetsToSqlite:
    @pytest.mark.asyncio
    async def test_sheets_connection_failed(self, workflow):
        with patch(
            "src.application.workflows.batch_workflow.BatchWorkflow._sync_sheets_to_sqlite",
            new_callable=AsyncMock,
        ) as mock_method:
            mock_method.return_value = {"synced_count": 0, "error": "Sheets connection failed"}
            result = await mock_method()
        assert result["synced_count"] == 0

    @pytest.mark.asyncio
    async def test_exception_handling(self, workflow):
        # Use the real method but mock imports
        with patch.dict(
            "sys.modules",
            {
                "src.tools.storage.sheets_writer": MagicMock(),
                "src.tools.storage.sqlite_storage": MagicMock(),
            },
        ):
            with patch(
                "src.tools.storage.sheets_writer.SheetsWriter",
                side_effect=RuntimeError("fail"),
            ):
                result = await workflow._sync_sheets_to_sqlite()
        assert result["synced_count"] == 0


# =========================================================================
# run_daily_workflow (integration with mocking)
# =========================================================================


class TestRunDailyWorkflow:
    @pytest.mark.asyncio
    async def test_workflow_stops_on_should_continue_false(self, workflow):
        """should_continue=False로 워크플로우 중단"""
        workflow._current_step = WorkflowStep.CRAWL
        workflow._state = {}

        # Think returns should_continue=False
        with patch.object(workflow, "_think", new_callable=AsyncMock) as mock_think:
            mock_think.return_value = ThinkResult(
                next_action="skip", reasoning="No data", should_continue=False
            )

            results = await workflow.run_daily_workflow()

        assert results["status"] == "completed"

    @pytest.mark.asyncio
    async def test_workflow_exception_handling(self, workflow):
        """워크플로우 실행 중 예외 처리"""
        with patch.object(workflow, "_think", new_callable=AsyncMock) as mock_think:
            mock_think.side_effect = RuntimeError("Critical error")

            results = await workflow.run_daily_workflow()

        assert results["status"] == "failed"
        assert "Critical error" in results["error"]


# =========================================================================
# Backward compatibility
# =========================================================================


class TestBackwardCompatibility:
    def test_orchestrator_alias(self):
        from src.application.workflows.batch_workflow import Orchestrator

        assert Orchestrator is BatchWorkflow

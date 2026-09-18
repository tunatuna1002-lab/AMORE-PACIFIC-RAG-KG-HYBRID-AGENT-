"""
Phase 5 지표 영속화 테스트 (§5, 결정 D3)

STORE_METRICS 스텝이 CALCULATE와 INSIGHT 사이에 실제로 들어가고,
MetricsAgent 산출 키가 SQLite 컬럼명으로 올바르게 매핑되는지 검증한다.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.workflows.batch_workflow import BatchWorkflow, WorkflowStep


@pytest.fixture
def workflow():
    wf = BatchWorkflow.__new__(BatchWorkflow)
    wf.logger = MagicMock()
    wf.spreadsheet_id = None
    return wf


METRICS_DATA = {
    "calculated_at": "2026-08-31T02:00:00",
    "brand_metrics": [
        {
            "brand_name": "LANEIGE",
            "category_id": "lip_care",
            "share_of_shelf": 2.0,
            "avg_rank": 26.2,
            "product_count": 2,
            "top10_count": 1,
            "is_laneige": True,
        },
        {
            "brand_name": "COSRX",
            "category_id": "lip_care",
            "share_of_shelf": 5.0,
            "avg_rank": 40.0,
            "product_count": 5,
        },
        # brand_name 없는 행은 건너뛴다
        {"category_id": "lip_care", "share_of_shelf": 1.0},
    ],
    "market_metrics": [
        {
            "category_id": "lip_care",
            "hhi": 0.0681,
            "churn_rate_7d": None,
            "avg_rating_gap": 0.1,
        },
        # category_id 없는 행은 건너뛴다
        {"hhi": 0.05},
    ],
}


class TestWorkflowStepOrder:
    def test_store_metrics_step_exists(self):
        assert WorkflowStep.STORE_METRICS.value == "store_metrics"

    def test_step_sits_between_calculate_and_insight(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[3]
            / "src"
            / "application"
            / "workflows"
            / "batch_workflow.py"
        ).read_text(encoding="utf-8")
        calc = src.index("WorkflowStep.CALCULATE.value,")
        store = src.index("WorkflowStep.STORE_METRICS.value,")
        insight = src.index("WorkflowStep.INSIGHT.value,")
        assert calc < store < insight


class TestStoreMetrics:
    @pytest.mark.asyncio
    async def test_maps_agent_keys_to_db_columns(self, workflow):
        storage = MagicMock()
        storage.initialize = AsyncMock()
        storage.save_brand_metrics = AsyncMock(return_value=2)
        storage.save_market_metrics = AsyncMock(return_value=1)

        with patch("src.tools.storage.sqlite_storage.get_sqlite_storage", return_value=storage):
            result = await workflow._store_metrics(METRICS_DATA, {"snapshot_date": "2026-08-31"})

        brand_rows = storage.save_brand_metrics.await_args.args[0]
        assert len(brand_rows) == 2  # brand_name 없는 행 제외
        laneige = next(r for r in brand_rows if r["brand"] == "LANEIGE")
        assert laneige["sos"] == 2.0  # share_of_shelf → sos
        assert laneige["brand_avg_rank"] == 26.2  # avg_rank → brand_avg_rank
        assert laneige["snapshot_date"] == "2026-08-31"

        market_rows = storage.save_market_metrics.await_args.args[0]
        assert len(market_rows) == 1  # category_id 없는 행 제외
        assert market_rows[0]["hhi"] == 0.0681
        assert market_rows[0]["churn_rate"] is None  # churn_rate_7d → churn_rate

        assert result == {"brand_rows": 2, "market_rows": 1, "snapshot_date": "2026-08-31"}

    @pytest.mark.asyncio
    async def test_empty_metrics_skips_writes(self, workflow):
        storage = MagicMock()
        storage.initialize = AsyncMock()
        storage.save_brand_metrics = AsyncMock()
        storage.save_market_metrics = AsyncMock()

        with patch("src.tools.storage.sqlite_storage.get_sqlite_storage", return_value=storage):
            result = await workflow._store_metrics({}, {})

        storage.save_brand_metrics.assert_not_awaited()
        storage.save_market_metrics.assert_not_awaited()
        assert result["brand_rows"] == 0


class TestResolveSnapshotDate:
    def test_prefers_crawl_snapshot_date(self, workflow):
        assert workflow._resolve_snapshot_date({}, {"snapshot_date": "2026-08-30"}) == "2026-08-30"

    def test_falls_back_to_crawl_date(self, workflow):
        assert workflow._resolve_snapshot_date({}, {"crawl_date": "2026-08-29"}) == "2026-08-29"

    def test_falls_back_to_calculated_at(self, workflow):
        assert (
            workflow._resolve_snapshot_date({"calculated_at": "2026-08-28T01:00:00"}, {})
            == "2026-08-28"
        )

    def test_defaults_to_today(self, workflow):
        from datetime import datetime

        assert workflow._resolve_snapshot_date({}, {}) == datetime.now().strftime("%Y-%m-%d")


class TestDeadParallelImplementationRemoved:
    """§5.3 / §5.4: 죽은 병렬 구현 제거"""

    def test_crawl_workflow_module_gone(self):
        with pytest.raises(ModuleNotFoundError):
            __import__("src.application.workflows.crawl_workflow")

    def test_container_factory_gone(self):
        from src.infrastructure.container import Container

        assert not hasattr(Container, "get_crawl_workflow")

    def test_storage_agent_save_metrics_gone(self):
        from src.agents.storage_agent import StorageAgent

        assert not hasattr(StorageAgent, "save_metrics")

    def test_protocol_no_longer_declares_save_metrics(self):
        from src.domain.interfaces.agent import StorageAgentProtocol

        assert not hasattr(StorageAgentProtocol, "save_metrics")

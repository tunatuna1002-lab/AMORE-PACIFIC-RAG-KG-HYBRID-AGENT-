"""
CrawlManager 단위 테스트
============================
src/core/crawl_manager.py 커버리지 15% → 50%+ 목표

테스트 대상:
- Enum (CrawlStatus)
- Dataclass (CrawlState, to_dict)
- CrawlManager 초기화, _load_state, _save_state (atomic writes)
- get_kst_today, get_data_date, is_today_data_available
- is_crawling (stale lock 감지 포함)
- needs_crawl, needs_crawl_with_sheets_check
- check_sheets_data_exists
- should_notify, mark_notified
- get_status_message, get_notification_message
- start_crawl, _run_crawl (complete lifecycle)
- Singleton pattern (get_crawl_manager)
- Edge cases and error handling
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.crawl_manager import (
    CrawlManager,
    CrawlState,
    CrawlStatus,
    get_crawl_manager,
)

# =========================================================================
# Fixtures
# =========================================================================


KST = timezone(timedelta(hours=9))


@pytest.fixture
def manager(tmp_path):
    """CrawlManager with tmp state/data files"""
    state_file = str(tmp_path / "crawl_state.json")
    data_file = str(tmp_path / "dashboard_data.json")

    with patch.object(CrawlManager, "STATE_FILE", state_file):
        with patch.object(CrawlManager, "DATA_FILE", data_file):
            mgr = CrawlManager()
            mgr.STATE_FILE = state_file
            mgr.DATA_FILE = data_file
    return mgr


# =========================================================================
# Enum / Dataclass
# =========================================================================


class TestCrawlStatus:
    def test_values(self):
        assert CrawlStatus.IDLE.value == "idle"
        assert CrawlStatus.RUNNING.value == "running"
        assert CrawlStatus.COMPLETED.value == "completed"
        assert CrawlStatus.FAILED.value == "failed"


class TestCrawlState:
    def test_defaults(self):
        state = CrawlState()
        assert state.status == CrawlStatus.IDLE
        assert state.date is None
        assert state.progress == 0
        assert state.products_collected == 0
        assert state.error is None

    def test_to_dict(self):
        state = CrawlState(
            status=CrawlStatus.COMPLETED,
            date="2025-01-15",
            started_at="2025-01-15T22:00:00",
            completed_at="2025-01-15T22:30:00",
            progress=100,
            categories_done=5,
            categories_total=5,
            products_collected=500,
        )
        d = state.to_dict()
        assert d["status"] == "completed"
        assert d["date"] == "2025-01-15"
        assert d["progress"] == 100
        assert d["products_collected"] == 500
        assert d["error"] is None


# =========================================================================
# CrawlManager 초기화
# =========================================================================


class TestCrawlManagerInit:
    def test_default_state(self, manager):
        assert manager.state.status == CrawlStatus.IDLE
        assert manager._crawl_task is None

    def test_load_state_from_file(self, tmp_path):
        state_file = str(tmp_path / "crawl_state.json")
        data = {
            "status": "completed",
            "date": "2025-01-15",
            "started_at": "2025-01-15T22:00:00",
            "completed_at": "2025-01-15T22:30:00",
            "progress": 100,
            "categories_done": 5,
            "categories_total": 5,
            "products_collected": 500,
        }
        with open(state_file, "w") as f:
            json.dump(data, f)

        with patch.object(CrawlManager, "STATE_FILE", state_file):
            mgr = CrawlManager()
        assert mgr.state.status == CrawlStatus.COMPLETED
        assert mgr.state.products_collected == 500

    def test_load_state_corrupt_file(self, tmp_path):
        state_file = str(tmp_path / "crawl_state.json")
        with open(state_file, "w") as f:
            f.write("not json!")

        with patch.object(CrawlManager, "STATE_FILE", state_file):
            mgr = CrawlManager()
        assert mgr.state.status == CrawlStatus.IDLE

    def test_load_state_no_file(self, tmp_path):
        state_file = str(tmp_path / "nonexistent.json")
        with patch.object(CrawlManager, "STATE_FILE", state_file):
            mgr = CrawlManager()
        assert mgr.state.status == CrawlStatus.IDLE


# =========================================================================
# _save_state
# =========================================================================


class TestSaveState:
    def test_save_creates_file(self, manager):
        manager.state = CrawlState(
            status=CrawlStatus.RUNNING,
            date="2025-01-15",
            progress=50,
        )
        manager._save_state()

        assert Path(manager.STATE_FILE).exists()
        with open(manager.STATE_FILE) as f:
            data = json.load(f)
        assert data["status"] == "running"
        assert data["progress"] == 50


# =========================================================================
# get_kst_today, get_data_date
# =========================================================================


class TestDateMethods:
    def test_get_kst_today(self, manager):
        today = manager.get_kst_today()
        # Should be a valid ISO date string
        datetime.fromisoformat(today)
        assert len(today) == 10  # YYYY-MM-DD

    def test_get_data_date_no_file(self, manager):
        result = manager.get_data_date()
        assert result is None

    def test_get_data_date_with_file(self, manager):
        data = {"metadata": {"data_date": "2025-01-15"}}
        with open(manager.DATA_FILE, "w") as f:
            json.dump(data, f)

        result = manager.get_data_date()
        assert result == "2025-01-15"

    def test_get_data_date_corrupt_file(self, manager):
        with open(manager.DATA_FILE, "w") as f:
            f.write("not json!")
        result = manager.get_data_date()
        assert result is None

    def test_is_today_data_available_true(self, manager):
        today = manager.get_kst_today()
        data = {"metadata": {"data_date": today}}
        with open(manager.DATA_FILE, "w") as f:
            json.dump(data, f)

        assert manager.is_today_data_available() is True

    def test_is_today_data_available_false(self, manager):
        data = {"metadata": {"data_date": "2020-01-01"}}
        with open(manager.DATA_FILE, "w") as f:
            json.dump(data, f)

        assert manager.is_today_data_available() is False


# =========================================================================
# is_crawling
# =========================================================================


class TestIsCrawling:
    def test_idle_not_crawling(self, manager):
        manager.state.status = CrawlStatus.IDLE
        assert manager.is_crawling() is False

    def test_running_is_crawling(self, manager):
        manager.state.status = CrawlStatus.RUNNING
        manager.state.started_at = datetime.now(KST).isoformat()
        assert manager.is_crawling() is True

    def test_stale_lock_detected(self, manager):
        manager.state.status = CrawlStatus.RUNNING
        manager.state.started_at = (datetime.now(KST) - timedelta(hours=3)).isoformat()
        assert manager.is_crawling() is False
        assert manager.state.status == CrawlStatus.FAILED
        assert "Stale lock" in manager.state.error

    def test_invalid_started_at(self, manager):
        manager.state.status = CrawlStatus.RUNNING
        manager.state.started_at = "not-a-date"
        assert manager.is_crawling() is False
        assert manager.state.status == CrawlStatus.FAILED

    def test_running_no_started_at(self, manager):
        manager.state.status = CrawlStatus.RUNNING
        manager.state.started_at = None
        assert manager.is_crawling() is True


# =========================================================================
# needs_crawl
# =========================================================================


class TestNeedsCrawl:
    def test_needs_crawl_no_data(self, manager):
        # No data file, not running, not completed today
        assert manager.needs_crawl() is True

    def test_needs_crawl_already_running(self, manager):
        manager.state.status = CrawlStatus.RUNNING
        manager.state.started_at = datetime.now(KST).isoformat()
        assert manager.needs_crawl() is False

    def test_needs_crawl_today_data(self, manager):
        today = manager.get_kst_today()
        data = {"metadata": {"data_date": today}}
        with open(manager.DATA_FILE, "w") as f:
            json.dump(data, f)
        assert manager.needs_crawl() is False

    def test_needs_crawl_completed_today(self, manager):
        today = manager.get_kst_today()
        manager.state.status = CrawlStatus.COMPLETED
        manager.state.date = today
        assert manager.needs_crawl() is False


# =========================================================================
# should_notify / mark_notified
# =========================================================================


class TestNotification:
    def test_should_notify_completed_today(self, manager):
        today = manager.get_kst_today()
        manager.state.status = CrawlStatus.COMPLETED
        manager.state.date = today
        manager.state.notified_sessions = set()
        assert manager.should_notify("sess-1") is True

    def test_should_notify_already_notified(self, manager):
        today = manager.get_kst_today()
        manager.state.status = CrawlStatus.COMPLETED
        manager.state.date = today
        manager.state.notified_sessions = {"sess-1"}
        assert manager.should_notify("sess-1") is False

    def test_should_notify_not_completed(self, manager):
        manager.state.status = CrawlStatus.RUNNING
        assert manager.should_notify("sess-1") is False

    def test_should_notify_wrong_date(self, manager):
        manager.state.status = CrawlStatus.COMPLETED
        manager.state.date = "2020-01-01"
        assert manager.should_notify("sess-1") is False

    def test_mark_notified(self, manager):
        manager.mark_notified("sess-1")
        assert "sess-1" in manager.state.notified_sessions


# =========================================================================
# get_status_message / get_notification_message
# =========================================================================


class TestStatusMessages:
    def test_idle_with_data(self, manager):
        data = {"metadata": {"data_date": "2025-01-15"}}
        with open(manager.DATA_FILE, "w") as f:
            json.dump(data, f)
        manager.state.status = CrawlStatus.IDLE
        msg = manager.get_status_message()
        assert "2025-01-15" in msg

    def test_idle_no_data(self, manager):
        manager.state.status = CrawlStatus.IDLE
        msg = manager.get_status_message()
        assert "데이터 없음" in msg

    def test_running(self, manager):
        manager.state.status = CrawlStatus.RUNNING
        manager.state.progress = 50
        msg = manager.get_status_message()
        assert "50%" in msg

    def test_completed(self, manager):
        manager.state.status = CrawlStatus.COMPLETED
        manager.state.products_collected = 100
        msg = manager.get_status_message()
        assert "100" in msg

    def test_failed(self, manager):
        manager.state.status = CrawlStatus.FAILED
        manager.state.error = "Timeout"
        msg = manager.get_status_message()
        assert "Timeout" in msg

    def test_notification_message(self, manager):
        manager.state.date = "2025-01-15"
        manager.state.products_collected = 500
        manager.state.categories_done = 5
        manager.state.completed_at = "2025-01-15T22:30:00"
        msg = manager.get_notification_message()
        assert "500" in msg
        assert "5" in msg


# =========================================================================
# start_crawl
# =========================================================================


class TestStartCrawl:
    @pytest.mark.asyncio
    async def test_start_crawl_already_running(self, manager):
        manager.state.status = CrawlStatus.RUNNING
        manager.state.started_at = datetime.now(KST).isoformat()
        result = await manager.start_crawl()
        assert result is False

    @pytest.mark.asyncio
    async def test_start_crawl_creates_task(self, manager):
        with patch.object(manager, "_run_crawl", new_callable=AsyncMock):
            result = await manager.start_crawl()
        assert result is True
        assert manager._crawl_task is not None


# =========================================================================
# needs_crawl_with_sheets_check
# =========================================================================


class TestNeedsCrawlWithSheetsCheck:
    @pytest.mark.asyncio
    async def test_sheets_check_data_exists(self, manager):
        with patch.object(manager, "is_crawling", return_value=False):
            with patch.object(manager, "is_today_data_available", return_value=False):
                with patch.object(
                    manager, "check_sheets_data_exists", new_callable=AsyncMock
                ) as mock_check:
                    mock_check.return_value = True
                    manager.state.status = CrawlStatus.IDLE
                    result = await manager.needs_crawl_with_sheets_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_sheets_check_no_data(self, manager):
        with patch.object(manager, "is_crawling", return_value=False):
            with patch.object(manager, "is_today_data_available", return_value=False):
                with patch.object(
                    manager, "check_sheets_data_exists", new_callable=AsyncMock
                ) as mock_check:
                    mock_check.return_value = False
                    manager.state.status = CrawlStatus.IDLE
                    result = await manager.needs_crawl_with_sheets_check()
        assert result is True


# =========================================================================
# check_sheets_data_exists
# =========================================================================


class TestCheckSheetsDataExists:
    @pytest.mark.asyncio
    async def test_sheets_data_exists(self, manager):
        """Should return True when Sheets has data for target date"""
        mock_sheets = MagicMock()
        mock_sheets.initialize = AsyncMock()
        mock_sheets.get_rank_history = AsyncMock(
            return_value=[
                {"snapshot_date": "2025-06-15", "rank": 1},
                {"snapshot_date": "2025-06-15", "rank": 2},
            ]
        )

        with patch("src.tools.storage.sheets_writer.SheetsWriter", return_value=mock_sheets):
            result = await manager.check_sheets_data_exists("2025-06-15")

        assert result is True
        mock_sheets.initialize.assert_called_once()
        mock_sheets.get_rank_history.assert_called_once_with(days=1)

    @pytest.mark.asyncio
    async def test_sheets_data_not_exists(self, manager):
        """Should return False when no data for target date"""
        mock_sheets = MagicMock()
        mock_sheets.initialize = AsyncMock()
        mock_sheets.get_rank_history = AsyncMock(
            return_value=[{"snapshot_date": "2025-06-14", "rank": 1}]
        )

        with patch("src.tools.storage.sheets_writer.SheetsWriter", return_value=mock_sheets):
            result = await manager.check_sheets_data_exists("2025-06-15")

        assert result is False

    @pytest.mark.asyncio
    async def test_sheets_check_error(self, manager, caplog):
        """Should return False and log warning on error"""
        mock_sheets = MagicMock()
        mock_sheets.initialize = AsyncMock(side_effect=Exception("API error"))

        with patch("src.tools.storage.sheets_writer.SheetsWriter", return_value=mock_sheets):
            result = await manager.check_sheets_data_exists("2025-06-15")

        assert result is False
        assert "Failed to check Sheets data" in caplog.text


# =========================================================================
# _save_state atomic writes
# =========================================================================


class TestSaveStateAtomic:
    def test_atomic_write_uses_tempfile(self, manager, tmp_path):
        """Should use atomic write pattern (tempfile + os.replace)"""
        manager.state.status = CrawlStatus.RUNNING

        with (
            patch("tempfile.NamedTemporaryFile") as mock_tempfile,
            patch("os.replace") as mock_replace,
        ):
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = str(tmp_path / "temp.tmp")
            mock_tempfile.return_value = mock_file

            manager._save_state()

            mock_tempfile.assert_called_once()
            mock_replace.assert_called_once()

    def test_save_state_handles_error(self, manager, caplog):
        """Should log error on save failure"""
        with patch("tempfile.NamedTemporaryFile", side_effect=OSError("disk full")):
            manager._save_state()

        assert "Failed to save crawl state" in caplog.text

    def test_save_state_cleans_temp_on_error(self, manager, tmp_path):
        """Should clean up temp file on os.replace failure"""
        temp_file = tmp_path / "temp.tmp"
        temp_file.write_text("test")

        with (
            patch("tempfile.NamedTemporaryFile") as mock_tempfile,
            patch("os.replace", side_effect=OSError("replace failed")),
            patch("os.path.exists", return_value=True),
            patch("os.remove") as mock_remove,
        ):
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = str(temp_file)
            mock_tempfile.return_value = mock_file

            manager._save_state()

            # Should attempt cleanup
            mock_remove.assert_called()


# =========================================================================
# _run_crawl complete lifecycle
# =========================================================================


class TestRunCrawlLifecycle:
    @pytest.mark.asyncio
    async def test_run_crawl_success_flow(self, manager):
        """Should execute complete crawl workflow successfully"""
        fixed_time = datetime(2025, 6, 15, 22, 30, tzinfo=KST)

        # Mock all dependencies
        mock_crawler = MagicMock()
        mock_crawler.scraper = MagicMock()
        mock_crawler.scraper.initialize = AsyncMock()
        mock_crawler.scraper.close = AsyncMock()
        mock_crawler.execute = AsyncMock(
            return_value={
                "status": "success",
                "total_products": 500,
                "categories": {
                    "cat1": {"products": [{"name": "p1"}]},
                    "cat2": {"products": [{"name": "p2"}]},
                },
                "snapshot_date": "2025-06-15",
            }
        )

        mock_storage = MagicMock()
        mock_storage.execute = AsyncMock(return_value={"raw_records": 500, "errors": []})

        mock_exporter = MagicMock()
        mock_exporter.initialize = AsyncMock()
        mock_exporter.export_dashboard_data = AsyncMock()

        mock_brain = MagicMock()
        mock_brain._response_pipeline = MagicMock()
        mock_brain._response_pipeline._cache = MagicMock()
        mock_brain._response_pipeline._cache.clear = MagicMock()

        with (
            patch("src.core.crawl_manager.datetime") as mock_dt,
            patch(
                "src.infrastructure.container.Container.get_crawler_agent",
                return_value=mock_crawler,
            ),
            patch(
                "src.infrastructure.container.Container.get_storage_agent",
                return_value=mock_storage,
            ),
            patch(
                "src.tools.exporters.dashboard_exporter.DashboardExporter",
                return_value=mock_exporter,
            ),
            patch("src.core.brain.get_brain", return_value=mock_brain),
        ):
            mock_dt.now.return_value = fixed_time

            await manager._run_crawl()

        # Verify final state
        assert manager.state.status == CrawlStatus.COMPLETED
        assert manager.state.products_collected == 500
        assert manager.state.categories_done == 2
        assert manager.state.progress == 100
        assert manager.state.date == "2025-06-15"

        # Verify workflow calls
        mock_crawler.scraper.initialize.assert_called_once()
        mock_crawler.execute.assert_called_once()
        mock_storage.execute.assert_called_once()
        mock_exporter.export_dashboard_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_crawl_failure(self, manager):
        """Should handle crawl failure and set error state"""
        fixed_time = datetime(2025, 6, 15, 22, 30, tzinfo=KST)

        mock_crawler = MagicMock()
        mock_crawler.scraper = MagicMock()
        mock_crawler.scraper.initialize = AsyncMock()
        mock_crawler.scraper.close = AsyncMock()
        mock_crawler.execute = AsyncMock(return_value={"status": "failed"})

        with (
            patch("src.core.crawl_manager.datetime") as mock_dt,
            patch(
                "src.infrastructure.container.Container.get_crawler_agent",
                return_value=mock_crawler,
            ),
        ):
            mock_dt.now.return_value = fixed_time

            await manager._run_crawl()

        assert manager.state.status == CrawlStatus.FAILED
        assert "All categories failed" in manager.state.error

    @pytest.mark.asyncio
    async def test_run_crawl_calls_callback(self, manager):
        """Should invoke completion callback after success"""
        fixed_time = datetime(2025, 6, 15, 22, 30, tzinfo=KST)
        callback = AsyncMock()
        manager._on_complete_callback = callback

        mock_crawler = MagicMock()
        mock_crawler.scraper = MagicMock()
        mock_crawler.scraper.initialize = AsyncMock()
        mock_crawler.scraper.close = AsyncMock()
        mock_crawler.execute = AsyncMock(
            return_value={
                "status": "success",
                "total_products": 100,
                "categories": {"cat1": {"products": []}},
                "snapshot_date": "2025-06-15",
            }
        )

        mock_storage = MagicMock()
        mock_storage.execute = AsyncMock(return_value={"raw_records": 100, "errors": []})

        mock_exporter = MagicMock()
        mock_exporter.initialize = AsyncMock()
        mock_exporter.export_dashboard_data = AsyncMock()

        with (
            patch("src.core.crawl_manager.datetime") as mock_dt,
            patch(
                "src.infrastructure.container.Container.get_crawler_agent",
                return_value=mock_crawler,
            ),
            patch(
                "src.infrastructure.container.Container.get_storage_agent",
                return_value=mock_storage,
            ),
            patch(
                "src.tools.exporters.dashboard_exporter.DashboardExporter",
                return_value=mock_exporter,
            ),
            patch("src.core.brain.get_brain", return_value=None),
        ):
            mock_dt.now.return_value = fixed_time

            await manager._run_crawl()

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_crawl_handles_callback_error(self, manager, caplog):
        """Should continue even if callback fails"""
        fixed_time = datetime(2025, 6, 15, 22, 30, tzinfo=KST)
        callback = AsyncMock(side_effect=Exception("Callback error"))
        manager._on_complete_callback = callback

        mock_crawler = MagicMock()
        mock_crawler.scraper = MagicMock()
        mock_crawler.scraper.initialize = AsyncMock()
        mock_crawler.scraper.close = AsyncMock()
        mock_crawler.execute = AsyncMock(
            return_value={
                "status": "success",
                "total_products": 100,
                "categories": {"cat1": {"products": []}},
                "snapshot_date": "2025-06-15",
            }
        )

        mock_storage = MagicMock()
        mock_storage.execute = AsyncMock(return_value={"raw_records": 100, "errors": []})

        mock_exporter = MagicMock()
        mock_exporter.initialize = AsyncMock()
        mock_exporter.export_dashboard_data = AsyncMock()

        with (
            patch("src.core.crawl_manager.datetime") as mock_dt,
            patch(
                "src.infrastructure.container.Container.get_crawler_agent",
                return_value=mock_crawler,
            ),
            patch(
                "src.infrastructure.container.Container.get_storage_agent",
                return_value=mock_storage,
            ),
            patch(
                "src.tools.exporters.dashboard_exporter.DashboardExporter",
                return_value=mock_exporter,
            ),
            patch("src.core.brain.get_brain", return_value=None),
        ):
            mock_dt.now.return_value = fixed_time

            await manager._run_crawl()

        # Should still complete
        assert manager.state.status == CrawlStatus.COMPLETED
        assert "Complete callback error" in caplog.text

    @pytest.mark.asyncio
    async def test_run_crawl_handles_export_failure(self, manager, caplog):
        """Should continue if dashboard export fails (non-fatal)"""
        fixed_time = datetime(2025, 6, 15, 22, 30, tzinfo=KST)

        mock_crawler = MagicMock()
        mock_crawler.scraper = MagicMock()
        mock_crawler.scraper.initialize = AsyncMock()
        mock_crawler.scraper.close = AsyncMock()
        mock_crawler.execute = AsyncMock(
            return_value={
                "status": "success",
                "total_products": 100,
                "categories": {"cat1": {"products": []}},
                "snapshot_date": "2025-06-15",
            }
        )

        mock_storage = MagicMock()
        mock_storage.execute = AsyncMock(return_value={"raw_records": 100, "errors": []})

        mock_exporter = MagicMock()
        mock_exporter.initialize = AsyncMock(side_effect=Exception("Export failed"))

        with (
            patch("src.core.crawl_manager.datetime") as mock_dt,
            patch(
                "src.infrastructure.container.Container.get_crawler_agent",
                return_value=mock_crawler,
            ),
            patch(
                "src.infrastructure.container.Container.get_storage_agent",
                return_value=mock_storage,
            ),
            patch(
                "src.tools.exporters.dashboard_exporter.DashboardExporter",
                return_value=mock_exporter,
            ),
            patch("src.core.brain.get_brain", return_value=None),
        ):
            mock_dt.now.return_value = fixed_time

            await manager._run_crawl()

        # Should still complete
        assert manager.state.status == CrawlStatus.COMPLETED
        assert "Dashboard export failed (non-fatal)" in caplog.text

    @pytest.mark.asyncio
    async def test_run_crawl_saves_json_files(self, manager):
        """Should save crawl results to JSON files"""
        fixed_time = datetime(2025, 6, 15, 22, 30, tzinfo=KST)

        mock_crawler = MagicMock()
        mock_crawler.scraper = MagicMock()
        mock_crawler.scraper.initialize = AsyncMock()
        mock_crawler.scraper.close = AsyncMock()
        mock_crawler.execute = AsyncMock(
            return_value={
                "status": "success",
                "total_products": 100,
                "categories": {"cat1": {"products": [{"name": "p1"}]}},
                "snapshot_date": "2025-06-15",
            }
        )

        mock_storage = MagicMock()
        mock_storage.execute = AsyncMock(return_value={"raw_records": 100, "errors": []})

        mock_exporter = MagicMock()
        mock_exporter.initialize = AsyncMock()
        mock_exporter.export_dashboard_data = AsyncMock()

        with (
            patch("src.core.crawl_manager.datetime") as mock_dt,
            patch(
                "src.infrastructure.container.Container.get_crawler_agent",
                return_value=mock_crawler,
            ),
            patch(
                "src.infrastructure.container.Container.get_storage_agent",
                return_value=mock_storage,
            ),
            patch(
                "src.tools.exporters.dashboard_exporter.DashboardExporter",
                return_value=mock_exporter,
            ),
            patch("src.core.brain.get_brain", return_value=None),
            patch("builtins.open", create=True) as mock_open,
            patch("pathlib.Path.mkdir"),
        ):
            mock_dt.now.return_value = fixed_time

            await manager._run_crawl()

        # Should have attempted to write JSON files
        assert mock_open.called


# =========================================================================
# Singleton pattern
# =========================================================================


class TestSingletonPattern:
    @pytest.mark.asyncio
    async def test_get_crawl_manager_returns_instance(self):
        """Should return CrawlManager instance"""
        manager = await get_crawl_manager()
        assert isinstance(manager, CrawlManager)

    @pytest.mark.asyncio
    async def test_get_crawl_manager_singleton(self):
        """Should return same instance (singleton)"""
        from src.core import crawl_manager as cm_module

        cm_module._crawl_manager = None  # Reset

        manager1 = await get_crawl_manager()
        manager2 = await get_crawl_manager()

        assert manager1 is manager2

        # Cleanup
        cm_module._crawl_manager = None

    @pytest.mark.asyncio
    async def test_get_crawl_manager_thread_safe(self):
        """Should be thread-safe with asyncio.Lock"""
        from src.core import crawl_manager as cm_module

        cm_module._crawl_manager = None  # Reset

        # Call concurrently
        results = await asyncio.gather(
            get_crawl_manager(),
            get_crawl_manager(),
            get_crawl_manager(),
        )

        # All should be same instance
        assert results[0] is results[1]
        assert results[1] is results[2]

        # Cleanup
        cm_module._crawl_manager = None


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    def test_load_state_missing_fields(self, tmp_path):
        """Should handle state file with partial data"""
        state_file = str(tmp_path / "crawl_state.json")
        data = {"status": "completed"}  # Missing other fields

        with open(state_file, "w") as f:
            json.dump(data, f)

        with patch.object(CrawlManager, "STATE_FILE", state_file):
            with patch.object(CrawlManager, "DATA_FILE", str(tmp_path / "data.json")):
                mgr = CrawlManager()

        # Should use defaults for missing fields
        assert mgr.state.status == CrawlStatus.COMPLETED
        assert mgr.state.progress == 0

    def test_get_data_date_no_metadata(self, manager):
        """Should handle data file without metadata"""
        data = {"categories": {}}
        with open(manager.DATA_FILE, "w") as f:
            json.dump(data, f)

        result = manager.get_data_date()
        assert result is None

    def test_is_crawling_no_started_at_running(self, manager):
        """Should return True if RUNNING but no started_at"""
        manager.state.status = CrawlStatus.RUNNING
        manager.state.started_at = None

        # Should consider it running (no timestamp to check)
        assert manager.is_crawling() is True

    def test_needs_crawl_completed_yesterday(self, manager):
        """Should need crawl if completed yesterday"""
        yesterday = (datetime.now(KST) - timedelta(days=1)).date().isoformat()
        manager.state.status = CrawlStatus.COMPLETED
        manager.state.date = yesterday

        assert manager.needs_crawl() is True

    @pytest.mark.asyncio
    async def test_needs_crawl_with_sheets_already_completed_today(self, manager):
        """Should skip sheets check if already completed today"""
        today = manager.get_kst_today()
        manager.state.status = CrawlStatus.COMPLETED
        manager.state.date = today

        # Should not even call check_sheets_data_exists
        with patch.object(manager, "check_sheets_data_exists") as mock_check:
            result = await manager.needs_crawl_with_sheets_check()

        assert result is False
        mock_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_crawl_with_callback(self, manager):
        """Should store callback for later invocation"""
        callback = AsyncMock()

        with patch.object(manager, "_run_crawl", new_callable=AsyncMock):
            result = await manager.start_crawl(callback)

        assert result is True
        assert manager._on_complete_callback == callback

        # Cleanup task
        if manager._crawl_task and not manager._crawl_task.done():
            manager._crawl_task.cancel()
            try:
                await manager._crawl_task
            except asyncio.CancelledError:
                pass

    def test_notified_sessions_set_operations(self, manager):
        """Should support set operations on notified_sessions"""
        manager.mark_notified("sess-1")
        manager.mark_notified("sess-2")
        manager.mark_notified("sess-1")  # Duplicate

        assert len(manager.state.notified_sessions) == 2
        assert "sess-1" in manager.state.notified_sessions
        assert "sess-2" in manager.state.notified_sessions

    def test_state_persistence_roundtrip(self, manager):
        """Should preserve state through save/load cycle"""
        # Set complex state
        manager.state = CrawlState(
            status=CrawlStatus.COMPLETED,
            date="2025-06-15",
            started_at="2025-06-15T22:00:00+09:00",
            completed_at="2025-06-15T22:30:00+09:00",
            progress=100,
            categories_done=5,
            categories_total=5,
            products_collected=500,
            error=None,
        )
        manager.state.notified_sessions.add("sess-1")

        # Save
        manager._save_state()

        # Load new instance
        with (
            patch.object(CrawlManager, "STATE_FILE", manager.STATE_FILE),
            patch.object(CrawlManager, "DATA_FILE", manager.DATA_FILE),
        ):
            new_manager = CrawlManager()

        # Verify
        assert new_manager.state.status == CrawlStatus.COMPLETED
        assert new_manager.state.date == "2025-06-15"
        assert new_manager.state.products_collected == 500
        # Note: notified_sessions not persisted (session-specific)

"""
CrawlManager 단위 테스트
============================
src/core/crawl_manager.py 커버리지 15% → 50%+ 목표

테스트 대상:
- Enum (CrawlStatus)
- Dataclass (CrawlState, to_dict)
- CrawlManager 초기화, _load_state, _save_state
- get_kst_today, get_data_date, is_today_data_available
- is_crawling (stale lock 감지 포함)
- needs_crawl, should_notify, mark_notified
- get_status_message, get_notification_message
- start_crawl (mocked)
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.core.crawl_manager import (
    CrawlManager,
    CrawlState,
    CrawlStatus,
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

        from pathlib import Path

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

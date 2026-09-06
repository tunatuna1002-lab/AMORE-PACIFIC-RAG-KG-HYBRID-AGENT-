"""Regression tests for CrawlManager defects D12 and D13.

D12: a partial crawl (crawler status "partial") and storage errors were reported as
     COMPLETED, so ``needs_crawl()`` treated a half-collected day as fully done.
D13: ``start_crawl`` only schedules a task; callers need ``wait_for_completion`` to
     await the actual crawl (the scheduler was marking tasks done immediately).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.crawl_manager import CrawlManager, CrawlState, CrawlStatus
from src.shared.constants import KST

FIXED_TIME = datetime(2025, 6, 15, 22, 30, tzinfo=KST)


@pytest.fixture
def manager(tmp_path) -> CrawlManager:
    state_file = str(tmp_path / "crawl_state.json")
    data_file = str(tmp_path / "dashboard_data.json")
    with patch.object(CrawlManager, "STATE_FILE", state_file):
        with patch.object(CrawlManager, "DATA_FILE", data_file):
            mgr = CrawlManager()
            mgr.STATE_FILE = state_file
            mgr.DATA_FILE = data_file
    return mgr


def _crawler(result: dict[str, Any]) -> MagicMock:
    crawler = MagicMock()
    crawler.scraper = MagicMock()
    crawler.scraper.initialize = AsyncMock()
    crawler.scraper.close = AsyncMock()
    crawler.execute = AsyncMock(return_value=result)
    return crawler


def _storage(result: dict[str, Any]) -> MagicMock:
    storage = MagicMock()
    storage.execute = AsyncMock(return_value=result)
    return storage


def _patches(crawler: MagicMock, storage: MagicMock) -> "_PatchSet":
    exporter = MagicMock()
    exporter.initialize = AsyncMock()
    exporter.export_dashboard_data = AsyncMock()
    return _PatchSet(crawler, storage, exporter)


class _PatchSet:
    """Context manager bundling the Container/exporter/brain patches."""

    def __init__(self, crawler: MagicMock, storage: MagicMock, exporter: MagicMock):
        self._patches = [
            patch("src.core.crawl_manager.datetime"),
            patch(
                "src.infrastructure.container.Container.get_crawler_agent", return_value=crawler
            ),
            patch(
                "src.infrastructure.container.Container.get_storage_agent", return_value=storage
            ),
            patch(
                "src.tools.exporters.dashboard_exporter.DashboardExporter", return_value=exporter
            ),
            patch("src.core.brain.get_brain", return_value=None),
        ]

    def __enter__(self):
        mocks = [p.start() for p in self._patches]
        mocks[0].now.return_value = FIXED_TIME
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False


GOOD_CRAWL = {
    "status": "completed",
    "total_products": 100,
    "categories": {"cat1": {"products": []}},
    "snapshot_date": "2025-06-15",
}


@pytest.mark.asyncio
async def test_partial_crawl_is_not_completed(manager: CrawlManager) -> None:
    partial = {
        **GOOD_CRAWL,
        "status": "partial",
        "errors": ["lip_makeup: timeout"],
        "categories": {"cat1": {"products": []}},
    }
    with _patches(_crawler(partial), _storage({"raw_records": 100, "errors": []})):
        await manager._run_crawl()

    assert manager.state.status == CrawlStatus.PARTIAL
    assert manager.state.status != CrawlStatus.COMPLETED
    assert manager.state.errors  # non-empty
    assert manager.state.products_collected == 100
    assert manager.state.to_dict()["status"] == "partial"
    assert manager.state.to_dict()["errors"] == manager.state.errors


@pytest.mark.asyncio
async def test_storage_errors_are_recorded_and_not_completed(manager: CrawlManager) -> None:
    with _patches(
        _crawler(GOOD_CRAWL), _storage({"raw_records": 0, "errors": ["Sheets quota exceeded"]})
    ):
        await manager._run_crawl()

    assert manager.state.errors == ["Sheets quota exceeded"]
    assert manager.state.status != CrawlStatus.COMPLETED
    assert manager.state.status == CrawlStatus.PARTIAL


@pytest.mark.asyncio
async def test_clean_crawl_is_still_completed(manager: CrawlManager) -> None:
    with _patches(_crawler(GOOD_CRAWL), _storage({"raw_records": 100, "errors": []})):
        await manager._run_crawl()

    assert manager.state.status == CrawlStatus.COMPLETED
    assert manager.state.errors == []


def test_needs_crawl_true_when_today_is_partial(manager: CrawlManager) -> None:
    today = manager.get_kst_today()
    manager.state = CrawlState(status=CrawlStatus.PARTIAL, date=today, errors=["x"])
    with patch.object(manager, "get_data_date", return_value=today):
        # even though dashboard data carries today's date, a partial day must be retried
        assert manager.needs_crawl() is True

    manager.state = CrawlState(status=CrawlStatus.COMPLETED, date=today)
    with patch.object(manager, "get_data_date", return_value=today):
        assert manager.needs_crawl() is False


def test_partial_state_round_trips_through_state_file(manager: CrawlManager) -> None:
    manager.state = CrawlState(status=CrawlStatus.PARTIAL, date="2025-06-15", errors=["boom"])
    manager._save_state()

    reloaded = CrawlManager.__new__(CrawlManager)
    reloaded.STATE_FILE = manager.STATE_FILE
    reloaded.DATA_FILE = manager.DATA_FILE
    reloaded.state = CrawlState()
    reloaded._crawl_task = None
    reloaded._on_complete_callback = None
    reloaded._load_state()

    assert reloaded.state.status == CrawlStatus.PARTIAL
    assert reloaded.state.errors == ["boom"]


def test_should_notify_only_for_completed(manager: CrawlManager) -> None:
    today = manager.get_kst_today()
    manager.state = CrawlState(status=CrawlStatus.PARTIAL, date=today)
    assert manager.should_notify("s1") is False


def test_status_message_for_partial(manager: CrawlManager) -> None:
    manager.state = CrawlState(status=CrawlStatus.PARTIAL, products_collected=42, errors=["e"])
    msg = manager.get_status_message()
    assert "42" in msg
    assert msg != "알 수 없음"


# ---------------------------------------------------------------------------
# D13: wait_for_completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_for_completion_awaits_internal_task(manager: CrawlManager) -> None:
    finished = False

    async def fake_run() -> None:
        nonlocal finished
        manager.state.status = CrawlStatus.RUNNING
        await asyncio.sleep(0.02)
        manager.state.status = CrawlStatus.COMPLETED
        finished = True

    with patch.object(manager, "_run_crawl", side_effect=fake_run):
        assert await manager.start_crawl() is True
        assert finished is False
        assert await manager.wait_for_completion() is True
        assert finished is True


@pytest.mark.asyncio
async def test_wait_for_completion_returns_false_on_failure(manager: CrawlManager) -> None:
    async def fake_run() -> None:
        manager.state.status = CrawlStatus.FAILED
        manager.state.error = "boom"

    with patch.object(manager, "_run_crawl", side_effect=fake_run):
        await manager.start_crawl()
        assert await manager.wait_for_completion() is False


@pytest.mark.asyncio
async def test_wait_for_completion_timeout(manager: CrawlManager) -> None:
    async def fake_run() -> None:
        manager.state.status = CrawlStatus.RUNNING
        await asyncio.sleep(0.5)
        manager.state.status = CrawlStatus.COMPLETED

    with patch.object(manager, "_run_crawl", side_effect=fake_run):
        await manager.start_crawl()
        assert await manager.wait_for_completion(timeout=0.01) is False
        # the crawl task itself keeps running (not cancelled by a timeout)
        assert manager._crawl_task is not None and not manager._crawl_task.done()
        manager._crawl_task.cancel()


@pytest.mark.asyncio
async def test_wait_for_completion_without_task(manager: CrawlManager) -> None:
    assert manager._crawl_task is None
    assert await manager.wait_for_completion() is False

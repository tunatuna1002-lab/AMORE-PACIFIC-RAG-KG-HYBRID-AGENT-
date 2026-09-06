"""Regression test for defect D13.

The scheduler marks a task completed right after ``await callback(task)``; the brain's
``crawl_workflow`` handler only kicked off ``CrawlManager.start_crawl()`` (a background
task) and returned immediately, so the task was marked done before the crawl finished.
The handler now awaits ``CrawlManager.wait_for_completion()``.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.core.brain import UnifiedBrain
from src.core.scheduler import AutonomousScheduler

# ``src.core.scheduler.asyncio`` is the global asyncio module, so patching its ``sleep``
# also patches ``asyncio.sleep`` everywhere; keep a reference to the real coroutine.
_REAL_SLEEP = asyncio.sleep


class FakeCrawlManager:
    """start_crawl schedules a short background task; completion is observable."""

    def __init__(self, events: list[str]):
        self.events = events
        self._task: asyncio.Task | None = None
        self.completed = False

    def is_crawling(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _run(self) -> None:
        await _REAL_SLEEP(0.02)
        self.completed = True
        self.events.append("crawl_finished")

    async def start_crawl(self, on_complete: Any = None) -> bool:
        self.events.append("crawl_started")
        self._task = asyncio.create_task(self._run())
        return True

    async def wait_for_completion(self, timeout: float | None = None) -> bool:
        if self._task is None:
            return False
        await asyncio.wait_for(asyncio.shield(self._task), timeout)
        return self.completed


@pytest.fixture
def scheduler(tmp_path) -> AutonomousScheduler:
    state_file = str(tmp_path / "scheduler_state.json")
    with patch.object(AutonomousScheduler, "STATE_FILE", state_file):
        s = AutonomousScheduler()
    s.STATE_FILE = state_file
    return s


@pytest.mark.asyncio
async def test_scheduler_marks_crawl_completed_only_after_crawl_finishes(
    scheduler: AutonomousScheduler,
) -> None:
    events: list[str] = []
    fake_cm = FakeCrawlManager(events)

    brain = UnifiedBrain()
    brain.scheduler = scheduler
    brain.collect_market_intelligence = AsyncMock(return_value={"status": "ok"})

    task = {"id": "daily_crawl", "name": "Daily Crawl", "action": "crawl_workflow"}
    calls = 0

    def due_tasks(*_a: Any) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [task]
        scheduler.running = False
        return []

    def mark_completed(schedule_id: str) -> None:
        events.append(f"mark_completed:{schedule_id}")

    async def fast_sleep(_seconds: float) -> None:
        await _REAL_SLEEP(0)

    with (
        patch("src.core.crawl_manager.get_crawl_manager", AsyncMock(return_value=fake_cm)),
        patch("src.core.brain.get_brain", AsyncMock(return_value=brain)),
        patch.object(scheduler, "get_due_tasks", side_effect=due_tasks),
        patch.object(scheduler, "mark_completed", side_effect=mark_completed),
        patch("src.core.scheduler.asyncio.sleep", side_effect=fast_sleep),
    ):
        await brain.start_scheduler()
        loop_task = scheduler._task
        assert loop_task is not None
        await asyncio.wait_for(loop_task, timeout=2.0)

    assert fake_cm.completed is True
    assert events == ["crawl_started", "crawl_finished", "mark_completed:daily_crawl"]
    assert brain._stats["autonomous_tasks"] == 1

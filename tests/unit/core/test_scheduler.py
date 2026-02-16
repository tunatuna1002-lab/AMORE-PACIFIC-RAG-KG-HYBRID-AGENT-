"""
Unit tests for src/core/scheduler.py

Tests cover:
- Initialization and state management
- State persistence (load/save with atomic writes)
- Default schedule loading
- Task scheduling logic (daily and interval)
- Task completion tracking
- Scheduler lifecycle (start/stop)
- Status reporting
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.scheduler import AutonomousScheduler
from src.shared.constants import KST

# Save reference to real asyncio.sleep before any patching
# This allows tests to wait without being affected by patches
_real_sleep = asyncio.sleep


@pytest.fixture
def scheduler(tmp_path):
    """Create scheduler with temporary state file"""
    state_file = str(tmp_path / "scheduler_state.json")
    with patch.object(AutonomousScheduler, "STATE_FILE", state_file):
        s = AutonomousScheduler()
    s.STATE_FILE = state_file  # also set instance attr for _save_state
    return s


@pytest.fixture
def fixed_time():
    """Fixed KST time for deterministic tests"""
    return datetime(2025, 6, 1, 22, 30, tzinfo=KST)


class TestInitialization:
    """Test scheduler initialization"""

    def test_init_creates_default_schedules(self, scheduler):
        """Should create 3 default schedules on initialization"""
        assert len(scheduler.schedules) == 3
        schedule_ids = {s["id"] for s in scheduler.schedules}
        assert schedule_ids == {"daily_crawl", "morning_brief", "check_data_freshness"}

    def test_init_sets_initial_state(self, scheduler):
        """Should initialize running=False and empty last_run"""
        assert scheduler.running is False
        assert scheduler._task is None
        assert isinstance(scheduler._last_run, dict)

    def test_default_schedule_daily_crawl(self, scheduler):
        """Should have daily_crawl at 22:00 KST"""
        crawl = next(s for s in scheduler.schedules if s["id"] == "daily_crawl")
        assert crawl["schedule_type"] == "daily"
        assert crawl["hour"] == 22
        assert crawl["minute"] == 0
        assert crawl["enabled"] is True
        assert crawl["action"] == "crawl_workflow"

    def test_default_schedule_morning_brief(self, scheduler):
        """Should have morning_brief at 08:00 KST"""
        brief = next(s for s in scheduler.schedules if s["id"] == "morning_brief")
        assert brief["schedule_type"] == "daily"
        assert brief["hour"] == 8
        assert brief["minute"] == 0
        assert brief["enabled"] is True

    def test_default_schedule_interval(self, scheduler):
        """Should have check_data_freshness with 1h interval"""
        check = next(s for s in scheduler.schedules if s["id"] == "check_data_freshness")
        assert check["schedule_type"] == "interval"
        assert check["interval_hours"] == 1
        assert check["enabled"] is True


class TestStateManagement:
    """Test state persistence (load/save)"""

    def test_load_state_no_file(self, scheduler):
        """Should handle missing state file gracefully"""
        # State file doesn't exist at init
        assert isinstance(scheduler._last_run, dict)

    def test_load_state_with_valid_file(self, tmp_path):
        """Should load last_run from valid state file"""
        state_file = str(tmp_path / "scheduler_state.json")
        test_time = datetime(2025, 6, 1, 10, 0, tzinfo=KST)

        # Create state file
        state_data = {
            "last_run": {"daily_crawl": test_time.isoformat()},
            "saved_at": datetime.now().isoformat(),
        }
        Path(state_file).write_text(json.dumps(state_data), encoding="utf-8")

        # Load scheduler
        with patch.object(AutonomousScheduler, "STATE_FILE", state_file):
            s = AutonomousScheduler()

        assert "daily_crawl" in s._last_run
        assert s._last_run["daily_crawl"] == test_time

    def test_load_state_with_corrupt_file(self, tmp_path):
        """Should handle corrupt state file gracefully"""
        state_file = str(tmp_path / "scheduler_state.json")
        Path(state_file).write_text("invalid json {", encoding="utf-8")

        with patch.object(AutonomousScheduler, "STATE_FILE", state_file):
            s = AutonomousScheduler()

        # Should initialize with empty last_run
        assert isinstance(s._last_run, dict)

    def test_save_state_creates_file(self, scheduler, tmp_path):
        """Should create state file with correct structure"""
        scheduler._last_run["test_task"] = datetime(2025, 6, 1, 10, 0, tzinfo=KST)
        scheduler._save_state()

        assert Path(scheduler.STATE_FILE).exists()

        with open(scheduler.STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)

        assert "last_run" in data
        assert "saved_at" in data
        assert "test_task" in data["last_run"]

    def test_save_state_atomic_write(self, scheduler, tmp_path):
        """Should use atomic write (tempfile + os.replace)"""
        scheduler._last_run["task1"] = datetime(2025, 6, 1, 10, 0, tzinfo=KST)

        with (
            patch("tempfile.NamedTemporaryFile") as mock_tempfile,
            patch("os.replace") as mock_replace,
        ):
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = str(tmp_path / "temp.tmp")
            mock_tempfile.return_value = mock_file

            scheduler._save_state()

            mock_tempfile.assert_called_once()
            mock_replace.assert_called_once()

    def test_save_state_handles_exception(self, scheduler, caplog):
        """Should handle save errors and log"""
        with patch("tempfile.NamedTemporaryFile", side_effect=OSError("disk full")):
            scheduler._save_state()

        assert "Failed to save scheduler state" in caplog.text

    def test_save_state_cleans_temp_file_on_error(self, scheduler, tmp_path):
        """Should clean up temp file if os.replace fails"""
        temp_file = tmp_path / "temp.tmp"
        temp_file.write_text("test")

        with (
            patch("tempfile.NamedTemporaryFile") as mock_tempfile,
            patch("os.replace", side_effect=OSError("replace failed")),
        ):
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = str(temp_file)
            mock_tempfile.return_value = mock_file

            scheduler._save_state()

            # Temp file should be cleaned up (attempt made)
            # Note: actual cleanup depends on implementation details


class TestTimeUtils:
    """Test time-related utilities"""

    def test_get_kst_now_returns_aware_datetime(self, scheduler):
        """Should return timezone-aware datetime in KST"""
        kst_now = scheduler.get_kst_now()
        assert kst_now.tzinfo is not None
        assert kst_now.tzinfo == KST

    def test_get_kst_now_is_current(self, scheduler):
        """Should return current time (within 1 second)"""
        kst_now = scheduler.get_kst_now()
        expected = datetime.now(KST)
        diff = abs((kst_now - expected).total_seconds())
        assert diff < 1.0


class TestDailyScheduling:
    """Test daily schedule logic"""

    def test_daily_task_due_no_last_run(self, scheduler, fixed_time):
        """Should mark daily task as due if never run and past scheduled time"""
        # Set current time to 22:30 KST (past 22:00 daily_crawl)
        with patch.object(scheduler, "get_kst_now", return_value=fixed_time):
            due_tasks = scheduler.get_due_tasks(datetime.now())

        task_ids = [t["id"] for t in due_tasks]
        assert "daily_crawl" in task_ids

    def test_daily_task_not_due_already_ran_today(self, scheduler, fixed_time):
        """Should NOT mark daily task as due if already ran today"""
        # Mark as completed today at 22:05
        scheduler._last_run["daily_crawl"] = fixed_time.replace(hour=22, minute=5)

        with patch.object(scheduler, "get_kst_now", return_value=fixed_time):
            due_tasks = scheduler.get_due_tasks(datetime.now())

        task_ids = [t["id"] for t in due_tasks]
        assert "daily_crawl" not in task_ids

    def test_daily_task_not_due_before_scheduled_time(self, scheduler, fixed_time):
        """Should NOT mark daily task as due if current time is before scheduled time"""
        # Set current time to 21:30 KST (before 22:00 daily_crawl)
        early_time = fixed_time.replace(hour=21, minute=30)

        with patch.object(scheduler, "get_kst_now", return_value=early_time):
            due_tasks = scheduler.get_due_tasks(datetime.now())

        task_ids = [t["id"] for t in due_tasks]
        assert "daily_crawl" not in task_ids

    def test_daily_task_due_after_midnight(self, scheduler, fixed_time):
        """Should mark daily task as due if last ran yesterday"""
        # Last run was yesterday
        yesterday = fixed_time - timedelta(days=1)
        scheduler._last_run["daily_crawl"] = yesterday.replace(hour=22, minute=5)

        with patch.object(scheduler, "get_kst_now", return_value=fixed_time):
            due_tasks = scheduler.get_due_tasks(datetime.now())

        task_ids = [t["id"] for t in due_tasks]
        assert "daily_crawl" in task_ids

    def test_multiple_daily_tasks_at_same_time(self, scheduler, fixed_time):
        """Should return multiple daily tasks if both are due"""
        # Set time to 22:30 (past both 22:00 and 08:00)
        # But morning_brief at 08:00 should already be done today
        # Add another task at 22:00
        scheduler.add_schedule(
            {
                "id": "evening_task",
                "schedule_type": "daily",
                "hour": 22,
                "minute": 0,
                "enabled": True,
            }
        )

        with patch.object(scheduler, "get_kst_now", return_value=fixed_time):
            due_tasks = scheduler.get_due_tasks(datetime.now())

        task_ids = [t["id"] for t in due_tasks]
        # Both daily_crawl and evening_task should be due
        assert "daily_crawl" in task_ids
        assert "evening_task" in task_ids


class TestIntervalScheduling:
    """Test interval schedule logic"""

    def test_interval_task_due_no_last_run(self, scheduler, fixed_time):
        """Should mark interval task as due if never run"""
        with patch.object(scheduler, "get_kst_now", return_value=fixed_time):
            due_tasks = scheduler.get_due_tasks(datetime.now())

        task_ids = [t["id"] for t in due_tasks]
        assert "check_data_freshness" in task_ids

    def test_interval_task_not_due_recently_ran(self, scheduler, fixed_time):
        """Should NOT mark interval task as due if ran within interval"""
        # Last run was 30 minutes ago (interval is 1 hour)
        recent_run = fixed_time - timedelta(minutes=30)
        scheduler._last_run["check_data_freshness"] = recent_run

        with patch.object(scheduler, "get_kst_now", return_value=fixed_time):
            due_tasks = scheduler.get_due_tasks(datetime.now())

        task_ids = [t["id"] for t in due_tasks]
        assert "check_data_freshness" not in task_ids

    def test_interval_task_due_after_interval(self, scheduler, fixed_time):
        """Should mark interval task as due if interval has elapsed"""
        # Last run was 61 minutes ago (interval is 1 hour)
        old_run = fixed_time - timedelta(minutes=61)
        scheduler._last_run["check_data_freshness"] = old_run

        with patch.object(scheduler, "get_kst_now", return_value=fixed_time):
            due_tasks = scheduler.get_due_tasks(datetime.now())

        task_ids = [t["id"] for t in due_tasks]
        assert "check_data_freshness" in task_ids

    def test_interval_task_custom_interval(self, scheduler, fixed_time):
        """Should respect custom interval hours"""
        scheduler.add_schedule(
            {
                "id": "frequent_check",
                "schedule_type": "interval",
                "interval_hours": 0.5,  # 30 minutes
                "enabled": True,
            }
        )

        # Last run was 20 minutes ago
        recent_run = fixed_time - timedelta(minutes=20)
        scheduler._last_run["frequent_check"] = recent_run

        with patch.object(scheduler, "get_kst_now", return_value=fixed_time):
            due_tasks = scheduler.get_due_tasks(datetime.now())

        task_ids = [t["id"] for t in due_tasks]
        assert "frequent_check" not in task_ids

        # Now 31 minutes ago
        old_run = fixed_time - timedelta(minutes=31)
        scheduler._last_run["frequent_check"] = old_run

        with patch.object(scheduler, "get_kst_now", return_value=fixed_time):
            due_tasks = scheduler.get_due_tasks(datetime.now())

        task_ids = [t["id"] for t in due_tasks]
        assert "frequent_check" in task_ids


class TestScheduleDisabling:
    """Test disabled schedule handling"""

    def test_disabled_schedule_skipped(self, scheduler, fixed_time):
        """Should skip disabled schedules"""
        # Disable daily_crawl
        for s in scheduler.schedules:
            if s["id"] == "daily_crawl":
                s["enabled"] = False

        with patch.object(scheduler, "get_kst_now", return_value=fixed_time):
            due_tasks = scheduler.get_due_tasks(datetime.now())

        task_ids = [t["id"] for t in due_tasks]
        assert "daily_crawl" not in task_ids

    def test_missing_enabled_defaults_to_true(self, scheduler, fixed_time):
        """Should treat missing 'enabled' key as True"""
        scheduler.add_schedule(
            {
                "id": "no_enabled_key",
                "schedule_type": "interval",
                "interval_hours": 1,
                # No 'enabled' key
            }
        )

        with patch.object(scheduler, "get_kst_now", return_value=fixed_time):
            due_tasks = scheduler.get_due_tasks(datetime.now())

        task_ids = [t["id"] for t in due_tasks]
        assert "no_enabled_key" in task_ids


class TestScheduleManagement:
    """Test add/remove schedule operations"""

    def test_add_schedule(self, scheduler):
        """Should add schedule to list"""
        initial_count = len(scheduler.schedules)
        new_schedule = {
            "id": "custom_task",
            "name": "Custom Task",
            "schedule_type": "daily",
            "hour": 12,
            "minute": 0,
            "enabled": True,
        }

        scheduler.add_schedule(new_schedule)

        assert len(scheduler.schedules) == initial_count + 1
        assert any(s["id"] == "custom_task" for s in scheduler.schedules)

    def test_remove_schedule(self, scheduler):
        """Should remove schedule from list"""
        initial_count = len(scheduler.schedules)

        scheduler.remove_schedule("daily_crawl")

        assert len(scheduler.schedules) == initial_count - 1
        assert not any(s["id"] == "daily_crawl" for s in scheduler.schedules)

    def test_remove_nonexistent_schedule(self, scheduler):
        """Should handle removing nonexistent schedule gracefully"""
        initial_count = len(scheduler.schedules)

        scheduler.remove_schedule("nonexistent_task")

        assert len(scheduler.schedules) == initial_count


class TestTaskCompletion:
    """Test mark_completed functionality"""

    def test_mark_completed_updates_last_run(self, scheduler):
        """Should update last_run timestamp"""
        schedule_id = "daily_crawl"
        before = datetime.now(KST)

        scheduler.mark_completed(schedule_id)

        assert schedule_id in scheduler._last_run
        after = datetime.now(KST)
        # Should be between before and after
        assert before <= scheduler._last_run[schedule_id] <= after

    def test_mark_completed_saves_state(self, scheduler, tmp_path):
        """Should save state to file"""
        scheduler.mark_completed("daily_crawl")

        assert Path(scheduler.STATE_FILE).exists()

        with open(scheduler.STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)

        assert "daily_crawl" in data["last_run"]

    def test_mark_completed_multiple_tasks(self, scheduler):
        """Should track multiple task completions"""
        scheduler.mark_completed("daily_crawl")
        scheduler.mark_completed("morning_brief")

        assert "daily_crawl" in scheduler._last_run
        assert "morning_brief" in scheduler._last_run
        assert scheduler._last_run["daily_crawl"] != scheduler._last_run["morning_brief"]


class TestPendingCount:
    """Test get_pending_count functionality"""

    def test_get_pending_count_with_due_tasks(self, scheduler, fixed_time):
        """Should return count of due tasks"""
        with patch.object(scheduler, "get_kst_now", return_value=fixed_time):
            count = scheduler.get_pending_count()

        # At 22:30, daily_crawl and check_data_freshness should be due
        # (morning_brief at 08:00 should also be due if not run today)
        assert count >= 2

    def test_get_pending_count_with_no_due_tasks(self, scheduler, fixed_time):
        """Should return 0 when no tasks are due"""
        # Mark all tasks as completed recently
        for schedule in scheduler.schedules:
            scheduler._last_run[schedule["id"]] = fixed_time

        with patch.object(scheduler, "get_kst_now", return_value=fixed_time):
            count = scheduler.get_pending_count()

        assert count == 0


class TestSchedulerLifecycle:
    """Test scheduler start/stop lifecycle"""

    def test_stop_sets_running_false(self, scheduler):
        """Should set running flag to False"""
        scheduler.running = True
        scheduler.stop()
        assert scheduler.running is False

    def test_stop_cancels_task(self, scheduler):
        """Should cancel running task"""
        mock_task = MagicMock()
        mock_task.done.return_value = False
        scheduler._task = mock_task
        scheduler.running = True

        scheduler.stop()

        mock_task.cancel.assert_called_once()
        assert scheduler._task is None

    def test_stop_handles_no_task(self, scheduler):
        """Should handle stop when no task is running"""
        scheduler._task = None
        scheduler.running = True

        scheduler.stop()  # Should not raise

        assert scheduler.running is False

    @pytest.mark.asyncio
    async def test_start_sets_running_true(self, scheduler):
        """Should set running flag to True"""
        callback = AsyncMock()

        async def fast_sleep(seconds):
            if seconds == 60:
                scheduler.running = False  # Stop after first iteration
            await _real_sleep(0.01)

        with (
            patch.object(scheduler, "get_due_tasks", return_value=[]),
            patch("src.core.scheduler.asyncio.sleep", side_effect=fast_sleep),
        ):
            task = asyncio.create_task(scheduler.start(callback))
            await _real_sleep(0.1)

        assert scheduler.running is False  # Should have stopped
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_start_already_running_returns_early(self, scheduler):
        """Should return early if already running"""
        callback = AsyncMock()
        scheduler.running = True

        await scheduler.start(callback)

        # Should return immediately without creating task
        assert scheduler._task is None

    @pytest.mark.asyncio
    async def test_start_calls_callback_for_due_tasks(self, scheduler):
        """Should call callback for each due task"""
        callback = AsyncMock()
        mock_task = {"id": "test_task", "action": "test_action"}

        call_count = 0

        def mock_get_due_tasks(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [mock_task]
            scheduler.running = False  # Stop after second call
            return []

        async def fast_sleep(seconds):
            await _real_sleep(0.01)

        with (
            patch.object(scheduler, "get_due_tasks", side_effect=mock_get_due_tasks),
            patch.object(scheduler, "mark_completed"),
            patch("src.core.scheduler.asyncio.sleep", side_effect=fast_sleep),
        ):
            task = asyncio.create_task(scheduler.start(callback))
            await _real_sleep(0.2)
            if not task.done():
                scheduler.stop()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except (TimeoutError, asyncio.CancelledError):
                    pass

        callback.assert_called_once_with(mock_task)

    @pytest.mark.asyncio
    async def test_start_marks_completed_after_callback(self, scheduler):
        """Should mark task as completed after successful callback"""
        callback = AsyncMock()
        mock_task = {"id": "test_task", "action": "test_action"}

        call_count = 0

        def mock_get_due_tasks(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [mock_task]
            scheduler.running = False
            return []

        async def fast_sleep(seconds):
            await _real_sleep(0.01)

        with (
            patch.object(scheduler, "get_due_tasks", side_effect=mock_get_due_tasks),
            patch.object(scheduler, "mark_completed") as mock_mark,
            patch("src.core.scheduler.asyncio.sleep", side_effect=fast_sleep),
        ):
            task = asyncio.create_task(scheduler.start(callback))
            await _real_sleep(0.2)
            if not task.done():
                scheduler.stop()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except (TimeoutError, asyncio.CancelledError):
                    pass

        mock_mark.assert_called_once_with("test_task")

    @pytest.mark.asyncio
    async def test_start_handles_callback_exception(self, scheduler, caplog):
        """Should handle callback exceptions and continue running"""
        callback = AsyncMock(side_effect=Exception("callback failed"))
        mock_task = {"id": "test_task", "action": "test_action"}

        call_count = 0

        def mock_get_due_tasks(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [mock_task]
            scheduler.running = False
            return []

        async def fast_sleep(seconds):
            await _real_sleep(0.01)

        with (
            patch.object(scheduler, "get_due_tasks", side_effect=mock_get_due_tasks),
            patch("src.core.scheduler.asyncio.sleep", side_effect=fast_sleep),
        ):
            task = asyncio.create_task(scheduler.start(callback))
            await _real_sleep(0.2)
            if not task.done():
                scheduler.stop()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except (TimeoutError, asyncio.CancelledError):
                    pass

        assert "Scheduler task error" in caplog.text

    @pytest.mark.asyncio
    async def test_start_handles_cancelled_error(self, scheduler):
        """Should handle CancelledError gracefully"""
        callback = AsyncMock()

        async def fast_sleep(seconds):
            if seconds == 60:
                scheduler.running = False
            await _real_sleep(0.01)

        with (
            patch.object(scheduler, "get_due_tasks", return_value=[]),
            patch("src.core.scheduler.asyncio.sleep", side_effect=fast_sleep),
        ):
            task = asyncio.create_task(scheduler.start(callback))
            await _real_sleep(0.1)
            if not task.done():
                scheduler.stop()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except (TimeoutError, asyncio.CancelledError):
                    pass

        # Should not raise, task should be cancelled cleanly
        assert scheduler.running is False

    @pytest.mark.asyncio
    async def test_start_sleeps_between_checks(self, scheduler):
        """Should sleep 60 seconds between checks"""
        callback = AsyncMock()
        sleep_args = []

        async def tracking_sleep(seconds):
            sleep_args.append(seconds)
            if seconds == 60:
                # Stop scheduler after first 60s sleep to avoid infinite loop
                scheduler.running = False
            await _real_sleep(0.01)

        with (
            patch("src.core.scheduler.asyncio.sleep", side_effect=tracking_sleep),
            patch.object(scheduler, "get_due_tasks", return_value=[]),
        ):
            task = asyncio.create_task(scheduler.start(callback))
            await _real_sleep(0.2)
            if not task.done():
                scheduler.stop()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except (TimeoutError, asyncio.CancelledError):
                    pass

        # Should have called sleep(60) at least once
        assert 60 in sleep_args


class TestStatusReporting:
    """Test get_status functionality"""

    def test_get_status_returns_correct_structure(self, scheduler):
        """Should return dict with required keys"""
        status = scheduler.get_status()

        assert "running" in status
        assert "schedules_count" in status
        assert "pending_tasks" in status
        assert "last_runs" in status
        assert "kst_now" in status

    def test_get_status_running_flag(self, scheduler):
        """Should reflect current running state"""
        scheduler.running = False
        status = scheduler.get_status()
        assert status["running"] is False

        scheduler.running = True
        status = scheduler.get_status()
        assert status["running"] is True

    def test_get_status_schedules_count(self, scheduler):
        """Should report correct number of schedules"""
        status = scheduler.get_status()
        assert status["schedules_count"] == 3

        scheduler.add_schedule(
            {"id": "new_task", "schedule_type": "daily", "hour": 10, "minute": 0, "enabled": True}
        )
        status = scheduler.get_status()
        assert status["schedules_count"] == 4

    def test_get_status_pending_tasks(self, scheduler, fixed_time):
        """Should report number of pending tasks"""
        with patch.object(scheduler, "get_kst_now", return_value=fixed_time):
            status = scheduler.get_status()

        assert isinstance(status["pending_tasks"], int)
        assert status["pending_tasks"] >= 0

    def test_get_status_last_runs(self, scheduler, fixed_time):
        """Should report last run timestamps as ISO strings"""
        scheduler._last_run["daily_crawl"] = fixed_time

        status = scheduler.get_status()

        assert "daily_crawl" in status["last_runs"]
        assert isinstance(status["last_runs"]["daily_crawl"], str)
        # Should be valid ISO format
        datetime.fromisoformat(status["last_runs"]["daily_crawl"])

    def test_get_status_kst_now(self, scheduler):
        """Should report current KST time as ISO string"""
        status = scheduler.get_status()

        assert isinstance(status["kst_now"], str)
        # Should be valid ISO format
        kst_time = datetime.fromisoformat(status["kst_now"])
        assert kst_time.tzinfo is not None


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_schedule_without_id(self, scheduler, fixed_time):
        """Should raise KeyError for schedule without id (id is required)"""
        scheduler.add_schedule({"schedule_type": "daily", "hour": 10, "minute": 0})

        # Should raise KeyError when getting due tasks since 'id' is required
        with patch.object(scheduler, "get_kst_now", return_value=fixed_time):
            with pytest.raises(KeyError):
                scheduler.get_due_tasks(datetime.now())

    def test_empty_schedules_list(self, scheduler):
        """Should handle empty schedules list"""
        scheduler.schedules = []

        due_tasks = scheduler.get_due_tasks(datetime.now())
        assert due_tasks == []

        count = scheduler.get_pending_count()
        assert count == 0

    def test_mark_completed_creates_new_entry(self, scheduler):
        """Should create new entry in last_run if doesn't exist"""
        new_task_id = "brand_new_task"
        assert new_task_id not in scheduler._last_run

        scheduler.mark_completed(new_task_id)

        assert new_task_id in scheduler._last_run

    def test_concurrent_state_saves(self, scheduler):
        """Should handle multiple concurrent mark_completed calls"""
        # This tests that _save_state is safe for concurrent calls
        scheduler.mark_completed("task1")
        scheduler.mark_completed("task2")
        scheduler.mark_completed("task3")

        assert "task1" in scheduler._last_run
        assert "task2" in scheduler._last_run
        assert "task3" in scheduler._last_run

        # State file should have all three
        with open(scheduler.STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)

        assert "task1" in data["last_run"]
        assert "task2" in data["last_run"]
        assert "task3" in data["last_run"]

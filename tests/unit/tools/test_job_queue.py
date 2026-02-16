"""
Unit tests for src/tools/utilities/job_queue.py

Tests cover:
- Initialization and database setup
- Job creation and status tracking
- Progress updates
- Job execution and handler registration
- Worker lifecycle (start/stop)
- Job cleanup and expiration
- Error handling
- Singleton pattern
"""

import asyncio
import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.shared.constants import KST
from src.tools.utilities.job_queue import JobQueue, JobStatus, JobType, get_job_queue

# Save reference to real asyncio.sleep
_real_sleep = asyncio.sleep


@pytest.fixture
def temp_db(tmp_path):
    """Create temporary database path"""
    return str(tmp_path / "test_job_queue.db")


@pytest.fixture
def job_queue(temp_db):
    """Create JobQueue instance with temporary database"""
    queue = JobQueue(db_path=temp_db)
    return queue


@pytest.fixture
def initialized_queue(temp_db):
    """Create and initialize JobQueue"""
    queue = JobQueue(db_path=temp_db)
    asyncio.run(queue.initialize())
    return queue


@pytest.fixture
def mock_handler():
    """Create mock job handler"""

    async def handler(job_id: str, params: dict, queue: JobQueue) -> str:
        # Simulate some work
        await queue.update_progress(job_id, 50, "Processing...")
        await asyncio.sleep(0.01)
        result_file = os.path.join(queue.output_dir, f"{job_id}_result.txt")
        Path(result_file).write_text("test result", encoding="utf-8")
        return result_file

    return handler


@pytest.fixture
def failing_handler():
    """Create handler that always fails"""

    async def handler(job_id: str, params: dict, queue: JobQueue) -> str:
        raise ValueError("Simulated failure")

    return handler


class TestInitialization:
    """Test JobQueue initialization"""

    def test_init_default_path(self):
        """Should use default database path when none provided"""
        queue = JobQueue()
        expected = os.path.join(os.getenv("DATA_DIR", "./data"), "job_queue.db")
        assert queue.db_path == expected

    def test_init_custom_path(self, temp_db):
        """Should use custom database path when provided"""
        queue = JobQueue(db_path=temp_db)
        assert queue.db_path == temp_db

    def test_init_creates_output_dir(self, temp_db):
        """Should create output directory on init"""
        queue = JobQueue(db_path=temp_db)
        expected_dir = os.path.join(os.path.dirname(temp_db), "exports")
        assert queue.output_dir == expected_dir
        assert Path(queue.output_dir).exists()

    def test_init_sets_initial_state(self, job_queue):
        """Should initialize internal state"""
        assert job_queue._initialized is False
        assert job_queue._worker_task is None
        assert isinstance(job_queue._handlers, dict)
        assert len(job_queue._handlers) == 0

    @pytest.mark.asyncio
    async def test_initialize_creates_database(self, job_queue):
        """Should create database and tables"""
        await job_queue.initialize()

        assert Path(job_queue.db_path).exists()
        assert job_queue._initialized is True

        # Verify table exists
        with job_queue._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='job_queue'"
            )
            assert cursor.fetchone() is not None

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, job_queue):
        """Should be safe to call initialize multiple times"""
        await job_queue.initialize()
        await job_queue.initialize()
        await job_queue.initialize()

        assert job_queue._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_creates_indexes(self, job_queue):
        """Should create database indexes"""
        await job_queue.initialize()

        with job_queue._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_job_queue_%'"
            )
            indexes = cursor.fetchall()
            index_names = {row[0] for row in indexes}

            assert "idx_job_queue_status" in index_names
            assert "idx_job_queue_created" in index_names
            assert "idx_job_queue_expires" in index_names


class TestDatabaseConnection:
    """Test database connection management"""

    def test_get_connection_context_manager(self, initialized_queue):
        """Should provide working connection context manager"""
        with initialized_queue._get_connection() as conn:
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1

    def test_get_connection_commits_on_success(self, initialized_queue):
        """Should commit transaction on success"""
        with initialized_queue._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO job_queue (id, job_type, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    "test123",
                    "test_type",
                    datetime.now(KST).isoformat(),
                    datetime.now(KST).isoformat(),
                ),
            )

        # Verify data was committed
        with initialized_queue._get_connection() as conn:
            cursor = conn.execute("SELECT id FROM job_queue WHERE id = ?", ("test123",))
            assert cursor.fetchone() is not None

    def test_get_connection_rolls_back_on_error(self, initialized_queue):
        """Should rollback transaction on error"""
        try:
            with initialized_queue._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO job_queue (id, job_type, created_at, expires_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        "test456",
                        "test_type",
                        datetime.now(KST).isoformat(),
                        datetime.now(KST).isoformat(),
                    ),
                )
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Verify data was NOT committed
        with initialized_queue._get_connection() as conn:
            cursor = conn.execute("SELECT id FROM job_queue WHERE id = ?", ("test456",))
            assert cursor.fetchone() is None

    def test_get_connection_closes_properly(self, initialized_queue):
        """Should close connection after use"""
        with initialized_queue._get_connection() as conn:
            pass

        # Connection should be closed, attempting to use it should fail
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


class TestJobCreation:
    """Test job creation functionality"""

    @pytest.mark.asyncio
    async def test_create_job_basic(self, job_queue):
        """Should create job with basic parameters"""
        job_id = await job_queue.create_job("export_docx", {"param1": "value1"})

        assert isinstance(job_id, str)
        assert len(job_id) == 8  # Short UUID

        # Verify job in database
        with job_queue._get_connection() as conn:
            row = conn.execute("SELECT * FROM job_queue WHERE id = ?", (job_id,)).fetchone()
            assert row is not None
            assert row["job_type"] == "export_docx"
            assert row["status"] == JobStatus.PENDING.value

    @pytest.mark.asyncio
    async def test_create_job_with_params(self, job_queue):
        """Should store job parameters as JSON"""
        params = {"start_date": "2026-01-01", "end_date": "2026-01-31", "format": "detailed"}
        job_id = await job_queue.create_job("export_analyst_report", params)

        with job_queue._get_connection() as conn:
            row = conn.execute("SELECT params FROM job_queue WHERE id = ?", (job_id,)).fetchone()
            stored_params = json.loads(row["params"])
            assert stored_params == params

    @pytest.mark.asyncio
    async def test_create_job_without_params(self, job_queue):
        """Should handle job creation without parameters"""
        job_id = await job_queue.create_job("export_excel")

        with job_queue._get_connection() as conn:
            row = conn.execute("SELECT params FROM job_queue WHERE id = ?", (job_id,)).fetchone()
            stored_params = json.loads(row["params"])
            assert stored_params == {}

    @pytest.mark.asyncio
    async def test_create_job_sets_timestamps(self, job_queue):
        """Should set created_at and expires_at timestamps"""
        before = datetime.now(KST)
        job_id = await job_queue.create_job("export_docx")
        after = datetime.now(KST)

        with job_queue._get_connection() as conn:
            row = conn.execute(
                "SELECT created_at, expires_at FROM job_queue WHERE id = ?", (job_id,)
            ).fetchone()

            created_at = datetime.fromisoformat(row["created_at"])
            expires_at = datetime.fromisoformat(row["expires_at"])

            assert before <= created_at <= after
            # Default retention is 24 hours
            expected_expiry = created_at + timedelta(hours=job_queue.FILE_RETENTION_HOURS)
            assert abs((expires_at - expected_expiry).total_seconds()) < 1

    @pytest.mark.asyncio
    async def test_create_job_custom_retention(self, job_queue):
        """Should respect custom retention period"""
        job_id = await job_queue.create_job("export_docx", retention_hours=48)

        with job_queue._get_connection() as conn:
            row = conn.execute(
                "SELECT created_at, expires_at FROM job_queue WHERE id = ?", (job_id,)
            ).fetchone()

            created_at = datetime.fromisoformat(row["created_at"])
            expires_at = datetime.fromisoformat(row["expires_at"])

            expected_expiry = created_at + timedelta(hours=48)
            assert abs((expires_at - expected_expiry).total_seconds()) < 1

    @pytest.mark.asyncio
    async def test_create_job_initializes_queue(self, job_queue):
        """Should auto-initialize queue if not initialized"""
        assert job_queue._initialized is False

        await job_queue.create_job("export_docx")

        assert job_queue._initialized is True


class TestJobStatus:
    """Test job status retrieval"""

    @pytest.mark.asyncio
    async def test_get_job_status_existing(self, job_queue):
        """Should return status for existing job"""
        job_id = await job_queue.create_job("export_docx", {"test": "data"})

        status = await job_queue.get_job_status(job_id)

        assert status is not None
        assert status["id"] == job_id
        assert status["job_type"] == "export_docx"
        assert status["status"] == JobStatus.PENDING.value
        assert status["progress"] == 0

    @pytest.mark.asyncio
    async def test_get_job_status_nonexistent(self, job_queue):
        """Should return None for nonexistent job"""
        status = await job_queue.get_job_status("nonexistent123")
        assert status is None

    @pytest.mark.asyncio
    async def test_get_job_status_includes_download_url(self, job_queue):
        """Should include download_url for completed jobs"""
        job_id = await job_queue.create_job("export_docx")

        # Mark as completed with result file
        await job_queue._mark_completed(job_id, "/path/to/result.docx")

        status = await job_queue.get_job_status(job_id)

        assert status["status"] == JobStatus.COMPLETED.value
        assert status["download_url"] == f"/api/export/download/{job_id}"

    @pytest.mark.asyncio
    async def test_get_job_status_no_download_url_for_pending(self, job_queue):
        """Should not include download_url for non-completed jobs"""
        job_id = await job_queue.create_job("export_docx")

        status = await job_queue.get_job_status(job_id)

        assert "download_url" not in status


class TestProgressUpdates:
    """Test progress update functionality"""

    @pytest.mark.asyncio
    async def test_update_progress(self, job_queue):
        """Should update job progress"""
        job_id = await job_queue.create_job("export_docx")

        await job_queue.update_progress(job_id, 50, "Generating document...")

        with job_queue._get_connection() as conn:
            row = conn.execute(
                "SELECT progress, progress_message FROM job_queue WHERE id = ?", (job_id,)
            ).fetchone()

            assert row["progress"] == 50
            assert row["progress_message"] == "Generating document..."

    @pytest.mark.asyncio
    async def test_update_progress_without_message(self, job_queue):
        """Should allow progress update without message"""
        job_id = await job_queue.create_job("export_docx")

        await job_queue.update_progress(job_id, 75)

        with job_queue._get_connection() as conn:
            row = conn.execute("SELECT progress FROM job_queue WHERE id = ?", (job_id,)).fetchone()
            assert row["progress"] == 75

    @pytest.mark.asyncio
    async def test_update_progress_multiple_times(self, job_queue):
        """Should handle multiple progress updates"""
        job_id = await job_queue.create_job("export_docx")

        await job_queue.update_progress(job_id, 25, "Step 1")
        await job_queue.update_progress(job_id, 50, "Step 2")
        await job_queue.update_progress(job_id, 75, "Step 3")

        status = await job_queue.get_job_status(job_id)
        assert status["progress"] == 75
        assert status["progress_message"] == "Step 3"


class TestJobStateTransitions:
    """Test job state transition methods"""

    @pytest.mark.asyncio
    async def test_mark_running(self, job_queue):
        """Should transition job to running state"""
        job_id = await job_queue.create_job("export_docx")

        await job_queue._mark_running(job_id)

        with job_queue._get_connection() as conn:
            row = conn.execute(
                "SELECT status, started_at, progress FROM job_queue WHERE id = ?", (job_id,)
            ).fetchone()

            assert row["status"] == JobStatus.RUNNING.value
            assert row["started_at"] is not None
            assert row["progress"] == 0

    @pytest.mark.asyncio
    async def test_mark_completed(self, job_queue):
        """Should transition job to completed state"""
        job_id = await job_queue.create_job("export_docx")
        result_file = "/path/to/result.docx"

        await job_queue._mark_completed(job_id, result_file)

        with job_queue._get_connection() as conn:
            row = conn.execute(
                "SELECT status, completed_at, result_file, progress FROM job_queue WHERE id = ?",
                (job_id,),
            ).fetchone()

            assert row["status"] == JobStatus.COMPLETED.value
            assert row["completed_at"] is not None
            assert row["result_file"] == result_file
            assert row["progress"] == 100

    @pytest.mark.asyncio
    async def test_mark_failed(self, job_queue):
        """Should transition job to failed state"""
        job_id = await job_queue.create_job("export_docx")
        error_msg = "Something went wrong"

        await job_queue._mark_failed(job_id, error_msg)

        with job_queue._get_connection() as conn:
            row = conn.execute(
                "SELECT status, completed_at, error_message FROM job_queue WHERE id = ?", (job_id,)
            ).fetchone()

            assert row["status"] == JobStatus.FAILED.value
            assert row["completed_at"] is not None
            assert row["error_message"] == error_msg


class TestHandlerRegistration:
    """Test job handler registration"""

    def test_register_handler(self, job_queue, mock_handler):
        """Should register handler for job type"""
        job_queue.register_handler("export_docx", mock_handler)

        assert "export_docx" in job_queue._handlers
        assert job_queue._handlers["export_docx"] == mock_handler

    def test_register_multiple_handlers(self, job_queue, mock_handler):
        """Should register multiple handlers"""
        handler2 = AsyncMock()

        job_queue.register_handler("export_docx", mock_handler)
        job_queue.register_handler("export_excel", handler2)

        assert len(job_queue._handlers) == 2
        assert job_queue._handlers["export_docx"] == mock_handler
        assert job_queue._handlers["export_excel"] == handler2

    def test_register_handler_overwrites(self, job_queue, mock_handler):
        """Should allow overwriting existing handler"""
        handler2 = AsyncMock()

        job_queue.register_handler("export_docx", mock_handler)
        job_queue.register_handler("export_docx", handler2)

        assert job_queue._handlers["export_docx"] == handler2


class TestJobProcessing:
    """Test job processing logic"""

    @pytest.mark.asyncio
    async def test_process_job_success(self, job_queue, mock_handler):
        """Should successfully process job with registered handler"""
        job_queue.register_handler("export_docx", mock_handler)
        job_id = await job_queue.create_job("export_docx", {"test": "data"})

        result = await job_queue.process_job(job_id)

        assert result is True

        status = await job_queue.get_job_status(job_id)
        assert status["status"] == JobStatus.COMPLETED.value
        assert status["result_file"] is not None
        assert Path(status["result_file"]).exists()

    @pytest.mark.asyncio
    async def test_process_job_nonexistent(self, job_queue):
        """Should return False for nonexistent job"""
        result = await job_queue.process_job("nonexistent123")
        assert result is False

    @pytest.mark.asyncio
    async def test_process_job_no_handler(self, job_queue):
        """Should fail job when no handler registered"""
        job_id = await job_queue.create_job("unknown_type")

        result = await job_queue.process_job(job_id)

        assert result is False

        status = await job_queue.get_job_status(job_id)
        assert status["status"] == JobStatus.FAILED.value
        assert "No handler for job type" in status["error_message"]

    @pytest.mark.asyncio
    async def test_process_job_handler_exception(self, job_queue, failing_handler):
        """Should mark job as failed when handler raises exception"""
        job_queue.register_handler("failing_job", failing_handler)
        job_id = await job_queue.create_job("failing_job")

        result = await job_queue.process_job(job_id)

        assert result is False

        status = await job_queue.get_job_status(job_id)
        assert status["status"] == JobStatus.FAILED.value
        assert "ValueError" in status["error_message"]
        assert "Simulated failure" in status["error_message"]

    @pytest.mark.asyncio
    async def test_process_job_marks_running(self, job_queue, mock_handler):
        """Should mark job as running before execution"""
        job_queue.register_handler("export_docx", mock_handler)
        job_id = await job_queue.create_job("export_docx")

        # Capture status during execution
        async def tracking_handler(job_id, params, queue):
            status = await queue.get_job_status(job_id)
            assert status["status"] == JobStatus.RUNNING.value
            return await mock_handler(job_id, params, queue)

        job_queue.register_handler("export_docx", tracking_handler)
        await job_queue.process_job(job_id)


class TestPendingJobs:
    """Test pending job retrieval"""

    @pytest.mark.asyncio
    async def test_get_pending_jobs_empty(self, job_queue):
        """Should return empty list when no pending jobs"""
        await job_queue.initialize()

        pending = await job_queue.get_pending_jobs()
        assert pending == []

    @pytest.mark.asyncio
    async def test_get_pending_jobs_single(self, job_queue):
        """Should return pending jobs"""
        job_id = await job_queue.create_job("export_docx")

        pending = await job_queue.get_pending_jobs()

        assert len(pending) == 1
        assert pending[0]["id"] == job_id
        assert pending[0]["status"] == JobStatus.PENDING.value

    @pytest.mark.asyncio
    async def test_get_pending_jobs_multiple(self, job_queue):
        """Should return multiple pending jobs in order"""
        await job_queue.initialize()

        # Create jobs with small delays to ensure ordering
        job_id1 = await job_queue.create_job("export_docx")
        await asyncio.sleep(0.01)
        job_id2 = await job_queue.create_job("export_excel")
        await asyncio.sleep(0.01)
        job_id3 = await job_queue.create_job("export_analyst_report")

        pending = await job_queue.get_pending_jobs()

        assert len(pending) == 3
        # Should be ordered by created_at ASC (oldest first)
        assert pending[0]["id"] == job_id1
        assert pending[1]["id"] == job_id2
        assert pending[2]["id"] == job_id3

    @pytest.mark.asyncio
    async def test_get_pending_jobs_excludes_running(self, job_queue):
        """Should not include running jobs"""
        job_id1 = await job_queue.create_job("export_docx")
        job_id2 = await job_queue.create_job("export_excel")

        await job_queue._mark_running(job_id1)

        pending = await job_queue.get_pending_jobs()

        assert len(pending) == 1
        assert pending[0]["id"] == job_id2

    @pytest.mark.asyncio
    async def test_get_pending_jobs_limit(self, job_queue):
        """Should respect limit parameter"""
        await job_queue.initialize()

        for _i in range(5):
            await job_queue.create_job("export_docx")

        pending = await job_queue.get_pending_jobs(limit=2)

        assert len(pending) == 2


class TestWorkerLifecycle:
    """Test worker start/stop lifecycle"""

    @pytest.mark.asyncio
    async def test_start_worker(self, job_queue):
        """Should start worker task"""
        assert job_queue._worker_task is None

        await job_queue.start_worker(poll_interval=0.1)

        assert job_queue._worker_task is not None
        assert not job_queue._worker_task.done()

        await job_queue.stop_worker()

    @pytest.mark.asyncio
    async def test_start_worker_already_running(self, job_queue, caplog):
        """Should not start duplicate worker"""
        await job_queue.start_worker(poll_interval=0.1)

        await job_queue.start_worker(poll_interval=0.1)

        assert "Worker already running" in caplog.text

        await job_queue.stop_worker()

    @pytest.mark.asyncio
    async def test_stop_worker(self, job_queue):
        """Should stop worker task"""
        await job_queue.start_worker(poll_interval=0.1)

        await job_queue.stop_worker()

        assert job_queue._worker_task is None

    @pytest.mark.asyncio
    async def test_stop_worker_not_running(self, job_queue):
        """Should handle stop when worker not running"""
        await job_queue.stop_worker()  # Should not raise
        assert job_queue._worker_task is None

    @pytest.mark.asyncio
    async def test_worker_processes_pending_jobs(self, job_queue, mock_handler):
        """Should process pending jobs in worker loop"""
        job_queue.register_handler("export_docx", mock_handler)
        job_id = await job_queue.create_job("export_docx")

        await job_queue.start_worker(poll_interval=0.1)

        # Wait for job to be processed
        for _ in range(20):
            status = await job_queue.get_job_status(job_id)
            if status["status"] == JobStatus.COMPLETED.value:
                break
            await asyncio.sleep(0.1)

        await job_queue.stop_worker()

        status = await job_queue.get_job_status(job_id)
        assert status["status"] == JobStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_worker_handles_errors(self, job_queue, failing_handler):
        """Should continue running after job failure"""
        job_queue.register_handler("failing_job", failing_handler)
        job_id = await job_queue.create_job("failing_job")

        await job_queue.start_worker(poll_interval=0.1)

        # Wait for job to be processed
        for _ in range(20):
            status = await job_queue.get_job_status(job_id)
            if status["status"] == JobStatus.FAILED.value:
                break
            await asyncio.sleep(0.1)

        await job_queue.stop_worker()

        status = await job_queue.get_job_status(job_id)
        assert status["status"] == JobStatus.FAILED.value


class TestJobCleanup:
    """Test job cleanup and expiration"""

    @pytest.mark.asyncio
    async def test_cleanup_expired_jobs(self, job_queue, tmp_path):
        """Should delete expired jobs and files"""
        await job_queue.initialize()

        # Create job that expired 1 hour ago
        job_id = await job_queue.create_job("export_docx", retention_hours=1)
        result_file = os.path.join(job_queue.output_dir, f"{job_id}_result.txt")
        Path(result_file).write_text("test", encoding="utf-8")

        await job_queue._mark_completed(job_id, result_file)

        # Manually set expiry to past
        past_time = (datetime.now(KST) - timedelta(hours=2)).isoformat()
        with job_queue._get_connection() as conn:
            conn.execute("UPDATE job_queue SET expires_at = ? WHERE id = ?", (past_time, job_id))

        await job_queue._cleanup_expired_jobs()

        # Job should be deleted
        status = await job_queue.get_job_status(job_id)
        assert status is None

        # File should be deleted
        assert not Path(result_file).exists()

    @pytest.mark.asyncio
    async def test_cleanup_keeps_active_jobs(self, job_queue):
        """Should not delete non-expired jobs"""
        job_id = await job_queue.create_job("export_docx", retention_hours=24)

        await job_queue._cleanup_expired_jobs()

        status = await job_queue.get_job_status(job_id)
        assert status is not None

    @pytest.mark.asyncio
    async def test_cleanup_handles_missing_file(self, job_queue, caplog):
        """Should handle cleanup when result file doesn't exist"""
        await job_queue.initialize()

        job_id = await job_queue.create_job("export_docx", retention_hours=1)
        await job_queue._mark_completed(job_id, "/nonexistent/file.txt")

        # Set expiry to past
        past_time = (datetime.now(KST) - timedelta(hours=2)).isoformat()
        with job_queue._get_connection() as conn:
            conn.execute("UPDATE job_queue SET expires_at = ? WHERE id = ?", (past_time, job_id))

        await job_queue._cleanup_expired_jobs()

        # Should still delete the job record
        status = await job_queue.get_job_status(job_id)
        assert status is None

    @pytest.mark.asyncio
    async def test_cleanup_only_affects_completed_failed(self, job_queue):
        """Should only cleanup completed and failed jobs"""
        await job_queue.initialize()

        pending_id = await job_queue.create_job("export_docx", retention_hours=1)
        running_id = await job_queue.create_job("export_excel", retention_hours=1)
        await job_queue._mark_running(running_id)

        # Set expiry to past
        past_time = (datetime.now(KST) - timedelta(hours=2)).isoformat()
        with job_queue._get_connection() as conn:
            conn.execute("UPDATE job_queue SET expires_at = ?", (past_time,))

        await job_queue._cleanup_expired_jobs()

        # Pending and running jobs should still exist
        assert await job_queue.get_job_status(pending_id) is not None
        assert await job_queue.get_job_status(running_id) is not None


class TestResultFileRetrieval:
    """Test result file path retrieval"""

    @pytest.mark.asyncio
    async def test_get_result_file_path_completed(self, job_queue):
        """Should return file path for completed job"""
        job_id = await job_queue.create_job("export_docx")
        result_file = "/path/to/result.docx"
        await job_queue._mark_completed(job_id, result_file)

        path = job_queue.get_result_file_path(job_id)
        assert path == result_file

    @pytest.mark.asyncio
    async def test_get_result_file_path_pending(self, job_queue):
        """Should return None for non-completed job"""
        job_id = await job_queue.create_job("export_docx")

        path = job_queue.get_result_file_path(job_id)
        assert path is None

    @pytest.mark.asyncio
    async def test_get_result_file_path_nonexistent(self, job_queue):
        """Should return None for nonexistent job"""
        await job_queue.initialize()
        path = job_queue.get_result_file_path("nonexistent123")
        assert path is None


class TestGetAllJobs:
    """Test get_all_jobs functionality"""

    @pytest.mark.asyncio
    async def test_get_all_jobs_empty(self, job_queue):
        """Should return empty list when no jobs"""
        jobs = await job_queue.get_all_jobs()
        assert jobs == []

    @pytest.mark.asyncio
    async def test_get_all_jobs_multiple(self, job_queue):
        """Should return all jobs"""
        job_id1 = await job_queue.create_job("export_docx")
        job_id2 = await job_queue.create_job("export_excel")

        jobs = await job_queue.get_all_jobs()

        assert len(jobs) == 2
        job_ids = {job["id"] for job in jobs}
        assert job_id1 in job_ids
        assert job_id2 in job_ids

    @pytest.mark.asyncio
    async def test_get_all_jobs_ordered_by_created_desc(self, job_queue):
        """Should return jobs ordered by created_at DESC (newest first)"""
        await job_queue.initialize()

        job_id1 = await job_queue.create_job("export_docx")
        await asyncio.sleep(0.01)
        job_id2 = await job_queue.create_job("export_excel")
        await asyncio.sleep(0.01)
        job_id3 = await job_queue.create_job("export_analyst_report")

        jobs = await job_queue.get_all_jobs()

        # Newest first
        assert jobs[0]["id"] == job_id3
        assert jobs[1]["id"] == job_id2
        assert jobs[2]["id"] == job_id1

    @pytest.mark.asyncio
    async def test_get_all_jobs_filter_by_status(self, job_queue):
        """Should filter jobs by status"""
        job_id1 = await job_queue.create_job("export_docx")
        job_id2 = await job_queue.create_job("export_excel")
        await job_queue._mark_running(job_id1)

        pending_jobs = await job_queue.get_all_jobs(status=JobStatus.PENDING.value)
        running_jobs = await job_queue.get_all_jobs(status=JobStatus.RUNNING.value)

        assert len(pending_jobs) == 1
        assert pending_jobs[0]["id"] == job_id2

        assert len(running_jobs) == 1
        assert running_jobs[0]["id"] == job_id1

    @pytest.mark.asyncio
    async def test_get_all_jobs_limit(self, job_queue):
        """Should respect limit parameter"""
        await job_queue.initialize()

        for _i in range(10):
            await job_queue.create_job("export_docx")

        jobs = await job_queue.get_all_jobs(limit=5)

        assert len(jobs) == 5


class TestSingletonPattern:
    """Test singleton get_job_queue function"""

    def test_get_job_queue_returns_instance(self):
        """Should return JobQueue instance"""
        queue = get_job_queue()
        assert isinstance(queue, JobQueue)

    def test_get_job_queue_singleton(self):
        """Should return same instance on multiple calls"""
        queue1 = get_job_queue()
        queue2 = get_job_queue()
        assert queue1 is queue2

    def test_get_job_queue_uses_default_path(self):
        """Should use default database path"""
        queue = get_job_queue()
        expected = os.path.join(os.getenv("DATA_DIR", "./data"), "job_queue.db")
        assert queue.db_path == expected


class TestEnums:
    """Test enum definitions"""

    def test_job_status_enum(self):
        """Should have all expected job statuses"""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_job_type_enum(self):
        """Should have all expected job types"""
        assert JobType.EXPORT_DOCX.value == "export_docx"
        assert JobType.EXPORT_ANALYST_REPORT.value == "export_analyst_report"
        assert JobType.EXPORT_EXCEL.value == "export_excel"


class TestEdgeCases:
    """Test edge cases and error scenarios"""

    @pytest.mark.asyncio
    async def test_concurrent_job_creation(self, job_queue):
        """Should handle concurrent job creation"""
        jobs = await asyncio.gather(*[job_queue.create_job("export_docx") for _ in range(10)])

        # All should have unique IDs
        assert len(set(jobs)) == 10

    @pytest.mark.asyncio
    async def test_process_job_with_empty_params(self, job_queue, mock_handler):
        """Should handle job with null params"""
        job_queue.register_handler("export_docx", mock_handler)

        # Create job with None params
        job_id = await job_queue.create_job("export_docx", None)

        result = await job_queue.process_job(job_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_worker_loop_continues_after_exception(self, job_queue, caplog):
        """Should continue worker loop after exception in cleanup"""
        await job_queue.initialize()

        # Mock cleanup to raise exception
        original_cleanup = job_queue._cleanup_expired_jobs
        call_count = 0

        async def failing_cleanup():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Cleanup error")
            await original_cleanup()

        with patch.object(job_queue, "_cleanup_expired_jobs", side_effect=failing_cleanup):
            await job_queue.start_worker(poll_interval=0.1)
            await asyncio.sleep(0.3)
            await job_queue.stop_worker()

        # Worker should have continued after error
        assert call_count >= 2
        assert "Worker error" in caplog.text

    @pytest.mark.asyncio
    async def test_update_progress_nonexistent_job(self, job_queue):
        """Should handle progress update for nonexistent job gracefully"""
        await job_queue.initialize()

        # Should not raise exception
        await job_queue.update_progress("nonexistent123", 50, "test")

        # No job should be created
        status = await job_queue.get_job_status("nonexistent123")
        assert status is None

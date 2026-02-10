"""
SQLite 기반 백그라운드 작업 큐

페이지 새로고침에도 다운로드가 지속되도록 하는 비동기 작업 시스템.
작업 상태를 SQLite에 저장하여 서버 재시작 후에도 복구 가능.

Usage:
    queue = JobQueue()
    await queue.initialize()

    # 작업 생성
    job_id = await queue.create_job("export_analyst_report", {"start_date": "2026-01-01"})

    # 작업 상태 확인
    status = await queue.get_job_status(job_id)

    # 완료된 파일 다운로드
    if status["status"] == "completed":
        file_path = status["result_file"]
"""

import asyncio
import json
import logging
import os
import sqlite3
import traceback
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# 한국 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))


class JobStatus(str, Enum):
    """작업 상태"""

    PENDING = "pending"  # 대기 중
    RUNNING = "running"  # 실행 중
    COMPLETED = "completed"  # 완료
    FAILED = "failed"  # 실패
    CANCELLED = "cancelled"  # 취소됨


class JobType(str, Enum):
    """작업 유형"""

    EXPORT_DOCX = "export_docx"
    EXPORT_ANALYST_REPORT = "export_analyst_report"
    EXPORT_EXCEL = "export_excel"


class JobQueue:
    """SQLite 기반 작업 큐"""

    # 작업 큐 스키마
    SCHEMA = """
    -- 백그라운드 작업 큐
    CREATE TABLE IF NOT EXISTS job_queue (
        id TEXT PRIMARY KEY,
        job_type TEXT NOT NULL,
        status TEXT DEFAULT 'pending',
        params TEXT,
        result_file TEXT,
        error_message TEXT,
        progress INTEGER DEFAULT 0,
        progress_message TEXT,
        created_at TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        expires_at TEXT
    );

    -- 인덱스
    CREATE INDEX IF NOT EXISTS idx_job_queue_status ON job_queue(status);
    CREATE INDEX IF NOT EXISTS idx_job_queue_created ON job_queue(created_at);
    CREATE INDEX IF NOT EXISTS idx_job_queue_expires ON job_queue(expires_at);
    """

    # 완료된 파일 보관 기간 (시간)
    FILE_RETENTION_HOURS = 24

    def __init__(self, db_path: str | None = None):
        """
        Args:
            db_path: SQLite 데이터베이스 경로. None이면 기본 경로 사용.
        """
        if db_path:
            self.db_path = db_path
        else:
            # Railway 환경 또는 로컬 환경
            data_dir = os.getenv("DATA_DIR", "./data")
            self.db_path = os.path.join(data_dir, "job_queue.db")

        # 출력 디렉토리
        self.output_dir = os.path.join(os.path.dirname(self.db_path), "exports")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self._initialized = False
        self._worker_task: asyncio.Task | None = None
        self._handlers: dict[str, Callable] = {}

    @contextmanager
    def _get_connection(self):
        """SQLite 연결 컨텍스트 매니저"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    async def initialize(self) -> None:
        """데이터베이스 초기화"""
        if self._initialized:
            return

        # 디렉토리 생성
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            conn.executescript(self.SCHEMA)

        self._initialized = True
        logger.info(f"JobQueue initialized: {self.db_path}")

    def register_handler(self, job_type: str, handler: Callable) -> None:
        """작업 핸들러 등록

        Args:
            job_type: 작업 유형 (JobType enum 값)
            handler: async def handler(job_id: str, params: dict) -> str (result_file_path)
        """
        self._handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")

    async def create_job(
        self, job_type: str, params: dict | None = None, retention_hours: int | None = None
    ) -> str:
        """새 작업 생성

        Args:
            job_type: 작업 유형
            params: 작업 파라미터
            retention_hours: 파일 보관 기간 (시간). None이면 기본값 사용.

        Returns:
            job_id: 생성된 작업 ID
        """
        await self.initialize()

        job_id = str(uuid.uuid4())[:8]  # 짧은 ID
        now = datetime.now(KST)
        retention = retention_hours or self.FILE_RETENTION_HOURS
        expires_at = now + timedelta(hours=retention)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO job_queue (id, job_type, status, params, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    job_type,
                    JobStatus.PENDING.value,
                    json.dumps(params or {}),
                    now.isoformat(),
                    expires_at.isoformat(),
                ),
            )

        logger.info(f"Created job: {job_id} (type={job_type})")
        return job_id

    async def get_job_status(self, job_id: str) -> dict | None:
        """작업 상태 조회

        Returns:
            {
                "id": "abc123",
                "job_type": "export_analyst_report",
                "status": "running",
                "progress": 50,
                "progress_message": "차트 생성 중...",
                "result_file": null,
                "error_message": null,
                "created_at": "2026-01-28T10:00:00+09:00",
                "download_url": "/api/export/download/abc123"
            }
        """
        await self.initialize()

        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM job_queue WHERE id = ?", (job_id,)).fetchone()

        if not row:
            return None

        result = dict(row)

        # 다운로드 URL 추가 (완료된 경우)
        if result["status"] == JobStatus.COMPLETED.value and result["result_file"]:
            result["download_url"] = f"/api/export/download/{job_id}"

        return result

    async def update_progress(self, job_id: str, progress: int, message: str | None = None) -> None:
        """작업 진행률 업데이트

        Args:
            job_id: 작업 ID
            progress: 진행률 (0-100)
            message: 진행 상태 메시지
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE job_queue
                SET progress = ?, progress_message = ?
                WHERE id = ?
                """,
                (progress, message, job_id),
            )

    async def _mark_running(self, job_id: str) -> None:
        """작업 시작 표시"""
        now = datetime.now(KST)
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE job_queue
                SET status = ?, started_at = ?, progress = 0
                WHERE id = ?
                """,
                (JobStatus.RUNNING.value, now.isoformat(), job_id),
            )

    async def _mark_completed(self, job_id: str, result_file: str) -> None:
        """작업 완료 표시"""
        now = datetime.now(KST)
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE job_queue
                SET status = ?, completed_at = ?, result_file = ?, progress = 100
                WHERE id = ?
                """,
                (JobStatus.COMPLETED.value, now.isoformat(), result_file, job_id),
            )

    async def _mark_failed(self, job_id: str, error_message: str) -> None:
        """작업 실패 표시"""
        now = datetime.now(KST)
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE job_queue
                SET status = ?, completed_at = ?, error_message = ?
                WHERE id = ?
                """,
                (JobStatus.FAILED.value, now.isoformat(), error_message, job_id),
            )

    async def get_pending_jobs(self, limit: int = 10) -> list[dict]:
        """대기 중인 작업 목록"""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM job_queue
                WHERE status = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (JobStatus.PENDING.value, limit),
            ).fetchall()

        return [dict(row) for row in rows]

    async def process_job(self, job_id: str) -> bool:
        """단일 작업 처리

        Returns:
            True if successful, False otherwise
        """
        # 작업 조회
        job = await self.get_job_status(job_id)
        if not job:
            logger.error(f"Job not found: {job_id}")
            return False

        job_type = job["job_type"]
        params = json.loads(job["params"]) if job["params"] else {}

        # 핸들러 확인
        handler = self._handlers.get(job_type)
        if not handler:
            await self._mark_failed(job_id, f"No handler for job type: {job_type}")
            return False

        # 실행
        await self._mark_running(job_id)

        try:
            result_file = await handler(job_id, params, self)
            await self._mark_completed(job_id, result_file)
            logger.info(f"Job completed: {job_id} -> {result_file}")
            return True
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            await self._mark_failed(job_id, error_msg)
            logger.error(f"Job failed: {job_id} - {error_msg}")
            return False

    async def start_worker(self, poll_interval: float = 2.0) -> None:
        """백그라운드 워커 시작

        Args:
            poll_interval: 폴링 간격 (초)
        """
        if self._worker_task and not self._worker_task.done():
            logger.warning("Worker already running")
            return

        self._worker_task = asyncio.create_task(self._worker_loop(poll_interval))
        logger.info("JobQueue worker started")

    async def stop_worker(self) -> None:
        """워커 중지"""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
            logger.info("JobQueue worker stopped")

    async def _worker_loop(self, poll_interval: float) -> None:
        """워커 루프"""
        await self.initialize()

        while True:
            try:
                # 대기 중인 작업 조회
                pending_jobs = await self.get_pending_jobs(limit=1)

                if pending_jobs:
                    job = pending_jobs[0]
                    await self.process_job(job["id"])

                # 만료된 파일 정리
                await self._cleanup_expired_jobs()

            except Exception as e:
                logger.error(f"Worker error: {e}")

            await asyncio.sleep(poll_interval)

    async def _cleanup_expired_jobs(self) -> None:
        """만료된 작업 및 파일 정리"""
        now = datetime.now(KST)

        with self._get_connection() as conn:
            # 만료된 완료 작업 조회
            expired = conn.execute(
                """
                SELECT id, result_file FROM job_queue
                WHERE status IN (?, ?) AND expires_at < ?
                """,
                (JobStatus.COMPLETED.value, JobStatus.FAILED.value, now.isoformat()),
            ).fetchall()

            for row in expired:
                job_id = row["id"]
                result_file = row["result_file"]

                # 파일 삭제
                if result_file and os.path.exists(result_file):
                    try:
                        os.remove(result_file)
                        logger.info(f"Deleted expired file: {result_file}")
                    except Exception as e:
                        logger.warning(f"Failed to delete file: {result_file} - {e}")

                # DB 레코드 삭제
                conn.execute("DELETE FROM job_queue WHERE id = ?", (job_id,))
                logger.info(f"Cleaned up expired job: {job_id}")

    def get_result_file_path(self, job_id: str) -> str | None:
        """완료된 작업의 결과 파일 경로"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT result_file FROM job_queue WHERE id = ? AND status = ?",
                (job_id, JobStatus.COMPLETED.value),
            ).fetchone()

        if row and row["result_file"]:
            return row["result_file"]
        return None

    async def get_all_jobs(self, status: str | None = None, limit: int = 50) -> list[dict]:
        """모든 작업 목록 조회

        Args:
            status: 필터링할 상태 (None이면 전체)
            limit: 최대 개수
        """
        await self.initialize()

        with self._get_connection() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT * FROM job_queue
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM job_queue
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        return [dict(row) for row in rows]


# 싱글톤 인스턴스
_job_queue: JobQueue | None = None


def get_job_queue() -> JobQueue:
    """JobQueue 싱글톤 인스턴스 반환"""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
    return _job_queue

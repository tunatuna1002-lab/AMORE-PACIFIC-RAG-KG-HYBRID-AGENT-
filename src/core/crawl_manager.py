"""
Crawl Manager
일일 크롤링 **작업 제어** (job control) 및 백그라운드 실행 서비스

플로우:
1. 첫 질문 시 오늘 데이터 체크
2. 없으면 백그라운드 크롤링 시작
3. 크롤링 중에도 과거 데이터로 응답 가능
4. 완료 시 다음 응답에 알림 포함

책임 분리 (F1/F7):
- 이 클래스는 작업 제어만 담당한다: 시작/중복 방지/stale lock/진행률/완료 대기/알림,
  그리고 ``crawl_state.json`` (작업 상태) 영속화.
- 실제 파이프라인(crawl → store → kg → metrics → insight → alert → export)은
  ``BatchWorkflow.run_daily_workflow()`` 하나뿐이며, ``_run_crawl`` 은 그 결과의
  status("completed"/"partial"/"failed")를 ``CrawlStatus`` 로 매핑한다.
- 데이터 신선도/KG 상태는 ``StateManager`` (``system_state.json``) 가 단일 출처이며
  BatchWorkflow가 기록한다.
"""

import asyncio
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from src.shared.constants import KST

logger = logging.getLogger(__name__)


def _get_data_dir() -> str:
    """Railway Volume 또는 로컬 데이터 디렉토리 반환"""
    if os.environ.get("RAILWAY_ENVIRONMENT"):
        return "/data"
    return "./data"


# 한국 시간대 (UTC+9)
class CrawlStatus(Enum):
    """크롤링 상태"""

    IDLE = "idle"  # 대기 중
    RUNNING = "running"  # 크롤링 진행 중
    COMPLETED = "completed"  # 완료 (모든 카테고리 수집 + 저장 성공)
    PARTIAL = "partial"  # 부분 완료 (일부 카테고리 실패 또는 저장 오류) -> 재크롤링 대상
    FAILED = "failed"  # 실패


@dataclass
class CrawlState:
    """크롤링 상태 정보"""

    status: CrawlStatus = CrawlStatus.IDLE
    date: str | None = None  # 크롤링 대상 날짜
    started_at: str | None = None
    completed_at: str | None = None
    progress: int = 0  # 0-100
    categories_done: int = 0
    categories_total: int = 0
    products_collected: int = 0
    error: str | None = None
    # 부분 실패/저장 오류 목록 (PARTIAL/FAILED 시 비어있지 않음)
    errors: list[str] = field(default_factory=list)

    # 알림 플래그 (세션별로 관리)
    notified_sessions: set = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "date": self.date,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
            "categories_done": self.categories_done,
            "categories_total": self.categories_total,
            "products_collected": self.products_collected,
            "error": self.error,
            "errors": list(self.errors),
        }


class CrawlManager:
    """일일 크롤링 관리자"""

    STATE_FILE = f"{_get_data_dir()}/crawl_state.json"
    DATA_FILE = f"{_get_data_dir()}/dashboard_data.json"

    # BatchWorkflow 스텝 -> 진행률(%)
    STEP_PROGRESS: dict[str, int] = {
        "crawl": 30,
        "store": 50,
        "update_kg": 55,
        "calculate": 65,
        "insight": 80,
        "alert": 90,
        "export": 100,
    }

    def __init__(
        self,
        state_manager: Any | None = None,
        workflow_factory: Callable[[], Any] | None = None,
    ):
        """
        Args:
            state_manager: 단일 시스템 상태 (None이면 싱글톤). BatchWorkflow에 전달된다.
            workflow_factory: BatchWorkflow 생성 팩토리 (테스트/커스텀 DI용).
                None이면 ``Container.get_batch_workflow(state_manager=...)``.
        """
        self.state = CrawlState()
        self._crawl_task: asyncio.Task | None = None
        self._on_complete_callback: Callable | None = None
        self._state_manager = state_manager
        self._workflow_factory = workflow_factory
        self._load_state()

    @property
    def state_manager(self):
        """단일 시스템 상태 (StateManager)"""
        if self._state_manager is None:
            from src.core.state_manager import get_state_manager

            self._state_manager = get_state_manager()
        return self._state_manager

    def _create_workflow(self):
        """배치 파이프라인(BatchWorkflow) 생성 — 유일한 파이프라인 진입점"""
        if self._workflow_factory is not None:
            return self._workflow_factory()

        from src.infrastructure.container import Container

        return Container.get_batch_workflow(
            state_manager=self.state_manager, data_dir=_get_data_dir()
        )

    def _load_state(self):
        """저장된 상태 로드"""
        try:
            if Path(self.STATE_FILE).exists():
                with open(self.STATE_FILE, encoding="utf-8") as f:
                    data = json.load(f)
                    self.state = CrawlState(
                        status=CrawlStatus(data.get("status", "idle")),
                        date=data.get("date"),
                        started_at=data.get("started_at"),
                        completed_at=data.get("completed_at"),
                        progress=data.get("progress", 0),
                        categories_done=data.get("categories_done", 0),
                        categories_total=data.get("categories_total", 0),
                        products_collected=data.get("products_collected", 0),
                        error=data.get("error"),
                        errors=[str(e) for e in (data.get("errors") or [])],
                    )
        except Exception as e:
            logger.warning(f"Failed to load crawl state: {e}")

    def _save_state(self):
        """상태를 파일에 원자적으로 저장 (crash-safe)"""
        try:
            import tempfile

            Path(self.STATE_FILE).parent.mkdir(parents=True, exist_ok=True)

            # 원자적 쓰기: 임시 파일에 쓴 후 rename (crash-safe)
            dir_path = str(Path(self.STATE_FILE).parent)
            with tempfile.NamedTemporaryFile(
                mode="w", dir=dir_path, delete=False, suffix=".tmp", encoding="utf-8"
            ) as f:
                json.dump(self.state.to_dict(), f, ensure_ascii=False, indent=2)
                temp_path = f.name
            os.replace(temp_path, self.STATE_FILE)  # 원자적 교체
        except Exception as e:
            logger.error(f"Failed to save crawl state: {e}")
            # 임시 파일 정리
            try:
                if "temp_path" in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                logger.debug(f"임시 파일 정리 실패 (무시됨): {e}")

    def get_kst_today(self) -> str:
        """한국 시간 기준 오늘 날짜 반환"""
        return datetime.now(KST).date().isoformat()

    def get_data_date(self) -> str | None:
        """현재 데이터의 날짜 반환"""
        try:
            if Path(self.DATA_FILE).exists():
                with open(self.DATA_FILE, encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("metadata", {}).get("data_date")
        except Exception as e:
            logger.warning(f"Failed to read data date: {e}")
        return None

    def is_today_data_available(self) -> bool:
        """오늘(한국시간 기준) 데이터가 있는지 확인"""
        data_date = self.get_data_date()
        kst_today = self.get_kst_today()
        logger.info(f"Data date check: data={data_date}, kst_today={kst_today}")
        return data_date == kst_today

    async def check_sheets_data_exists(self, target_date: str) -> bool:
        """
        Google Sheets에서 해당 날짜의 데이터가 있는지 확인

        Args:
            target_date: 확인할 날짜 (YYYY-MM-DD)

        Returns:
            해당 날짜 데이터 존재 여부
        """
        try:
            from src.tools.storage.sheets_writer import SheetsWriter

            sheets = SheetsWriter()
            await sheets.initialize()

            # 최근 1일 데이터만 가져와서 확인
            records = await sheets.get_rank_history(days=1)

            # 해당 날짜 데이터가 있는지 확인
            for record in records:
                if record.get("snapshot_date") == target_date:
                    logger.info(f"Found data for {target_date} in Google Sheets")
                    return True

            logger.info(f"No data found for {target_date} in Google Sheets")
            return False

        except Exception as e:
            logger.warning(f"Failed to check Sheets data: {e}")
            return False

    def is_crawling(self) -> bool:
        """크롤링 진행 중인지 확인 (stale lock 감지 포함)"""
        if self.state.status != CrawlStatus.RUNNING:
            return False

        # Stale lock 감지: 2시간 이상 running 상태면 죽은 것으로 판단
        if self.state.started_at:
            try:
                started = datetime.fromisoformat(self.state.started_at)
                elapsed = datetime.now(KST) - started
                if elapsed > timedelta(hours=2):
                    logger.warning(
                        f"Stale crawl lock detected: started {self.state.started_at}, "
                        f"elapsed {elapsed}. Resetting to FAILED."
                    )
                    self.state.status = CrawlStatus.FAILED
                    self.state.error = f"Stale lock: process unresponsive for {elapsed}"
                    self.state.completed_at = datetime.now(KST).isoformat()
                    self._save_state()
                    return False
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse started_at: {e}, resetting stale lock")
                self.state.status = CrawlStatus.FAILED
                self.state.error = "Stale lock: invalid started_at timestamp"
                self.state.completed_at = datetime.now(KST).isoformat()
                self._save_state()
                return False

        return True

    def _is_partial_today(self, kst_today: str) -> bool:
        """오늘(KST) 크롤링이 PARTIAL 상태로 끝났는지 확인 (D12)

        PARTIAL은 일부 카테고리 실패 또는 저장 오류를 뜻한다. 대시보드 JSON이 오늘
        날짜로 내보내졌더라도 데이터가 불완전하므로 "완료"로 취급하지 않고 재크롤링한다.
        """
        return self.state.status == CrawlStatus.PARTIAL and self.state.date == kst_today

    def needs_crawl(self) -> bool:
        """크롤링이 필요한지 확인 (한국시간 기준)

        - RUNNING: 불필요 (진행 중)
        - 오늘 PARTIAL: 필요 (부분 데이터는 완료로 간주하지 않음, D12)
        - 오늘 데이터 존재 / 오늘 COMPLETED: 불필요
        """
        kst_today = self.get_kst_today()

        # 이미 진행 중이면 필요 없음
        if self.is_crawling():
            logger.info("Crawl not needed: already running")
            return False

        # 오늘 부분 완료(PARTIAL)면 데이터 파일과 무관하게 재크롤링 필요
        if self._is_partial_today(kst_today):
            logger.info(f"Crawl needed: today's crawl was partial ({self.state.errors})")
            return True

        # 오늘(KST) 데이터가 있으면 필요 없음
        if self.is_today_data_available():
            logger.info("Crawl not needed: today's data available")
            return False

        # 오늘(KST) 이미 완료했으면 필요 없음
        if self.state.status == CrawlStatus.COMPLETED and self.state.date == kst_today:
            logger.info("Crawl not needed: already completed today")
            return False

        logger.info(f"Crawl needed: no data for {kst_today}")
        return True

    async def needs_crawl_with_sheets_check(self) -> bool:
        """
        크롤링이 필요한지 확인 (Google Sheets까지 확인)

        로컬 파일 체크 후, 확실하지 않으면 Sheets까지 확인
        """
        kst_today = self.get_kst_today()

        # 이미 진행 중이면 필요 없음
        if self.is_crawling():
            return False

        # 로컬 파일에 오늘 데이터가 있으면 필요 없음
        if self.is_today_data_available():
            return False

        # 오늘 이미 완료했으면 필요 없음
        if self.state.status == CrawlStatus.COMPLETED and self.state.date == kst_today:
            return False

        # Google Sheets에서 최종 확인
        if await self.check_sheets_data_exists(kst_today):
            logger.info(f"Data exists in Sheets for {kst_today}, skipping crawl")
            return False

        return True

    def should_notify(self, session_id: str) -> bool:
        """해당 세션에 크롤링 완료 알림이 필요한지 확인"""
        if self.state.status != CrawlStatus.COMPLETED:
            return False
        if self.state.date != self.get_kst_today():
            return False
        if session_id in self.state.notified_sessions:
            return False
        return True

    def mark_notified(self, session_id: str):
        """세션에 알림 완료 표시"""
        self.state.notified_sessions.add(session_id)

    async def start_crawl(self, on_complete: Callable | None = None) -> bool:
        """
        백그라운드 크롤링 시작

        Returns:
            True if crawl started, False if already running
        """
        if self.is_crawling():
            logger.info("Crawl already in progress")
            return False

        self._on_complete_callback = on_complete
        self._crawl_task = asyncio.create_task(self._run_crawl())
        return True

    async def wait_for_completion(self, timeout: float | None = None) -> bool:
        """
        start_crawl()로 시작된 백그라운드 크롤링의 실제 종료를 대기 (D13)

        start_crawl()은 태스크만 생성하고 즉시 반환하므로, 스케줄러처럼 "크롤링이
        끝난 뒤" 완료 마킹을 해야 하는 호출자는 이 메서드를 await 한다.
        타임아웃 시에도 크롤링 태스크는 취소되지 않고 계속 실행된다.

        Args:
            timeout: 최대 대기 시간(초). None이면 무제한.

        Returns:
            True: 크롤링이 COMPLETED 상태로 종료됨
            False: 진행 중인 태스크 없음 / 타임아웃 / FAILED 또는 PARTIAL로 종료
        """
        task = self._crawl_task
        if task is None:
            return False

        try:
            # shield: 타임아웃으로 인해 내부 크롤링 태스크가 취소되지 않도록 보호
            await asyncio.wait_for(asyncio.shield(task), timeout)
        except TimeoutError:
            logger.warning(f"wait_for_completion timed out after {timeout}s (crawl still running)")
            return False
        except Exception as e:
            logger.error(f"Crawl task raised: {e}")
            return False

        return self.state.status == CrawlStatus.COMPLETED

    def _on_workflow_step(self, step: str, payload: dict[str, Any]) -> None:
        """BatchWorkflow 진행률 콜백: 작업 상태(progress/products)만 갱신"""
        if step == "crawl" and payload.get("status") == "completed":
            crawl_result = payload.get("result") or {}
            self.state.products_collected = crawl_result.get("total_products", 0)
            self.state.categories_done = len(crawl_result.get("categories", {}))
        self.state.progress = self.STEP_PROGRESS.get(step, self.state.progress)
        self._save_state()

    async def _run_crawl(self):
        """백그라운드 크롤링 작업 실행 (BatchWorkflow에 위임)

        - 파이프라인 status 매핑: completed → COMPLETED, partial → PARTIAL,
          그 외(failed) → FAILED. ``result["errors"]`` 는 ``state.errors`` 로 복사된다.
        - PARTIAL 은 ``needs_crawl()`` 이 재크롤링 대상으로 취급한다 (D12).
        """
        kst_today = self.get_kst_today()

        # 상태 초기화
        self.state = CrawlState(
            status=CrawlStatus.RUNNING,
            date=kst_today,
            started_at=datetime.now(KST).isoformat(),
            categories_total=5,  # config에서 가져오면 더 좋음
        )
        self._save_state()

        logger.info(f"Starting daily crawl for {kst_today} (KST)")

        workflow = None
        try:
            workflow = self._create_workflow()
            result = await workflow.run_daily_workflow(progress_callback=self._on_workflow_step)

            crawl_step = (result.get("steps") or {}).get("crawl") or {}
            crawl_result = crawl_step.get("result") or {}
            self.state.products_collected = crawl_result.get("total_products", 0)
            self.state.categories_done = len(crawl_result.get("categories", {}))
            self.state.errors = [str(e) for e in (result.get("errors") or [])]

            workflow_status = result.get("status")
            if workflow_status == "completed":
                self.state.status = CrawlStatus.COMPLETED
                logger.info(
                    f"Crawl completed for {kst_today}: {self.state.products_collected} products"
                )
            elif workflow_status == "partial":
                # D12: 일부 카테고리 실패/저장 오류 -> PARTIAL (needs_crawl()이 재시도)
                self.state.status = CrawlStatus.PARTIAL
                logger.warning(
                    f"Crawl finished PARTIAL for {kst_today}: {len(self.state.errors)} error(s)"
                )
            else:
                self.state.status = CrawlStatus.FAILED
                self.state.error = str(result.get("error") or "batch workflow failed")
                logger.error(f"Crawl failed for {kst_today}: {self.state.error}")

            self.state.progress = 100
            self.state.completed_at = datetime.now(KST).isoformat()
            self.state.notified_sessions = set()  # 알림 초기화
            self._save_state()

            if self.state.status is CrawlStatus.FAILED:
                return

            # Brain 캐시 무효화
            try:
                from src.core.brain import get_brain

                brain = await get_brain()
                if brain and hasattr(brain, "_response_pipeline") and brain._response_pipeline:
                    brain._response_pipeline._cache.clear()
                    logger.info("Brain response cache invalidated")
            except Exception as e:
                logger.warning(f"Failed to invalidate brain cache: {e}")

            # 완료 콜백 실행
            if self._on_complete_callback:
                try:
                    await self._on_complete_callback(self.state)
                except Exception as e:
                    logger.error(f"Complete callback error: {e}")

        except Exception as e:
            logger.error(f"Crawl failed: {e}")
            self.state.status = CrawlStatus.FAILED
            self.state.error = str(e)
            self.state.completed_at = datetime.now(KST).isoformat()
            self._save_state()

        finally:
            cleanup = getattr(workflow, "cleanup", None)
            if callable(cleanup):
                try:
                    await cleanup()
                except Exception as e:
                    logger.warning(f"Workflow cleanup failed (ignored): {e}")

    def get_status_message(self) -> str:
        """현재 상태 메시지 반환"""
        if self.state.status == CrawlStatus.IDLE:
            data_date = self.get_data_date()
            if data_date:
                return f"마지막 데이터: {data_date}"
            return "데이터 없음"

        elif self.state.status == CrawlStatus.RUNNING:
            return f"데이터 수집 중... ({self.state.progress}%)"

        elif self.state.status == CrawlStatus.COMPLETED:
            return f"오늘 데이터 수집 완료 ({self.state.products_collected}개 제품)"

        elif self.state.status == CrawlStatus.PARTIAL:
            return (
                f"오늘 데이터 일부 수집 ({self.state.products_collected}개 제품, "
                f"오류 {len(self.state.errors)}건)"
            )

        elif self.state.status == CrawlStatus.FAILED:
            return f"데이터 수집 실패: {self.state.error}"

        return "알 수 없음"

    def get_notification_message(self) -> str:
        """크롤링 완료 알림 메시지"""
        return (
            f"📊 **오늘({self.state.date}) 데이터 수집이 완료되었습니다!**\n\n"
            f"- 수집 제품: {self.state.products_collected}개\n"
            f"- 수집 카테고리: {self.state.categories_done}개\n"
            f"- 완료 시간: {self.state.completed_at}\n\n"
            "이제 최신 데이터로 분석을 진행합니다."
        )


# 싱글톤 인스턴스 (스레드 안전)
_crawl_manager: CrawlManager | None = None
_crawl_manager_lock = asyncio.Lock()


async def get_crawl_manager() -> CrawlManager:
    """CrawlManager 싱글톤 반환 (스레드 안전)"""
    global _crawl_manager
    async with _crawl_manager_lock:
        if _crawl_manager is None:
            _crawl_manager = CrawlManager()
    return _crawl_manager

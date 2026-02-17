"""
Crawl Manager
일일 크롤링 상태 관리 및 백그라운드 크롤링 서비스

플로우:
1. 첫 질문 시 오늘 데이터 체크
2. 없으면 백그라운드 크롤링 시작
3. 크롤링 중에도 과거 데이터로 응답 가능
4. 완료 시 다음 응답에 알림 포함
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
    COMPLETED = "completed"  # 완료
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
        }


class CrawlManager:
    """일일 크롤링 관리자"""

    STATE_FILE = f"{_get_data_dir()}/crawl_state.json"
    DATA_FILE = f"{_get_data_dir()}/dashboard_data.json"

    def __init__(self):
        self.state = CrawlState()
        self._crawl_task: asyncio.Task | None = None
        self._on_complete_callback: Callable | None = None
        self._load_state()

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

    def needs_crawl(self) -> bool:
        """크롤링이 필요한지 확인 (한국시간 기준)"""
        kst_today = self.get_kst_today()

        # 이미 진행 중이면 필요 없음
        if self.is_crawling():
            logger.info("Crawl not needed: already running")
            return False

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

    async def _run_crawl(self):
        """실제 크롤링 실행"""
        from src.infrastructure.container import Container
        from src.tools.exporters.dashboard_exporter import DashboardExporter

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

        try:
            # 1. 크롤링 실행
            crawler = Container.get_crawler_agent()
            await crawler.scraper.initialize()

            result = await crawler.execute()

            await crawler.scraper.close()

            if result.get("status") == "failed":
                raise Exception("All categories failed")

            self.state.products_collected = result.get("total_products", 0)
            self.state.categories_done = len(result.get("categories", {}))
            self.state.progress = 30
            self._save_state()

            logger.info(f"Crawl completed: {self.state.products_collected} products")

            # 크롤링 원본 데이터를 JSON으로 저장 (Excel export용)
            try:
                crawl_json_path = Path(f"{_get_data_dir()}/latest_crawl_result.json")

                # JSON 직렬화 가능한 형태로 변환 (datetime, Decimal 등 처리)
                def json_serializer(obj):
                    if hasattr(obj, "isoformat"):
                        return obj.isoformat()
                    if hasattr(obj, "__str__"):
                        return str(obj)
                    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

                with open(crawl_json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2, default=json_serializer)
                logger.info(f"Crawl result saved to {crawl_json_path}")

                # 날짜별 히스토리 데이터 저장 (raw_products 폴더)
                raw_products_dir = Path(f"{_get_data_dir()}/raw_products")
                raw_products_dir.mkdir(parents=True, exist_ok=True)

                snapshot_date = result.get("snapshot_date", datetime.now(KST).strftime("%Y-%m-%d"))
                history_path = raw_products_dir / f"{snapshot_date}.json"

                # 모든 카테고리의 제품을 플랫 리스트로 저장
                all_products = []
                for cat_id, cat_data in result.get("categories", {}).items():
                    for product in cat_data.get("products", []):
                        product["category_id"] = cat_id
                        all_products.append(product)

                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(
                        all_products, f, ensure_ascii=False, indent=2, default=json_serializer
                    )
                logger.info(
                    f"Historical data saved to {history_path} ({len(all_products)} products)"
                )
            except Exception as save_error:
                logger.error(f"Failed to save crawl result JSON: {save_error}")

            # 2. Google Sheets에 데이터 저장
            logger.info("Saving data to Google Sheets...")
            storage = Container.get_storage_agent()
            storage_result = await storage.execute(result)

            self.state.progress = 60
            self._save_state()

            if storage_result.get("errors"):
                logger.warning(f"Storage warnings: {storage_result['errors']}")
            else:
                logger.info(
                    f"Saved {storage_result.get('raw_records', 0)} records to Google Sheets"
                )

            # 3. Dashboard 데이터 생성 (Google Sheets에서 읽어옴)
            logger.info("Starting Dashboard data export...")
            try:
                exporter = DashboardExporter()
                logger.info("DashboardExporter created")
                await exporter.initialize()
                logger.info("DashboardExporter initialized")
                await exporter.export_dashboard_data(self.DATA_FILE)
                logger.info(f"Dashboard data exported to {self.DATA_FILE}")
            except Exception as export_error:
                logger.error(f"Dashboard export failed (non-fatal): {export_error}")
                # 크롤링+저장은 성공 → export 실패는 치명적이지 않으므로 계속 진행

            self.state.progress = 100
            self.state.status = CrawlStatus.COMPLETED
            self.state.completed_at = datetime.now(KST).isoformat()
            self.state.notified_sessions = set()  # 알림 초기화
            self._save_state()

            logger.info(f"Dashboard data exported for {kst_today}")

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

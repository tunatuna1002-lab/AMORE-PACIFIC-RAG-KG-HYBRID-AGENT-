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
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CrawlStatus(Enum):
    """크롤링 상태"""
    IDLE = "idle"               # 대기 중
    RUNNING = "running"         # 크롤링 진행 중
    COMPLETED = "completed"     # 완료
    FAILED = "failed"           # 실패


@dataclass
class CrawlState:
    """크롤링 상태 정보"""
    status: CrawlStatus = CrawlStatus.IDLE
    date: Optional[str] = None  # 크롤링 대상 날짜
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: int = 0  # 0-100
    categories_done: int = 0
    categories_total: int = 0
    products_collected: int = 0
    error: Optional[str] = None

    # 알림 플래그 (세션별로 관리)
    notified_sessions: set = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "date": self.date,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
            "categories_done": self.categories_done,
            "categories_total": self.categories_total,
            "products_collected": self.products_collected,
            "error": self.error
        }


class CrawlManager:
    """일일 크롤링 관리자"""

    STATE_FILE = "./data/crawl_state.json"
    DATA_FILE = "./data/dashboard_data.json"

    def __init__(self):
        self.state = CrawlState()
        self._crawl_task: Optional[asyncio.Task] = None
        self._on_complete_callback: Optional[Callable] = None
        self._load_state()

    def _load_state(self):
        """저장된 상태 로드"""
        try:
            if Path(self.STATE_FILE).exists():
                with open(self.STATE_FILE, "r", encoding="utf-8") as f:
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
                        error=data.get("error")
                    )
        except Exception as e:
            logger.warning(f"Failed to load crawl state: {e}")

    def _save_state(self):
        """상태 저장"""
        try:
            Path(self.STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
            with open(self.STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.state.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save crawl state: {e}")

    def get_data_date(self) -> Optional[str]:
        """현재 데이터의 날짜 반환"""
        try:
            if Path(self.DATA_FILE).exists():
                with open(self.DATA_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("metadata", {}).get("data_date")
        except Exception as e:
            logger.warning(f"Failed to read data date: {e}")
        return None

    def is_today_data_available(self) -> bool:
        """오늘 데이터가 있는지 확인"""
        data_date = self.get_data_date()
        today = date.today().isoformat()
        return data_date == today

    def is_crawling(self) -> bool:
        """크롤링 진행 중인지 확인"""
        return self.state.status == CrawlStatus.RUNNING

    def needs_crawl(self) -> bool:
        """크롤링이 필요한지 확인"""
        # 이미 진행 중이면 필요 없음
        if self.is_crawling():
            return False

        # 오늘 데이터가 있으면 필요 없음
        if self.is_today_data_available():
            return False

        # 오늘 이미 완료했으면 필요 없음
        if (self.state.status == CrawlStatus.COMPLETED and
            self.state.date == date.today().isoformat()):
            return False

        return True

    def should_notify(self, session_id: str) -> bool:
        """해당 세션에 크롤링 완료 알림이 필요한지 확인"""
        if self.state.status != CrawlStatus.COMPLETED:
            return False
        if self.state.date != date.today().isoformat():
            return False
        if session_id in self.state.notified_sessions:
            return False
        return True

    def mark_notified(self, session_id: str):
        """세션에 알림 완료 표시"""
        self.state.notified_sessions.add(session_id)

    async def start_crawl(self, on_complete: Optional[Callable] = None) -> bool:
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
        from agents.crawler_agent import CrawlerAgent
        from agents.storage_agent import StorageAgent
        from tools.dashboard_exporter import DashboardExporter

        today = date.today().isoformat()

        # 상태 초기화
        self.state = CrawlState(
            status=CrawlStatus.RUNNING,
            date=today,
            started_at=datetime.now().isoformat(),
            categories_total=5  # config에서 가져오면 더 좋음
        )
        self._save_state()

        logger.info(f"Starting daily crawl for {today}")

        try:
            # 1. 크롤링 실행
            crawler = CrawlerAgent()
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

            # 2. Google Sheets에 데이터 저장
            logger.info("Saving data to Google Sheets...")
            storage = StorageAgent()
            storage_result = await storage.execute(result)

            self.state.progress = 60
            self._save_state()

            if storage_result.get("errors"):
                logger.warning(f"Storage warnings: {storage_result['errors']}")
            else:
                logger.info(f"Saved {storage_result.get('raw_records', 0)} records to Google Sheets")

            # 3. Dashboard 데이터 생성 (Google Sheets에서 읽어옴)
            exporter = DashboardExporter()
            await exporter.initialize()
            await exporter.export_dashboard_data(self.DATA_FILE)

            self.state.progress = 100
            self.state.status = CrawlStatus.COMPLETED
            self.state.completed_at = datetime.now().isoformat()
            self.state.notified_sessions = set()  # 알림 초기화
            self._save_state()

            logger.info(f"Dashboard data exported for {today}")

            # SimpleChatService 캐시 무효화
            try:
                from core.simple_chat import get_chat_service
                chat_service = get_chat_service()
                chat_service.invalidate_cache()
                logger.info("Chat service cache invalidated")
            except Exception as e:
                logger.warning(f"Failed to invalidate chat cache: {e}")

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
            self.state.completed_at = datetime.now().isoformat()
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


# 싱글톤 인스턴스
_crawl_manager: Optional[CrawlManager] = None


def get_crawl_manager() -> CrawlManager:
    """CrawlManager 싱글톤 반환"""
    global _crawl_manager
    if _crawl_manager is None:
        _crawl_manager = CrawlManager()
    return _crawl_manager

"""
오케스트레이터 상태 관리
========================
시스템 상태 추적 및 판단 지원

관리 상태:
- 크롤링 시간 및 데이터 신선도
- KG 초기화 상태
- 세션 정보
- 실행 중인 도구 목록

Usage:
    state = OrchestratorState()

    if state.is_crawl_needed():
        await crawler.execute()
        state.mark_crawled()
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorState:
    """
    오케스트레이터 상태

    시스템의 현재 상태를 추적하여 LLM 판단에 활용.
    파일 기반 영속화를 지원하여 재시작 시에도 상태 유지.

    Attributes:
        last_crawl_time: 마지막 크롤링 완료 시간
        data_freshness: 데이터 신선도 (fresh/stale/unknown)
        kg_initialized: KG 초기화 완료 여부
        kg_triple_count: KG에 저장된 트리플 수
        current_session_id: 현재 활성 세션 ID
        active_tools: 현재 실행 중인 도구 목록
        last_metrics_time: 마지막 지표 계산 시간
    """

    last_crawl_time: datetime | None = None
    data_freshness: str = "unknown"
    kg_initialized: bool = False
    kg_triple_count: int = 0
    current_session_id: str | None = None
    active_tools: list[str] = field(default_factory=list)
    last_metrics_time: datetime | None = None

    # 영속화 경로
    _persist_path: Path | None = field(default=None, repr=False)

    def __post_init__(self):
        """초기화 후 영속화된 상태 로드 시도"""
        if self._persist_path is None:
            self._persist_path = Path("./data/orchestrator_state.json")

        self._load_state()

    # =========================================================================
    # 크롤링 상태 관리
    # =========================================================================

    def is_crawl_needed(self) -> bool:
        """
        오늘 크롤링이 필요한지 판단

        Returns:
            True면 크롤링 필요
        """
        if self.last_crawl_time is None:
            return True

        # 오늘 날짜와 비교
        return self.last_crawl_time.date() < date.today()

    def mark_crawled(self, products_count: int = 0) -> None:
        """
        크롤링 완료 표시

        Args:
            products_count: 수집된 제품 수
        """
        self.last_crawl_time = datetime.now()
        self.data_freshness = "fresh"

        logger.info(f"Crawl completed: {products_count} products at {self.last_crawl_time}")
        self._save_state()

    def mark_data_stale(self) -> None:
        """데이터를 stale 상태로 표시"""
        self.data_freshness = "stale"
        self._save_state()

    def get_data_age_hours(self) -> float | None:
        """
        데이터 경과 시간 (시간 단위)

        Returns:
            경과 시간 (크롤링 기록 없으면 None)
        """
        if self.last_crawl_time is None:
            return None

        delta = datetime.now() - self.last_crawl_time
        return delta.total_seconds() / 3600

    # =========================================================================
    # 지표 계산 상태
    # =========================================================================

    def mark_metrics_calculated(self) -> None:
        """지표 계산 완료 표시"""
        self.last_metrics_time = datetime.now()
        self._save_state()

    def is_metrics_fresh(self, max_age_hours: float = 24) -> bool:
        """
        지표가 최신인지 확인

        Args:
            max_age_hours: 최대 허용 경과 시간

        Returns:
            True면 최신
        """
        if self.last_metrics_time is None:
            return False

        delta = datetime.now() - self.last_metrics_time
        return delta.total_seconds() / 3600 < max_age_hours

    # =========================================================================
    # KG 상태 관리
    # =========================================================================

    def mark_kg_initialized(self, triple_count: int = 0) -> None:
        """
        KG 초기화 완료 표시

        Args:
            triple_count: KG 트리플 수
        """
        self.kg_initialized = True
        self.kg_triple_count = triple_count
        logger.info(f"KG initialized: {triple_count} triples")
        self._save_state()

    def update_kg_stats(self, triple_count: int) -> None:
        """KG 통계 업데이트"""
        self.kg_triple_count = triple_count
        self._save_state()

    # =========================================================================
    # 도구 실행 상태
    # =========================================================================

    def start_tool(self, tool_name: str) -> None:
        """도구 실행 시작 표시"""
        if tool_name not in self.active_tools:
            self.active_tools.append(tool_name)
            logger.debug(f"Tool started: {tool_name}")

    def end_tool(self, tool_name: str) -> None:
        """도구 실행 완료 표시"""
        if tool_name in self.active_tools:
            self.active_tools.remove(tool_name)
            logger.debug(f"Tool ended: {tool_name}")

    def is_tool_running(self, tool_name: str) -> bool:
        """특정 도구가 실행 중인지 확인"""
        return tool_name in self.active_tools

    def has_active_tools(self) -> bool:
        """실행 중인 도구가 있는지 확인"""
        return len(self.active_tools) > 0

    # =========================================================================
    # 세션 관리
    # =========================================================================

    def set_session(self, session_id: str) -> None:
        """현재 세션 설정"""
        self.current_session_id = session_id

    def clear_session(self) -> None:
        """세션 초기화"""
        self.current_session_id = None

    # =========================================================================
    # 직렬화
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """상태를 딕셔너리로 변환"""
        return {
            "last_crawl_time": self.last_crawl_time.isoformat() if self.last_crawl_time else None,
            "data_freshness": self.data_freshness,
            "kg_initialized": self.kg_initialized,
            "kg_triple_count": self.kg_triple_count,
            "current_session_id": self.current_session_id,
            "active_tools": self.active_tools,
            "last_metrics_time": self.last_metrics_time.isoformat()
            if self.last_metrics_time
            else None,
        }

    def to_context_summary(self) -> str:
        """LLM 컨텍스트용 요약 생성"""
        parts = []

        # 크롤링 상태
        if self.last_crawl_time:
            age = self.get_data_age_hours()
            age_str = f"{age:.1f}시간 전" if age else "알 수 없음"
            parts.append(f"마지막 크롤링: {age_str}")
        else:
            parts.append("마지막 크롤링: 없음")

        parts.append(f"데이터 신선도: {self.data_freshness}")

        # KG 상태
        if self.kg_initialized:
            parts.append(f"KG: 초기화됨 ({self.kg_triple_count} 트리플)")
        else:
            parts.append("KG: 미초기화")

        return " | ".join(parts)

    # =========================================================================
    # 영속화
    # =========================================================================

    def _save_state(self) -> None:
        """상태를 파일에 저장"""
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)

            state_dict = self.to_dict()
            state_dict["saved_at"] = datetime.now().isoformat()

            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, ensure_ascii=False, indent=2)

            logger.debug(f"State saved to {self._persist_path}")

        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def _load_state(self) -> None:
        """파일에서 상태 로드"""
        try:
            if not self._persist_path.exists():
                return

            with open(self._persist_path, encoding="utf-8") as f:
                data = json.load(f)

            # 값 복원
            if data.get("last_crawl_time"):
                self.last_crawl_time = datetime.fromisoformat(data["last_crawl_time"])

            if data.get("last_metrics_time"):
                self.last_metrics_time = datetime.fromisoformat(data["last_metrics_time"])

            self.data_freshness = data.get("data_freshness", "unknown")
            self.kg_initialized = data.get("kg_initialized", False)
            self.kg_triple_count = data.get("kg_triple_count", 0)

            logger.debug(f"State loaded from {self._persist_path}")

        except Exception as e:
            logger.warning(f"Failed to load state: {e}")

    def reset(self) -> None:
        """상태 전체 초기화"""
        self.last_crawl_time = None
        self.data_freshness = "unknown"
        self.kg_initialized = False
        self.kg_triple_count = 0
        self.current_session_id = None
        self.active_tools = []
        self.last_metrics_time = None

        # 파일도 삭제
        if self._persist_path.exists():
            self._persist_path.unlink()

        logger.info("State reset")

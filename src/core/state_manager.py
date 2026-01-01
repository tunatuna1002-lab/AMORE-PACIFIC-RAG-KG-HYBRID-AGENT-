"""
상태 관리자 (State Manager)
===========================
Level 4 Autonomous Agent의 통합 상태 관리

역할:
1. 시스템 전체 상태 통합 관리
2. 이벤트/알림 구독 관리
3. 사용자 설정 (이메일 동의 등) 관리
4. 상태 영속화

관리하는 상태:
- 크롤링 상태 (마지막 시간, 성공 여부)
- KG 상태 (초기화, 트리플 수)
- 에이전트 상태 (실행 중, 완료, 실패)
- 사용자 설정 (이메일 동의, 알림 설정)
- 세션 정보

Usage:
    state_manager = StateManager()

    # 크롤링 완료 표시
    state_manager.mark_crawled(products_count=150)

    # 이메일 동의 등록
    state_manager.register_email("user@example.com", consent=True)

    # 알림 대상자 조회
    recipients = state_manager.get_alert_recipients("rank_change")
"""

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# 타입 정의
# =============================================================================

class AlertType(Enum):
    """알림 유형"""
    RANK_CHANGE = "rank_change"          # 순위 변동
    IMPORTANT_INSIGHT = "important_insight"  # 중요 인사이트
    CRAWL_COMPLETE = "crawl_complete"    # 크롤링 완료
    ERROR = "error"                       # 에러
    DAILY_SUMMARY = "daily_summary"      # 일일 요약


class DataFreshness(Enum):
    """데이터 신선도"""
    FRESH = "fresh"        # 오늘 데이터
    STALE = "stale"        # 오래된 데이터
    UNKNOWN = "unknown"    # 알 수 없음


@dataclass
class EmailSubscription:
    """이메일 구독 정보"""
    email: str
    consent: bool                    # 명시적 동의 여부
    consent_date: Optional[datetime] = None
    alert_types: List[str] = field(default_factory=list)  # 구독 알림 유형
    active: bool = True              # 활성 상태

    def to_dict(self) -> Dict[str, Any]:
        return {
            "email": self.email,
            "consent": self.consent,
            "consent_date": self.consent_date.isoformat() if self.consent_date else None,
            "alert_types": self.alert_types,
            "active": self.active
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmailSubscription":
        consent_date = None
        if data.get("consent_date"):
            consent_date = datetime.fromisoformat(data["consent_date"])

        return cls(
            email=data["email"],
            consent=data.get("consent", False),
            consent_date=consent_date,
            alert_types=data.get("alert_types", []),
            active=data.get("active", True)
        )


@dataclass
class AgentStatus:
    """에이전트 상태"""
    name: str
    status: str  # idle, running, completed, failed
    last_run: Optional[datetime] = None
    last_error: Optional[str] = None
    run_count: int = 0
    success_count: int = 0
    fail_count: int = 0


# =============================================================================
# 상태 관리자
# =============================================================================

class StateManager:
    """
    통합 상태 관리자

    시스템 전체의 상태를 관리하고 영속화합니다.

    주요 기능:
    1. 크롤링/KG 상태 관리
    2. 이메일 구독 관리 (명시적 동의 기반)
    3. 에이전트 상태 추적
    4. 상태 영속화 (JSON 파일)
    """

    def __init__(
        self,
        persist_dir: Optional[Path] = None
    ):
        """
        Args:
            persist_dir: 영속화 디렉토리 (기본: ./data)
        """
        if persist_dir is None:
            persist_dir = Path("./data")

        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # 크롤링 상태
        self.last_crawl_time: Optional[datetime] = None
        self.last_crawl_success: bool = False
        self.last_crawl_count: int = 0
        self.data_freshness: DataFreshness = DataFreshness.UNKNOWN

        # KG 상태
        self.kg_initialized: bool = False
        self.kg_triple_count: int = 0
        self.kg_last_update: Optional[datetime] = None

        # 에이전트 상태
        self._agent_status: Dict[str, AgentStatus] = {}
        self._active_tools: Set[str] = set()

        # 이메일 구독
        self._email_subscriptions: Dict[str, EmailSubscription] = {}

        # 세션
        self.current_session_id: Optional[str] = None

        # 메트릭 상태
        self.last_metrics_time: Optional[datetime] = None

        # 로드
        self._load_state()
        self._load_subscriptions()

    # =========================================================================
    # 크롤링 상태 관리
    # =========================================================================

    def is_crawl_needed(self) -> bool:
        """오늘 크롤링이 필요한지 판단"""
        if self.last_crawl_time is None:
            return True

        return self.last_crawl_time.date() < date.today()

    def mark_crawled(
        self,
        success: bool = True,
        products_count: int = 0
    ) -> None:
        """
        크롤링 완료 표시

        Args:
            success: 성공 여부
            products_count: 수집된 제품 수
        """
        self.last_crawl_time = datetime.now()
        self.last_crawl_success = success
        self.last_crawl_count = products_count
        self.data_freshness = DataFreshness.FRESH if success else DataFreshness.UNKNOWN

        logger.info(f"Crawl marked: success={success}, count={products_count}")
        self._save_state()

    def mark_data_stale(self) -> None:
        """데이터를 stale로 표시"""
        self.data_freshness = DataFreshness.STALE
        self._save_state()

    def get_data_age_hours(self) -> Optional[float]:
        """데이터 경과 시간 (시간 단위)"""
        if self.last_crawl_time is None:
            return None

        delta = datetime.now() - self.last_crawl_time
        return delta.total_seconds() / 3600

    # =========================================================================
    # KG 상태 관리
    # =========================================================================

    def mark_kg_initialized(self, triple_count: int = 0) -> None:
        """KG 초기화 완료"""
        self.kg_initialized = True
        self.kg_triple_count = triple_count
        self.kg_last_update = datetime.now()
        logger.info(f"KG initialized: {triple_count} triples")
        self._save_state()

    def update_kg_stats(self, triple_count: int) -> None:
        """KG 통계 업데이트"""
        self.kg_triple_count = triple_count
        self.kg_last_update = datetime.now()
        self._save_state()

    # =========================================================================
    # 에이전트 상태 관리
    # =========================================================================

    def start_agent(self, name: str) -> None:
        """에이전트 시작"""
        if name not in self._agent_status:
            self._agent_status[name] = AgentStatus(name=name, status="idle")

        self._agent_status[name].status = "running"
        self._agent_status[name].run_count += 1
        self._active_tools.add(name)
        logger.debug(f"Agent started: {name}")

    def complete_agent(self, name: str, success: bool = True, error: Optional[str] = None) -> None:
        """에이전트 완료"""
        if name not in self._agent_status:
            self._agent_status[name] = AgentStatus(name=name, status="idle")

        status = self._agent_status[name]
        status.last_run = datetime.now()

        if success:
            status.status = "completed"
            status.success_count += 1
            status.last_error = None
        else:
            status.status = "failed"
            status.fail_count += 1
            status.last_error = error

        self._active_tools.discard(name)
        logger.debug(f"Agent completed: {name}, success={success}")

    def is_agent_running(self, name: str) -> bool:
        """에이전트 실행 중 여부"""
        return name in self._active_tools

    def has_active_agents(self) -> bool:
        """실행 중인 에이전트 있는지"""
        return len(self._active_tools) > 0

    def get_agent_status(self, name: str) -> Optional[AgentStatus]:
        """에이전트 상태 조회"""
        return self._agent_status.get(name)

    def get_all_agent_statuses(self) -> Dict[str, AgentStatus]:
        """모든 에이전트 상태"""
        return self._agent_status.copy()

    # =========================================================================
    # 이메일 구독 관리 (명시적 동의 기반)
    # =========================================================================

    def register_email(
        self,
        email: str,
        consent: bool,
        alert_types: Optional[List[str]] = None
    ) -> bool:
        """
        이메일 등록 (명시적 동의 필수)

        Args:
            email: 이메일 주소
            consent: 동의 여부 (체크박스)
            alert_types: 구독할 알림 유형

        Returns:
            등록 성공 여부
        """
        if not consent:
            logger.warning(f"Email registration rejected: consent=False for {email}")
            return False

        if not self._validate_email(email):
            logger.warning(f"Invalid email format: {email}")
            return False

        # 기본 알림 유형
        if alert_types is None:
            alert_types = [
                AlertType.RANK_CHANGE.value,
                AlertType.IMPORTANT_INSIGHT.value,
                AlertType.ERROR.value
            ]

        subscription = EmailSubscription(
            email=email,
            consent=True,
            consent_date=datetime.now(),
            alert_types=alert_types,
            active=True
        )

        self._email_subscriptions[email] = subscription
        logger.info(f"Email registered: {email}, alerts={alert_types}")
        self._save_subscriptions()
        return True

    def update_email_subscription(
        self,
        email: str,
        alert_types: Optional[List[str]] = None,
        active: Optional[bool] = None
    ) -> bool:
        """
        이메일 구독 설정 변경

        Args:
            email: 이메일 주소
            alert_types: 새 알림 유형
            active: 활성 상태
        """
        if email not in self._email_subscriptions:
            return False

        subscription = self._email_subscriptions[email]

        if alert_types is not None:
            subscription.alert_types = alert_types

        if active is not None:
            subscription.active = active

        self._save_subscriptions()
        return True

    def revoke_email_consent(self, email: str) -> bool:
        """이메일 동의 철회"""
        if email not in self._email_subscriptions:
            return False

        self._email_subscriptions[email].consent = False
        self._email_subscriptions[email].active = False
        logger.info(f"Email consent revoked: {email}")
        self._save_subscriptions()
        return True

    def get_alert_recipients(self, alert_type: str) -> List[str]:
        """
        특정 알림 유형의 수신자 목록

        Args:
            alert_type: 알림 유형

        Returns:
            이메일 주소 목록
        """
        recipients = []
        for email, sub in self._email_subscriptions.items():
            if sub.active and sub.consent:
                if alert_type in sub.alert_types:
                    recipients.append(email)
        return recipients

    def get_subscription(self, email: str) -> Optional[EmailSubscription]:
        """구독 정보 조회"""
        return self._email_subscriptions.get(email)

    def get_all_subscriptions(self) -> Dict[str, EmailSubscription]:
        """모든 구독 정보"""
        return self._email_subscriptions.copy()

    def _validate_email(self, email: str) -> bool:
        """이메일 형식 검증"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    # =========================================================================
    # 세션 관리
    # =========================================================================

    def set_session(self, session_id: str) -> None:
        """세션 설정"""
        self.current_session_id = session_id

    def clear_session(self) -> None:
        """세션 초기화"""
        self.current_session_id = None

    # =========================================================================
    # 메트릭 상태
    # =========================================================================

    def mark_metrics_calculated(self) -> None:
        """지표 계산 완료"""
        self.last_metrics_time = datetime.now()
        self._save_state()

    def is_metrics_fresh(self, max_age_hours: float = 24) -> bool:
        """지표가 최신인지"""
        if self.last_metrics_time is None:
            return False

        delta = datetime.now() - self.last_metrics_time
        return delta.total_seconds() / 3600 < max_age_hours

    # =========================================================================
    # 직렬화
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """상태를 딕셔너리로"""
        return {
            "last_crawl_time": self.last_crawl_time.isoformat() if self.last_crawl_time else None,
            "last_crawl_success": self.last_crawl_success,
            "last_crawl_count": self.last_crawl_count,
            "data_freshness": self.data_freshness.value,
            "kg_initialized": self.kg_initialized,
            "kg_triple_count": self.kg_triple_count,
            "kg_last_update": self.kg_last_update.isoformat() if self.kg_last_update else None,
            "last_metrics_time": self.last_metrics_time.isoformat() if self.last_metrics_time else None
        }

    def to_context_summary(self) -> str:
        """LLM 컨텍스트용 요약"""
        parts = []

        # 크롤링 상태
        if self.last_crawl_time:
            age = self.get_data_age_hours()
            age_str = f"{age:.1f}시간 전" if age else "알 수 없음"
            parts.append(f"크롤링: {age_str}")
        else:
            parts.append("크롤링: 없음")

        parts.append(f"데이터: {self.data_freshness.value}")

        # KG 상태
        if self.kg_initialized:
            parts.append(f"KG: {self.kg_triple_count} 트리플")
        else:
            parts.append("KG: 미초기화")

        # 활성 에이전트
        if self._active_tools:
            parts.append(f"실행 중: {', '.join(self._active_tools)}")

        return " | ".join(parts)

    # =========================================================================
    # 영속화
    # =========================================================================

    def _save_state(self) -> None:
        """상태 저장"""
        try:
            state_path = self.persist_dir / "system_state.json"
            state_dict = self.to_dict()
            state_dict["saved_at"] = datetime.now().isoformat()

            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def _load_state(self) -> None:
        """상태 로드"""
        try:
            state_path = self.persist_dir / "system_state.json"
            if not state_path.exists():
                return

            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("last_crawl_time"):
                self.last_crawl_time = datetime.fromisoformat(data["last_crawl_time"])

            if data.get("kg_last_update"):
                self.kg_last_update = datetime.fromisoformat(data["kg_last_update"])

            if data.get("last_metrics_time"):
                self.last_metrics_time = datetime.fromisoformat(data["last_metrics_time"])

            self.last_crawl_success = data.get("last_crawl_success", False)
            self.last_crawl_count = data.get("last_crawl_count", 0)
            self.data_freshness = DataFreshness(data.get("data_freshness", "unknown"))
            self.kg_initialized = data.get("kg_initialized", False)
            self.kg_triple_count = data.get("kg_triple_count", 0)

            logger.debug("State loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load state: {e}")

    def _save_subscriptions(self) -> None:
        """구독 정보 저장"""
        try:
            sub_path = self.persist_dir / "email_subscriptions.json"
            data = {
                email: sub.to_dict()
                for email, sub in self._email_subscriptions.items()
            }
            data["saved_at"] = datetime.now().isoformat()

            with open(sub_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save subscriptions: {e}")

    def _load_subscriptions(self) -> None:
        """구독 정보 로드"""
        try:
            sub_path = self.persist_dir / "email_subscriptions.json"
            if not sub_path.exists():
                return

            with open(sub_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for email, sub_data in data.items():
                if email == "saved_at":
                    continue
                self._email_subscriptions[email] = EmailSubscription.from_dict(sub_data)

            logger.debug(f"Loaded {len(self._email_subscriptions)} email subscriptions")

        except Exception as e:
            logger.warning(f"Failed to load subscriptions: {e}")

    def reset(self) -> None:
        """전체 상태 초기화"""
        self.last_crawl_time = None
        self.last_crawl_success = False
        self.last_crawl_count = 0
        self.data_freshness = DataFreshness.UNKNOWN
        self.kg_initialized = False
        self.kg_triple_count = 0
        self.kg_last_update = None
        self._agent_status = {}
        self._active_tools = set()
        self.current_session_id = None
        self.last_metrics_time = None

        # 파일 삭제
        state_path = self.persist_dir / "system_state.json"
        if state_path.exists():
            state_path.unlink()

        logger.info("State manager reset")


# =============================================================================
# 싱글톤
# =============================================================================

_state_manager_instance: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """상태 관리자 싱글톤"""
    global _state_manager_instance

    if _state_manager_instance is None:
        _state_manager_instance = StateManager()
        logger.info("StateManager singleton created")

    return _state_manager_instance


def reset_state_manager() -> None:
    """싱글톤 리셋"""
    global _state_manager_instance
    _state_manager_instance = None

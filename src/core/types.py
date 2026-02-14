"""
Core Type Definitions
=====================
Brain 및 Orchestrator용 공통 타입 정의

이 모듈은 순환 참조를 방지하기 위해 타입 정의만 포함합니다.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class BrainMode(Enum):
    """두뇌 동작 모드"""

    IDLE = "idle"  # 대기
    AUTONOMOUS = "autonomous"  # 자율 작업 중
    RESPONDING = "responding"  # 사용자 응답 중
    EXECUTING = "executing"  # 에이전트 실행 중
    ALERTING = "alerting"  # 알림 처리 중


class TaskPriority(Enum):
    """작업 우선순위"""

    USER_REQUEST = 0  # 사용자 요청 (최우선)
    CRITICAL_ALERT = 1  # 중요 알림
    SCHEDULED = 2  # 예약 작업
    BACKGROUND = 3  # 백그라운드 작업


class ErrorStrategy(Enum):
    """에러 처리 전략"""

    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    NOTIFY_USER = "notify_user"


@dataclass
class BrainTask:
    """두뇌가 처리할 작업"""

    id: str
    type: str  # query, scheduled, alert, autonomous
    priority: TaskPriority
    payload: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any | None = None
    error: str | None = None

    def __lt__(self, other):
        """우선순위 기반 정렬 (힙큐용)"""
        return self.priority.value < other.priority.value

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "id": self.id,
            "type": self.type,
            "priority": self.priority.value,
            "payload": self.payload,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class AgentError:
    """에이전트 에러 정보"""

    agent_name: str
    error_message: str
    error_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent_name,
            "message": self.error_message,
            "type": self.error_type,
            "timestamp": self.timestamp.isoformat(),
            "retry_count": self.retry_count,
        }


# 에이전트별 에러 전략 매핑
AGENT_ERROR_STRATEGIES: dict[str, ErrorStrategy] = {
    "crawl_amazon": ErrorStrategy.FALLBACK,
    "calculate_metrics": ErrorStrategy.RETRY,
    "query_data": ErrorStrategy.FALLBACK,
    "query_knowledge_graph": ErrorStrategy.SKIP,
    "generate_insight": ErrorStrategy.RETRY,
    "send_alert": ErrorStrategy.RETRY,
    "workflow": ErrorStrategy.RETRY,
}


# 타입 별칭
MetricsDict = dict[str, Any]
ContextDict = dict[str, Any]
ToolParams = dict[str, Any]

"""
Session Manager
현재 실행 세션 상태 관리
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SessionStatus(str, Enum):
    """세션 상태"""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStatus(str, Enum):
    """에이전트 상태"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AgentState:
    """개별 에이전트 상태"""

    agent_name: str
    status: AgentStatus = AgentStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    input_data: dict | None = None
    output_data: dict | None = None
    error_message: str | None = None


@dataclass
class Session:
    """실행 세션"""

    session_id: str
    status: SessionStatus = SessionStatus.CREATED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    agents: dict[str, AgentState] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


class SessionManager:
    """세션 관리자"""

    AGENT_EXECUTION_ORDER = ["crawler_agent", "storage_agent", "metrics_agent", "insight_agent"]

    def __init__(self):
        """세션 관리자 초기화"""
        self._current_session: Session | None = None
        self._sessions: dict[str, Session] = {}

    def create_session(self) -> str:
        """
        새 세션 생성

        Returns:
            생성된 세션 ID
        """
        session_id = str(uuid.uuid4())
        session = Session(session_id=session_id, status=SessionStatus.CREATED)

        # 에이전트 상태 초기화
        for agent_name in self.AGENT_EXECUTION_ORDER:
            session.agents[agent_name] = AgentState(agent_name=agent_name)

        self._sessions[session_id] = session
        self._current_session = session

        return session_id

    def get_current_session(self) -> Session | None:
        """현재 세션 반환"""
        return self._current_session

    def get_session(self, session_id: str) -> Session | None:
        """특정 세션 반환"""
        return self._sessions.get(session_id)

    def start_session(self) -> None:
        """세션 시작"""
        if self._current_session:
            self._current_session.status = SessionStatus.RUNNING
            self._current_session.started_at = datetime.now()

    def complete_session(self, success: bool = True, error: str | None = None) -> None:
        """세션 완료"""
        if self._current_session:
            self._current_session.completed_at = datetime.now()
            if success:
                self._current_session.status = SessionStatus.COMPLETED
            else:
                self._current_session.status = SessionStatus.FAILED
                self._current_session.error_message = error

    def end_session(self, session_id: str) -> dict[str, Any]:
        """
        세션 종료 및 요약 반환

        Args:
            session_id: 세션 ID

        Returns:
            세션 요약
        """
        self.complete_session(success=True)
        return self.get_session_summary()

    def fail_agent(self, session_id: str, agent_name: str, error: str) -> None:
        """에이전트 실패 처리"""
        self.complete_agent(session_id, agent_name, success=False, error=error)

    def start_agent(self, session_id: str, agent_name: str, input_data: dict | None = None) -> None:
        """
        에이전트 시작

        Args:
            session_id: 세션 ID
            agent_name: 에이전트 이름
            input_data: 입력 데이터
        """
        if not self._current_session:
            return

        if agent_name in self._current_session.agents:
            agent = self._current_session.agents[agent_name]
            agent.status = AgentStatus.RUNNING
            agent.started_at = datetime.now()
            agent.input_data = input_data

    def complete_agent(
        self,
        session_id: str,
        agent_name: str,
        output_data: dict | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """
        에이전트 완료

        Args:
            session_id: 세션 ID
            agent_name: 에이전트 이름
            output_data: 출력 데이터
            success: 성공 여부
            error: 에러 메시지
        """
        if not self._current_session:
            return

        if agent_name in self._current_session.agents:
            agent = self._current_session.agents[agent_name]
            agent.completed_at = datetime.now()
            agent.output_data = output_data

            if success:
                agent.status = AgentStatus.COMPLETED
            else:
                agent.status = AgentStatus.FAILED
                agent.error_message = error

    def set_context(self, key: str, value: Any) -> None:
        """컨텍스트 값 설정"""
        if self._current_session:
            self._current_session.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """컨텍스트 값 조회"""
        if self._current_session:
            return self._current_session.context.get(key, default)
        return default

    def get_agent_output(self, agent_name: str) -> dict | None:
        """에이전트 출력 조회"""
        if not self._current_session:
            return None

        agent = self._current_session.agents.get(agent_name)
        return agent.output_data if agent else None

    def get_next_agent(self) -> str | None:
        """다음 실행할 에이전트 반환"""
        if not self._current_session:
            return None

        for agent_name in self.AGENT_EXECUTION_ORDER:
            agent = self._current_session.agents.get(agent_name)
            if agent and agent.status == AgentStatus.PENDING:
                return agent_name

        return None

    def get_session_summary(self) -> dict[str, Any]:
        """세션 요약 반환"""
        if not self._current_session:
            return {}

        session = self._current_session
        agent_summaries = {}

        for name, agent in session.agents.items():
            duration = None
            if agent.started_at and agent.completed_at:
                duration = (agent.completed_at - agent.started_at).total_seconds()

            agent_summaries[name] = {
                "status": agent.status.value,
                "duration_seconds": duration,
                "has_output": agent.output_data is not None,
                "error": agent.error_message,
            }

        total_duration = None
        if session.started_at and session.completed_at:
            total_duration = (session.completed_at - session.started_at).total_seconds()

        return {
            "session_id": session.session_id,
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "total_duration_seconds": total_duration,
            "agents": agent_summaries,
        }

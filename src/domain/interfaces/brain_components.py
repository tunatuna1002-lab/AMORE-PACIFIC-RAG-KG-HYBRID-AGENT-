"""
Brain Component Protocols
=========================
UnifiedBrain 분해를 위한 Protocol 정의

SRP(Single Responsibility Principle)에 따라 분리된 컴포넌트들의 인터페이스입니다.

컴포넌트:
- QueryProcessorProtocol: 사용자 질문 처리
- DecisionMakerProtocol: LLM 의사결정
- ToolCoordinatorProtocol: 도구 실행 조율
- AlertManagerProtocol: 알림 처리
"""

from typing import Any, Protocol, runtime_checkable

from src.core.models import Context, Response, ToolResult

# =============================================================================
# Query Processor Protocol
# =============================================================================


@runtime_checkable
class QueryProcessorProtocol(Protocol):
    """
    사용자 질문 처리 Protocol

    책임:
    - 사용자 질문 전처리
    - 의사결정 → 도구 실행 → 응답 생성 파이프라인 조율
    """

    async def process(
        self,
        query: str,
        context: Context,
        session_id: str | None = None,
        current_metrics: dict[str, Any] | None = None,
    ) -> Response:
        """
        질문 처리

        Args:
            query: 사용자 질문
            context: 수집된 컨텍스트
            session_id: 세션 ID
            current_metrics: 현재 지표 데이터

        Returns:
            Response 객체
        """
        ...


# =============================================================================
# Decision Maker Protocol
# =============================================================================


@runtime_checkable
class DecisionMakerProtocol(Protocol):
    """
    LLM 의사결정 Protocol

    책임:
    - 질문 분석
    - 도구 선택 결정
    - 신뢰도 판단
    """

    async def decide(
        self, query: str, context: Context, system_state: dict[str, Any]
    ) -> dict[str, Any]:
        """
        의사결정

        Args:
            query: 사용자 질문
            context: 수집된 컨텍스트
            system_state: 시스템 상태

        Returns:
            의사결정 결과 (tool, tool_params, reason, confidence, key_points)
        """
        ...


# =============================================================================
# Tool Coordinator Protocol
# =============================================================================


@runtime_checkable
class ToolCoordinatorProtocol(Protocol):
    """
    도구 실행 조율 Protocol

    책임:
    - 도구 실행
    - 에러 핸들링 (재시도, 폴백)
    - 실행 상태 추적
    """

    async def execute(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """
        도구 실행

        Args:
            tool_name: 실행할 도구 이름
            params: 도구 파라미터

        Returns:
            ToolResult 객체
        """
        ...

    def get_available_tools(self) -> list[str]:
        """사용 가능한 도구 목록"""
        ...

    def get_failed_tools(self) -> list[str]:
        """최근 실패한 도구 목록"""
        ...


# =============================================================================
# Alert Manager Protocol
# =============================================================================


@runtime_checkable
class AlertManagerProtocol(Protocol):
    """
    알림 처리 Protocol

    책임:
    - 알림 조건 체크
    - 알림 생성
    - 알림 발송 (이메일, Telegram 등)
    """

    async def check_conditions(self, event_name: str, data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        알림 조건 체크

        Args:
            event_name: 이벤트 이름
            data: 이벤트 데이터

        Returns:
            발생한 알림 목록
        """
        ...

    async def process_alert(self, alert: dict[str, Any]) -> bool:
        """
        알림 처리 및 발송

        Args:
            alert: 알림 데이터

        Returns:
            발송 성공 여부
        """
        ...

    async def check_metrics_alerts(self, metrics_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        지표 데이터 기반 알림 조건 확인

        Args:
            metrics_data: 지표 데이터

        Returns:
            발생한 알림 목록
        """
        ...


# =============================================================================
# Response Generator Protocol
# =============================================================================


@runtime_checkable
class ResponseGeneratorProtocol(Protocol):
    """
    응답 생성 Protocol

    책임:
    - 컨텍스트 기반 응답 생성
    - 도구 결과 통합
    - 후처리
    """

    async def generate(
        self,
        query: str,
        context: Context,
        decision: dict[str, Any] | None = None,
        tool_result: ToolResult | None = None,
    ) -> Response:
        """
        응답 생성

        Args:
            query: 사용자 질문
            context: 수집된 컨텍스트
            decision: 의사결정 결과
            tool_result: 도구 실행 결과

        Returns:
            Response 객체
        """
        ...

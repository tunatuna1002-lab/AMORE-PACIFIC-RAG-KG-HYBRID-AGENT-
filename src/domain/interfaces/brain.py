"""
Brain Protocol
==============
UnifiedBrain에 대한 추상 인터페이스

구현체:
- UnifiedBrain (src/core/brain.py)
"""

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BrainProtocol(Protocol):
    """
    통합 두뇌(Unified Brain) Protocol

    Level 4 Autonomous Agent의 핵심 두뇌 인터페이스.
    모든 에이전트를 통제하는 단일 중앙 제어를 제공합니다.

    Methods:
        initialize: 두뇌 초기화
        process_query: 사용자 질문 처리
        process_query_stream: 사용자 질문 스트리밍 처리
        run_autonomous_cycle: 자율 작업 사이클 실행
        collect_market_intelligence: 시장 정보 수집
        check_alerts: 알림 조건 확인
        start_scheduler: 스케줄러 시작
        stop_scheduler: 스케줄러 중지
        on_event: 이벤트 핸들러 등록
        emit_event: 이벤트 발생
    """

    async def initialize(self) -> None:
        """
        두뇌를 초기화합니다.

        모든 컴포넌트(ContextGatherer, ToolCoordinator, AlertManager 등)를
        초기화하고 이벤트 핸들러를 등록합니다.
        """
        ...

    async def process_query(
        self,
        query: str,
        session_id: str | None = None,
        current_metrics: dict[str, Any] | None = None,
    ) -> Any:
        """
        사용자 질문을 처리합니다.

        Args:
            query: 사용자 질문
            session_id: 세션 ID (선택)
            current_metrics: 현재 메트릭 데이터 (선택)

        Returns:
            Response 객체 (답변, 메타데이터, 신뢰도 등 포함)
        """
        ...

    async def process_query_stream(
        self,
        query: str,
        session_id: str | None = None,
        current_metrics: dict[str, Any] | None = None,
    ) -> Any:
        """
        사용자 질문을 스트리밍 방식으로 처리합니다.

        Args:
            query: 사용자 질문
            session_id: 세션 ID (선택)
            current_metrics: 현재 메트릭 데이터 (선택)

        Yields:
            StreamChunk 객체 (부분 응답 스트리밍)
        """
        ...

    async def run_autonomous_cycle(self) -> dict[str, Any]:
        """
        자율 작업 사이클을 실행합니다.

        스케줄된 작업(크롤링, 분석, 인사이트 생성 등)을
        순차적으로 실행합니다.

        Returns:
            실행 결과 딕셔너리
            {
                "status": "success" | "failed",
                "completed_tasks": [...],
                "failed_tasks": [...],
                ...
            }
        """
        ...

    async def collect_market_intelligence(self) -> dict[str, Any]:
        """
        시장 정보를 수집합니다.

        외부 신호, 소셜 미디어 트렌드, 뉴스 등을
        수집하여 통합합니다.

        Returns:
            수집된 시장 정보 딕셔너리
        """
        ...

    async def check_alerts(self, metrics_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        알림 조건을 확인하고 알림을 생성합니다.

        Args:
            metrics_data: 메트릭 데이터

        Returns:
            생성된 알림 목록
        """
        ...

    async def start_scheduler(self) -> None:
        """
        자율 스케줄러를 시작합니다.

        백그라운드에서 예약된 작업을 자동으로 실행합니다.
        """
        ...

    def stop_scheduler(self) -> None:
        """
        자율 스케줄러를 중지합니다.
        """
        ...

    def on_event(self, event_name: str, handler: Callable) -> None:
        """
        이벤트 핸들러를 등록합니다.

        Args:
            event_name: 이벤트 이름 (예: "crawl_complete", "error")
            handler: 이벤트 핸들러 함수
        """
        ...

    async def emit_event(self, event_name: str, data: dict[str, Any] | None = None) -> None:
        """
        이벤트를 발생시킵니다.

        Args:
            event_name: 이벤트 이름
            data: 이벤트 데이터 (선택)
        """
        ...

    def get_stats(self) -> dict[str, Any]:
        """
        통계 정보를 반환합니다.

        Returns:
            통계 딕셔너리 (실행 횟수, 에러 횟수, 캐시 히트율 등)
        """
        ...

    def get_state_summary(self) -> str:
        """
        현재 상태 요약을 반환합니다.

        Returns:
            상태 요약 문자열
        """
        ...

    def reset_failed_agents(self) -> None:
        """
        실패한 에이전트 상태를 리셋합니다.
        """
        ...

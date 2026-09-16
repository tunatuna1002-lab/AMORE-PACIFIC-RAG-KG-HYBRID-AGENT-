"""
Alert Agent Protocol
====================
AlertAgent에 대한 추상 인터페이스

구현체:
- AlertAgent (src/agents/alert_agent.py)

D16 정렬: 시그니처는 구현체(AlertAgent)의 실제 시그니처를 그대로 옮겨 적는다.
tests/unit/domain/test_protocol_conformance.py 가 이 일치를 고정한다.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AlertAgentProtocol(Protocol):
    """
    Alert Agent Protocol

    알림 처리 및 발송 전담 에이전트 인터페이스.
    알림 조건을 모니터링하고, 동의한 사용자에게 이메일을 발송합니다.

    핵심 원칙:
    1. 명시적 동의 없이는 절대 이메일 발송 안 함
    2. 모든 알림은 대시보드에도 표시
    3. 중복 알림 방지

    Methods:
        create_alert: 알림 생성
        process_metrics: 메트릭 기반 알림 처리
        send_pending_alerts: 대기 중인 알림 발송
        on_crawl_complete: 크롤링 완료 이벤트 핸들러
        on_crawl_failed: 크롤링 실패 이벤트 핸들러
        on_error: 에러 이벤트 핸들러
        send_daily_summary: 일일 요약 발송
        get_alerts: 알림 목록 조회
        get_pending_count: 대기 중인 알림 수 조회
        get_stats: 통계 조회
        clear_old_alerts: 오래된 알림 삭제
    """

    def create_alert(
        self,
        alert_type: str,
        title: str,
        message: str,
        data: dict[str, Any] | None = None,
        priority: Any = None,
    ) -> Any:
        """
        알림을 생성합니다.

        Args:
            alert_type: 알림 타입 (rank_change, sos_change, market_shift 등)
            title: 알림 제목
            message: 알림 메시지
            data: 추가 데이터 (선택)
            priority: 알림 우선순위 (AlertPriority enum, 기본 NORMAL)

        Returns:
            Alert 객체
        """
        ...

    async def process_metrics(self, metrics_data: dict[str, Any]) -> list[Any]:
        """
        메트릭 데이터를 분석하여 알림을 생성합니다.

        순위 변동, SoS 변화, 시장 변화 등의 조건을 체크하여
        알림을 생성하고 저장합니다.

        Args:
            metrics_data: 메트릭 데이터

        Returns:
            생성된 Alert 객체 리스트
        """
        ...

    async def send_pending_alerts(self) -> dict[str, Any]:
        """
        대기 중인 알림을 발송합니다.

        동의한 사용자에게만 이메일을 발송하며,
        발송 결과를 기록합니다.

        Returns:
            발송 결과 딕셔너리
            {
                "sent_count": int,
                "failed_count": int,
                "recipients": [...],
                ...
            }
        """
        ...

    async def on_crawl_complete(self, data: dict[str, Any]) -> None:
        """
        크롤링 완료 이벤트 핸들러.

        Args:
            data: 이벤트 데이터
        """
        ...

    async def on_crawl_failed(self, data: dict[str, Any]) -> None:
        """
        크롤링 실패 이벤트 핸들러.

        Args:
            data: 이벤트 데이터
        """
        ...

    async def on_error(self, data: dict[str, Any]) -> None:
        """
        에러 이벤트 핸들러.

        Args:
            data: 이벤트 데이터
        """
        ...

    async def send_daily_summary(
        self,
        highlights: list[str],
        avg_rank: float,
        sos: float,
        alert_count: int,
        action_items: list[str],
    ) -> Any:
        """
        일일 요약을 발송합니다.

        Args:
            highlights: 하이라이트 문구
            avg_rank: 평균 순위
            sos: Share of Shelf (퍼센트 - 표시 계층)
            alert_count: 당일 알림 수
            action_items: 액션 아이템

        Returns:
            SendResult (발송 성공 여부·수신자·오류)
        """
        ...

    def get_alerts(self, limit: int = 50, alert_type: str | None = None) -> list[dict[str, Any]]:
        """
        알림 목록을 조회합니다.

        Args:
            limit: 최대 조회 수
            alert_type: 알림 타입 필터 (선택)

        Returns:
            알림 딕셔너리 리스트
        """
        ...

    def get_pending_count(self) -> int:
        """
        대기 중인 알림 수를 반환합니다.

        Returns:
            대기 중인 알림 수
        """
        ...

    def get_stats(self) -> dict[str, Any]:
        """
        통계를 반환합니다.

        Returns:
            통계 딕셔너리 (총 알림 수, 발송 수, 실패 수 등)
        """
        ...

    def clear_old_alerts(self, hours: int = 24) -> int:
        """
        오래된 알림을 삭제합니다.

        Args:
            hours: 기준 시간 (시간)

        Returns:
            삭제된 알림 수
        """
        ...

"""
알림 에이전트 (Alert Agent)
============================
알림 처리 및 발송 전담 에이전트

역할:
- 알림 조건 모니터링
- 동의한 사용자에게 이메일 발송
- 대시보드 알림 표시

이 에이전트는 core/brain.py에서 호출되어 작동합니다.

중요:
- 이메일 발송은 반드시 명시적 동의(체크박스)가 있는 사용자에게만
- StateManager에서 동의 여부 확인

Usage:
    alert_agent = AlertAgent(state_manager)
    await alert_agent.process_alerts(metrics_data)
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from core.rules_engine import RulesEngine
from core.state_manager import StateManager, AlertType
from tools.email_sender import EmailSender, SendResult

logger = logging.getLogger(__name__)


# =============================================================================
# 타입 정의
# =============================================================================

class AlertPriority(Enum):
    """알림 우선순위"""
    CRITICAL = 1    # 즉시 발송
    HIGH = 2        # 빠른 발송
    NORMAL = 3      # 일반 발송
    LOW = 4         # 배치 발송


@dataclass
class Alert:
    """알림 정보"""
    id: str
    type: str
    priority: AlertPriority
    title: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    sent: bool = False
    sent_to: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "priority": self.priority.value,
            "title": self.title,
            "message": self.message,
            "data": self.data,
            "created_at": self.created_at.isoformat(),
            "sent": self.sent,
            "sent_to": self.sent_to
        }


# =============================================================================
# 알림 에이전트
# =============================================================================

class AlertAgent:
    """
    알림 처리 및 발송 전담 에이전트

    알림 조건을 모니터링하고, 동의한 사용자에게 이메일을 발송합니다.

    핵심 원칙:
    1. 명시적 동의 없이는 절대 이메일 발송 안 함
    2. 모든 알림은 대시보드에도 표시
    3. 중복 알림 방지
    """

    def __init__(
        self,
        state_manager: StateManager,
        rules_engine: Optional[RulesEngine] = None,
        email_sender: Optional[EmailSender] = None
    ):
        """
        Args:
            state_manager: 상태 관리자 (동의 정보 포함)
            rules_engine: 규칙 엔진
            email_sender: 이메일 발송기
        """
        self.state_manager = state_manager
        self.rules_engine = rules_engine or RulesEngine()
        self.email_sender = email_sender or EmailSender()

        # 알림 저장소
        self._alerts: List[Alert] = []
        self._pending_alerts: List[Alert] = []

        # 중복 방지 (최근 24시간 알림 ID)
        self._sent_alert_ids: set = set()

        # 통계
        self._stats = {
            "total_alerts": 0,
            "emails_sent": 0,
            "emails_failed": 0
        }

    # =========================================================================
    # 알림 생성
    # =========================================================================

    def create_alert(
        self,
        alert_type: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        priority: AlertPriority = AlertPriority.NORMAL
    ) -> Alert:
        """
        알림 생성

        Args:
            alert_type: 알림 유형
            title: 제목
            message: 메시지
            data: 추가 데이터
            priority: 우선순위

        Returns:
            Alert
        """
        alert_id = f"{alert_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self._alerts)}"

        alert = Alert(
            id=alert_id,
            type=alert_type,
            priority=priority,
            title=title,
            message=message,
            data=data or {}
        )

        self._alerts.append(alert)
        self._pending_alerts.append(alert)
        self._stats["total_alerts"] += 1

        logger.info(f"Alert created: {alert_id} - {title}")
        return alert

    # =========================================================================
    # 지표 기반 알림 생성
    # =========================================================================

    async def process_metrics(self, metrics_data: Dict[str, Any]) -> List[Alert]:
        """
        지표 데이터에서 알림 생성

        Args:
            metrics_data: 지표 데이터

        Returns:
            생성된 알림 목록
        """
        alerts = []

        # 제품별 순위 변동 확인
        products = metrics_data.get("products", [])
        for product in products:
            rank_change = product.get("rank_change", 0)

            # 순위 급락 (10등 이상)
            if rank_change >= 10:
                alert = self.create_alert(
                    alert_type="rank_change",
                    title=f"{product.get('name', '제품')} 순위 급락",
                    message=f"순위가 {rank_change}등 하락했습니다. ({product.get('previous_rank', '?')}등 → {product.get('current_rank', '?')}등)",
                    data={
                        "product_name": product.get("name"),
                        "brand": product.get("brand"),
                        "previous_rank": product.get("previous_rank"),
                        "current_rank": product.get("current_rank"),
                        "change": rank_change
                    },
                    priority=AlertPriority.HIGH
                )
                alerts.append(alert)

            # 순위 급등 (-10등 이하)
            elif rank_change <= -10:
                alert = self.create_alert(
                    alert_type="rank_change",
                    title=f"{product.get('name', '제품')} 순위 급등",
                    message=f"순위가 {abs(rank_change)}등 상승했습니다!",
                    data={
                        "product_name": product.get("name"),
                        "brand": product.get("brand"),
                        "previous_rank": product.get("previous_rank"),
                        "current_rank": product.get("current_rank"),
                        "change": rank_change
                    },
                    priority=AlertPriority.NORMAL
                )
                alerts.append(alert)

        # 신규 Top 10 진입 확인
        for product in products:
            current_rank = product.get("current_rank", 999)
            previous_rank = product.get("previous_rank", 999)

            if current_rank <= 10 and previous_rank > 10:
                alert = self.create_alert(
                    alert_type="important_insight",
                    title=f"{product.get('name', '제품')} Top 10 진입!",
                    message=f"Top 10에 새로 진입했습니다. ({previous_rank}등 → {current_rank}등)",
                    data={
                        "product_name": product.get("name"),
                        "brand": product.get("brand"),
                        "current_rank": current_rank
                    },
                    priority=AlertPriority.HIGH
                )
                alerts.append(alert)

        return alerts

    # =========================================================================
    # 알림 발송
    # =========================================================================

    async def send_pending_alerts(self) -> Dict[str, Any]:
        """
        대기 중인 알림 발송

        동의한 사용자에게만 이메일을 발송합니다.

        Returns:
            발송 결과
        """
        results = {
            "processed": 0,
            "sent": 0,
            "failed": 0,
            "skipped": 0,
            "details": []
        }

        for alert in self._pending_alerts.copy():
            # 중복 체크
            if alert.id in self._sent_alert_ids:
                results["skipped"] += 1
                self._pending_alerts.remove(alert)
                continue

            # 동의한 수신자 조회
            recipients = self.state_manager.get_alert_recipients(alert.type)

            if not recipients:
                logger.debug(f"No recipients for alert type: {alert.type}")
                results["skipped"] += 1
                self._pending_alerts.remove(alert)
                continue

            # 이메일 발송
            try:
                send_result = await self._send_alert_email(alert, recipients)

                alert.sent = send_result.success
                alert.sent_to = send_result.sent_to

                if send_result.success:
                    results["sent"] += 1
                    self._stats["emails_sent"] += len(send_result.sent_to)
                    self._sent_alert_ids.add(alert.id)
                else:
                    results["failed"] += 1
                    self._stats["emails_failed"] += len(send_result.failed)

                results["details"].append({
                    "alert_id": alert.id,
                    "type": alert.type,
                    "success": send_result.success,
                    "sent_to": send_result.sent_to,
                    "failed": send_result.failed
                })

            except Exception as e:
                logger.error(f"Failed to send alert {alert.id}: {e}")
                results["failed"] += 1

            results["processed"] += 1
            self._pending_alerts.remove(alert)

        return results

    async def _send_alert_email(self, alert: Alert, recipients: List[str]) -> SendResult:
        """알림 이메일 발송"""
        # 알림 유형별 처리
        if alert.type == "rank_change":
            return await self.email_sender.send_rank_change_alert(
                recipients=recipients,
                product_name=alert.data.get("product_name", "Unknown"),
                brand=alert.data.get("brand", "Unknown"),
                previous_rank=alert.data.get("previous_rank", 0),
                current_rank=alert.data.get("current_rank", 0)
            )

        elif alert.type == "error":
            return await self.email_sender.send_error_alert(
                recipients=recipients,
                error_message=alert.message,
                location=alert.data.get("location", "Unknown")
            )

        else:
            # 일반 알림
            return await self.email_sender.send_alert(
                alert_type=alert.type,
                subject=alert.title,
                content={
                    "insight": alert.message,
                    "action_items": alert.data.get("action_items", [])
                },
                recipients=recipients
            )

    # =========================================================================
    # 이벤트 핸들러
    # =========================================================================

    async def on_crawl_complete(self, data: Dict[str, Any]) -> None:
        """크롤링 완료 이벤트 처리"""
        alert = self.create_alert(
            alert_type="crawl_complete",
            title="크롤링 완료",
            message=f"{data.get('total_products', 0)}개 제품 수집 완료",
            data=data,
            priority=AlertPriority.LOW
        )

        await self.send_pending_alerts()

    async def on_crawl_failed(self, data: Dict[str, Any]) -> None:
        """크롤링 실패 이벤트 처리"""
        alert = self.create_alert(
            alert_type="error",
            title="크롤링 실패",
            message=data.get("error", "알 수 없는 오류"),
            data={"location": "crawler"},
            priority=AlertPriority.CRITICAL
        )

        await self.send_pending_alerts()

    async def on_error(self, data: Dict[str, Any]) -> None:
        """에러 이벤트 처리"""
        alert = self.create_alert(
            alert_type="error",
            title="시스템 에러",
            message=data.get("error", "알 수 없는 오류"),
            data=data,
            priority=AlertPriority.CRITICAL
        )

        await self.send_pending_alerts()

    # =========================================================================
    # 일일 요약
    # =========================================================================

    async def send_daily_summary(
        self,
        highlights: List[str],
        avg_rank: float,
        sos: float,
        alert_count: int,
        action_items: List[str]
    ) -> SendResult:
        """
        일일 요약 발송

        Args:
            highlights: 하이라이트
            avg_rank: 평균 순위
            sos: Share of Shelf
            alert_count: 알림 수
            action_items: 권장 액션

        Returns:
            발송 결과
        """
        recipients = self.state_manager.get_alert_recipients("daily_summary")

        if not recipients:
            return SendResult(
                success=True,
                sent_to=[],
                failed=[],
                message="일일 요약 수신자 없음"
            )

        return await self.email_sender.send_daily_summary(
            recipients=recipients,
            highlights=highlights,
            avg_rank=avg_rank,
            sos=sos,
            alert_count=alert_count,
            action_items=action_items
        )

    # =========================================================================
    # 조회
    # =========================================================================

    def get_alerts(
        self,
        limit: int = 50,
        alert_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        알림 목록 조회

        Args:
            limit: 최대 개수
            alert_type: 필터할 알림 유형

        Returns:
            알림 목록
        """
        alerts = self._alerts

        if alert_type:
            alerts = [a for a in alerts if a.type == alert_type]

        # 최신순 정렬
        alerts = sorted(alerts, key=lambda a: a.created_at, reverse=True)

        return [a.to_dict() for a in alerts[:limit]]

    def get_pending_count(self) -> int:
        """대기 중인 알림 수"""
        return len(self._pending_alerts)

    def get_stats(self) -> Dict[str, Any]:
        """통계"""
        return {
            **self._stats,
            "pending": len(self._pending_alerts),
            "email_enabled": self.email_sender.is_enabled()
        }

    def clear_old_alerts(self, hours: int = 24) -> int:
        """오래된 알림 정리"""
        cutoff = datetime.now().timestamp() - (hours * 3600)
        old_ids = [
            a.id for a in self._alerts
            if a.created_at.timestamp() < cutoff
        ]

        self._alerts = [a for a in self._alerts if a.id not in old_ids]
        self._sent_alert_ids -= set(old_ids)

        return len(old_ids)

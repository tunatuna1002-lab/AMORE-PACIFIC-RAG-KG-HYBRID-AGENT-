"""
Alert Manager - 알림 처리 전담
==============================
UnifiedBrain에서 분리된 알림 처리 컴포넌트

책임:
- 알림 조건 체크
- 알림 생성
- 알림 발송 (이메일, Telegram 등)
- 알림 히스토리 관리

관련 Protocol: AlertManagerProtocol
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..agents.alert_agent import AlertAgent
    from ..core.state_manager import StateManager

logger = logging.getLogger(__name__)


class AlertManager:
    """
    알림 처리 관리

    이벤트 기반 알림 조건 체크 및 발송을 담당합니다.

    Usage:
        alert_manager = AlertManager()
        alerts = await alert_manager.check_conditions("rank_changed", data)
        for alert in alerts:
            await alert_manager.process_alert(alert)
    """

    # 알림 조건 임계값
    RANK_CHANGE_THRESHOLD = 10  # 순위 변동 알림 임계값
    SOS_CHANGE_THRESHOLD = 2.0  # SoS 변동 알림 임계값 (%p)

    def __init__(
        self, state_manager: "StateManager | None" = None, alert_agent: "AlertAgent | None" = None
    ):
        """
        Args:
            state_manager: 상태 관리자
            alert_agent: AlertAgent (이메일 발송용)
        """
        self._state_manager = state_manager
        self._alert_agent = alert_agent
        self._initialized = False

        # 알림 히스토리
        self._alert_history: list[dict[str, Any]] = []

        # 통계
        self._stats = {"alerts_generated": 0, "alerts_sent": 0, "send_failures": 0}

    async def initialize(self) -> None:
        """비동기 초기화 (지연 임포트)"""
        if self._initialized:
            return

        try:
            # StateManager 지연 임포트
            if self._state_manager is None:
                from ..core.state_manager import StateManager

                self._state_manager = StateManager()

            # AlertAgent 지연 임포트
            if self._alert_agent is None:
                from ..agents.alert_agent import AlertAgent

                self._alert_agent = AlertAgent(self._state_manager)
                logger.info("AlertAgent initialized in AlertManager")

            self._initialized = True

        except Exception as e:
            logger.warning(f"AlertManager initialization failed: {e}")
            self._initialized = False

    async def check_conditions(self, event_name: str, data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        알림 조건 체크

        Args:
            event_name: 이벤트 이름
            data: 이벤트 데이터

        Returns:
            발생한 알림 목록
        """
        alerts = []

        if event_name == "rank_changed":
            alerts.extend(self._check_rank_change(data))

        elif event_name == "crawl_complete":
            alerts.extend(self._check_crawl_complete(data))

        elif event_name == "metrics_calculated":
            alerts.extend(self._check_metrics_calculated(data))

        return alerts

    def _check_rank_change(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """순위 변동 알림 체크"""
        alerts = []
        product = data.get("product", {})
        change = data.get("change", 0)

        if abs(change) >= self.RANK_CHANGE_THRESHOLD:
            alert_type = "rank_drop" if change > 0 else "rank_surge"
            direction = "급락" if change > 0 else "급등"

            alerts.append(
                {
                    "type": alert_type,
                    "severity": "critical" if abs(change) >= 20 else "warning",
                    "product": product.get("name"),
                    "asin": product.get("asin"),
                    "change": change,
                    "current_rank": product.get("rank"),
                    "message": f"{product.get('name')} 순위 {direction}: {abs(change)}단계",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return alerts

    def _check_crawl_complete(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """크롤링 완료 알림 체크"""
        alerts = []
        result = data.get("result", {})

        # 크롤링 실패 시 알림
        if not result.get("success", True):
            alerts.append(
                {
                    "type": "crawl_failed",
                    "severity": "critical",
                    "message": f"크롤링 실패: {result.get('error', 'Unknown error')}",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return alerts

    def _check_metrics_calculated(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """지표 계산 완료 후 알림 체크"""
        # metrics_data로 check_metrics_alerts 호출
        return []  # 상위 레벨에서 처리

    async def check_metrics_alerts(self, metrics_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        지표 데이터 기반 알림 조건 확인

        Args:
            metrics_data: 지표 데이터

        Returns:
            발생한 알림 목록
        """
        alerts = []

        # 제품별 순위 변동 확인
        products = metrics_data.get("products", {})
        for asin, product in products.items():
            rank_delta = product.get("rank_delta", "")

            if rank_delta:
                try:
                    change = int(rank_delta.replace("+", "").replace("-", ""))
                    if "-" in rank_delta:
                        change = -change

                    if abs(change) >= self.RANK_CHANGE_THRESHOLD:
                        direction = "급락" if change > 0 else "급등"
                        alerts.append(
                            {
                                "type": "rank_change",
                                "severity": "critical" if abs(change) >= 20 else "warning",
                                "product": product.get("name"),
                                "asin": asin,
                                "change": change,
                                "current_rank": product.get("rank"),
                                "message": f"{product.get('name')} 순위 {direction}: {abs(change)}단계",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                except ValueError:
                    pass

        # SoS 변동 확인
        brand_kpis = metrics_data.get("brand", {}).get("kpis", {})
        sos_delta = brand_kpis.get("sos_delta", "")
        if sos_delta:
            try:
                sos_change = float(sos_delta.replace("+", "").replace("%", "").replace("p", ""))
                if sos_change <= -self.SOS_CHANGE_THRESHOLD:
                    alerts.append(
                        {
                            "type": "sos_drop",
                            "severity": "warning",
                            "change": sos_change,
                            "current_sos": brand_kpis.get("sos"),
                            "message": f"LANEIGE SoS {sos_change}%p 하락",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
            except ValueError:
                pass

        return alerts

    async def process_alert(self, alert: dict[str, Any]) -> bool:
        """
        알림 처리 및 발송

        Args:
            alert: 알림 데이터

        Returns:
            발송 성공 여부
        """
        self._stats["alerts_generated"] += 1
        self._alert_history.append(alert)

        # 히스토리 크기 제한
        if len(self._alert_history) > 100:
            self._alert_history = self._alert_history[-50:]

        # AlertAgent 초기화 (lazy)
        if not self._initialized:
            await self.initialize()

        if not self._alert_agent:
            logger.warning("AlertAgent not available, skipping alert send")
            return False

        try:
            from ..agents.alert_agent import AlertPriority

            # 알림 타입에 따른 우선순위 매핑
            priority_map = {
                "rank_drop": AlertPriority.HIGH,
                "rank_surge": AlertPriority.NORMAL,
                "sos_drop": AlertPriority.HIGH,
                "sos_surge": AlertPriority.NORMAL,
                "error": AlertPriority.CRITICAL,
                "crawl_complete": AlertPriority.LOW,
                "crawl_failed": AlertPriority.CRITICAL,
            }
            priority = priority_map.get(alert.get("type"), AlertPriority.NORMAL)

            # AlertAgent를 통해 알림 생성
            self._alert_agent.create_alert(
                alert_type=alert.get("type", "unknown"),
                title=alert.get("message", "알림"),
                message=alert.get("details", alert.get("message", "")),
                data=alert,
                priority=priority,
            )

            # 즉시 발송
            send_result = await self._alert_agent.send_pending_alerts()
            sent_count = send_result.get("sent", 0)

            if sent_count > 0:
                self._stats["alerts_sent"] += sent_count
                logger.info(f"Alert processed: {alert.get('type')} - sent: {sent_count}")
                return True
            else:
                logger.warning(f"Alert created but not sent: {alert.get('type')}")
                return False

        except Exception as e:
            self._stats["send_failures"] += 1
            logger.error(f"Failed to send alert: {e}")
            return False

    def get_recent_alerts(self, limit: int = 10) -> list[dict[str, Any]]:
        """최근 알림 목록"""
        return self._alert_history[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """통계 반환"""
        return {
            **self._stats,
            "history_size": len(self._alert_history),
            "initialized": self._initialized,
        }

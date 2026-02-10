"""
Alert Workflow
==============
Orchestrates the alert processing workflow.

Flow:
1. Process metrics to detect alert conditions
2. Create alerts
3. Optionally send pending alerts

Clean Architecture:
- Depends only on domain interfaces
- No infrastructure dependencies
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.domain.interfaces.alert import AlertAgentProtocol


@dataclass
class AlertWorkflowResult:
    """Alert workflow execution result"""

    success: bool = False
    alerts_created: int = 0
    alerts_sent: int = 0
    execution_time: float = 0.0
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "alerts_created": self.alerts_created,
            "alerts_sent": self.alerts_sent,
            "execution_time": self.execution_time,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class AlertWorkflow:
    """
    Alert Workflow

    Orchestrates the alert processing using dependency injection.

    Usage:
        workflow = AlertWorkflow(alert_agent=alert_agent)
        result = await workflow.execute(metrics_data=metrics)
    """

    def __init__(self, alert_agent: AlertAgentProtocol):
        """
        Args:
            alert_agent: AlertAgent implementation
        """
        self.alert_agent = alert_agent

    async def execute(self, metrics_data: dict[str, Any]) -> AlertWorkflowResult:
        """
        Execute alert processing workflow.

        Args:
            metrics_data: Metrics data dictionary

        Returns:
            AlertWorkflowResult with alert statistics
        """
        start_time = time.time()
        result = AlertWorkflowResult()

        if not metrics_data:
            result.execution_time = time.time() - start_time
            return result

        try:
            # Process metrics and create alerts
            alerts = await self.alert_agent.process_metrics(metrics_data)
            result.alerts_created = len(alerts)
            result.success = True

        except Exception as e:
            result.error = str(e)
            result.success = False

        result.execution_time = time.time() - start_time
        return result

    async def send_pending_alerts(self) -> AlertWorkflowResult:
        """
        Send all pending alerts.

        Returns:
            AlertWorkflowResult with send statistics
        """
        start_time = time.time()
        result = AlertWorkflowResult()

        try:
            send_result = await self.alert_agent.send_pending_alerts()
            result.alerts_sent = send_result.get("sent_count", 0)
            result.success = True

        except Exception as e:
            result.error = str(e)
            result.success = False

        result.execution_time = time.time() - start_time
        return result

"""
Insight Workflow
================
Orchestrates the insight generation workflow.

Flow:
1. Validate metrics data
2. Generate insights via InsightAgent
3. Return structured results

Clean Architecture:
- Depends only on domain interfaces
- No infrastructure dependencies
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.domain.interfaces.insight import InsightAgentProtocol
from src.domain.interfaces.storage import StorageProtocol


@dataclass
class InsightWorkflowResult:
    """Insight workflow execution result"""

    success: bool = False
    insights: list[dict[str, Any]] = field(default_factory=list)
    summary: str | None = None
    highlights: list[str] = field(default_factory=list)
    execution_time: float = 0.0
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "insights": self.insights,
            "summary": self.summary,
            "highlights": self.highlights,
            "execution_time": self.execution_time,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class InsightWorkflow:
    """
    Insight Workflow

    Orchestrates the insight generation process using dependency injection.

    Usage:
        workflow = InsightWorkflow(
            insight_agent=insight_agent,
            storage=storage
        )
        result = await workflow.execute(metrics_data=metrics)
    """

    def __init__(
        self,
        insight_agent: InsightAgentProtocol,
        storage: StorageProtocol,
    ):
        """
        Args:
            insight_agent: InsightAgent implementation
            storage: Storage implementation (for context data)
        """
        self.insight_agent = insight_agent
        self.storage = storage

    async def execute(
        self,
        metrics_data: dict[str, Any],
        target_brand: str = "LANEIGE",
        category_id: str | None = None,
    ) -> InsightWorkflowResult:
        """
        Execute insight generation workflow.

        Args:
            metrics_data: Metrics data dictionary
            target_brand: Target brand for analysis
            category_id: Optional category filter

        Returns:
            InsightWorkflowResult with generated insights
        """
        start_time = time.time()
        result = InsightWorkflowResult()

        if not metrics_data:
            result.execution_time = time.time() - start_time
            return result

        try:
            # Execute insight generation
            insight_result = await self.insight_agent.execute(
                metrics_data=metrics_data,
                target_brand=target_brand,
                category_id=category_id,
            )

            # Populate result
            result.insights = insight_result.get("insights", [])
            result.summary = insight_result.get("summary")
            result.highlights = insight_result.get("highlights", [])
            result.success = True

        except Exception as e:
            result.error = str(e)
            result.success = False

        result.execution_time = time.time() - start_time
        return result

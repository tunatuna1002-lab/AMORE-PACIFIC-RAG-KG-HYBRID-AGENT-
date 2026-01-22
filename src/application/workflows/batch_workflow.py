"""
Batch Workflow with Dependency Injection
========================================
크롤링 → 저장 → 메트릭 계산 → 인사이트 생성 워크플로우

이전 batch_workflow.py와 달리, 의존성을 직접 생성하지 않고 주입받습니다.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from src.domain.interfaces.agent import (
    CrawlerAgentProtocol,
    StorageAgentProtocol,
    MetricsAgentProtocol,
    InsightAgentProtocol,
)


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowResult:
    """워크플로우 실행 결과"""
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    records_count: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    insights: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "records_count": self.records_count,
            "metrics": self.metrics,
            "insights": self.insights,
            "errors": self.errors,
        }


@dataclass
class WorkflowDependencies:
    """워크플로우 의존성 컨테이너 (DI)"""
    crawler: CrawlerAgentProtocol
    storage: StorageAgentProtocol
    metrics: MetricsAgentProtocol
    insight: Optional[InsightAgentProtocol] = None
    categories: List[str] = field(default_factory=list)


class BatchWorkflow:
    """
    배치 워크플로우 (의존성 주입 버전)

    Usage:
        deps = WorkflowDependencies(
            crawler=crawler_agent,
            storage=storage_agent,
            metrics=metrics_agent,
            categories=["lip_care", "skin_care"]
        )
        workflow = BatchWorkflow(deps)
        result = await workflow.execute()
    """

    def __init__(self, deps: WorkflowDependencies):
        self.deps = deps
        self._result: Optional[WorkflowResult] = None

    async def execute(self) -> WorkflowResult:
        """워크플로우 실행"""
        result = WorkflowResult(
            status=WorkflowStatus.RUNNING,
            started_at=datetime.now()
        )

        try:
            # Step 1: Crawl
            records = await self.deps.crawler.crawl(self.deps.categories)
            result.records_count = len(records)

            # Step 2: Save
            await self.deps.storage.save(records)

            # Step 3: Calculate metrics
            metrics = await self.deps.metrics.calculate(records)
            result.metrics = metrics
            await self.deps.storage.save_metrics(metrics)

            # Step 4: Generate insights (optional)
            if self.deps.insight:
                insights = await self.deps.insight.generate(metrics, {"records": records})
                result.insights = insights

            result.status = WorkflowStatus.COMPLETED
            result.completed_at = datetime.now()

        except Exception as e:
            result.status = WorkflowStatus.FAILED
            result.errors.append(str(e))
            result.completed_at = datetime.now()

        self._result = result
        return result

    @property
    def result(self) -> Optional[WorkflowResult]:
        return self._result

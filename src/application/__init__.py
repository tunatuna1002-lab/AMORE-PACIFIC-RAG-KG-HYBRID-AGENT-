"""
Application Layer
=================
Clean Architecture의 Use Case Layer (Application Business Rules)

이 레이어는 애플리케이션 특정 비즈니스 로직을 포함합니다.
Domain 레이어에만 의존하며, Infrastructure 레이어를 알지 못합니다.

구조:
- workflows/: 배치 워크플로우 (크롤링 → 저장 → 분석)
- services/: 애플리케이션 서비스 (채팅, 인사이트 등)
- orchestrators/: 오케스트레이션 로직 (Brain, CrawlManager)
"""

# Services
from src.application.services.query_analyzer import (
    ComplexityLevel,
    QueryAnalyzer,
    QueryIntent,
)

# Workflows
from src.application.workflows.alert_workflow import AlertWorkflow, AlertWorkflowResult
from src.application.workflows.batch_workflow import (
    ActResult,
    BatchWorkflow,
    CrawlResult,
    InsightResult,
    MetricsResult,
    ObserveResult,
    Orchestrator,
    ThinkResult,
    WorkflowDependencies,
    WorkflowResult,
    WorkflowState,
    WorkflowStep,
    run_full_workflow,
)
from src.application.workflows.chat_workflow import ChatWorkflow, ChatWorkflowResult
from src.application.workflows.crawl_workflow import CrawlWorkflow, CrawlWorkflowResult
from src.application.workflows.insight_workflow import (
    InsightWorkflow,
    InsightWorkflowResult,
)

__all__ = [
    # Services
    "QueryAnalyzer",
    "ComplexityLevel",
    "QueryIntent",
    # Workflows
    "BatchWorkflow",
    "WorkflowDependencies",
    "WorkflowResult",
    "WorkflowStep",
    "ThinkResult",
    "ActResult",
    "ObserveResult",
    "Orchestrator",
    "CrawlResult",
    "MetricsResult",
    "InsightResult",
    "WorkflowState",
    "run_full_workflow",
    "ChatWorkflow",
    "ChatWorkflowResult",
    "CrawlWorkflow",
    "CrawlWorkflowResult",
    "InsightWorkflow",
    "InsightWorkflowResult",
    "AlertWorkflow",
    "AlertWorkflowResult",
]

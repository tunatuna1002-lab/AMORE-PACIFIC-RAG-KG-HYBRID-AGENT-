"""
Orchestrator (Backward Compatibility)
=====================================
이 파일은 하위 호환성을 위해 유지됩니다.
실제 구현은 src/core/batch_workflow.py에 있습니다.

Usage:
    # 기존 방식 (여전히 작동)
    from orchestrator import Orchestrator, run_full_workflow

    # 새로운 방식 (권장)
    from src.core.batch_workflow import BatchWorkflow, run_full_workflow

Note:
    - Orchestrator 클래스는 BatchWorkflow의 별칭입니다
    - 모든 기능은 src/core/batch_workflow.py에 구현되어 있습니다
    - 기존 코드 호환성을 위해 이 파일을 통한 import도 지원합니다
"""

from src.core.batch_workflow import (
    BatchWorkflow,
    Orchestrator,
    run_full_workflow,
    WorkflowStep,
    ThinkResult,
    ActResult,
    ObserveResult,
    CrawlResult,
    MetricsResult,
    InsightResult,
    WorkflowState
)

__all__ = [
    "BatchWorkflow",
    "Orchestrator",  # Alias for BatchWorkflow
    "run_full_workflow",
    "WorkflowStep",
    "ThinkResult",
    "ActResult",
    "ObserveResult",
    "CrawlResult",
    "MetricsResult",
    "InsightResult",
    "WorkflowState"
]

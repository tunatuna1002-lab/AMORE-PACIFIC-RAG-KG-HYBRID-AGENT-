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

from src.application.workflows.batch_workflow import BatchWorkflow, WorkflowDependencies, WorkflowResult

__all__ = [
    "BatchWorkflow",
    "WorkflowDependencies",
    "WorkflowResult",
]

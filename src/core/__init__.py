"""
Core 모듈
=========
챗봇 질의 파이프라인과 자율 스케줄러의 핵심 컴포넌트

모듈 구조:
- models.py: 데이터 모델 (Context, Response, Decision 등)
- confidence.py: 신뢰도 평가
- cache.py: 응답 캐싱
- state_manager.py: 단일 시스템 상태 (StateManager)
- context_gatherer.py: RAG + KG 컨텍스트 수집
- tools.py: 에이전트 도구 정의
- response_pipeline.py: 응답 생성 파이프라인
- query_graph.py: 질의 처리 상태 그래프
- scheduler.py: 자율 작업 스케줄러
- decision_maker.py / tool_coordinator.py / alert_manager.py: SRP 분해 컴포넌트
- brain.py: UnifiedBrain (Facade)
- brain_scheduler.py: BrainTask / TaskPriority / BrainTaskQueue (작업 우선순위 큐)
- crawl_manager.py: 배치 작업 제어 (단계 실행은 application.workflows.batch_workflow)

패키지 최상위 이름은 지연 로딩된다 (brain ↔ agents ↔ state_manager 순환 import 방지, 경량 import).
"""

import importlib

_LAZY: dict[str, str] = {
    "Context": ".models",
    "Response": ".models",
    "ToolResult": ".models",
    "Decision": ".models",
    "ConfidenceLevel": ".models",
    "ConfidenceAssessor": ".confidence",
    "ResponseCache": ".cache",
    "ContextGatherer": ".context_gatherer",
    "AgentTool": ".tools",
    "ToolExecutor": ".tools",
    "AGENT_TOOLS": ".tools",
    "ResponsePipeline": ".response_pipeline",
    "AutonomousScheduler": ".scheduler",
    "StateManager": ".state_manager",
    "get_state_manager": ".state_manager",
    "UnifiedBrain": ".brain",
    "get_brain": ".brain",
    "get_initialized_brain": ".brain",
    "BrainMode": ".brain",
    "BrainTask": ".brain_scheduler",
    "BrainTaskQueue": ".brain_scheduler",
    "TaskPriority": ".brain_scheduler",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))

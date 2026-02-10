"""
Core 모듈
=========
LLM 기반 오케스트레이터 핵심 컴포넌트

모듈 구조:
- models.py: 데이터 모델 정의 (Context, Response, Decision 등)
- confidence.py: 신뢰도 평가 로직
- cache.py: 응답 캐싱
- state.py: 오케스트레이터 상태 관리
- context_gatherer.py: RAG + KG 컨텍스트 수집
- tools.py: 에이전트 도구 정의
- response_pipeline.py: 응답 생성 파이프라인
- scheduler.py: 자율 작업 스케줄러
- decision_maker.py: LLM 의사결정 (SRP 분해)
- tool_coordinator.py: 도구 실행 조율 (SRP 분해)
- alert_manager.py: 알림 관리 (SRP 분해)
- brain.py: Level 4 자율 에이전트 두뇌 (LLM-First Facade)
- batch_workflow.py: 배치 워크플로우 오케스트레이터 (Think-Act-Observe)
- llm_orchestrator.py: 메인 오케스트레이터 (Legacy)

주요 클래스:
- UnifiedBrain: 통합 두뇌 (Facade) - 모든 컴포넌트 조율
- DecisionMaker: LLM 의사결정
- ToolCoordinator: 도구 실행 조율
- AlertManager: 알림 처리
- ContextGatherer: 컨텍스트 수집
- ResponsePipeline: 응답 생성
"""

# 순환 import 방지: brain 관련 import는 lazy loading으로 처리
# brain.py → alert_agent.py → state_manager.py → core/__init__.py → brain.py 순환 방지
from .cache import ResponseCache
from .confidence import ConfidenceAssessor
from .context_gatherer import ContextGatherer
from .llm_orchestrator import LLMOrchestrator
from .models import ConfidenceLevel, Context, Decision, Response, ToolResult
from .response_pipeline import ResponsePipeline

# Autonomous Scheduler
from .scheduler import AutonomousScheduler
from .state import OrchestratorState
from .tools import AGENT_TOOLS, AgentTool, ToolExecutor

# Lazy loading for brain module to prevent circular imports
_brain_module = None


def _load_brain():
    global _brain_module
    if _brain_module is None:
        from . import brain as _brain_module
    return _brain_module


def __getattr__(name):
    """Lazy loading for brain-related exports"""
    brain_exports = {
        "UnifiedBrain",
        "get_brain",
        "get_initialized_brain",
        "BrainMode",
        "TaskPriority",
    }
    if name in brain_exports:
        brain = _load_brain()
        return getattr(brain, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Models
    "Context",
    "Response",
    "ToolResult",
    "Decision",
    "ConfidenceLevel",
    # Components
    "ConfidenceAssessor",
    "ResponseCache",
    "OrchestratorState",
    "ContextGatherer",
    "AgentTool",
    "ToolExecutor",
    "AGENT_TOOLS",
    "ResponsePipeline",
    "LLMOrchestrator",
    # Scheduler
    "AutonomousScheduler",
    # Level 4 Brain (lazy loaded)
    "UnifiedBrain",
    "get_brain",
    "get_initialized_brain",
    "BrainMode",
    "TaskPriority",
]

"""
Core 모듈
=========
LLM 기반 오케스트레이터 핵심 컴포넌트

모듈 구조:
- models: 데이터 모델 정의 (Context, Response 등)
- confidence: 신뢰도 평가 로직
- cache: 응답 캐싱
- state: 오케스트레이터 상태 관리
- context_gatherer: RAG + KG 컨텍스트 수집
- tools: 에이전트 도구 정의
- response_pipeline: 응답 생성 파이프라인
- scheduler: 자율 작업 스케줄러
- brain: Level 4 자율 에이전트 두뇌 (LLM-First)
- llm_orchestrator: 메인 오케스트레이터 (Legacy)
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

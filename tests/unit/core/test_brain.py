"""
UnifiedBrain 단위 테스트
========================
src/core/brain.py의 Facade 패턴 오케스트레이터 테스트

테스트 구조:
1. Enum and Dataclass Tests - BrainMode, TaskPriority, BrainTask
2. Initialization Tests - 기본 초기화, DI, lazy init
3. State and Mode Tests - 초기 모드, 전환, 통계
4. Event System Tests - 이벤트 등록 및 발생
5. Task Queue Tests - 작업 추가 및 우선순위 정렬
6. Helper Method Tests - get_system_state, get_stats
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.brain import (
    BrainMode,
    BrainTask,
    TaskPriority,
    UnifiedBrain,
    get_brain,
    reset_brain,
)
from src.core.models import Context, Response, ToolResult

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_context_gatherer():
    """Mock ContextGatherer"""
    mock = MagicMock()
    mock.initialize = AsyncMock()
    mock.gather = AsyncMock(
        return_value=Context(
            query="test query",
            entities={"brands": ["LANEIGE"]},
            rag_docs=[{"content": "test doc"}],
            kg_facts=[],
            summary="Test summary",
        )
    )
    return mock


@pytest.fixture
def mock_tool_executor():
    """Mock ToolExecutor"""
    mock = MagicMock()
    mock.execute = AsyncMock(
        return_value=ToolResult(tool_name="test_tool", success=True, data={"result": "ok"})
    )
    return mock


@pytest.fixture
def mock_response_pipeline():
    """Mock ResponsePipeline"""
    mock = MagicMock()
    mock.generate = AsyncMock(
        return_value=Response(
            text="Test response",
            confidence_score=0.8,
            query_type="test",
        )
    )
    return mock


@pytest.fixture
def brain_with_mocks(mock_context_gatherer, mock_tool_executor, mock_response_pipeline):
    """UnifiedBrain with mocked dependencies"""
    brain = UnifiedBrain(
        context_gatherer=mock_context_gatherer,
        tool_executor=mock_tool_executor,
        response_pipeline=mock_response_pipeline,
    )
    return brain


@pytest.fixture(autouse=True)
def cleanup():
    """Reset brain singleton after each test"""
    yield
    reset_brain()


# =============================================================================
# 1. Enum and Dataclass Tests
# =============================================================================


def test_brain_mode_values():
    """BrainMode enum should have all expected values"""
    assert BrainMode.IDLE.value == "idle"
    assert BrainMode.AUTONOMOUS.value == "autonomous"
    assert BrainMode.RESPONDING.value == "responding"
    assert BrainMode.EXECUTING.value == "executing"
    assert BrainMode.ALERTING.value == "alerting"


def test_task_priority_values():
    """TaskPriority enum should have all expected values"""
    assert TaskPriority.USER_REQUEST.value == 0
    assert TaskPriority.CRITICAL_ALERT.value == 1
    assert TaskPriority.SCHEDULED.value == 2
    assert TaskPriority.BACKGROUND.value == 3


def test_brain_task_creation():
    """BrainTask dataclass should initialize correctly"""
    task = BrainTask(
        id="task-1",
        type="query",
        priority=TaskPriority.USER_REQUEST,
        payload={"query": "test"},
    )

    assert task.id == "task-1"
    assert task.type == "query"
    assert task.priority == TaskPriority.USER_REQUEST
    assert task.payload == {"query": "test"}
    assert task.started_at is None
    assert task.completed_at is None
    assert task.result is None
    assert task.error is None
    assert isinstance(task.created_at, datetime)


def test_brain_task_priority_comparison():
    """BrainTask should support priority-based sorting"""
    task_low = BrainTask(id="low", type="bg", priority=TaskPriority.BACKGROUND, payload={})
    task_high = BrainTask(id="high", type="user", priority=TaskPriority.USER_REQUEST, payload={})

    # Lower priority value means higher priority
    assert task_high < task_low
    assert not task_low < task_high


# =============================================================================
# 2. Initialization Tests
# =============================================================================


def test_brain_default_initialization():
    """UnifiedBrain should initialize with default values"""
    brain = UnifiedBrain()

    # Check default attributes
    assert brain.model == "gpt-4o-mini"
    assert brain.max_retries == 2
    assert brain.mode == BrainMode.IDLE
    assert brain._current_task is None
    assert len(brain._task_queue) == 0
    assert len(brain._task_history) == 0
    assert brain._initialized is False

    # Check stats
    stats = brain._stats
    assert stats["total_queries"] == 0
    assert stats["llm_decisions"] == 0
    assert stats["cache_hits"] == 0
    assert stats["autonomous_tasks"] == 0
    assert stats["alerts_generated"] == 0
    assert stats["errors"] == 0


def test_brain_with_custom_components():
    """UnifiedBrain should accept custom components via DI"""
    mock_context_gatherer = MagicMock()
    mock_tool_executor = MagicMock()
    mock_response_pipeline = MagicMock()

    brain = UnifiedBrain(
        context_gatherer=mock_context_gatherer,
        tool_executor=mock_tool_executor,
        response_pipeline=mock_response_pipeline,
        model="gpt-4",
        max_retries=5,
    )

    assert brain._context_gatherer is mock_context_gatherer
    assert brain._tool_executor is mock_tool_executor
    assert brain._response_pipeline is mock_response_pipeline
    assert brain.model == "gpt-4"
    assert brain.max_retries == 5


def test_brain_lazy_init_decision_maker():
    """DecisionMaker should be lazy initialized"""
    brain = UnifiedBrain()

    # Initially None
    assert brain._decision_maker is None

    # Access via property triggers initialization
    decision_maker = brain.decision_maker
    assert decision_maker is not None
    assert brain._decision_maker is decision_maker

    # Second access returns same instance
    assert brain.decision_maker is decision_maker


def test_brain_lazy_init_tool_coordinator():
    """ToolCoordinator should be lazy initialized"""
    brain = UnifiedBrain()

    # Initially None
    assert brain._tool_coordinator is None

    # Access via property triggers initialization
    tool_coordinator = brain.tool_coordinator
    assert tool_coordinator is not None
    assert brain._tool_coordinator is tool_coordinator

    # Second access returns same instance
    assert brain.tool_coordinator is tool_coordinator


def test_brain_lazy_init_alert_manager():
    """AlertManager should be lazy initialized"""
    brain = UnifiedBrain()

    # Initially None
    assert brain._alert_manager is None

    # Access via property triggers initialization
    alert_manager = brain.alert_manager
    assert alert_manager is not None
    assert brain._alert_manager is alert_manager

    # Second access returns same instance
    assert brain.alert_manager is alert_manager


def test_brain_lazy_init_query_processor():
    """QueryProcessor should be lazy initialized"""
    brain = UnifiedBrain()

    # Initially None
    assert brain._query_processor is None

    # Access via property triggers initialization
    query_processor = brain.query_processor
    assert query_processor is not None
    assert brain._query_processor is query_processor

    # Second access returns same instance
    assert brain.query_processor is query_processor


# =============================================================================
# 3. State and Mode Tests
# =============================================================================


def test_brain_initial_mode_is_idle():
    """Brain should start in IDLE mode"""
    brain = UnifiedBrain()
    assert brain.mode == BrainMode.IDLE


@pytest.mark.asyncio
async def test_brain_mode_transition_to_responding(brain_with_mocks):
    """Brain mode should change to RESPONDING during query processing"""
    # Mock decision_maker
    brain_with_mocks._decision_maker = MagicMock()
    brain_with_mocks._decision_maker.decide = AsyncMock(
        return_value={
            "tool": "direct_answer",
            "confidence": 0.8,
            "reason": "test",
            "key_points": [],
        }
    )
    brain_with_mocks._initialized = True

    # Track mode changes by patching process_query to check mode mid-execution
    mode_during_processing = None

    original_generate = brain_with_mocks._response_pipeline.generate

    async def capture_mode_and_generate(*args, **kwargs):
        nonlocal mode_during_processing
        mode_during_processing = brain_with_mocks.mode
        return await original_generate(*args, **kwargs)

    brain_with_mocks._response_pipeline.generate = capture_mode_and_generate

    # Initial mode
    assert brain_with_mocks.mode == BrainMode.IDLE

    # Process query
    await brain_with_mocks.process_query("test query")

    # Should have been in RESPONDING mode during processing
    assert mode_during_processing == BrainMode.RESPONDING
    # Should return to IDLE after completion
    assert brain_with_mocks.mode == BrainMode.IDLE


def test_brain_stats_tracking():
    """Brain should track statistics correctly"""
    brain = UnifiedBrain()

    # Initial stats
    stats = brain.get_stats()
    assert stats["total_queries"] == 0
    assert stats["llm_decisions"] == 0
    assert stats["cache_hits"] == 0

    # Update stats
    brain._stats["total_queries"] += 1
    brain._stats["cache_hits"] += 1

    # Verify updates
    stats = brain.get_stats()
    assert stats["total_queries"] == 1
    assert stats["cache_hits"] == 1


# =============================================================================
# 4. Event System Tests
# =============================================================================


def test_brain_on_event_registration():
    """Event handlers should be registered correctly"""
    brain = UnifiedBrain()

    handler1 = MagicMock()
    handler2 = MagicMock()

    # Register handlers
    brain.on_event("test_event", handler1)
    brain.on_event("test_event", handler2)
    brain.on_event("other_event", handler1)

    # Verify registration
    assert "test_event" in brain._event_handlers
    assert len(brain._event_handlers["test_event"]) == 2
    assert handler1 in brain._event_handlers["test_event"]
    assert handler2 in brain._event_handlers["test_event"]

    assert "other_event" in brain._event_handlers
    assert len(brain._event_handlers["other_event"]) == 1


@pytest.mark.asyncio
async def test_brain_emit_event_calls_handlers():
    """Emitting events should call registered handlers"""
    brain = UnifiedBrain()

    # Sync handler
    sync_handler = MagicMock()
    brain.on_event("test_event", sync_handler)

    # Async handler
    async_handler = AsyncMock()
    brain.on_event("test_event", async_handler)

    # Emit event
    event_data = {"key": "value"}
    await brain.emit_event("test_event", event_data)

    # Verify calls
    sync_handler.assert_called_once_with(event_data)
    async_handler.assert_called_once_with(event_data)


@pytest.mark.asyncio
async def test_brain_emit_event_handles_handler_errors():
    """Event emission should handle handler errors gracefully"""
    brain = UnifiedBrain()

    # Handler that raises error
    def failing_handler(data):
        raise ValueError("Handler error")

    brain.on_event("test_event", failing_handler)

    # Should not raise exception
    try:
        await brain.emit_event("test_event", {"key": "value"})
    except Exception as e:
        pytest.fail(f"emit_event raised exception: {e}")


# =============================================================================
# 5. Task Queue Tests
# =============================================================================


def test_brain_add_task():
    """Adding tasks should use priority heap"""
    brain = UnifiedBrain()

    task1 = BrainTask(id="bg-1", type="background", priority=TaskPriority.BACKGROUND, payload={})
    task2 = BrainTask(id="user-1", type="user", priority=TaskPriority.USER_REQUEST, payload={})
    task3 = BrainTask(id="alert-1", type="alert", priority=TaskPriority.CRITICAL_ALERT, payload={})

    # Add tasks (not in priority order)
    brain.add_task(task1)
    brain.add_task(task2)
    brain.add_task(task3)

    # Queue should maintain priority
    assert len(brain._task_queue) == 3


def test_brain_task_queue_priority_ordering():
    """Task queue should maintain priority ordering"""
    brain = UnifiedBrain()

    # Add tasks in random order
    tasks = [
        BrainTask(id="bg-1", type="bg", priority=TaskPriority.BACKGROUND, payload={}),
        BrainTask(id="user-1", type="user", priority=TaskPriority.USER_REQUEST, payload={}),
        BrainTask(id="sched-1", type="sched", priority=TaskPriority.SCHEDULED, payload={}),
        BrainTask(id="alert-1", type="alert", priority=TaskPriority.CRITICAL_ALERT, payload={}),
    ]

    for task in tasks:
        brain.add_task(task)

    # Pop tasks - should come out in priority order
    import heapq

    first = heapq.heappop(brain._task_queue)
    assert first.priority == TaskPriority.USER_REQUEST

    second = heapq.heappop(brain._task_queue)
    assert second.priority == TaskPriority.CRITICAL_ALERT

    third = heapq.heappop(brain._task_queue)
    assert third.priority == TaskPriority.SCHEDULED

    fourth = heapq.heappop(brain._task_queue)
    assert fourth.priority == TaskPriority.BACKGROUND


# =============================================================================
# 6. Helper Method Tests
# =============================================================================


def test_brain_get_system_state():
    """get_system_state should return correct structure"""
    brain = UnifiedBrain()

    # Without metrics
    state = brain._get_system_state()
    assert "data_status" in state
    assert "data_date" in state
    assert "available_tools" in state
    assert "failed_tools" in state
    assert "mode" in state
    assert "cache_stats" in state

    # With metrics
    current_metrics = {
        "metadata": {"data_date": "2026-01-28"},
        "brand_metrics": [],
    }
    state = brain._get_system_state(current_metrics)
    assert state["data_date"] == "2026-01-28"


def test_brain_format_system_state():
    """format_system_state should return formatted string"""
    brain = UnifiedBrain()

    state = {
        "data_status": "최신",
        "mode": "idle",
        "available_tools": ["tool1", "tool2"],
        "failed_tools": ["tool3"],
    }

    formatted = brain._format_system_state(state)

    assert "데이터 상태: 최신" in formatted
    assert "동작 모드: idle" in formatted
    assert "tool1" in formatted
    assert "tool2" in formatted
    assert "실패 도구: tool3" in formatted


def test_brain_get_stats():
    """get_stats should return complete statistics"""
    brain = UnifiedBrain()

    stats = brain.get_stats()

    # Check basic stats
    assert "total_queries" in stats
    assert "llm_decisions" in stats
    assert "cache_hits" in stats
    assert "autonomous_tasks" in stats
    assert "alerts_generated" in stats
    assert "errors" in stats

    # Check derived stats
    assert "mode" in stats
    assert "queue_size" in stats
    assert "scheduled_tasks" in stats

    # Check component stats
    assert "components" in stats
    assert "decision_maker" in stats["components"]
    assert "tool_coordinator" in stats["components"]
    assert "alert_manager" in stats["components"]


def test_brain_get_state_summary():
    """get_state_summary should return formatted summary"""
    brain = UnifiedBrain()

    summary = brain.get_state_summary()
    assert isinstance(summary, str)
    assert "idle" in summary  # Should contain mode


# =============================================================================
# 7. Singleton Tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_brain_singleton():
    """get_brain should return singleton instance"""
    brain1 = await get_brain()
    brain2 = await get_brain()

    assert brain1 is brain2


def test_reset_brain():
    """reset_brain should clear singleton"""
    from src.core.brain import _brain_instance

    # Create singleton
    asyncio.run(get_brain())

    # Reset
    reset_brain()

    # Verify reset

    assert _brain_instance is None


# =============================================================================
# 8. Property Tests
# =============================================================================


def test_brain_context_gatherer_property():
    """context_gatherer property should work correctly"""
    brain = UnifiedBrain()
    mock_gatherer = MagicMock()

    # Setter
    brain.context_gatherer = mock_gatherer
    assert brain._context_gatherer is mock_gatherer

    # Getter
    assert brain.context_gatherer is mock_gatherer


def test_brain_tool_executor_property():
    """tool_executor property should return injected executor"""
    mock_executor = MagicMock()
    brain = UnifiedBrain(tool_executor=mock_executor)

    assert brain.tool_executor is mock_executor


def test_brain_response_pipeline_property():
    """response_pipeline property should return injected pipeline"""
    mock_pipeline = MagicMock()
    brain = UnifiedBrain(response_pipeline=mock_pipeline)

    assert brain.response_pipeline is mock_pipeline


# =============================================================================
# 9. Error Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_brain_process_query_error_handling(brain_with_mocks):
    """process_query should handle errors gracefully"""
    brain_with_mocks._initialized = True
    brain_with_mocks._context_gatherer.gather.side_effect = Exception("Gather error")

    response = await brain_with_mocks.process_query("test query")

    # Should return fallback response
    assert response.is_fallback
    assert "오류" in response.text

    # Error stat should increment
    assert brain_with_mocks._stats["errors"] == 1

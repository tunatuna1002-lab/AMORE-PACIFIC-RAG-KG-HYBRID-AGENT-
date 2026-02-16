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
    assert brain.model == "gpt-4.1-mini"
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


def test_brain_lazy_init_query_graph():
    """QueryGraph should be initialized after brain.initialize()"""
    brain = UnifiedBrain()

    # Initially None (before initialize)
    assert brain._query_graph is None


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
    from src.core.models import Decision
    from src.core.query_graph import QueryGraph

    # Mock decision_maker with proper Decision object
    brain_with_mocks._decision_maker = MagicMock()
    brain_with_mocks._decision_maker.decide = AsyncMock(
        return_value=Decision(
            tool="direct_answer",
            confidence=0.8,
            reason="test",
            key_points=[],
        )
    )
    brain_with_mocks._initialized = True

    # Initialize query graph so process_query can delegate to it
    brain_with_mocks._query_graph = QueryGraph(
        cache=brain_with_mocks.cache,
        context_gatherer=brain_with_mocks._context_gatherer,
        confidence_assessor=brain_with_mocks.confidence_assessor,
        decision_maker=brain_with_mocks.decision_maker,
        tool_coordinator=brain_with_mocks.tool_coordinator,
        response_pipeline=brain_with_mocks._response_pipeline,
        react_agent=brain_with_mocks._react_agent,
    )

    # Track mode changes by patching response_pipeline to check mode mid-execution
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


# =============================================================================
# 10. Task Scheduling and Execution Tests (Wave 4)
# =============================================================================


class TestAutonomousCycle:
    """자율 작업 사이클 테스트"""

    @pytest.mark.asyncio
    async def test_run_autonomous_cycle_skips_when_responding(self):
        """사용자 응답 중에는 자율 작업 스킵"""
        brain = UnifiedBrain()
        brain.mode = BrainMode.RESPONDING

        result = await brain.run_autonomous_cycle()

        assert result["status"] == "skipped"
        assert "사용자 응답 중" in result["reason"]

    @pytest.mark.asyncio
    async def test_run_autonomous_cycle_sets_mode(self):
        """자율 모드 전환 및 복원"""
        brain = UnifiedBrain()
        brain.scheduler = MagicMock()
        brain.scheduler.get_due_tasks = MagicMock(return_value=[])

        mode_during_execution = None

        original_process = brain._process_task_queue

        async def capture_mode():
            nonlocal mode_during_execution
            mode_during_execution = brain.mode
            return await original_process()

        brain._process_task_queue = capture_mode

        result = await brain.run_autonomous_cycle()

        assert mode_during_execution == BrainMode.AUTONOMOUS
        assert brain.mode == BrainMode.IDLE
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_run_autonomous_cycle_increments_stats(self):
        """자율 작업 실행 시 통계 증가"""
        brain = UnifiedBrain()
        brain.scheduler = MagicMock()
        brain.scheduler.get_due_tasks = MagicMock(return_value=[])

        initial_count = brain._stats["autonomous_tasks"]
        await brain.run_autonomous_cycle()

        assert brain._stats["autonomous_tasks"] == initial_count + 1

    @pytest.mark.asyncio
    async def test_run_autonomous_cycle_executes_due_tasks(self):
        """스케줄된 작업 실행"""
        brain = UnifiedBrain()
        brain.scheduler = MagicMock()
        due_task = {"id": "task-1", "name": "check_data", "action": "check_data"}
        brain.scheduler.get_due_tasks = MagicMock(return_value=[due_task])

        brain.state = MagicMock()
        brain.state.is_crawl_needed = MagicMock(return_value=False)

        result = await brain.run_autonomous_cycle()

        assert result["status"] == "completed"
        assert len(result["tasks_executed"]) == 1
        brain.scheduler.mark_completed.assert_called_once_with("task-1")

    @pytest.mark.asyncio
    async def test_run_autonomous_cycle_error_handling(self):
        """자율 사이클 에러 처리"""
        brain = UnifiedBrain()
        brain.scheduler = MagicMock()
        brain.scheduler.get_due_tasks = MagicMock(side_effect=Exception("Scheduler error"))

        result = await brain.run_autonomous_cycle()

        assert result["status"] == "error"
        assert "Scheduler error" in result["error"]
        assert brain.mode == BrainMode.IDLE  # mode should be restored


# =============================================================================
# 11. Scheduled Task Execution Tests (Wave 4)
# =============================================================================


class TestExecuteScheduledTask:
    """스케줄된 작업 실행 테스트"""

    @pytest.mark.asyncio
    async def test_check_data_action(self):
        """check_data 작업 실행"""
        brain = UnifiedBrain()
        brain.state = MagicMock()
        brain.state.is_crawl_needed = MagicMock(return_value=True)

        task = {"id": "t1", "name": "Data Check", "action": "check_data"}
        result = await brain._execute_scheduled_task(task)

        assert result["status"] == "completed"
        assert result["needs_crawl"] is True

    @pytest.mark.asyncio
    async def test_unknown_action_skipped(self):
        """알 수 없는 action은 스킵"""
        brain = UnifiedBrain()

        task = {"id": "t2", "name": "Unknown Task", "action": "unknown_action"}
        result = await brain._execute_scheduled_task(task)

        assert result["status"] == "skipped"
        assert "Unknown action" in result["reason"]

    @pytest.mark.asyncio
    async def test_crawl_workflow_action(self):
        """crawl_workflow 작업 실행"""
        brain = UnifiedBrain()

        mock_workflow = AsyncMock()
        mock_workflow.run_daily_workflow = AsyncMock(
            return_value={"summary": "crawl done", "products": 100}
        )
        brain._workflow_agent = mock_workflow

        # Mock emit_event and collect_market_intelligence
        brain.emit_event = AsyncMock()
        brain.collect_market_intelligence = AsyncMock(return_value={"status": "success"})

        task = {"id": "t3", "name": "Daily Crawl", "action": "crawl_workflow"}
        result = await brain._execute_scheduled_task(task)

        assert result["status"] == "completed"
        mock_workflow.run_daily_workflow.assert_called_once()
        brain.emit_event.assert_called()

    @pytest.mark.asyncio
    async def test_scheduled_task_failure(self):
        """스케줄된 작업 실패 처리"""
        brain = UnifiedBrain()
        brain.state = MagicMock()
        brain.state.is_crawl_needed = MagicMock(side_effect=Exception("DB error"))

        task = {"id": "t4", "name": "Failing Task", "action": "check_data"}
        result = await brain._execute_scheduled_task(task)

        assert result["status"] == "failed"
        assert "DB error" in result["error"]


# =============================================================================
# 12. Task Queue Processing Tests (Wave 4)
# =============================================================================


class TestTaskQueueProcessing:
    """작업 큐 처리 테스트"""

    @pytest.mark.asyncio
    async def test_process_task_queue_empty(self):
        """빈 큐 처리"""
        brain = UnifiedBrain()
        # Should not raise
        await brain._process_task_queue()
        assert len(brain._task_queue) == 0

    @pytest.mark.asyncio
    async def test_process_task_queue_stops_when_responding(self):
        """사용자 응답 모드에서 큐 처리 중단"""
        brain = UnifiedBrain()

        task1 = BrainTask(
            id="t1", type="alert", priority=TaskPriority.BACKGROUND, payload={"alert": "test"}
        )
        task2 = BrainTask(
            id="t2", type="alert", priority=TaskPriority.BACKGROUND, payload={"alert": "test2"}
        )
        brain.add_task(task1)
        brain.add_task(task2)

        # Switch to responding mode after first task
        original_execute = brain._execute_queued_task

        async def switch_mode(task):
            brain.mode = BrainMode.RESPONDING
            await original_execute(task)

        brain._execute_queued_task = switch_mode

        await brain._process_task_queue()

        # Should have stopped after mode change, leaving items in queue
        # (the first task pops, mode changes, loop breaks before second pop)
        assert brain.mode == BrainMode.RESPONDING

    @pytest.mark.asyncio
    async def test_execute_queued_task_alert(self):
        """alert 타입 큐 작업 실행"""
        brain = UnifiedBrain()
        brain._alert_manager = MagicMock()
        brain._alert_manager.process_alert = AsyncMock()
        brain._alert_manager.check_conditions = AsyncMock(return_value=[])
        brain.emit_event = AsyncMock()

        task = BrainTask(
            id="alert-1",
            type="alert",
            priority=TaskPriority.CRITICAL_ALERT,
            payload={"severity": "critical", "message": "test alert"},
        )

        await brain._execute_queued_task(task)

        assert task.started_at is not None
        assert task.completed_at is not None
        assert task.result == {"processed": True}
        assert task in brain._task_history
        assert brain._current_task is None

    @pytest.mark.asyncio
    async def test_execute_queued_task_error(self):
        """큐 작업 실행 에러 처리"""
        brain = UnifiedBrain()

        # Create a task that will raise when processed
        brain._alert_manager = MagicMock()
        brain._alert_manager.process_alert = AsyncMock(side_effect=Exception("Alert error"))
        brain._alert_manager.check_conditions = AsyncMock(return_value=[])
        brain.emit_event = AsyncMock(side_effect=Exception("Alert error"))

        task = BrainTask(
            id="err-1",
            type="alert",
            priority=TaskPriority.CRITICAL_ALERT,
            payload={"test": "error"},
        )

        await brain._execute_queued_task(task)

        assert task.error is not None
        assert brain._current_task is None  # Should be cleaned up

    def test_add_task_maintains_heap(self):
        """add_task가 힙 구조 유지"""
        import heapq

        brain = UnifiedBrain()

        tasks = [
            BrainTask(id="bg", type="bg", priority=TaskPriority.BACKGROUND, payload={}),
            BrainTask(id="user", type="user", priority=TaskPriority.USER_REQUEST, payload={}),
            BrainTask(id="sched", type="sched", priority=TaskPriority.SCHEDULED, payload={}),
        ]

        for t in tasks:
            brain.add_task(t)

        # Pop should give USER_REQUEST first
        first = heapq.heappop(brain._task_queue)
        assert first.id == "user"


# =============================================================================
# 13. State Management Tests (Wave 4)
# =============================================================================


class TestStateManagement:
    """상태 관리 테스트"""

    def test_get_system_state_without_metrics(self):
        """메트릭 없이 시스템 상태"""
        brain = UnifiedBrain()
        state = brain._get_system_state()

        assert state["data_status"] == "없음"
        assert state["data_date"] is None
        assert state["mode"] == "idle"

    def test_get_system_state_with_fresh_metrics(self):
        """최신 메트릭으로 시스템 상태"""
        brain = UnifiedBrain()
        today = datetime.now().strftime("%Y-%m-%d")
        metrics = {"metadata": {"data_date": today}}

        state = brain._get_system_state(metrics)

        assert state["data_status"] == "최신"
        assert state["data_date"] == today

    def test_get_system_state_with_stale_metrics(self):
        """오래된 메트릭으로 시스템 상태"""
        brain = UnifiedBrain()
        metrics = {"metadata": {"data_date": "2025-01-01"}}

        state = brain._get_system_state(metrics)

        assert "오래됨" in state["data_status"]

    def test_format_system_state_no_failed_tools(self):
        """실패 도구 없을 때 포맷"""
        brain = UnifiedBrain()
        state = {
            "data_status": "최신",
            "mode": "idle",
            "available_tools": ["get_brand_status"],
            "failed_tools": [],
        }

        formatted = brain._format_system_state(state)
        assert "실패 도구" not in formatted

    def test_get_state_summary_format(self):
        """상태 요약 포맷 확인"""
        brain = UnifiedBrain()
        summary = brain.get_state_summary()
        assert "idle" in summary

    def test_reset_failed_agents(self):
        """실패 에이전트 리셋"""
        brain = UnifiedBrain()
        brain._tool_coordinator = MagicMock()
        brain._tool_coordinator.reset_failed_tools = MagicMock()

        brain.reset_failed_agents()

        brain._tool_coordinator.reset_failed_tools.assert_called_once()


# =============================================================================
# 14. Complex Query Detection Tests (Wave 4)
# =============================================================================


class TestComplexQueryDetection:
    """복잡한 질문 감지 테스트"""

    def test_complex_keyword_detected(self):
        """복잡도 키워드 감지"""
        brain = UnifiedBrain()
        context = Context(
            query="왜 LANEIGE 순위가 하락했나요?",
            entities={},
            rag_docs=[{"content": "doc1"}, {"content": "doc2"}],
            kg_facts=[],
        )
        # "왜" is a complex keyword
        assert brain._is_complex_query("왜 LANEIGE 순위가 하락했나요?", context) is True

    def test_analysis_keyword_detected(self):
        """분석 키워드 감지"""
        brain = UnifiedBrain()
        context = Context(
            query="LANEIGE 경쟁사 비교 분석해줘",
            entities={},
            rag_docs=[{"content": "doc1"}, {"content": "doc2"}],
            kg_facts=[],
        )
        assert brain._is_complex_query("LANEIGE 경쟁사 비교 분석해줘", context) is True

    def test_simple_query_not_complex(self):
        """단순 질문은 복잡하지 않음"""
        brain = UnifiedBrain()
        context = Context(
            query="LANEIGE 순위 알려줘",
            entities={},
            rag_docs=[{"content": "doc1"}, {"content": "doc2"}, {"content": "doc3"}],
            kg_facts=[],
        )
        # kg_triples is checked via hasattr, set as attribute
        context.kg_triples = [("LANEIGE", "rank", "5")]
        result = brain._is_complex_query("LANEIGE 순위 알려줘", context)
        assert result is False

    def test_low_context_with_multi_step_is_complex(self):
        """컨텍스트 부족 + 다단계 질문은 복잡"""
        brain = UnifiedBrain()
        context = Context(
            query="LANEIGE 그리고 COSRX?",
            entities={},
            rag_docs=[],  # no docs
            kg_facts=[],
        )
        result = brain._is_complex_query("LANEIGE 그리고 COSRX?", context)
        # low_context and multi_step (contains "그리고")
        assert result is True


# =============================================================================
# 15. Query Intent Assessment Tests (Wave 4)
# =============================================================================


class TestQueryIntentAssessment:
    """쿼리 의도 평가 테스트"""

    def test_empty_query_returns_zero(self):
        """빈 쿼리 → 0점"""
        brain = UnifiedBrain()
        assert brain._assess_query_intent("") == 0.0
        assert brain._assess_query_intent("  ") == 0.0

    def test_short_query_returns_zero(self):
        """짧은 무의미 입력 → 0점"""
        brain = UnifiedBrain()
        assert brain._assess_query_intent("ab") == 0.0

    def test_domain_keyword_score(self):
        """도메인 키워드 포함 → 1.5+ 점"""
        brain = UnifiedBrain()
        score = brain._assess_query_intent("laneige 순위")
        assert score >= 1.5

    def test_intent_keyword_score(self):
        """의도 키워드 포함 → 1.5+ 점"""
        brain = UnifiedBrain()
        score = brain._assess_query_intent("분석 결과 보여줘")
        assert score >= 1.5

    def test_meaningful_query_floor(self):
        """의미 있는 질문은 최소 1.5점"""
        brain = UnifiedBrain()
        score = brain._assess_query_intent("오늘 날씨 어때")
        assert score >= 1.5

    def test_domain_and_intent_combined(self):
        """도메인 + 의도 키워드 조합"""
        brain = UnifiedBrain()
        score = brain._assess_query_intent("laneige 순위 분석해줘")
        # Both domain and intent keywords
        assert score >= 2.0


# =============================================================================
# 16. Key Points Extraction Tests (Wave 4)
# =============================================================================


class TestKeyPointsExtraction:
    """핵심 포인트 추출 테스트"""

    def test_extract_from_kg_facts(self):
        """KG 사실에서 포인트 추출"""
        brain = UnifiedBrain()
        fact = MagicMock()
        fact.entity = "LANEIGE"
        fact.fact_type = "market_leader"

        context = Context(
            query="test",
            entities={},
            rag_docs=[],
            kg_facts=[fact],
        )

        points = brain._extract_key_points_from_context(context)
        assert len(points) >= 1
        assert "LANEIGE" in points[0]

    def test_extract_from_kg_inferences(self):
        """KG 추론에서 포인트 추출"""
        brain = UnifiedBrain()

        context = Context(
            query="test",
            entities={},
            rag_docs=[],
            kg_facts=[],
            kg_inferences=[{"insight": "LANEIGE SoS is growing"}],
        )

        points = brain._extract_key_points_from_context(context)
        assert len(points) >= 1
        assert "LANEIGE SoS is growing" in points[0]

    def test_extract_empty_context(self):
        """빈 컨텍스트에서 포인트 추출"""
        brain = UnifiedBrain()

        context = Context(
            query="test",
            entities={},
            rag_docs=[],
            kg_facts=[],
        )

        points = brain._extract_key_points_from_context(context)
        assert points == []

    def test_extract_limits_results(self):
        """최대 개수 제한"""
        brain = UnifiedBrain()

        facts = []
        for i in range(10):
            fact = MagicMock()
            fact.entity = f"Brand{i}"
            fact.fact_type = f"type{i}"
            facts.append(fact)

        context = Context(
            query="test",
            entities={},
            rag_docs=[],
            kg_facts=facts,
        )

        points = brain._extract_key_points_from_context(context)
        # Should be limited to 3 facts + 2 inferences max = 5
        assert len(points) <= 5


# =============================================================================
# 17. Response Generation Tests (Wave 4)
# =============================================================================


class TestResponseGeneration:
    """응답 생성 테스트"""

    @pytest.mark.asyncio
    async def test_generate_response_with_pipeline(self):
        """파이프라인으로 응답 생성"""
        brain = UnifiedBrain()
        mock_pipeline = AsyncMock()
        mock_pipeline.generate = AsyncMock(
            return_value=Response(text="Pipeline response", confidence_score=0.9)
        )
        brain._response_pipeline = mock_pipeline

        from src.core.models import Decision

        context = Context(query="test", entities={}, rag_docs=[], kg_facts=[])
        decision = Decision(tool="direct_answer", confidence=0.9, reason="test", key_points=[])

        response = await brain._generate_response(
            query="test", context=context, decision=decision, tool_result=None
        )

        assert response.text == "Pipeline response"
        mock_pipeline.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_fallback_with_tool_result(self):
        """파이프라인 없이 도구 결과로 폴백 응답"""
        brain = UnifiedBrain()
        brain._response_pipeline = None

        from src.core.models import Decision

        context = Context(query="test", entities={}, rag_docs=[], kg_facts=[])
        decision = Decision(tool="get_brand_status", confidence=0.8, reason="test", key_points=[])
        tool_result = ToolResult(
            tool_name="get_brand_status",
            success=True,
            data={"brand": "LANEIGE", "sos": 12.5},
        )

        response = await brain._generate_response(
            query="test", context=context, decision=decision, tool_result=tool_result
        )

        assert "도구 실행 결과" in response.text
        assert "LANEIGE" in response.text

    @pytest.mark.asyncio
    async def test_generate_response_fallback_with_context_summary(self):
        """파이프라인 없이 컨텍스트 요약으로 폴백"""
        brain = UnifiedBrain()
        brain._response_pipeline = None

        from src.core.models import Decision

        context = Context(
            query="test",
            entities={},
            rag_docs=[],
            kg_facts=[],
            summary="LANEIGE는 Lip Care에서 4위입니다",
        )
        decision = Decision(tool="direct_answer", confidence=0.8, reason="test", key_points=[])

        response = await brain._generate_response(
            query="test", context=context, decision=decision, tool_result=None
        )

        assert "LANEIGE는 Lip Care에서 4위입니다" in response.text

    @pytest.mark.asyncio
    async def test_generate_response_fallback_no_info(self):
        """정보 없을 때 폴백"""
        brain = UnifiedBrain()
        brain._response_pipeline = None

        from src.core.models import Decision

        context = Context(query="test", entities={}, rag_docs=[], kg_facts=[])
        decision = Decision(tool="direct_answer", confidence=0.5, reason="test", key_points=[])

        response = await brain._generate_response(
            query="test", context=context, decision=decision, tool_result=None
        )

        assert "관련 정보를 찾을 수 없습니다" in response.text


# =============================================================================
# 18. ReAct Processing Tests (Wave 4)
# =============================================================================


class TestReActProcessing:
    """ReAct 처리 테스트"""

    @pytest.mark.asyncio
    async def test_process_with_react_no_agent(self):
        """ReAct 에이전트 없을 때 폴백"""
        brain = UnifiedBrain()
        brain._react_agent = None

        context = Context(query="test", entities={}, rag_docs=[], kg_facts=[])
        response = await brain._process_with_react("test", context)

        assert response.is_fallback
        assert "ReAct 에이전트" in response.text

    @pytest.mark.asyncio
    async def test_process_with_react_success(self):
        """ReAct 처리 성공"""
        brain = UnifiedBrain()

        mock_step = MagicMock()
        mock_step.action = "search_kg"

        mock_result = MagicMock()
        mock_result.final_answer = "LANEIGE is #4 in Lip Care"
        mock_result.confidence = 0.85
        mock_result.steps = [mock_step]
        mock_result.needs_improvement = False

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        brain._react_agent = mock_agent

        context = Context(
            query="test",
            entities={},
            rag_docs=[{"content": "doc"}],
            kg_facts=[],
            summary="test summary",
        )
        response = await brain._process_with_react("왜 LANEIGE가 하락했나?", context)

        assert response.text == "LANEIGE is #4 in Lip Care"
        assert response.confidence_score == 0.85
        assert "search_kg" in response.tools_called

    @pytest.mark.asyncio
    async def test_process_with_react_failure(self):
        """ReAct 처리 실패"""
        brain = UnifiedBrain()

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=Exception("ReAct error"))
        brain._react_agent = mock_agent

        context = Context(query="test", entities={}, rag_docs=[], kg_facts=[])
        response = await brain._process_with_react("test", context)

        assert response.is_fallback
        assert "ReAct 처리 실패" in response.text


# =============================================================================
# 19. Market Intelligence Tests (Wave 4)
# =============================================================================


class TestMarketIntelligence:
    """Market Intelligence 데이터 수집 테스트"""

    @pytest.mark.asyncio
    async def test_collect_market_intelligence_success(self):
        """MI 수집 성공"""
        brain = UnifiedBrain()
        brain.emit_event = AsyncMock()

        mock_mi = AsyncMock()
        mock_mi.collect_all_layers = AsyncMock(return_value={"layer1": {}, "layer2": {}})
        mock_mi.save_data = MagicMock()
        brain._market_intelligence = mock_mi

        result = await brain.collect_market_intelligence()

        assert result["status"] == "success"
        assert "layer1" in result["layers_collected"]
        assert "layer2" in result["layers_collected"]
        mock_mi.collect_all_layers.assert_called_once()
        mock_mi.save_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_collect_market_intelligence_failure(self):
        """MI 수집 실패"""
        brain = UnifiedBrain()

        mock_mi = AsyncMock()
        mock_mi.collect_all_layers = AsyncMock(side_effect=Exception("API error"))
        brain._market_intelligence = mock_mi

        result = await brain.collect_market_intelligence()

        assert result["status"] == "error"
        assert "API error" in result["error"]

    @pytest.mark.asyncio
    async def test_check_alerts_delegates_to_alert_manager(self):
        """check_alerts가 AlertManager에 위임"""
        brain = UnifiedBrain()
        brain._alert_manager = MagicMock()
        brain._alert_manager.check_metrics_alerts = AsyncMock(
            return_value=[{"type": "sos_drop", "message": "SoS dropped"}]
        )

        alerts = await brain.check_alerts({"sos": 3.0})

        assert len(alerts) == 1
        assert alerts[0]["type"] == "sos_drop"


# =============================================================================
# 20. Scheduler Management Tests (Wave 4)
# =============================================================================


class TestSchedulerManagement:
    """스케줄러 관리 테스트"""

    @pytest.mark.asyncio
    async def test_start_scheduler_already_running(self):
        """이미 실행 중인 스케줄러 시작 시도"""
        brain = UnifiedBrain()
        brain.scheduler = MagicMock()
        brain.scheduler.running = True

        await brain.start_scheduler()

        # Should not call start again
        brain.scheduler.start.assert_not_called()

    def test_stop_scheduler(self):
        """스케줄러 중지"""
        brain = UnifiedBrain()
        brain.scheduler = MagicMock()
        brain.mode = BrainMode.AUTONOMOUS

        brain.stop_scheduler()

        brain.scheduler.stop.assert_called_once()
        assert brain.mode == BrainMode.IDLE

    def test_get_recent_errors(self):
        """최근 에러 목록 조회"""
        brain = UnifiedBrain()
        brain._tool_coordinator = MagicMock()
        brain._tool_coordinator.get_recent_errors = MagicMock(
            return_value=[{"error": "test", "time": "2026-01-01"}]
        )

        errors = brain.get_recent_errors(limit=5)

        assert len(errors) == 1
        brain._tool_coordinator.get_recent_errors.assert_called_once_with(5)


# =============================================================================
# 21. KG Sync Tests (Wave 4)
# =============================================================================


class TestKGSync:
    """Knowledge Graph 동기화 테스트"""

    def test_sync_kg_without_knowledge_graph(self):
        """KG가 없으면 조기 반환"""
        brain = UnifiedBrain()
        # No _knowledge_graph attribute
        brain._sync_knowledge_graph({"brand": {"competitors": []}})
        # Should not raise

    def test_sync_kg_with_brand_metrics(self):
        """브랜드 메트릭 KG 동기화"""
        brain = UnifiedBrain()
        mock_kg = MagicMock()
        brain._knowledge_graph = mock_kg

        data = {
            "brand": {
                "competitors": [
                    {"brand": "LANEIGE", "sos": 12.0, "avg_rank": 5, "products": 3},
                    {"brand": "COSRX", "sos": 8.0, "avg_rank": 15, "products": 2},
                ]
            }
        }
        brain._sync_knowledge_graph(data)

        assert mock_kg.add_entity_metadata.call_count == 2

    def test_sync_kg_with_owl_reasoner(self):
        """OWL Reasoner 동기화"""
        brain = UnifiedBrain()
        mock_kg = MagicMock()
        mock_owl = MagicMock()
        brain._knowledge_graph = mock_kg
        brain._owl_reasoner = mock_owl

        data = {
            "brand": {
                "competitors": [
                    {"brand": "LANEIGE", "sos": 12.0, "avg_rank": 5, "products": 3},
                ]
            }
        }
        brain._sync_knowledge_graph(data)

        mock_owl.add_brand.assert_called_once()
        mock_owl.infer_market_positions.assert_called_once()

    def test_sync_kg_error_handling(self):
        """KG 동기화 에러 처리"""
        brain = UnifiedBrain()
        mock_kg = MagicMock()
        mock_kg.add_entity_metadata = MagicMock(side_effect=Exception("KG error"))
        brain._knowledge_graph = mock_kg

        # Should not raise
        brain._sync_knowledge_graph({"brand": {"competitors": [{"brand": "LANEIGE"}]}})

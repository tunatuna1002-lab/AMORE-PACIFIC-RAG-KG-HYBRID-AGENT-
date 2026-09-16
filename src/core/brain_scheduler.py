"""
Brain task queue
================
``UnifiedBrain`` 이 처리할 작업의 우선순위 큐 (Phase 4).

brain 에서 분리한 것은 **큐 기계장치**다: 작업 타입(``BrainTask``), 우선순위
(``TaskPriority``), 힙 정렬, 실행 이력 보관. 스케줄된 작업의 *액션 분기*
(crawl_workflow / morning_brief / ...)는 brain 에 남는다 — 그 분기는 모든 줄이
brain 의 협력자(crawl manager, workflow, alert manager)를 호출하므로 옮겨도
brain 참조를 들고 되돌아 호출하는 클래스가 생길 뿐 결합이 줄지 않는다.

Usage:
    queue = BrainTaskQueue()
    queue.add(BrainTask(id="a1", type="alert", priority=TaskPriority.CRITICAL_ALERT, payload={}))
    while queue.pending:
        await queue.execute(queue.pop(), {"alert": handle_alert})
"""

from __future__ import annotations

import heapq
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """작업 우선순위"""

    USER_REQUEST = 0  # 사용자 요청 (최우선)
    CRITICAL_ALERT = 1  # 중요 알림
    SCHEDULED = 2  # 예약 작업
    BACKGROUND = 3  # 백그라운드 작업


@dataclass
class BrainTask:
    """두뇌가 처리할 작업"""

    id: str
    type: str  # query, scheduled, alert, autonomous
    priority: TaskPriority
    payload: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any | None = None
    error: str | None = None

    def __lt__(self, other: BrainTask) -> bool:
        """우선순위 기반 정렬"""
        return self.priority.value < other.priority.value


TaskHandler = Callable[[dict[str, Any]], Awaitable[Any]]


class BrainTaskQueue:
    """우선순위 큐 + 실행 이력.

    ``pending`` 은 heapq 가 직접 다루는 리스트다 (호출부가 ``heappop`` 으로 꺼내
    쓰는 경로가 있어 그대로 노출한다).
    """

    def __init__(self) -> None:
        self.pending: list[BrainTask] = []
        self.history: list[BrainTask] = []
        self.current: BrainTask | None = None

    def __len__(self) -> int:
        return len(self.pending)

    def __bool__(self) -> bool:
        return bool(self.pending)

    def add(self, task: BrainTask) -> None:
        """작업 큐에 추가 (우선순위 힙)"""
        heapq.heappush(self.pending, task)

    def pop(self) -> BrainTask:
        """가장 우선순위 높은 작업 꺼내기"""
        return heapq.heappop(self.pending)

    async def execute(self, task: BrainTask, handlers: dict[str, TaskHandler]) -> None:
        """작업 1건 실행 (예외는 ``task.error`` 에 기록하고 삼키지 않고 로깅)"""
        task.started_at = datetime.now()
        self.current = task

        try:
            handler = handlers.get(task.type)
            if handler is not None:
                await handler(task.payload)
                task.result = {"processed": True}

            task.completed_at = datetime.now()
            self.history.append(task)

        except Exception as e:
            task.error = str(e)
            logger.error(f"Queued task failed: {e}")

        finally:
            self.current = None


__all__ = ["BrainTask", "BrainTaskQueue", "TaskHandler", "TaskPriority"]

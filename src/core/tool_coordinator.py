"""
Tool Coordinator - 도구 실행 조율 전담
======================================
UnifiedBrain에서 분리된 도구 실행 조율 컴포넌트

책임:
- 도구 실행
- 에러 핸들링 (재시도, 폴백)
- 실행 상태 추적
- 실패 도구 관리

관련 Protocol: ToolCoordinatorProtocol
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any

from .cache import ResponseCache
from .models import ToolResult
from .state import OrchestratorState
from .tools import ToolExecutor

logger = logging.getLogger(__name__)


class ErrorStrategy(Enum):
    """에러 처리 전략"""

    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    NOTIFY_USER = "notify_user"


# 도구별 에러 전략
TOOL_ERROR_STRATEGIES: dict[str, ErrorStrategy] = {
    "crawl_amazon": ErrorStrategy.FALLBACK,
    "calculate_metrics": ErrorStrategy.RETRY,
    "query_data": ErrorStrategy.FALLBACK,
    "query_knowledge_graph": ErrorStrategy.SKIP,
    "generate_insight": ErrorStrategy.RETRY,
    "send_alert": ErrorStrategy.RETRY,
    "workflow": ErrorStrategy.RETRY,
    "query_deals": ErrorStrategy.FALLBACK,
    "query_deals_summary": ErrorStrategy.FALLBACK,
}


class ToolCoordinator:
    """
    도구 실행 조율

    에러 핸들링, 재시도, 폴백을 포함한 안정적인 도구 실행을 제공합니다.

    Usage:
        coordinator = ToolCoordinator()
        result = await coordinator.execute("crawl_amazon", {"categories": ["lip_care"]})
    """

    def __init__(
        self,
        tool_executor: ToolExecutor | None = None,
        state: OrchestratorState | None = None,
        cache: ResponseCache | None = None,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ):
        """
        Args:
            tool_executor: 도구 실행기
            state: 오케스트레이터 상태
            cache: 응답 캐시 (폴백용)
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 간 대기 시간 (초)
        """
        self.tool_executor = tool_executor or ToolExecutor()
        self.state = state or OrchestratorState()
        self.cache = cache or ResponseCache()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 에러 추적
        self._error_history: list[dict[str, Any]] = []
        self._failed_tools: dict[str, datetime] = {}

        # 통계
        self._stats = {
            "total_executions": 0,
            "successful": 0,
            "failed": 0,
            "retried": 0,
            "fallbacks_used": 0,
        }

    async def execute(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """
        도구 실행 (에러 처리 포함)

        Args:
            tool_name: 실행할 도구 이름
            params: 도구 파라미터

        Returns:
            ToolResult 객체
        """
        self._stats["total_executions"] += 1
        strategy = TOOL_ERROR_STRATEGIES.get(tool_name, ErrorStrategy.NOTIFY_USER)
        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                self.state.start_tool(tool_name)
                result = await self.tool_executor.execute(tool_name, params)
                self.state.end_tool(tool_name)

                if result.success:
                    self._stats["successful"] += 1
                    self._failed_tools.pop(tool_name, None)
                    return result

                # 실패 처리
                error_info = self._record_error(
                    tool_name=tool_name,
                    error_message=result.error or "Unknown error",
                    error_type="execution",
                    retry_count=retry_count,
                )
                return await self._handle_error(error_info, strategy, tool_name, params)

            except TimeoutError:
                self.state.end_tool(tool_name)
                error_info = self._record_error(
                    tool_name=tool_name,
                    error_message="Timeout",
                    error_type="timeout",
                    retry_count=retry_count,
                )

                if strategy == ErrorStrategy.RETRY and retry_count < self.max_retries:
                    retry_count += 1
                    self._stats["retried"] += 1
                    await asyncio.sleep(self.retry_delay)
                    continue

                return await self._handle_error(error_info, strategy, tool_name, params)

            except Exception as e:
                self.state.end_tool(tool_name)
                error_info = self._record_error(
                    tool_name=tool_name,
                    error_message=str(e),
                    error_type="exception",
                    retry_count=retry_count,
                )

                if strategy == ErrorStrategy.RETRY and retry_count < self.max_retries:
                    retry_count += 1
                    self._stats["retried"] += 1
                    await asyncio.sleep(self.retry_delay)
                    continue

                return await self._handle_error(error_info, strategy, tool_name, params)

        self._stats["failed"] += 1
        return ToolResult(
            tool_name=tool_name, success=False, error=f"최대 재시도 초과 ({self.max_retries}회)"
        )

    async def _handle_error(
        self,
        error_info: dict[str, Any],
        strategy: ErrorStrategy,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult:
        """에러 전략에 따른 처리"""
        self._failed_tools[tool_name] = datetime.now()

        logger.warning(
            f"Error in {tool_name}: {error_info['error_message']} " f"(strategy: {strategy.value})"
        )

        if strategy == ErrorStrategy.FALLBACK:
            cached = self.cache.get_tool_result(tool_name)
            if cached:
                self._stats["fallbacks_used"] += 1
                logger.info(f"Using cached result for {tool_name}")
                return ToolResult(
                    tool_name=tool_name, success=True, data={**cached, "_from_cache": True}
                )

        elif strategy == ErrorStrategy.SKIP:
            logger.info(f"Skipping failed tool: {tool_name}")
            return ToolResult(tool_name=tool_name, success=True, data={"_skipped": True})

        # NOTIFY_USER 또는 기타
        self._stats["failed"] += 1
        return ToolResult(tool_name=tool_name, success=False, error=error_info["error_message"])

    def _record_error(
        self, tool_name: str, error_message: str, error_type: str, retry_count: int
    ) -> dict[str, Any]:
        """에러 기록"""
        error_info = {
            "tool_name": tool_name,
            "error_message": error_message,
            "error_type": error_type,
            "retry_count": retry_count,
            "timestamp": datetime.now().isoformat(),
        }

        self._error_history.append(error_info)

        # 히스토리 크기 제한
        if len(self._error_history) > 100:
            self._error_history = self._error_history[-50:]

        return error_info

    def get_available_tools(self) -> list[str]:
        """사용 가능한 도구 목록"""
        all_tools = self.tool_executor.get_available_tools()
        failed_recently = self.get_failed_tools()
        return [t for t in all_tools if t not in failed_recently]

    def get_failed_tools(self, timeout_seconds: int = 300) -> list[str]:
        """
        최근 실패한 도구 목록

        Args:
            timeout_seconds: 실패 후 재사용 불가 시간 (초)

        Returns:
            최근 실패한 도구 이름 목록
        """
        now = datetime.now()
        return [
            name
            for name, time in self._failed_tools.items()
            if (now - time).seconds < timeout_seconds
        ]

    def reset_failed_tools(self) -> None:
        """실패 도구 목록 초기화"""
        self._failed_tools.clear()
        logger.info("Failed tools list cleared")

    def get_recent_errors(self, limit: int = 10) -> list[dict[str, Any]]:
        """최근 에러 목록"""
        return self._error_history[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """통계 반환"""
        return {
            **self._stats,
            "error_history_size": len(self._error_history),
            "failed_tools": list(self._failed_tools.keys()),
        }

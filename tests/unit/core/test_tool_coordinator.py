"""
ToolCoordinator 단위 테스트
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.models import ToolResult
from src.core.tool_coordinator import (
    TOOL_ERROR_STRATEGIES,
    ErrorStrategy,
    ToolCoordinator,
)


class TestErrorStrategy:
    """ErrorStrategy 열거형 테스트"""

    def test_strategies_exist(self):
        assert ErrorStrategy.RETRY.value == "retry"
        assert ErrorStrategy.FALLBACK.value == "fallback"
        assert ErrorStrategy.SKIP.value == "skip"
        assert ErrorStrategy.NOTIFY_USER.value == "notify_user"

    def test_tool_error_strategies_mapping(self):
        assert TOOL_ERROR_STRATEGIES["crawl_amazon"] == ErrorStrategy.FALLBACK
        assert TOOL_ERROR_STRATEGIES["calculate_metrics"] == ErrorStrategy.RETRY
        assert TOOL_ERROR_STRATEGIES["query_knowledge_graph"] == ErrorStrategy.SKIP


class TestToolCoordinator:
    """ToolCoordinator 클래스 테스트"""

    def setup_method(self):
        self.mock_executor = MagicMock()
        self.mock_executor.get_available_tools.return_value = [
            "crawl_amazon",
            "calculate_metrics",
            "query_data",
        ]
        self.mock_state = MagicMock()
        self.mock_cache = MagicMock()

        self.coordinator = ToolCoordinator(
            tool_executor=self.mock_executor,
            state=self.mock_state,
            cache=self.mock_cache,
            max_retries=2,
            retry_delay=0.01,  # 빠른 테스트
        )

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """성공적인 도구 실행"""
        self.mock_executor.execute = AsyncMock(
            return_value=ToolResult(tool_name="query_data", success=True, data={"count": 5})
        )

        result = await self.coordinator.execute("query_data", {"sql": "SELECT 1"})

        assert result.success is True
        assert result.data["count"] == 5
        assert self.coordinator._stats["successful"] == 1

    @pytest.mark.asyncio
    async def test_execute_failure_notify_user(self):
        """실패 시 NOTIFY_USER 전략 (기본)"""
        self.mock_executor.execute = AsyncMock(
            return_value=ToolResult(tool_name="unknown_tool", success=False, error="Not found")
        )

        result = await self.coordinator.execute("unknown_tool", {})

        assert result.success is False
        assert "Not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_fallback_with_cache(self):
        """FALLBACK 전략 + 캐시 있음"""
        self.mock_executor.execute = AsyncMock(
            return_value=ToolResult(tool_name="crawl_amazon", success=False, error="Timeout")
        )
        self.mock_cache.get_tool_result.return_value = {"products": []}

        result = await self.coordinator.execute("crawl_amazon", {})

        assert result.success is True
        assert result.data.get("_from_cache") is True

    @pytest.mark.asyncio
    async def test_execute_skip_strategy(self):
        """SKIP 전략"""
        self.mock_executor.execute = AsyncMock(
            return_value=ToolResult(
                tool_name="query_knowledge_graph", success=False, error="KG unavailable"
            )
        )

        result = await self.coordinator.execute("query_knowledge_graph", {})

        assert result.success is True
        assert result.data.get("_skipped") is True

    @pytest.mark.asyncio
    async def test_execute_timeout_with_retry(self):
        """타임아웃 + 재시도"""
        call_count = 0

        async def mock_execute(tool_name, params):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("Timed out")
            return ToolResult(tool_name=tool_name, success=True, data={})

        self.mock_executor.execute = mock_execute

        result = await self.coordinator.execute("calculate_metrics", {})

        assert result.success is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_exception_with_retry(self):
        """예외 + 재시도 소진"""
        self.mock_executor.execute = AsyncMock(side_effect=RuntimeError("Connection refused"))

        result = await self.coordinator.execute("calculate_metrics", {})

        assert result.success is False

    def test_get_available_tools(self):
        """사용 가능한 도구 목록"""
        tools = self.coordinator.get_available_tools()
        assert "crawl_amazon" in tools

    def test_get_failed_tools(self):
        """실패한 도구 추적"""
        self.coordinator._failed_tools["crawl_amazon"] = datetime.now()
        failed = self.coordinator.get_failed_tools(timeout_seconds=300)
        assert "crawl_amazon" in failed

    def test_reset_failed_tools(self):
        """실패 도구 초기화"""
        self.coordinator._failed_tools["test"] = datetime.now()
        self.coordinator.reset_failed_tools()
        assert len(self.coordinator._failed_tools) == 0

    def test_get_stats(self):
        """통계 반환"""
        stats = self.coordinator.get_stats()
        assert "total_executions" in stats
        assert "successful" in stats
        assert "failed" in stats

    def test_get_recent_errors(self):
        """최근 에러 목록"""
        self.coordinator._record_error("test", "error msg", "exception", 0)
        errors = self.coordinator.get_recent_errors(limit=5)
        assert len(errors) == 1
        assert errors[0]["tool_name"] == "test"

    def test_error_history_size_limit(self):
        """에러 히스토리 크기 제한"""
        for i in range(110):
            self.coordinator._record_error(f"tool_{i}", f"error_{i}", "exception", 0)

        # 히스토리는 100개 제한 (실제 구현 확인)
        assert len(self.coordinator._error_history) <= 100

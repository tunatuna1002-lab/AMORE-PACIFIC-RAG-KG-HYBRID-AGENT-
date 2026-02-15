"""
Unit tests for QueryProcessor
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.query_processor import QueryProcessor


@pytest.fixture
def mock_decision_maker():
    dm = AsyncMock()
    dm.decide = AsyncMock(return_value={"tool": "direct_answer", "answer": "test answer"})
    dm.get_stats = MagicMock(return_value={})
    return dm


@pytest.fixture
def mock_tool_coordinator():
    tc = AsyncMock()
    tc.execute = AsyncMock(return_value={"result": "tool_output"})
    tc.get_available_tools = MagicMock(return_value=["search", "calculate"])
    tc.get_failed_tools = MagicMock(return_value=[])
    tc.get_stats = MagicMock(return_value={})
    return tc


@pytest.fixture
def mock_response_pipeline():
    rp = AsyncMock()
    response = MagicMock()
    response.is_fallback = False
    response.processing_time_ms = 0
    rp.generate = AsyncMock(return_value=response)
    return rp


@pytest.fixture
def mock_context_gatherer():
    cg = AsyncMock()
    cg.gather = AsyncMock(return_value=MagicMock(query="test"))
    return cg


@pytest.fixture
def processor(mock_decision_maker, mock_tool_coordinator, mock_response_pipeline):
    return QueryProcessor(
        decision_maker=mock_decision_maker,
        tool_coordinator=mock_tool_coordinator,
        response_pipeline=mock_response_pipeline,
    )


class TestQueryProcessor:
    """Test QueryProcessor functionality"""

    @pytest.mark.asyncio
    async def test_process_basic_query(self, processor):
        """Test basic query processing"""
        result = await processor.process("LANEIGE 순위는?")

        assert result is not None
        processor.decision_maker.decide.assert_called_once()
        processor.response_pipeline.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_with_tool_call(
        self, mock_decision_maker, mock_tool_coordinator, mock_response_pipeline
    ):
        """Test query that requires a tool call"""
        mock_decision_maker.decide = AsyncMock(
            return_value={"tool": "search", "tool_params": {"query": "LANEIGE"}}
        )
        processor = QueryProcessor(
            decision_maker=mock_decision_maker,
            tool_coordinator=mock_tool_coordinator,
            response_pipeline=mock_response_pipeline,
        )

        await processor.process("LANEIGE 검색해줘")

        mock_tool_coordinator.execute.assert_called_once_with(
            tool_name="search", params={"query": "LANEIGE"}
        )

    @pytest.mark.asyncio
    async def test_process_with_cache_hit(self, processor):
        """Test cache hit"""
        cached_response = MagicMock()
        processor.cache.get = MagicMock(return_value=cached_response)

        result = await processor.process("cached query")

        assert result == cached_response
        processor.decision_maker.decide.assert_not_called()
        assert processor._stats["cache_hits"] == 1

    @pytest.mark.asyncio
    async def test_process_skip_cache(self, processor):
        """Test skipping cache"""
        processor.cache.get = MagicMock(return_value=MagicMock())

        await processor.process("query", skip_cache=True)

        # Should not check cache when skip_cache=True
        processor.decision_maker.decide.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_with_context_gatherer(
        self,
        mock_decision_maker,
        mock_tool_coordinator,
        mock_response_pipeline,
        mock_context_gatherer,
    ):
        """Test context gathering when no context provided"""
        processor = QueryProcessor(
            decision_maker=mock_decision_maker,
            tool_coordinator=mock_tool_coordinator,
            response_pipeline=mock_response_pipeline,
            context_gatherer=mock_context_gatherer,
        )

        await processor.process("query without context")

        mock_context_gatherer.gather.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_error_returns_fallback(self, processor):
        """Test error handling returns fallback response"""
        processor.decision_maker.decide = AsyncMock(side_effect=Exception("LLM error"))

        result = await processor.process("failing query")

        assert result.is_fallback is True
        assert processor._stats["errors"] == 1

    def test_get_stats(self, processor):
        """Test stats reporting"""
        stats = processor.get_stats()

        assert "total_queries" in stats
        assert "cache_hits" in stats
        assert "tool_calls" in stats
        assert "errors" in stats

    def test_get_system_state(self, processor):
        """Test system state collection"""
        state = processor._get_system_state({"metadata": {"data_date": "2026-02-16"}})

        assert state["data_status"] in ["최신", "오래됨 (2026-02-16)"]
        assert "available_tools" in state
        assert "cache_stats" in state

    def test_get_system_state_no_metrics(self, processor):
        """Test system state with no metrics"""
        state = processor._get_system_state(None)

        assert state["data_status"] == "없음"
        assert state["data_date"] is None

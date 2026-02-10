"""Tests for src.monitoring.logger module."""

import logging

import pytest

from src.monitoring.logger import AgentLogger, SensitiveDataFilter, log_execution


class TestSensitiveDataFilter:
    @pytest.fixture
    def filter_(self):
        return SensitiveDataFilter()

    def test_masks_openai_key(self, filter_):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="API key: sk-abc123def456ghi789jkl012mno345",
            args=(),
            exc_info=None,
        )
        filter_.filter(record)
        assert "sk-abc123" not in record.msg
        assert "sk-****" in record.msg

    def test_masks_apify_key(self, filter_):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Using apify_api_abc123def456ghi789jkl",
            args=(),
            exc_info=None,
        )
        filter_.filter(record)
        assert "apify_api_****" in record.msg

    def test_masks_tavily_key(self, filter_):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Key: tvly-abc123def456ghi789jklmno",
            args=(),
            exc_info=None,
        )
        filter_.filter(record)
        assert "tvly-****" in record.msg

    def test_masks_bearer_token(self, filter_):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            args=(),
            exc_info=None,
        )
        filter_.filter(record)
        assert "Bearer ****" in record.msg

    def test_masks_dict_args(self, filter_):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        # Manually set args to dict (Python 3.13 LogRecord constructor rejects dict args)
        record.args = {"key": "sk-abc123def456ghi789jkl012mno345"}
        filter_.filter(record)
        assert "sk-abc123" not in str(record.args)

    def test_masks_tuple_args(self, filter_):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test %s",
            args=("sk-abc123def456ghi789jkl012mno345",),
            exc_info=None,
        )
        filter_.filter(record)
        assert "sk-abc123" not in str(record.args)

    def test_always_returns_true(self, filter_):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="safe message",
            args=(),
            exc_info=None,
        )
        assert filter_.filter(record) is True

    def test_mask_nested_dict(self, filter_):
        value = filter_._mask_value({"nested": {"key": "sk-abc123def456ghi789jkl012mno345"}})
        assert "sk-abc123" not in str(value)

    def test_mask_list(self, filter_):
        value = filter_._mask_value(["sk-abc123def456ghi789jkl012mno345"])
        assert "sk-abc123" not in str(value)

    def test_mask_non_string(self, filter_):
        assert filter_._mask_value(42) == 42
        assert filter_._mask_value(None) is None


class TestAgentLogger:
    @pytest.fixture(autouse=True)
    def clear_instances(self):
        """Clear singleton instances between tests."""
        AgentLogger._instances.clear()
        yield
        AgentLogger._instances.clear()

    @pytest.fixture
    def logger(self, tmp_path):
        return AgentLogger(name="test_agent", log_dir=str(tmp_path / "logs"))

    def test_singleton_per_name(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        l1 = AgentLogger(name="agent_a", log_dir=log_dir)
        l2 = AgentLogger(name="agent_a", log_dir=log_dir)
        l3 = AgentLogger(name="agent_b", log_dir=log_dir)
        assert l1 is l2
        assert l1 is not l3

    def test_creates_log_directory(self, tmp_path):
        log_dir = tmp_path / "new_logs"
        AgentLogger(name="test", log_dir=str(log_dir))
        assert log_dir.exists()

    def test_log_methods_dont_raise(self, logger):
        logger.debug("debug msg")
        logger.info("info msg")
        logger.warning("warn msg")
        logger.error("error msg")
        logger.critical("critical msg")

    def test_log_with_extra(self, logger):
        logger.info("test", extra={"key": "value"})

    def test_format_extra_none(self, logger):
        assert logger._format_extra(None) == ""

    def test_format_extra_dict(self, logger):
        result = logger._format_extra({"key": "val"})
        assert "key" in result
        assert "val" in result

    def test_agent_lifecycle_logs(self, logger):
        logger.agent_start("crawler", task="crawl_lip_care")
        logger.agent_complete("crawler", duration=5.5, result="100 products")
        logger.agent_error("crawler", error="Timeout", duration=10.0)

    def test_tool_logs(self, logger):
        logger.tool_call("scraper", params={"url": "test"})
        logger.tool_result("scraper", success=True, result_summary="OK")

    def test_llm_logs(self, logger):
        logger.llm_request("gpt-4.1-mini", prompt_tokens=100)
        logger.llm_response("gpt-4.1-mini", completion_tokens=50, latency_ms=1200.5)

    def test_workflow_step_log(self, logger):
        logger.workflow_step("crawl", "start")
        logger.workflow_step("crawl", "complete")
        logger.workflow_step("crawl", "error")

    def test_metric_log(self, logger):
        logger.metric("response_time", 1.5, unit="seconds")

    def test_chat_request_and_response(self, logger):
        ctx = logger.chat_request("What is LANEIGE?", session_id="s1")
        assert "request_id" in ctx
        assert ctx["session_id"] == "s1"

        logger.chat_response(
            request_context=ctx,
            response="LANEIGE is a K-beauty brand",
            kg_facts_count=3,
            rag_chunks_count=5,
            inferences_count=2,
        )

    def test_chat_response_failure(self, logger):
        ctx = logger.chat_request("broken query")
        logger.chat_response(
            request_context=ctx,
            response="",
            success=False,
            error="LLM timeout",
        )

    def test_audit_log_written(self, logger, tmp_path):
        ctx = logger.chat_request("test query")
        logger.chat_response(request_context=ctx, response="test response")

        audit_files = list((tmp_path / "logs").glob("chatbot_audit_*.jsonl"))
        assert len(audit_files) == 1

        import json

        with open(audit_files[0]) as f:
            line = f.readline()
            record = json.loads(line)
            assert record["success"] is True


class TestLogExecution:
    @pytest.fixture(autouse=True)
    def clear_instances(self):
        AgentLogger._instances.clear()
        yield
        AgentLogger._instances.clear()

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        @log_execution()
        async def my_async_func():
            return 42

        result = await my_async_func()
        assert result == 42

    @pytest.mark.asyncio
    async def test_async_decorator_exception(self):
        @log_execution()
        async def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await failing_func()

    def test_sync_decorator(self):
        @log_execution()
        def my_sync_func():
            return 99

        result = my_sync_func()
        assert result == 99

    def test_sync_decorator_exception(self):
        @log_execution()
        def failing_func():
            raise RuntimeError("oops")

        with pytest.raises(RuntimeError, match="oops"):
            failing_func()

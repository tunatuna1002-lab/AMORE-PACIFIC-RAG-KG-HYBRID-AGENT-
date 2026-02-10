"""Tests for src.shared.llm_client module."""

from unittest.mock import MagicMock, patch

import pytest

from src.monitoring.logger import AgentLogger
from src.shared.llm_client import LLMClient, LLMError, get_default_client


def _make_mock_response(content="test response", prompt_tokens=10, completion_tokens=5):
    """Create a mock LLM response."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    mock.usage = MagicMock()
    mock.usage.prompt_tokens = prompt_tokens
    mock.usage.completion_tokens = completion_tokens
    mock.usage.total_tokens = prompt_tokens + completion_tokens
    return mock


class TestLLMError:
    def test_is_exception(self):
        assert issubclass(LLMError, Exception)

    def test_message(self):
        err = LLMError("test error")
        assert str(err) == "test error"


class TestLLMClient:
    @pytest.fixture(autouse=True)
    def clear_logger_instances(self):
        AgentLogger._instances.clear()
        yield
        AgentLogger._instances.clear()

    @pytest.fixture
    def client(self, tmp_path):
        logger = AgentLogger("test_llm", log_dir=str(tmp_path / "logs"))
        return LLMClient(logger=logger)

    def test_defaults(self, client):
        assert client.model == "gpt-4.1-mini"
        assert client.default_temperature == 0.7
        assert client.max_retries == 3
        assert client.retry_delay == 1.0

    def test_custom_init(self, tmp_path):
        logger = AgentLogger("custom", log_dir=str(tmp_path / "logs"))
        c = LLMClient(
            model="gpt-4",
            default_temperature=0.5,
            max_retries=5,
            retry_delay=2.0,
            logger=logger,
        )
        assert c.model == "gpt-4"
        assert c.default_temperature == 0.5
        assert c.max_retries == 5

    @pytest.mark.asyncio
    @patch("src.shared.llm_client.acompletion")
    async def test_complete(self, mock_acompletion, client):
        mock_acompletion.return_value = _make_mock_response("Hello world")

        result = await client.complete(system_prompt="Be helpful", user_prompt="Say hello")
        assert result == "Hello world"
        assert client._total_calls == 1

    @pytest.mark.asyncio
    @patch("src.shared.llm_client.acompletion")
    async def test_complete_with_custom_model(self, mock_acompletion, client):
        mock_acompletion.return_value = _make_mock_response("Custom")

        result = await client.complete(
            system_prompt="test", user_prompt="test", model="gpt-4", temperature=0.1
        )
        assert result == "Custom"
        call_kwargs = mock_acompletion.call_args[1]
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["temperature"] == 0.1

    @pytest.mark.asyncio
    @patch("src.shared.llm_client.acompletion")
    async def test_complete_with_messages(self, mock_acompletion, client):
        mock_acompletion.return_value = _make_mock_response("From messages")
        messages = [{"role": "user", "content": "test"}]

        result = await client.complete(
            system_prompt="ignored", user_prompt="ignored", messages=messages
        )
        assert result == "From messages"
        call_kwargs = mock_acompletion.call_args[1]
        assert call_kwargs["messages"] == messages

    @pytest.mark.asyncio
    @patch("src.shared.llm_client.acompletion")
    async def test_complete_empty_choices(self, mock_acompletion, client):
        mock = MagicMock()
        mock.choices = []
        mock_acompletion.return_value = mock

        with pytest.raises(LLMError, match="empty choices"):
            await client.complete("sys", "user")

    @pytest.mark.asyncio
    @patch("src.shared.llm_client.acompletion")
    async def test_complete_retry_on_failure(self, mock_acompletion, client):
        client.max_retries = 2
        client.retry_delay = 0.01

        mock_acompletion.side_effect = [
            Exception("timeout"),
            _make_mock_response("recovered"),
        ]

        result = await client.complete("sys", "user")
        assert result == "recovered"
        assert mock_acompletion.call_count == 2
        assert client._total_errors == 1

    @pytest.mark.asyncio
    @patch("src.shared.llm_client.acompletion")
    async def test_complete_all_retries_fail(self, mock_acompletion, client):
        client.max_retries = 2
        client.retry_delay = 0.01
        mock_acompletion.side_effect = Exception("persistent failure")

        with pytest.raises(LLMError, match="Failed to complete"):
            await client.complete("sys", "user")
        assert client._total_errors == 2

    @pytest.mark.asyncio
    @patch("src.shared.llm_client.acompletion")
    async def test_complete_no_usage(self, mock_acompletion, client):
        mock = MagicMock()
        mock.choices = [MagicMock()]
        mock.choices[0].message.content = "no usage"
        del mock.usage  # Remove usage attribute
        mock_acompletion.return_value = mock

        result = await client.complete("sys", "user")
        assert result == "no usage"

    @pytest.mark.asyncio
    @patch("src.shared.llm_client.acompletion")
    async def test_complete_json(self, mock_acompletion, client):
        mock_acompletion.return_value = _make_mock_response('{"key": "value"}')

        result = await client.complete_json("sys", "user")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    @patch("src.shared.llm_client.acompletion")
    async def test_complete_json_with_markdown(self, mock_acompletion, client):
        mock_acompletion.return_value = _make_mock_response('```json\n{"key": "value"}\n```')

        result = await client.complete_json("sys", "user")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    @patch("src.shared.llm_client.acompletion")
    async def test_complete_json_invalid(self, mock_acompletion, client):
        mock_acompletion.return_value = _make_mock_response("not json at all")

        with pytest.raises(LLMError, match="not valid JSON"):
            await client.complete_json("sys", "user")

    @pytest.mark.asyncio
    @patch("src.shared.llm_client.acompletion")
    async def test_complete_with_usage(self, mock_acompletion, client):
        mock_acompletion.return_value = _make_mock_response("usage test", 100, 50)

        result = await client.complete_with_usage("sys", "user")
        assert result["content"] == "usage test"
        assert result["model"] == "gpt-4.1-mini"
        assert result["latency_ms"] > 0
        assert result["usage"]["prompt_tokens"] == 100
        assert result["usage"]["completion_tokens"] == 50

    @pytest.mark.asyncio
    @patch("src.shared.llm_client.acompletion")
    async def test_complete_with_usage_no_usage(self, mock_acompletion, client):
        mock = MagicMock()
        mock.choices = [MagicMock()]
        mock.choices[0].message.content = "no usage"
        del mock.usage
        mock_acompletion.return_value = mock

        result = await client.complete_with_usage("sys", "user")
        assert result["usage"] is None

    def test_estimate_cost_default_model(self, client):
        cost = client.estimate_cost(1_000_000, 1_000_000)
        # gpt-4.1-mini: $0.40/1M input + $1.60/1M output = $2.00
        assert cost == 2.0

    def test_estimate_cost_gpt4(self, client):
        cost = client.estimate_cost(1_000_000, 1_000_000, model="gpt-4")
        # gpt-4: $30/1M input + $60/1M output = $90
        assert cost == 90.0

    def test_estimate_cost_unknown_model(self, client):
        cost = client.estimate_cost(1_000_000, 1_000_000, model="unknown-model")
        # Falls back to gpt-4.1-mini pricing
        assert cost == 2.0

    def test_get_statistics(self, client):
        stats = client.get_statistics()
        assert stats["total_calls"] == 0
        assert stats["total_prompt_tokens"] == 0
        assert stats["total_completion_tokens"] == 0
        assert stats["total_errors"] == 0
        assert stats["estimated_cost"] == 0.0

    def test_reset_statistics(self, client):
        client._total_calls = 10
        client._total_errors = 2
        client.reset_statistics()
        assert client._total_calls == 0
        assert client._total_errors == 0

    @pytest.mark.asyncio
    @patch("src.shared.llm_client.acompletion")
    async def test_statistics_tracking(self, mock_acompletion, client):
        mock_acompletion.return_value = _make_mock_response("test", 100, 50)

        await client.complete("sys", "user")
        await client.complete("sys", "user")

        stats = client.get_statistics()
        assert stats["total_calls"] == 2
        assert stats["total_prompt_tokens"] == 200
        assert stats["total_completion_tokens"] == 100


class TestGetDefaultClient:
    @pytest.fixture(autouse=True)
    def clear_global(self):
        import src.shared.llm_client as mod

        if hasattr(mod, "_default_client"):
            delattr(mod, "_default_client")
        AgentLogger._instances.clear()
        yield
        if hasattr(mod, "_default_client"):
            delattr(mod, "_default_client")
        AgentLogger._instances.clear()

    def test_returns_singleton(self):
        c1 = get_default_client()
        c2 = get_default_client()
        assert c1 is c2

    def test_returns_llm_client(self):
        c = get_default_client()
        assert isinstance(c, LLMClient)

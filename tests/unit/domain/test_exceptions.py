"""
TDD Phase 1: 커스텀 예외 타입 테스트 (RED → GREEN)

테스트 대상: src/domain/exceptions.py
"""
import pytest
from typing import Any


class TestAmoreAgentError:
    """Base exception 테스트"""

    def test_base_exception_is_exception_subclass(self):
        """AmoreAgentError는 Exception의 서브클래스여야 함"""
        from src.domain.exceptions import AmoreAgentError

        assert issubclass(AmoreAgentError, Exception)

    def test_base_exception_stores_message(self):
        """AmoreAgentError는 메시지를 저장해야 함"""
        from src.domain.exceptions import AmoreAgentError

        error = AmoreAgentError("Test error message")
        assert str(error) == "Test error message"


class TestNetworkError:
    """네트워크 에러 테스트"""

    def test_network_error_is_amore_error_subclass(self):
        """NetworkError는 AmoreAgentError의 서브클래스여야 함"""
        from src.domain.exceptions import NetworkError, AmoreAgentError

        assert issubclass(NetworkError, AmoreAgentError)

    def test_network_error_attributes(self):
        """NetworkError는 url, status_code, retry_count 속성 가져야 함"""
        from src.domain.exceptions import NetworkError

        error = NetworkError(
            message="Connection timeout",
            url="https://amazon.com/product",
            status_code=503,
            retry_count=3
        )

        assert error.url == "https://amazon.com/product"
        assert error.status_code == 503
        assert error.retry_count == 3
        assert "Connection timeout" in str(error)

    def test_network_error_default_values(self):
        """NetworkError는 기본값을 가져야 함"""
        from src.domain.exceptions import NetworkError

        error = NetworkError("Simple network error")

        assert error.url is None
        assert error.status_code is None
        assert error.retry_count == 0

    def test_network_error_can_be_raised_and_caught(self):
        """NetworkError는 raise/catch 가능해야 함"""
        from src.domain.exceptions import NetworkError

        with pytest.raises(NetworkError) as exc_info:
            raise NetworkError("Test", url="http://test.com", status_code=404)

        assert exc_info.value.status_code == 404


class TestLLMAPIError:
    """LLM API 에러 테스트"""

    def test_llm_api_error_is_amore_error_subclass(self):
        """LLMAPIError는 AmoreAgentError의 서브클래스여야 함"""
        from src.domain.exceptions import LLMAPIError, AmoreAgentError

        assert issubclass(LLMAPIError, AmoreAgentError)

    def test_llm_api_error_attributes(self):
        """LLMAPIError는 model, error_code, is_retryable 속성 가져야 함"""
        from src.domain.exceptions import LLMAPIError

        error = LLMAPIError(
            message="Rate limit exceeded",
            model="gpt-4.1-mini",
            error_code="rate_limit_exceeded",
            is_retryable=True
        )

        assert error.model == "gpt-4.1-mini"
        assert error.error_code == "rate_limit_exceeded"
        assert error.is_retryable is True
        assert "Rate limit exceeded" in str(error)

    def test_llm_api_error_default_values(self):
        """LLMAPIError는 기본값을 가져야 함"""
        from src.domain.exceptions import LLMAPIError

        error = LLMAPIError("API error")

        assert error.model is None
        assert error.error_code is None
        assert error.is_retryable is False

    def test_llm_api_error_retryable_flag(self):
        """is_retryable 플래그가 올바르게 동작해야 함"""
        from src.domain.exceptions import LLMAPIError

        retryable_error = LLMAPIError("Rate limit", is_retryable=True)
        non_retryable_error = LLMAPIError("Invalid key", is_retryable=False)

        assert retryable_error.is_retryable is True
        assert non_retryable_error.is_retryable is False


class TestDataValidationError:
    """데이터 검증 에러 테스트"""

    def test_data_validation_error_is_amore_error_subclass(self):
        """DataValidationError는 AmoreAgentError의 서브클래스여야 함"""
        from src.domain.exceptions import DataValidationError, AmoreAgentError

        assert issubclass(DataValidationError, AmoreAgentError)

    def test_data_validation_error_attributes(self):
        """DataValidationError는 field, value, constraint 속성 가져야 함"""
        from src.domain.exceptions import DataValidationError

        error = DataValidationError(
            message="Invalid price value",
            field="price",
            value=-10.5,
            constraint="price >= 0"
        )

        assert error.field == "price"
        assert error.value == -10.5
        assert error.constraint == "price >= 0"
        assert "Invalid price value" in str(error)

    def test_data_validation_error_default_values(self):
        """DataValidationError는 기본값을 가져야 함"""
        from src.domain.exceptions import DataValidationError

        error = DataValidationError("Validation failed")

        assert error.field is None
        assert error.value is None
        assert error.constraint is None

    def test_data_validation_error_with_complex_value(self):
        """복잡한 값(dict, list)도 저장 가능해야 함"""
        from src.domain.exceptions import DataValidationError

        complex_value = {"nested": {"data": [1, 2, 3]}}
        error = DataValidationError(
            "Invalid structure",
            field="config",
            value=complex_value
        )

        assert error.value == complex_value


class TestScraperError:
    """스크레이퍼 에러 테스트"""

    def test_scraper_error_is_amore_error_subclass(self):
        """ScraperError는 AmoreAgentError의 서브클래스여야 함"""
        from src.domain.exceptions import ScraperError, AmoreAgentError

        assert issubclass(ScraperError, AmoreAgentError)

    def test_scraper_error_attributes(self):
        """ScraperError는 category, asin, error_type 속성 가져야 함"""
        from src.domain.exceptions import ScraperError

        error = ScraperError(
            message="Amazon blocked request",
            category="lip_care",
            asin="B0BSHRYY1S",
            error_type="BLOCKED"
        )

        assert error.category == "lip_care"
        assert error.asin == "B0BSHRYY1S"
        assert error.error_type == "BLOCKED"
        assert "Amazon blocked request" in str(error)

    def test_scraper_error_default_values(self):
        """ScraperError는 기본값을 가져야 함"""
        from src.domain.exceptions import ScraperError

        error = ScraperError("Scraping failed")

        assert error.category is None
        assert error.asin is None
        assert error.error_type is None

    def test_scraper_error_types(self):
        """error_type은 BLOCKED, TIMEOUT, PARSE_ERROR 등이 될 수 있음"""
        from src.domain.exceptions import ScraperError

        blocked_error = ScraperError("Blocked", error_type="BLOCKED")
        timeout_error = ScraperError("Timeout", error_type="TIMEOUT")
        parse_error = ScraperError("Parse failed", error_type="PARSE_ERROR")

        assert blocked_error.error_type == "BLOCKED"
        assert timeout_error.error_type == "TIMEOUT"
        assert parse_error.error_type == "PARSE_ERROR"


class TestKnowledgeGraphError:
    """Knowledge Graph 에러 테스트"""

    def test_knowledge_graph_error_is_amore_error_subclass(self):
        """KnowledgeGraphError는 AmoreAgentError의 서브클래스여야 함"""
        from src.domain.exceptions import KnowledgeGraphError, AmoreAgentError

        assert issubclass(KnowledgeGraphError, AmoreAgentError)

    def test_knowledge_graph_error_attributes(self):
        """KnowledgeGraphError는 entity, relation, operation 속성 가져야 함"""
        from src.domain.exceptions import KnowledgeGraphError

        error = KnowledgeGraphError(
            message="Entity not found",
            entity="LANEIGE",
            relation="competitor_of",
            operation="query"
        )

        assert error.entity == "LANEIGE"
        assert error.relation == "competitor_of"
        assert error.operation == "query"


class TestReasonerError:
    """Reasoner 에러 테스트"""

    def test_reasoner_error_is_amore_error_subclass(self):
        """ReasonerError는 AmoreAgentError의 서브클래스여야 함"""
        from src.domain.exceptions import ReasonerError, AmoreAgentError

        assert issubclass(ReasonerError, AmoreAgentError)

    def test_reasoner_error_attributes(self):
        """ReasonerError는 rule_name, context 속성 가져야 함"""
        from src.domain.exceptions import ReasonerError

        error = ReasonerError(
            message="Rule execution failed",
            rule_name="sos_threshold_rule",
            context={"sos": 0.35, "threshold": 0.3}
        )

        assert error.rule_name == "sos_threshold_rule"
        assert error.context == {"sos": 0.35, "threshold": 0.3}


class TestRetrieverError:
    """Retriever 에러 테스트"""

    def test_retriever_error_is_amore_error_subclass(self):
        """RetrieverError는 AmoreAgentError의 서브클래스여야 함"""
        from src.domain.exceptions import RetrieverError, AmoreAgentError

        assert issubclass(RetrieverError, AmoreAgentError)

    def test_retriever_error_attributes(self):
        """RetrieverError는 query, retriever_type 속성 가져야 함"""
        from src.domain.exceptions import RetrieverError

        error = RetrieverError(
            message="Vector search failed",
            query="LANEIGE 경쟁력 분석",
            retriever_type="hybrid"
        )

        assert error.query == "LANEIGE 경쟁력 분석"
        assert error.retriever_type == "hybrid"
        assert "Vector search failed" in str(error)

    def test_retriever_error_default_values(self):
        """RetrieverError는 기본값을 가져야 함"""
        from src.domain.exceptions import RetrieverError

        error = RetrieverError("Retrieval failed")

        assert error.query is None
        assert error.retriever_type is None


class TestConfigurationError:
    """Configuration 에러 테스트"""

    def test_configuration_error_is_amore_error_subclass(self):
        """ConfigurationError는 AmoreAgentError의 서브클래스여야 함"""
        from src.domain.exceptions import ConfigurationError, AmoreAgentError

        assert issubclass(ConfigurationError, AmoreAgentError)

    def test_configuration_error_attributes(self):
        """ConfigurationError는 config_key, expected, actual 속성 가져야 함"""
        from src.domain.exceptions import ConfigurationError

        error = ConfigurationError(
            message="Missing required API key",
            config_key="OPENAI_API_KEY",
            expected="string",
            actual=None
        )

        assert error.config_key == "OPENAI_API_KEY"
        assert error.expected == "string"
        assert error.actual is None
        assert "Missing required API key" in str(error)

    def test_configuration_error_default_values(self):
        """ConfigurationError는 기본값을 가져야 함"""
        from src.domain.exceptions import ConfigurationError

        error = ConfigurationError("Config error")

        assert error.config_key is None
        assert error.expected is None
        assert error.actual is None


class TestExceptionHierarchy:
    """예외 계층 구조 테스트"""

    def test_all_exceptions_inherit_from_base(self):
        """모든 커스텀 예외는 AmoreAgentError를 상속해야 함"""
        from src.domain.exceptions import (
            AmoreAgentError,
            NetworkError,
            LLMAPIError,
            DataValidationError,
            ScraperError,
            KnowledgeGraphError,
            ReasonerError,
            RetrieverError,
            ConfigurationError
        )

        exceptions = [
            NetworkError,
            LLMAPIError,
            DataValidationError,
            ScraperError,
            KnowledgeGraphError,
            ReasonerError,
            RetrieverError,
            ConfigurationError
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, AmoreAgentError), \
                f"{exc_class.__name__} should inherit from AmoreAgentError"

    def test_catch_all_with_base_exception(self):
        """AmoreAgentError로 모든 서브 예외를 잡을 수 있어야 함"""
        from src.domain.exceptions import (
            AmoreAgentError,
            NetworkError,
            LLMAPIError,
            ScraperError
        )

        errors_to_test = [
            NetworkError("Network failed"),
            LLMAPIError("LLM failed"),
            ScraperError("Scraping failed")
        ]

        for error in errors_to_test:
            with pytest.raises(AmoreAgentError):
                raise error

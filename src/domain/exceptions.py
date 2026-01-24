"""
AMORE Agent 커스텀 예외 타입

이 모듈은 프로젝트 전체에서 사용되는 구체적인 예외 타입을 정의합니다.
광범위한 `except Exception`을 대체하여 더 명확한 에러 처리를 가능하게 합니다.

사용 예:
    from src.domain.exceptions import NetworkError, LLMAPIError

    try:
        response = await fetch_data(url)
    except NetworkError as e:
        logger.error(f"Network failed: {e.url}, status: {e.status_code}")
        if e.retry_count < 3:
            retry()
"""
from typing import Any, Optional, Dict


class AmoreAgentError(Exception):
    """
    Base exception for all AMORE agent errors.

    모든 커스텀 예외의 기본 클래스입니다.
    이 클래스를 사용하면 모든 AMORE 관련 예외를 한 번에 잡을 수 있습니다.

    Example:
        try:
            await agent.execute()
        except AmoreAgentError as e:
            logger.error(f"Agent error: {e}")
    """
    pass


class NetworkError(AmoreAgentError):
    """
    Network-related errors (timeout, connection, HTTP errors).

    네트워크 관련 에러를 처리합니다:
    - Connection timeout
    - DNS resolution failure
    - HTTP 4xx/5xx errors
    - SSL/TLS errors

    Attributes:
        url: 요청한 URL
        status_code: HTTP 상태 코드 (해당시)
        retry_count: 재시도 횟수

    Example:
        raise NetworkError(
            "Connection timeout after 30s",
            url="https://amazon.com/product/123",
            status_code=None,
            retry_count=3
        )
    """

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        retry_count: int = 0
    ):
        super().__init__(message)
        self.url = url
        self.status_code = status_code
        self.retry_count = retry_count


class LLMAPIError(AmoreAgentError):
    """
    LLM API errors (rate limit, invalid response, authentication).

    LLM API 호출 관련 에러를 처리합니다:
    - Rate limit exceeded
    - Invalid API key
    - Model not found
    - Context length exceeded
    - Invalid response format

    Attributes:
        model: 사용한 모델명 (예: "gpt-4.1-mini")
        error_code: API 에러 코드 (예: "rate_limit_exceeded")
        is_retryable: 재시도 가능 여부

    Example:
        raise LLMAPIError(
            "Rate limit exceeded",
            model="gpt-4.1-mini",
            error_code="rate_limit_exceeded",
            is_retryable=True
        )
    """

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        error_code: Optional[str] = None,
        is_retryable: bool = False
    ):
        super().__init__(message)
        self.model = model
        self.error_code = error_code
        self.is_retryable = is_retryable


class DataValidationError(AmoreAgentError):
    """
    Data validation errors.

    데이터 검증 실패 시 발생하는 에러입니다:
    - Pydantic 검증 실패
    - 필수 필드 누락
    - 타입 불일치
    - 제약 조건 위반

    Attributes:
        field: 검증 실패한 필드명
        value: 실패한 값
        constraint: 위반한 제약 조건

    Example:
        raise DataValidationError(
            "Price must be non-negative",
            field="price",
            value=-10.5,
            constraint="price >= 0"
        )
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        constraint: Optional[str] = None
    ):
        super().__init__(message)
        self.field = field
        self.value = value
        self.constraint = constraint


class ScraperError(AmoreAgentError):
    """
    Amazon scraping errors.

    Amazon 크롤링 중 발생하는 에러입니다:
    - BLOCKED: Amazon의 봇 차단
    - TIMEOUT: 페이지 로드 타임아웃
    - PARSE_ERROR: HTML 파싱 실패
    - CAPTCHA: CAPTCHA 발생

    Attributes:
        category: 크롤링 중인 카테고리 (예: "lip_care")
        asin: 관련 제품의 ASIN
        error_type: 에러 유형 (BLOCKED, TIMEOUT, PARSE_ERROR, CAPTCHA)

    Example:
        raise ScraperError(
            "Amazon blocked request",
            category="lip_care",
            asin="B0BSHRYY1S",
            error_type="BLOCKED"
        )
    """

    def __init__(
        self,
        message: str,
        category: Optional[str] = None,
        asin: Optional[str] = None,
        error_type: Optional[str] = None
    ):
        super().__init__(message)
        self.category = category
        self.asin = asin
        self.error_type = error_type


class KnowledgeGraphError(AmoreAgentError):
    """
    Knowledge Graph related errors.

    지식 그래프 작업 중 발생하는 에러입니다:
    - Entity not found
    - Invalid relation type
    - Duplicate entity
    - Query execution failed

    Attributes:
        entity: 관련 엔티티명
        relation: 관련 관계 타입
        operation: 수행 중인 작업 (query, insert, update, delete)

    Example:
        raise KnowledgeGraphError(
            "Entity not found",
            entity="LANEIGE",
            relation="competitor_of",
            operation="query"
        )
    """

    def __init__(
        self,
        message: str,
        entity: Optional[str] = None,
        relation: Optional[str] = None,
        operation: Optional[str] = None
    ):
        super().__init__(message)
        self.entity = entity
        self.relation = relation
        self.operation = operation


class ReasonerError(AmoreAgentError):
    """
    Ontology Reasoner related errors.

    온톨로지 추론 중 발생하는 에러입니다:
    - Rule execution failed
    - Invalid rule definition
    - Inference cycle detected
    - Missing prerequisites

    Attributes:
        rule_name: 실패한 규칙명
        context: 규칙 실행 컨텍스트 (변수, 조건 등)

    Example:
        raise ReasonerError(
            "Rule execution failed",
            rule_name="sos_threshold_rule",
            context={"sos": 0.35, "threshold": 0.3}
        )
    """

    def __init__(
        self,
        message: str,
        rule_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.rule_name = rule_name
        self.context = context or {}


class RetrieverError(AmoreAgentError):
    """
    RAG Retriever related errors.

    RAG 검색 중 발생하는 에러입니다:
    - Embedding generation failed
    - Vector store query failed
    - Document chunking failed

    Attributes:
        query: 검색 쿼리
        retriever_type: 검색기 유형 (document, hybrid, kg)

    Example:
        raise RetrieverError(
            "Vector search failed",
            query="LANEIGE 경쟁력 분석",
            retriever_type="hybrid"
        )
    """

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        retriever_type: Optional[str] = None
    ):
        super().__init__(message)
        self.query = query
        self.retriever_type = retriever_type


class ConfigurationError(AmoreAgentError):
    """
    Configuration related errors.

    설정 관련 에러입니다:
    - Missing required configuration
    - Invalid configuration value
    - Configuration file not found

    Attributes:
        config_key: 문제가 된 설정 키
        expected: 기대하는 값/형식
        actual: 실제 값

    Example:
        raise ConfigurationError(
            "Missing required API key",
            config_key="OPENAI_API_KEY",
            expected="string",
            actual=None
        )
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None
    ):
        super().__init__(message)
        self.config_key = config_key
        self.expected = expected
        self.actual = actual


# 편의를 위한 __all__ 정의
__all__ = [
    "AmoreAgentError",
    "NetworkError",
    "LLMAPIError",
    "DataValidationError",
    "ScraperError",
    "KnowledgeGraphError",
    "ReasonerError",
    "RetrieverError",
    "ConfigurationError",
]

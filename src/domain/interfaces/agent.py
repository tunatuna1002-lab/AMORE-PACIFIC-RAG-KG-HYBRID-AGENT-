"""
Agent Protocols
===============
에이전트에 대한 추상 인터페이스
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CrawlerAgentProtocol(Protocol):
    """크롤러 에이전트 Protocol"""

    async def crawl(self, categories: list[str]) -> list[Any]: ...
    async def crawl_single(self, category_id: str, url: str) -> list[Any]: ...
    async def initialize(self) -> None: ...
    async def close(self) -> None: ...


@runtime_checkable
class StorageAgentProtocol(Protocol):
    """저장소 에이전트 Protocol"""

    async def save(self, records: list[Any]) -> bool: ...
    async def save_metrics(self, metrics: dict[str, Any]) -> bool: ...
    async def initialize(self) -> None: ...


@runtime_checkable
class MetricsAgentProtocol(Protocol):
    """메트릭 계산 에이전트 Protocol"""

    async def calculate(self, records: list[Any]) -> dict[str, Any]: ...
    async def calculate_brand_metrics(
        self, records: list[Any], brand: str, category_id: str
    ) -> Any: ...
    async def calculate_market_metrics(self, records: list[Any], category_id: str) -> Any: ...


@runtime_checkable
class InsightAgentProtocol(Protocol):
    """인사이트 생성 에이전트 Protocol"""

    async def generate(
        self, metrics: dict[str, Any], records: list[Any] | None = None
    ) -> list[dict[str, Any]]: ...
    async def generate_for_brand(
        self, brand: str, metrics: dict[str, Any], competitors: list[str] | None = None
    ) -> list[dict[str, Any]]: ...
    async def generate_alerts(
        self, metrics: dict[str, Any], thresholds: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]: ...

"""
Agent Protocols
===============
배치 파이프라인(``src.application.workflows.batch_workflow``)이 주입받는 에이전트의
추상 인터페이스.

D16 정렬: 이전 버전은 실제 구현에 존재하지 않는 메서드(``crawl`` / ``save`` /
``calculate`` / ``generate``)를 선언하고 있어서 ``isinstance(agent, Protocol)`` 이 항상
False 였고, 프로토콜을 신뢰해 호출하면 TypeError 가 났다. 지금은 구현체
(``src.agents.*``)의 실제 시그니처(= ``execute`` 기반)를 그대로 옮겨 적는다.

``InsightAgentProtocol`` 의 유일한 정의는 ``src.domain.interfaces.insight`` 에 있고
여기서는 하위 호환을 위해 재수출만 한다.
"""

from typing import Any, Protocol, runtime_checkable

from src.domain.interfaces.insight import InsightAgentProtocol

__all__ = [
    "CrawlerAgentProtocol",
    "InsightAgentProtocol",
    "MetricsAgentProtocol",
    "StorageAgentProtocol",
]


@runtime_checkable
class CrawlerAgentProtocol(Protocol):
    """크롤러 에이전트 Protocol (구현: ``src.agents.crawler_agent.CrawlerAgent``)"""

    async def execute(self, categories: list[str] | None = None) -> dict[str, Any]:
        """카테고리별 크롤링 실행 → ``{"categories": {...}, "status": ..., ...}``"""
        ...

    async def close(self) -> None:
        """브라우저/세션 정리"""
        ...

    def get_results(self) -> dict[str, Any]:
        """마지막 ``execute`` 결과"""
        ...


@runtime_checkable
class StorageAgentProtocol(Protocol):
    """저장소 에이전트 Protocol (구현: ``src.agents.storage_agent.StorageAgent``)"""

    async def execute(self, crawl_data: dict[str, Any]) -> dict[str, Any]:
        """크롤 결과 저장 → ``{"raw_records": int, "products_upserted": int, ...}``"""
        ...

    async def save_metrics(
        self,
        brand_metrics: list[Any] | None = None,
        product_metrics: list[Any] | None = None,
        market_metrics: list[Any] | None = None,
    ) -> dict[str, Any]:
        """계산된 지표 저장"""
        ...

    def get_results(self) -> dict[str, Any]:
        """마지막 ``execute`` 결과"""
        ...


@runtime_checkable
class MetricsAgentProtocol(Protocol):
    """지표 계산 에이전트 Protocol (구현: ``src.agents.metrics_agent.MetricsAgent``)"""

    async def execute(
        self,
        crawl_data: dict[str, Any],
        historical_data: dict[str, list] | None = None,
    ) -> dict[str, Any]:
        """SoS/HHI/CPI 계산 → ``{"brand_metrics": [...], "alerts": [...], ...}``"""
        ...

    def get_results(self) -> dict[str, Any]:
        """마지막 ``execute`` 결과"""
        ...

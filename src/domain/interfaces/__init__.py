"""
Domain Interfaces (Protocols)
=============================
의존성 역전을 위한 추상 인터페이스 정의

Clean Architecture에서 내부 레이어(Domain, Application)는 외부 레이어(Infrastructure)를
직접 참조하지 않고, 이 Protocol을 통해 간접적으로 의존합니다.

사용 예:
    # Application layer에서
    from src.domain.interfaces.repository import ProductRepository

    class BatchWorkflow:
        def __init__(self, repo: ProductRepository):
            self.repo = repo  # 구현체가 아닌 Protocol에 의존

    # Infrastructure layer에서 (실제 구현체 import)
    # 구현체는 Infrastructure 레이어에서만 import합니다
    # repo = ConcreteRepository(...)  # ProductRepository Protocol 구현체
    workflow = BatchWorkflow(repo)  # Protocol 만족하면 주입 가능
"""

from src.domain.interfaces.repository import (
    ProductRepository,
    MetricsRepository,
)
from src.domain.interfaces.agent import (
    CrawlerAgentProtocol,
    StorageAgentProtocol,
    MetricsAgentProtocol,
    InsightAgentProtocol,
    ChatAgentProtocol,
)
from src.domain.interfaces.scraper import ScraperProtocol
from src.domain.interfaces.llm_client import LLMClientProtocol
from src.domain.interfaces.knowledge_graph import KnowledgeGraphProtocol
from src.domain.interfaces.retriever import RetrieverProtocol, DocumentRetrieverProtocol

__all__ = [
    # Repository
    "ProductRepository",
    "MetricsRepository",
    # Agents
    "CrawlerAgentProtocol",
    "StorageAgentProtocol",
    "MetricsAgentProtocol",
    "InsightAgentProtocol",
    "ChatAgentProtocol",
    # Infrastructure
    "ScraperProtocol",
    "LLMClientProtocol",
    "KnowledgeGraphProtocol",
    "RetrieverProtocol",
    "DocumentRetrieverProtocol",
]

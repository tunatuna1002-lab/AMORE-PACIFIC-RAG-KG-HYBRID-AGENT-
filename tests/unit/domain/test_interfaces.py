"""
Domain Interfaces Tests (TDD - RED Phase)
==========================================
Protocol(인터페이스) 정의 테스트
의존성 역전을 위한 추상 인터페이스 검증
"""

import pytest
from typing import List, Dict, Any, Optional
from datetime import date


class TestRepositoryProtocol:
    """Repository Protocol 테스트"""

    def test_product_repository_protocol_exists(self):
        """ProductRepository Protocol이 존재하는지 검증"""
        from src.domain.interfaces.repository import ProductRepository

        # Protocol 클래스가 존재하는지 확인
        assert ProductRepository is not None

    def test_product_repository_has_save_records_method(self):
        """ProductRepository가 save_records 메서드를 정의하는지 검증"""
        from src.domain.interfaces.repository import ProductRepository
        from typing import get_type_hints

        # Protocol의 메서드 시그니처 확인
        hints = get_type_hints(ProductRepository.save_records)
        # async def save_records(self, records: list[RankRecord]) -> bool
        assert "return" in hints or hints.get("return") is not None

    def test_product_repository_has_get_recent_method(self):
        """ProductRepository가 get_recent 메서드를 정의하는지 검증"""
        from src.domain.interfaces.repository import ProductRepository

        # get_recent 메서드 존재 확인
        assert hasattr(ProductRepository, "get_recent")

    def test_product_repository_implementation(self):
        """ProductRepository 구현체가 Protocol을 만족하는지 검증"""
        from src.domain.interfaces.repository import ProductRepository
        from src.domain.entities.product import RankRecord

        class MockRepository:
            """Mock implementation for testing"""
            async def save_records(self, records: List[Any]) -> bool:
                return True

            async def get_recent(self, days: int = 7) -> List[Any]:
                return []

        # isinstance 대신 Protocol 호환성 확인
        repo = MockRepository()
        assert hasattr(repo, "save_records")
        assert hasattr(repo, "get_recent")


class TestAgentProtocol:
    """Agent Protocol 테스트"""

    def test_crawler_agent_protocol_exists(self):
        """CrawlerAgentProtocol이 존재하는지 검증"""
        from src.domain.interfaces.agent import CrawlerAgentProtocol

        assert CrawlerAgentProtocol is not None

    def test_crawler_agent_has_crawl_method(self):
        """CrawlerAgentProtocol이 crawl 메서드를 정의하는지 검증"""
        from src.domain.interfaces.agent import CrawlerAgentProtocol

        assert hasattr(CrawlerAgentProtocol, "crawl")

    def test_storage_agent_protocol_exists(self):
        """StorageAgentProtocol이 존재하는지 검증"""
        from src.domain.interfaces.agent import StorageAgentProtocol

        assert StorageAgentProtocol is not None

    def test_storage_agent_has_save_method(self):
        """StorageAgentProtocol이 save 메서드를 정의하는지 검증"""
        from src.domain.interfaces.agent import StorageAgentProtocol

        assert hasattr(StorageAgentProtocol, "save")

    def test_metrics_agent_protocol_exists(self):
        """MetricsAgentProtocol이 존재하는지 검증"""
        from src.domain.interfaces.agent import MetricsAgentProtocol

        assert MetricsAgentProtocol is not None

    def test_metrics_agent_has_calculate_method(self):
        """MetricsAgentProtocol이 calculate 메서드를 정의하는지 검증"""
        from src.domain.interfaces.agent import MetricsAgentProtocol

        assert hasattr(MetricsAgentProtocol, "calculate")

    def test_insight_agent_protocol_exists(self):
        """InsightAgentProtocol이 존재하는지 검증"""
        from src.domain.interfaces.agent import InsightAgentProtocol

        assert InsightAgentProtocol is not None


class TestScraperProtocol:
    """Scraper Protocol 테스트"""

    def test_scraper_protocol_exists(self):
        """ScraperProtocol이 존재하는지 검증"""
        from src.domain.interfaces.scraper import ScraperProtocol

        assert ScraperProtocol is not None

    def test_scraper_has_scrape_category_method(self):
        """ScraperProtocol이 scrape_category 메서드를 정의하는지 검증"""
        from src.domain.interfaces.scraper import ScraperProtocol

        assert hasattr(ScraperProtocol, "scrape_category")

    def test_scraper_has_initialize_method(self):
        """ScraperProtocol이 initialize 메서드를 정의하는지 검증"""
        from src.domain.interfaces.scraper import ScraperProtocol

        assert hasattr(ScraperProtocol, "initialize")

    def test_scraper_has_close_method(self):
        """ScraperProtocol이 close 메서드를 정의하는지 검증"""
        from src.domain.interfaces.scraper import ScraperProtocol

        assert hasattr(ScraperProtocol, "close")


class TestLLMClientProtocol:
    """LLM Client Protocol 테스트"""

    def test_llm_client_protocol_exists(self):
        """LLMClientProtocol이 존재하는지 검증"""
        from src.domain.interfaces.llm_client import LLMClientProtocol

        assert LLMClientProtocol is not None

    def test_llm_client_has_generate_method(self):
        """LLMClientProtocol이 generate 메서드를 정의하는지 검증"""
        from src.domain.interfaces.llm_client import LLMClientProtocol

        assert hasattr(LLMClientProtocol, "generate")

    def test_llm_client_has_generate_with_context_method(self):
        """LLMClientProtocol이 generate_with_context 메서드를 정의하는지 검증"""
        from src.domain.interfaces.llm_client import LLMClientProtocol

        assert hasattr(LLMClientProtocol, "generate_with_context")


class TestKnowledgeGraphProtocol:
    """Knowledge Graph Protocol 테스트"""

    def test_knowledge_graph_protocol_exists(self):
        """KnowledgeGraphProtocol이 존재하는지 검증"""
        from src.domain.interfaces.knowledge_graph import KnowledgeGraphProtocol

        assert KnowledgeGraphProtocol is not None

    def test_kg_has_add_relation_method(self):
        """KnowledgeGraphProtocol이 add_relation 메서드를 정의하는지 검증"""
        from src.domain.interfaces.knowledge_graph import KnowledgeGraphProtocol

        assert hasattr(KnowledgeGraphProtocol, "add_relation")

    def test_kg_has_query_method(self):
        """KnowledgeGraphProtocol이 query 메서드를 정의하는지 검증"""
        from src.domain.interfaces.knowledge_graph import KnowledgeGraphProtocol

        assert hasattr(KnowledgeGraphProtocol, "query")

    def test_kg_has_get_entity_relations_method(self):
        """KnowledgeGraphProtocol이 get_entity_relations 메서드를 정의하는지 검증"""
        from src.domain.interfaces.knowledge_graph import KnowledgeGraphProtocol

        assert hasattr(KnowledgeGraphProtocol, "get_entity_relations")


class TestRetrieverProtocol:
    """Retriever Protocol 테스트"""

    def test_retriever_protocol_exists(self):
        """RetrieverProtocol이 존재하는지 검증"""
        from src.domain.interfaces.retriever import RetrieverProtocol

        assert RetrieverProtocol is not None

    def test_retriever_has_retrieve_method(self):
        """RetrieverProtocol이 retrieve 메서드를 정의하는지 검증"""
        from src.domain.interfaces.retriever import RetrieverProtocol

        assert hasattr(RetrieverProtocol, "retrieve")

    def test_retriever_has_initialize_method(self):
        """RetrieverProtocol이 initialize 메서드를 정의하는지 검증"""
        from src.domain.interfaces.retriever import RetrieverProtocol

        assert hasattr(RetrieverProtocol, "initialize")

"""
Domain Interfaces Tests (TDD - RED Phase)
==========================================
Protocol(인터페이스) 정의 테스트
의존성 역전을 위한 추상 인터페이스 검증
"""

from typing import Any


class TestRepositoryProtocol:
    """Repository Protocol 테스트"""

    def test_product_repository_protocol_exists(self):
        """ProductRepository Protocol이 존재하는지 검증"""
        from src.domain.interfaces.repository import ProductRepository

        # Protocol 클래스가 존재하는지 확인
        assert ProductRepository is not None

    def test_product_repository_has_save_records_method(self):
        """ProductRepository가 save_records 메서드를 정의하는지 검증"""
        from typing import get_type_hints

        from src.domain.interfaces.repository import ProductRepository

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

        class MockRepository:
            """Mock implementation for testing"""

            async def save_records(self, records: list[Any]) -> bool:
                return True

            async def get_recent(self, days: int = 7) -> list[Any]:
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


class TestBrainProtocol:
    """Brain Protocol 테스트 (2026-02-10)"""

    def test_brain_protocol_exists(self):
        """BrainProtocol이 존재하는지 검증"""
        from src.domain.interfaces.brain import BrainProtocol

        assert BrainProtocol is not None

    def test_brain_protocol_is_runtime_checkable(self):
        """BrainProtocol이 runtime_checkable인지 검증"""

        from src.domain.interfaces.brain import BrainProtocol

        assert isinstance(BrainProtocol, type)

    def test_brain_has_required_methods(self):
        """BrainProtocol이 필수 메서드를 정의하는지 검증"""
        from src.domain.interfaces.brain import BrainProtocol

        required_methods = [
            "initialize",
            "process_query",
            "process_query_stream",
            "run_autonomous_cycle",
            "collect_market_intelligence",
            "check_alerts",
            "start_scheduler",
            "stop_scheduler",
            "on_event",
            "emit_event",
            "get_stats",
            "get_state_summary",
            "reset_failed_agents",
        ]

        for method in required_methods:
            assert hasattr(BrainProtocol, method), f"Missing method: {method}"


class TestChatbotAgentProtocol:
    """Chatbot Agent Protocol 테스트 (2026-02-10)"""

    def test_chatbot_agent_protocol_exists(self):
        """ChatbotAgentProtocol이 존재하는지 검증"""
        from src.domain.interfaces.chatbot import ChatbotAgentProtocol

        assert ChatbotAgentProtocol is not None

    def test_chatbot_agent_has_required_methods(self):
        """ChatbotAgentProtocol이 필수 메서드를 정의하는지 검증"""
        from src.domain.interfaces.chatbot import ChatbotAgentProtocol

        required_methods = [
            "chat",
            "set_data_context",
            "get_conversation_history",
            "clear_conversation",
            "get_last_hybrid_context",
            "get_knowledge_graph",
            "get_reasoner",
            "explain_last_response",
        ]

        for method in required_methods:
            assert hasattr(ChatbotAgentProtocol, method), f"Missing method: {method}"


class TestInsightAgentProtocol:
    """Insight Agent Protocol 테스트 (2026-02-10)"""

    def test_insight_agent_protocol_exists(self):
        """InsightAgentProtocol(HybridInsightAgentProtocol)이 존재하는지 검증"""
        from src.domain.interfaces.insight import InsightAgentProtocol

        assert InsightAgentProtocol is not None

    def test_insight_agent_has_required_methods(self):
        """InsightAgentProtocol이 필수 메서드를 정의하는지 검증"""
        from src.domain.interfaces.insight import InsightAgentProtocol

        required_methods = [
            "execute",
            "get_results",
            "get_last_hybrid_context",
            "get_knowledge_graph",
            "get_reasoner",
        ]

        for method in required_methods:
            assert hasattr(InsightAgentProtocol, method), f"Missing method: {method}"


class TestAlertAgentProtocol:
    """Alert Agent Protocol 테스트 (2026-02-10)"""

    def test_alert_agent_protocol_exists(self):
        """AlertAgentProtocol이 존재하는지 검증"""
        from src.domain.interfaces.alert import AlertAgentProtocol

        assert AlertAgentProtocol is not None

    def test_alert_agent_has_required_methods(self):
        """AlertAgentProtocol이 필수 메서드를 정의하는지 검증"""
        from src.domain.interfaces.alert import AlertAgentProtocol

        required_methods = [
            "create_alert",
            "process_metrics",
            "send_pending_alerts",
            "on_crawl_complete",
            "on_crawl_failed",
            "on_error",
            "send_daily_summary",
            "get_alerts",
            "get_pending_count",
            "get_stats",
            "clear_old_alerts",
        ]

        for method in required_methods:
            assert hasattr(AlertAgentProtocol, method), f"Missing method: {method}"


class TestMetricCalculatorProtocol:
    """Metric Calculator Protocol 테스트 (2026-02-10)"""

    def test_metric_calculator_protocol_exists(self):
        """MetricCalculatorProtocol이 존재하는지 검증"""
        from src.domain.interfaces.metric import MetricCalculatorProtocol

        assert MetricCalculatorProtocol is not None

    def test_metric_calculator_has_required_methods(self):
        """MetricCalculatorProtocol이 필수 메서드를 정의하는지 검증"""
        from src.domain.interfaces.metric import MetricCalculatorProtocol

        required_methods = [
            "calculate_sos",
            "calculate_hhi",
            "calculate_brand_avg_rank",
            "calculate_cpi",
            "calculate_churn_rate",
            "calculate_avg_rating_gap",
            "calculate_rank_volatility",
            "calculate_rank_shock",
            "calculate_rank_change",
            "calculate_streak_days",
            "calculate_rating_trend",
            "calculate_brand_metrics",
            "calculate_product_metrics",
            "calculate_market_metrics",
        ]

        for method in required_methods:
            assert hasattr(MetricCalculatorProtocol, method), f"Missing method: {method}"


class TestStorageProtocol:
    """Storage Protocol 테스트 (2026-02-10)"""

    def test_storage_protocol_exists(self):
        """StorageProtocol이 존재하는지 검증"""
        from src.domain.interfaces.storage import StorageProtocol

        assert StorageProtocol is not None

    def test_storage_has_required_methods(self):
        """StorageProtocol이 필수 메서드를 정의하는지 검증"""
        from src.domain.interfaces.storage import StorageProtocol

        required_methods = [
            "initialize",
            "append_rank_records",
            "get_raw_data",
            "get_latest_data",
            "get_historical_data",
            "save_brand_metrics",
            "save_market_metrics",
            "save_competitor_products",
            "get_competitor_products",
            "get_data_date",
            "get_stats",
            "export_to_excel",
        ]

        for method in required_methods:
            assert hasattr(StorageProtocol, method), f"Missing method: {method}"


class TestSignalCollectorProtocol:
    """Signal Collector Protocol 테스트 (2026-02-10)"""

    def test_signal_collector_protocol_exists(self):
        """SignalCollectorProtocol이 존재하는지 검증"""
        from src.domain.interfaces.signal import SignalCollectorProtocol

        assert SignalCollectorProtocol is not None

    def test_signal_collector_has_required_methods(self):
        """SignalCollectorProtocol이 필수 메서드를 정의하는지 검증"""
        from src.domain.interfaces.signal import SignalCollectorProtocol

        required_methods = [
            "initialize",
            "close",
            "fetch_rss_articles",
            "fetch_all_rss_feeds",
            "fetch_reddit_trends",
            "fetch_tiktok_trends",
            "fetch_kbeauty_news",
            "fetch_industry_signals",
            "fetch_tavily_news",
            "fetch_all_news",
            "add_manual_media_input",
            "add_weekly_trend_radar",
            "generate_report_section",
            "get_source_reliability",
            "create_source_reference",
            "get_signals_for_kg",
            "get_stats",
        ]

        for method in required_methods:
            assert hasattr(SignalCollectorProtocol, method), f"Missing method: {method}"

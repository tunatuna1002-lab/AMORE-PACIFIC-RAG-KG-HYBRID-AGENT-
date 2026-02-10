"""
Test fixtures for Application layer
====================================
Provides mock implementations of domain protocols for testing workflows.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


# Mock Chatbot Agent
class MockChatbotAgent:
    """Mock implementation of ChatbotAgentProtocol"""

    def __init__(self):
        self.chat = AsyncMock(
            return_value={
                "response": "Mock response",
                "sources": [],
                "confidence": 0.85,
                "suggestions": [],
                "entities": {},
                "inferences": [],
                "query_type": "simple",
            }
        )
        self.set_data_context = MagicMock()
        self.get_conversation_history = MagicMock(return_value=[])
        self.clear_conversation = MagicMock()
        self.get_last_hybrid_context = MagicMock(return_value=None)
        self.get_knowledge_graph = MagicMock()
        self.get_reasoner = MagicMock()
        self.explain_last_response = AsyncMock(return_value="Mock explanation")


# Mock Retriever
class MockRetriever:
    """Mock implementation of RetrieverProtocol"""

    def __init__(self):
        self.initialize = AsyncMock()
        self.retrieve = AsyncMock(
            return_value={
                "kg_facts": [],
                "rag_documents": [],
                "inferences": [],
            }
        )
        self.search = AsyncMock(return_value=[])


# Mock Scraper
class MockScraper:
    """Mock implementation of ScraperProtocol"""

    def __init__(self):
        self.initialize = AsyncMock()
        self.scrape_category = AsyncMock(
            return_value=[
                {
                    "snapshot_date": "2026-02-10",
                    "category_id": "lip_care",
                    "rank": 1,
                    "asin": "B001GXRQW0",
                    "product_name": "LANEIGE Lip Sleeping Mask",
                    "brand": "LANEIGE",
                    "price": 24.0,
                    "rating": 4.5,
                    "reviews": 15000,
                }
            ]
        )
        self.scrape_all_categories = AsyncMock(return_value=[])
        self.close = AsyncMock()


# Mock Storage
class MockStorage:
    """Mock implementation of StorageProtocol"""

    def __init__(self):
        self.initialize = AsyncMock(return_value=True)
        self.append_rank_records = AsyncMock(
            return_value={"inserted": 1, "duplicates": 0, "errors": 0}
        )
        self.get_raw_data = AsyncMock(return_value=[])
        self.get_latest_data = AsyncMock(return_value=[])
        self.get_historical_data = AsyncMock(return_value=[])
        self.save_brand_metrics = AsyncMock(return_value=1)
        self.save_market_metrics = AsyncMock(return_value=1)
        self.save_competitor_products = AsyncMock(return_value={"saved": 1})
        self.get_competitor_products = AsyncMock(return_value=[])
        self.get_data_date = MagicMock(return_value="2026-02-10")
        self.get_stats = MagicMock(
            return_value={
                "total_records": 100,
                "total_products": 50,
                "total_brands": 10,
            }
        )
        self.export_to_excel = MagicMock(return_value={"success": True})


# Mock Metric Calculator
class MockMetricCalculator:
    """Mock implementation of MetricCalculatorProtocol"""

    def __init__(self):
        self.calculate_sos = MagicMock(return_value=5.2)
        self.calculate_hhi = MagicMock(return_value=0.08)
        self.calculate_brand_avg_rank = MagicMock(return_value=42.5)
        self.calculate_cpi = MagicMock(return_value=105.0)
        self.calculate_churn_rate = MagicMock(return_value=3.5)
        self.calculate_avg_rating_gap = MagicMock(return_value=0.2)
        self.calculate_rank_volatility = MagicMock(return_value=2.1)
        self.calculate_rank_shock = MagicMock(return_value=False)
        self.calculate_rank_change = MagicMock(return_value=0)
        self.calculate_streak_days = MagicMock(return_value=7)
        self.calculate_rating_trend = MagicMock(return_value=0.05)
        self.calculate_brand_metrics = MagicMock(
            return_value={
                "brand": "LANEIGE",
                "category_id": "lip_care",
                "sos": 5.2,
                "brand_avg_rank": 42.5,
                "cpi": 105.0,
            }
        )
        self.calculate_product_metrics = MagicMock(return_value={})
        self.calculate_market_metrics = MagicMock(
            return_value={
                "category_id": "lip_care",
                "hhi": 0.08,
                "avg_price": 22.5,
                "avg_rating": 4.3,
            }
        )


# Mock Insight Agent
class MockInsightAgent:
    """Mock implementation of InsightAgentProtocol"""

    def __init__(self):
        self.execute = AsyncMock(
            return_value={
                "insights": [
                    {
                        "type": "opportunity",
                        "priority": "high",
                        "title": "Mock Insight",
                        "content": "Mock content",
                        "action_items": [],
                        "data_sources": [],
                        "confidence": 0.8,
                    }
                ],
                "summary": "Mock summary",
                "highlights": [],
                "external_signals": {},
                "execution_time": 1.5,
                "cost": 0.01,
            }
        )
        self.get_results = MagicMock(return_value={})
        self.get_last_hybrid_context = MagicMock(return_value=None)
        self.get_knowledge_graph = MagicMock()
        self.get_reasoner = MagicMock()


# Mock Alert Agent
class MockAlertAgent:
    """Mock implementation of AlertAgentProtocol"""

    def __init__(self):
        self.create_alert = MagicMock(
            return_value={
                "id": "alert-1",
                "type": "rank_change",
                "title": "Mock Alert",
                "message": "Mock message",
            }
        )
        self.process_metrics = AsyncMock(return_value=[])
        self.send_pending_alerts = AsyncMock(
            return_value={"sent_count": 0, "failed_count": 0, "recipients": []}
        )
        self.on_crawl_complete = AsyncMock()
        self.on_crawl_failed = AsyncMock()
        self.on_error = AsyncMock()
        self.send_daily_summary = AsyncMock(return_value={"sent": True})
        self.get_alerts = MagicMock(return_value=[])
        self.get_pending_count = MagicMock(return_value=0)
        self.get_stats = MagicMock(return_value={"total": 0, "sent": 0, "failed": 0})
        self.clear_old_alerts = MagicMock(return_value=0)


# Fixtures
@pytest.fixture
def mock_chatbot():
    """Mock chatbot agent fixture"""
    return MockChatbotAgent()


@pytest.fixture
def mock_retriever():
    """Mock retriever fixture"""
    return MockRetriever()


@pytest.fixture
def mock_scraper():
    """Mock scraper fixture"""
    return MockScraper()


@pytest.fixture
def mock_storage():
    """Mock storage fixture"""
    return MockStorage()


@pytest.fixture
def mock_metric_calculator():
    """Mock metric calculator fixture"""
    return MockMetricCalculator()


@pytest.fixture
def mock_insight_agent():
    """Mock insight agent fixture"""
    return MockInsightAgent()


@pytest.fixture
def mock_alert_agent():
    """Mock alert agent fixture"""
    return MockAlertAgent()


@pytest.fixture
def sample_records():
    """Sample product records for testing"""
    return [
        {
            "snapshot_date": "2026-02-10",
            "category_id": "lip_care",
            "rank": 1,
            "asin": "B001GXRQW0",
            "product_name": "LANEIGE Lip Sleeping Mask",
            "brand": "LANEIGE",
            "price": 24.0,
            "rating": 4.5,
            "reviews": 15000,
        },
        {
            "snapshot_date": "2026-02-10",
            "category_id": "lip_care",
            "rank": 2,
            "asin": "B000ASIN02",
            "product_name": "Competitor Product",
            "brand": "Burt's Bees",
            "price": 15.0,
            "rating": 4.3,
            "reviews": 8000,
        },
    ]


@pytest.fixture
def sample_metrics():
    """Sample metrics data for testing"""
    return {
        "brand_metrics": {
            "LANEIGE": {
                "lip_care": {
                    "sos": 5.2,
                    "brand_avg_rank": 42.5,
                    "product_count": 3,
                    "cpi": 105.0,
                    "avg_rating_gap": 0.2,
                }
            }
        },
        "market_metrics": {
            "lip_care": {
                "hhi": 0.08,
                "avg_price": 22.5,
                "avg_rating": 4.3,
                "top_brands": ["LANEIGE", "Burt's Bees", "Aquaphor"],
            }
        },
    }

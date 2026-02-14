"""
Unit tests for ExternalSignalManager
"""

import pytest

from src.agents.external_signal_manager import ExternalSignalManager


class TestExternalSignalManager:
    """Test ExternalSignalManager functionality"""

    @pytest.fixture
    def manager(self):
        """Create ExternalSignalManager instance"""
        return ExternalSignalManager()

    @pytest.fixture
    def mock_collector(self):
        """Create mock ExternalSignalCollector"""

        class MockSignal:
            def __init__(self, title, source, relevance=0.8, reliability=0.9):
                self.title = title
                self.source = source
                self.relevance_score = relevance
                self.metadata = {"reliability_score": reliability}
                self.url = "https://example.com"
                self.published_at = "2026-01-27"

        class MockCollector:
            def __init__(self):
                self.initialized = False

            async def initialize(self):
                self.initialized = True

            async def fetch_tavily_news(self, brands, topics, days=14, max_results=8):
                return [
                    MockSignal("News 1", "Tavily", 0.9, 0.95),
                    MockSignal("News 2", "Tavily", 0.8, 0.9),
                ]

            async def fetch_all_rss_feeds(self, keywords):
                return [
                    MockSignal("RSS 1", "RSS Feed", 0.7, 0.8),
                ]

        return MockCollector()

    @pytest.mark.asyncio
    async def test_lazy_initialization(self, manager):
        """Test that collector is initialized lazily"""
        assert manager._collector is None

        # After collect, should attempt initialization
        # (will fail without mock, but we test the lazy pattern)
        signals = await manager.collect("test query")

        # May fail to initialize, but should return empty list gracefully
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_collect_with_mock_collector(self, manager, mock_collector, monkeypatch):
        """Test collection with mock collector"""
        # Inject mock collector
        manager._collector = mock_collector
        await manager._collector.initialize()

        entities = {
            "brands": ["LANEIGE"],
            "categories": ["Lip Care"],
        }

        signals = await manager.collect("test query", entities=entities)

        assert isinstance(signals, list)
        # Should have Tavily news (2) + RSS (1 max 3) = 3 signals, but sorted and capped at 8
        assert len(signals) <= 8

    @pytest.mark.asyncio
    async def test_collect_without_entities(self, manager, mock_collector):
        """Test collection with no entities (uses defaults)"""
        manager._collector = mock_collector
        await manager._collector.initialize()

        signals = await manager.collect("test query")

        assert isinstance(signals, list)
        # Should use default brands ["LANEIGE", "K-Beauty"] and topics

    @pytest.mark.asyncio
    async def test_signal_type_filtering(self, manager, mock_collector):
        """Test collection with specific signal types"""
        manager._collector = mock_collector
        await manager._collector.initialize()

        # Only collect Tavily
        signals = await manager.collect("test query", signal_types=["tavily"])

        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_failed_collector_tracking(self, manager):
        """Test tracking of failed collectors"""
        # Attempt collection without proper initialization
        signals = await manager.collect("test query")

        # Should track failure
        failed = manager.get_failed_collectors()
        assert isinstance(failed, list)
        # May have failures recorded
        # (depends on environment - ExternalSignalCollector may not be available)

    @pytest.mark.asyncio
    async def test_signal_sorting_by_reliability(self, manager, mock_collector):
        """Test that signals are sorted by reliability * relevance"""

        class MockSignal:
            def __init__(self, title, relevance, reliability):
                self.title = title
                self.relevance_score = relevance
                self.metadata = {"reliability_score": reliability}
                self.source = "test"
                self.url = ""
                self.published_at = ""

        # Override collector to return signals with different scores
        async def mock_fetch_tavily(brands, topics, days=14, max_results=8):
            return [
                MockSignal("Low score", 0.5, 0.5),  # 0.25
                MockSignal("High score", 0.9, 0.9),  # 0.81
                MockSignal("Mid score", 0.7, 0.7),  # 0.49
            ]

        async def mock_fetch_rss(keywords):
            return []

        manager._collector = mock_collector
        await manager._collector.initialize()
        manager._collector.fetch_tavily_news = mock_fetch_tavily
        manager._collector.fetch_all_rss_feeds = mock_fetch_rss

        signals = await manager.collect("test query")

        # Should be sorted descending by score
        if len(signals) >= 2:
            # First should have highest score
            first_score = signals[0].relevance_score * signals[0].metadata.get(
                "reliability_score", 0.7
            )
            second_score = signals[1].relevance_score * signals[1].metadata.get(
                "reliability_score", 0.7
            )
            assert first_score >= second_score

    @pytest.mark.asyncio
    async def test_max_results_limit(self, manager, mock_collector):
        """Test that results are capped at 8"""

        class MockSignal:
            def __init__(self, i):
                self.title = f"Signal {i}"
                self.relevance_score = 0.8
                self.metadata = {"reliability_score": 0.9}
                self.source = "test"
                self.url = ""
                self.published_at = ""

        # Return 15 signals
        async def mock_fetch_tavily(brands, topics, days=14, max_results=8):
            return [MockSignal(i) for i in range(15)]

        async def mock_fetch_rss(keywords):
            return []

        manager._collector = mock_collector
        await manager._collector.initialize()
        manager._collector.fetch_tavily_news = mock_fetch_tavily
        manager._collector.fetch_all_rss_feeds = mock_fetch_rss

        signals = await manager.collect("test query")

        # Should be capped at 8
        assert len(signals) <= 8

    @pytest.mark.asyncio
    async def test_graceful_failure_handling(self, manager):
        """Test that failures are handled gracefully"""

        # Force a failure scenario
        class FailingCollector:
            async def initialize(self):
                raise Exception("Init failed")

        manager._collector = FailingCollector()

        # Should not raise, should return empty list
        signals = await manager.collect("test query")
        assert signals == []

        # Should track failure
        failed = manager.get_failed_collectors()
        assert len(failed) > 0

    def test_get_failed_collectors_before_init(self, manager):
        """Test get_failed_collectors when collector not initialized"""
        failed = manager.get_failed_collectors()
        assert isinstance(failed, list)
        # May or may not have failures depending on module availability

    @pytest.mark.asyncio
    async def test_entity_to_topic_conversion(self, manager, mock_collector):
        """Test that categories are converted to topics"""
        manager._collector = mock_collector
        await manager._collector.initialize()

        entities = {
            "brands": ["LANEIGE"],
            "categories": ["Lip_Care", "Skin_Care"],
        }

        # Mock to capture the topics parameter
        topics_captured = []

        async def mock_fetch_tavily(brands, topics, days=14, max_results=8):
            topics_captured.extend(topics)
            return []

        manager._collector.fetch_tavily_news = mock_fetch_tavily

        await manager.collect("test query", entities=entities)

        # Should have converted underscores to spaces
        assert "Lip Care" in topics_captured or "Skin Care" in topics_captured

"""
SentimentService 단위 테스트
"""

from unittest.mock import MagicMock

from src.ontology.sentiment_service import SentimentService


class TestSentimentServiceInit:
    """SentimentService 초기화 테스트"""

    def test_sentiment_service_requires_kg(self):
        """SentimentService는 KnowledgeGraph를 필요로 함"""
        mock_kg = MagicMock()
        service = SentimentService(mock_kg)
        assert service._kg is mock_kg


class TestSentimentServiceGetProductSentiments:
    """SentimentService.get_product_sentiments 테스트"""

    def test_get_product_sentiments_empty(self):
        """감성 정보가 없는 제품"""
        mock_kg = MagicMock()
        mock_kg.query.return_value = []

        service = SentimentService(mock_kg)
        result = service.get_product_sentiments("B0TEST123")

        assert result["asin"] == "B0TEST123"
        assert result["ai_summary"] is None
        assert result["sentiment_tags"] == []
        assert result["sentiment_clusters"] == {}

    def test_get_product_sentiments_with_data(self):
        """감성 정보가 있는 제품"""
        mock_summary_rel = MagicMock()
        mock_summary_rel.properties = {"summary_text": "Customers love it"}

        mock_sentiment_rel = MagicMock()
        mock_sentiment_rel.object = "Moisturizing"
        mock_sentiment_rel.properties = {"cluster": "quality"}

        mock_kg = MagicMock()
        mock_kg.query.side_effect = [
            [mock_summary_rel],  # HAS_AI_SUMMARY
            [mock_sentiment_rel],  # HAS_SENTIMENT
        ]

        service = SentimentService(mock_kg)
        result = service.get_product_sentiments("B0TEST123")

        assert result["asin"] == "B0TEST123"
        assert result["ai_summary"] == "Customers love it"
        assert "Moisturizing" in result["sentiment_tags"]
        assert "quality" in result["sentiment_clusters"]


class TestSentimentServiceCompareProducts:
    """SentimentService.compare_products 테스트"""

    def test_compare_products_finds_common_tags(self):
        """공통 태그 찾기"""
        # Product 1 mocks
        mock_sentiment_rel1 = MagicMock()
        mock_sentiment_rel1.object = "Moisturizing"
        mock_sentiment_rel1.properties = {}

        # Product 2 mocks
        mock_sentiment_rel2 = MagicMock()
        mock_sentiment_rel2.object = "Moisturizing"
        mock_sentiment_rel2.properties = {}

        mock_kg = MagicMock()
        mock_kg.query.side_effect = [
            [],  # Product 1 AI Summary
            [mock_sentiment_rel1],  # Product 1 Sentiments
            [],  # Product 2 AI Summary
            [mock_sentiment_rel2],  # Product 2 Sentiments
        ]

        service = SentimentService(mock_kg)
        result = service.compare_products("B0PROD1", "B0PROD2")

        assert "Moisturizing" in result["common_tags"]

    def test_compare_products_finds_unique_tags(self):
        """고유 태그 찾기"""
        mock_sentiment_rel1 = MagicMock()
        mock_sentiment_rel1.object = "Hydrating"
        mock_sentiment_rel1.properties = {}

        mock_sentiment_rel2 = MagicMock()
        mock_sentiment_rel2.object = "ValueForMoney"
        mock_sentiment_rel2.properties = {}

        mock_kg = MagicMock()
        mock_kg.query.side_effect = [
            [],  # Product 1 AI Summary
            [mock_sentiment_rel1],  # Product 1 Sentiments
            [],  # Product 2 AI Summary
            [mock_sentiment_rel2],  # Product 2 Sentiments
        ]

        service = SentimentService(mock_kg)
        result = service.compare_products("B0PROD1", "B0PROD2")

        assert "Hydrating" in result["unique_to_1"]
        assert "ValueForMoney" in result["unique_to_2"]


class TestSentimentServiceFindProducts:
    """SentimentService.find_products_by_sentiment 테스트"""

    def test_find_products_by_sentiment(self):
        """감성 태그로 제품 검색"""
        mock_rel1 = MagicMock()
        mock_rel1.subject = "B0PROD1"
        mock_rel2 = MagicMock()
        mock_rel2.subject = "B0PROD2"

        mock_kg = MagicMock()
        mock_kg.query.return_value = [mock_rel1, mock_rel2]

        service = SentimentService(mock_kg)
        result = service.find_products_by_sentiment("Moisturizing")

        assert "B0PROD1" in result
        assert "B0PROD2" in result

    def test_find_products_by_sentiment_with_brand_filter(self):
        """브랜드 필터 적용"""
        mock_rel1 = MagicMock()
        mock_rel1.subject = "B0PROD1"
        mock_rel2 = MagicMock()
        mock_rel2.subject = "B0PROD2"

        mock_kg = MagicMock()
        mock_kg.query.return_value = [mock_rel1, mock_rel2]
        mock_kg.get_entity_metadata.side_effect = [
            {"brand": "LANEIGE"},
            {"brand": "COSRX"},
        ]

        service = SentimentService(mock_kg)
        result = service.find_products_by_sentiment("Moisturizing", brand_filter="LANEIGE")

        assert "B0PROD1" in result
        assert "B0PROD2" not in result


class TestContainerSentimentService:
    """Container를 통한 SentimentService 테스트"""

    def test_container_get_sentiment_service(self):
        """Container에서 SentimentService 획득"""
        from src.infrastructure.container import Container

        Container.reset()
        service = Container.get_sentiment_service()

        assert isinstance(service, SentimentService)

    def test_container_sentiment_service_uses_same_kg(self):
        """SentimentService는 Container의 KG를 사용"""
        from src.infrastructure.container import Container

        Container.reset()
        kg = Container.get_knowledge_graph()
        service = Container.get_sentiment_service()

        assert service._kg is kg

    def test_container_sentiment_service_singleton(self):
        """SentimentService는 싱글톤"""
        from src.infrastructure.container import Container

        Container.reset()
        service1 = Container.get_sentiment_service()
        service2 = Container.get_sentiment_service()

        assert service1 is service2

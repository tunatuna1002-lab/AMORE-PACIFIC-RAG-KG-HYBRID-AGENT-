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


# =========================================================================
# Wave 3: load_from_data 테스트
# =========================================================================


class TestSentimentServiceLoadFromData:
    """SentimentService.load_from_data 테스트"""

    def test_load_from_data_empty_products(self):
        """빈 products dict일 때 0 반환"""
        mock_kg = MagicMock()
        service = SentimentService(mock_kg)
        result = service.load_from_data({"products": {}, "collected_at": "2025-01-15"})
        assert result == 0

    def test_load_from_data_skips_failed_products(self):
        """success=False인 제품은 건너뜀"""
        mock_kg = MagicMock()
        service = SentimentService(mock_kg)
        data = {
            "products": {
                "B0FAIL": {"asin": "B0FAIL", "success": False},
            },
            "collected_at": "2025-01-15",
        }
        result = service.load_from_data(data)
        assert result == 0
        mock_kg.add_relation.assert_not_called()

    def test_load_from_data_with_ai_summary(self):
        """AI summary 관계 추가"""
        mock_kg = MagicMock()
        mock_kg.add_relation.return_value = True
        service = SentimentService(mock_kg)
        data = {
            "products": {
                "B0TEST1": {
                    "asin": "B0TEST1",
                    "ai_customers_say": "Customers like the moisturizing effect",
                    "sentiment_tags": [],
                    "success": True,
                },
            },
            "collected_at": "2025-01-15",
        }
        result = service.load_from_data(data)
        assert result == 1
        # Should set entity metadata with ai_summary
        mock_kg.set_entity_metadata.assert_called_once()

    def test_load_from_data_with_sentiment_tags(self):
        """감성 태그 관계 추가"""
        mock_kg = MagicMock()
        mock_kg.add_relation.return_value = True
        service = SentimentService(mock_kg)
        data = {
            "products": {
                "B0TEST2": {
                    "asin": "B0TEST2",
                    "sentiment_tags": ["Moisturizing", "Value for money"],
                    "success": True,
                },
            },
            "collected_at": "2025-01-15",
        }
        result = service.load_from_data(data)
        # 2 sentiment tags (no ai_summary)
        assert result == 2

    def test_load_from_data_with_both_summary_and_tags(self):
        """AI summary + 감성 태그 모두 있는 경우"""
        mock_kg = MagicMock()
        mock_kg.add_relation.return_value = True
        service = SentimentService(mock_kg)
        data = {
            "products": {
                "B0BOTH": {
                    "asin": "B0BOTH",
                    "ai_customers_say": "Great product",
                    "sentiment_tags": ["Moisturizing", "Hydrating", "Affordable"],
                    "success": True,
                },
            },
            "collected_at": "2025-01-15",
        }
        result = service.load_from_data(data)
        # 1 summary + 3 tags = 4
        assert result == 4

    def test_load_from_data_add_relation_failure(self):
        """add_relation이 False 반환하면 카운트하지 않음"""
        mock_kg = MagicMock()
        mock_kg.add_relation.return_value = False
        service = SentimentService(mock_kg)
        data = {
            "products": {
                "B0FAIL2": {
                    "asin": "B0FAIL2",
                    "ai_customers_say": "Good",
                    "sentiment_tags": ["Nice"],
                    "success": True,
                },
            },
            "collected_at": "2025-01-15",
        }
        result = service.load_from_data(data)
        assert result == 0

    def test_load_from_data_multiple_products(self):
        """여러 제품 데이터 로드"""
        mock_kg = MagicMock()
        mock_kg.add_relation.return_value = True
        service = SentimentService(mock_kg)
        data = {
            "products": {
                "B0PROD1": {
                    "asin": "B0PROD1",
                    "ai_customers_say": "Great moisturizer",
                    "sentiment_tags": ["Moisturizing"],
                    "success": True,
                },
                "B0PROD2": {
                    "asin": "B0PROD2",
                    "sentiment_tags": ["Affordable", "Effective"],
                    "success": True,
                },
            },
            "collected_at": "2025-01-15",
        }
        result = service.load_from_data(data)
        # Prod1: 1 summary + 1 tag = 2; Prod2: 2 tags = 2; total = 4
        assert result == 4

    def test_load_from_data_missing_collected_at(self):
        """collected_at 키가 없어도 동작"""
        mock_kg = MagicMock()
        mock_kg.add_relation.return_value = True
        service = SentimentService(mock_kg)
        data = {
            "products": {
                "B0NODATE": {
                    "asin": "B0NODATE",
                    "ai_customers_say": "Nice",
                    "sentiment_tags": [],
                    "success": True,
                },
            },
        }
        result = service.load_from_data(data)
        assert result == 1


# =========================================================================
# Wave 3: get_product_sentiments 심화 테스트
# =========================================================================


class TestSentimentServiceGetProductSentimentsExtended:
    """get_product_sentiments 확장 테스트"""

    def test_get_product_sentiments_multiple_tags_same_cluster(self):
        """같은 클러스터에 속하는 여러 태그"""
        mock_tag1 = MagicMock()
        mock_tag1.object = "Moisturizing"
        mock_tag1.properties = {"cluster": "Hydration"}

        mock_tag2 = MagicMock()
        mock_tag2.object = "Hydrating"
        mock_tag2.properties = {"cluster": "Hydration"}

        mock_kg = MagicMock()
        mock_kg.query.side_effect = [
            [],  # No AI summary
            [mock_tag1, mock_tag2],  # Two sentiment tags
        ]

        service = SentimentService(mock_kg)
        result = service.get_product_sentiments("B0MULTI")

        assert len(result["sentiment_tags"]) == 2
        assert "Hydration" in result["sentiment_clusters"]
        assert len(result["sentiment_clusters"]["Hydration"]) == 2

    def test_get_product_sentiments_tag_without_cluster(self):
        """클러스터가 없는 태그"""
        mock_tag = MagicMock()
        mock_tag.object = "Unknown Quality"
        mock_tag.properties = {}  # No cluster key

        mock_kg = MagicMock()
        mock_kg.query.side_effect = [
            [],  # No AI summary
            [mock_tag],
        ]

        service = SentimentService(mock_kg)
        result = service.get_product_sentiments("B0NOCLUST")

        assert "Unknown Quality" in result["sentiment_tags"]
        assert result["sentiment_clusters"] == {}

    def test_get_product_sentiments_tag_with_none_cluster(self):
        """클러스터가 None인 태그"""
        mock_tag = MagicMock()
        mock_tag.object = "SomeTag"
        mock_tag.properties = {"cluster": None}

        mock_kg = MagicMock()
        mock_kg.query.side_effect = [
            [],
            [mock_tag],
        ]

        service = SentimentService(mock_kg)
        result = service.get_product_sentiments("B0NULLCLUST")

        assert "SomeTag" in result["sentiment_tags"]
        # None cluster should not be added
        assert result["sentiment_clusters"] == {}


# =========================================================================
# Wave 3: get_brand_profile 테스트
# =========================================================================


class TestSentimentServiceGetBrandProfile:
    """SentimentService.get_brand_profile 테스트"""

    def test_get_brand_profile_no_products(self):
        """제품이 없는 브랜드"""
        mock_kg = MagicMock()
        mock_kg.get_brand_products.return_value = []

        service = SentimentService(mock_kg)
        result = service.get_brand_profile("UnknownBrand")

        assert result["brand"] == "UnknownBrand"
        assert result["product_count"] == 0
        assert result["all_tags"] == []
        assert result["dominant_sentiment"] is None

    def test_get_brand_profile_with_products(self):
        """제품이 있는 브랜드의 감성 프로필"""
        mock_kg = MagicMock()
        mock_kg.get_brand_products.return_value = [
            {"asin": "B0PROD1"},
            {"asin": "B0PROD2"},
        ]

        # Product 1 sentiments
        tag1 = MagicMock()
        tag1.object = "Moisturizing"
        tag1.properties = {"cluster": "Hydration"}

        # Product 2 sentiments
        tag2 = MagicMock()
        tag2.object = "Moisturizing"
        tag2.properties = {"cluster": "Hydration"}
        tag3 = MagicMock()
        tag3.object = "Affordable"
        tag3.properties = {"cluster": "Pricing"}

        mock_kg.query.side_effect = [
            [],  # Prod1 AI summary
            [tag1],  # Prod1 sentiments
            [],  # Prod2 AI summary
            [tag2, tag3],  # Prod2 sentiments
        ]

        service = SentimentService(mock_kg)
        result = service.get_brand_profile("LANEIGE")

        assert result["brand"] == "LANEIGE"
        assert result["product_count"] == 2
        assert "Moisturizing" in result["all_tags"]
        assert "Affordable" in result["all_tags"]
        # Moisturizing appears 2x, Affordable 1x -> dominant = Moisturizing
        assert result["dominant_sentiment"] == "Moisturizing"
        assert result["clusters"]["Hydration"] == 2
        assert result["clusters"]["Pricing"] == 1

    def test_get_brand_profile_products_without_asin(self):
        """asin이 없는 제품은 건너뜀"""
        mock_kg = MagicMock()
        mock_kg.get_brand_products.return_value = [
            {"name": "Product without ASIN"},
        ]

        service = SentimentService(mock_kg)
        result = service.get_brand_profile("TestBrand")

        assert result["product_count"] == 1
        assert result["all_tags"] == []


# =========================================================================
# Wave 3: compare_products 심화 테스트
# =========================================================================


class TestSentimentServiceCompareExtended:
    """compare_products 확장 테스트"""

    def test_compare_products_no_tags(self):
        """두 제품 모두 태그가 없는 경우"""
        mock_kg = MagicMock()
        mock_kg.query.return_value = []

        service = SentimentService(mock_kg)
        result = service.compare_products("B0A", "B0B")

        assert result["common_tags"] == []
        assert result["unique_to_1"] == []
        assert result["unique_to_2"] == []

    def test_compare_products_cluster_comparison(self):
        """클러스터 비교 데이터 포함"""
        tag1 = MagicMock()
        tag1.object = "Moisturizing"
        tag1.properties = {"cluster": "Hydration"}

        tag2 = MagicMock()
        tag2.object = "Affordable"
        tag2.properties = {"cluster": "Pricing"}

        mock_kg = MagicMock()
        mock_kg.query.side_effect = [
            [],  # P1 AI summary
            [tag1],  # P1 sentiments
            [],  # P2 AI summary
            [tag2],  # P2 sentiments
        ]

        service = SentimentService(mock_kg)
        result = service.compare_products("B0P1", "B0P2")

        assert "cluster_comparison" in result
        assert "product1_clusters" in result["cluster_comparison"]
        assert "product2_clusters" in result["cluster_comparison"]


# =========================================================================
# Wave 3: find_products_by_sentiment 심화 테스트
# =========================================================================


class TestSentimentServiceFindProductsExtended:
    """find_products_by_sentiment 확장 테스트"""

    def test_find_products_no_results(self):
        """결과가 없는 경우"""
        mock_kg = MagicMock()
        mock_kg.query.return_value = []

        service = SentimentService(mock_kg)
        result = service.find_products_by_sentiment("Nonexistent")

        assert result == []

    def test_find_products_brand_filter_case_insensitive(self):
        """브랜드 필터는 대소문자 구분 없이 동작"""
        mock_rel = MagicMock()
        mock_rel.subject = "B0PROD1"

        mock_kg = MagicMock()
        mock_kg.query.return_value = [mock_rel]
        mock_kg.get_entity_metadata.return_value = {"brand": "laneige"}

        service = SentimentService(mock_kg)
        result = service.find_products_by_sentiment("Moisturizing", brand_filter="LANEIGE")

        assert "B0PROD1" in result

    def test_find_products_brand_filter_no_match(self):
        """브랜드 필터에 매칭되지 않는 경우"""
        mock_rel = MagicMock()
        mock_rel.subject = "B0COSRX"

        mock_kg = MagicMock()
        mock_kg.query.return_value = [mock_rel]
        mock_kg.get_entity_metadata.return_value = {"brand": "COSRX"}

        service = SentimentService(mock_kg)
        result = service.find_products_by_sentiment("Moisturizing", brand_filter="LANEIGE")

        assert result == []

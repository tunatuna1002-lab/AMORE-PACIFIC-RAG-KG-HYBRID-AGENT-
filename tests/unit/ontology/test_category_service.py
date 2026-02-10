"""
CategoryService 단위 테스트
"""

from unittest.mock import MagicMock

from src.ontology.category_service import CategoryService


class TestCategoryServiceInit:
    """CategoryService 초기화 테스트"""

    def test_category_service_requires_kg(self):
        """CategoryService는 KnowledgeGraph를 필요로 함"""
        mock_kg = MagicMock()
        service = CategoryService(mock_kg)
        assert service._kg is mock_kg

    def test_category_service_composition(self):
        """CategoryService는 KnowledgeGraph를 composition으로 사용"""
        mock_kg = MagicMock()
        service = CategoryService(mock_kg)

        # 내부에서 KG 메서드 호출 가능
        service._kg.query.return_value = []
        result = service._kg.query(subject="test")
        mock_kg.query.assert_called_once()


class TestCategoryServiceGetHierarchy:
    """CategoryService.get_hierarchy 테스트"""

    def test_get_hierarchy_not_found(self):
        """존재하지 않는 카테고리 조회 시 에러 반환"""
        mock_kg = MagicMock()
        mock_kg.get_entity_metadata.return_value = {}

        service = CategoryService(mock_kg)
        result = service.get_hierarchy("nonexistent")

        assert result["category"] == "nonexistent"
        assert "error" in result

    def test_get_hierarchy_with_metadata(self):
        """메타데이터가 있는 카테고리 조회"""
        mock_kg = MagicMock()
        mock_kg.get_entity_metadata.return_value = {
            "type": "category",
            "name": "Lip Care",
            "level": 2,
            "path": ["beauty", "skin_care", "lip_care"],
        }
        mock_kg.query.return_value = []

        service = CategoryService(mock_kg)
        result = service.get_hierarchy("lip_care")

        assert result["category"] == "lip_care"
        assert result["name"] == "Lip Care"
        assert result["level"] == 2
        assert "ancestors" in result
        assert "descendants" in result


class TestCategoryServiceGetProductContext:
    """CategoryService.get_product_category_context 테스트"""

    def test_get_product_category_context_no_ranks(self):
        """순위 정보가 없는 제품"""
        mock_kg = MagicMock()
        mock_kg.query.return_value = []

        service = CategoryService(mock_kg)
        result = service.get_product_category_context("B0TEST123")

        assert result["asin"] == "B0TEST123"
        assert result["categories"] == []

    def test_get_product_category_context_with_ranks(self):
        """순위 정보가 있는 제품"""

        mock_rel = MagicMock()
        mock_rel.object = "lip_care"
        mock_rel.properties = {"rank": 5}

        mock_kg = MagicMock()
        mock_kg.query.side_effect = [
            [mock_rel],  # RANKED_IN query
            [],  # PARENT_CATEGORY query (for hierarchy)
            [],  # HAS_SUBCATEGORY query (for hierarchy)
        ]
        mock_kg.get_entity_metadata.return_value = {
            "type": "category",
            "name": "Lip Care",
            "level": 2,
            "path": [],
        }

        service = CategoryService(mock_kg)
        result = service.get_product_category_context("B0TEST123")

        assert result["asin"] == "B0TEST123"
        assert len(result["categories"]) == 1
        assert result["categories"][0]["category_id"] == "lip_care"
        assert result["categories"][0]["rank"] == 5


class TestContainerCategoryService:
    """Container를 통한 CategoryService 테스트"""

    def test_container_get_category_service(self):
        """Container에서 CategoryService 획득"""
        from src.infrastructure.container import Container

        Container.reset()
        service = Container.get_category_service()

        from src.ontology.category_service import CategoryService

        assert isinstance(service, CategoryService)

    def test_container_category_service_uses_same_kg(self):
        """CategoryService는 Container의 KG를 사용"""
        from src.infrastructure.container import Container

        Container.reset()
        kg = Container.get_knowledge_graph()
        service = Container.get_category_service()

        assert service._kg is kg

    def test_container_category_service_singleton(self):
        """CategoryService는 싱글톤"""
        from src.infrastructure.container import Container

        Container.reset()
        service1 = Container.get_category_service()
        service2 = Container.get_category_service()

        assert service1 is service2

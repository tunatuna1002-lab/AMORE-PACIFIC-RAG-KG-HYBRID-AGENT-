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


# =========================================================================
# Wave 3: load_hierarchy 테스트
# =========================================================================


class TestCategoryServiceLoadHierarchy:
    """CategoryService.load_hierarchy 테스트"""

    def test_load_hierarchy_file_not_found_returns_zero(self, tmp_path):
        """존재하지 않는 파일 경로 시 0 반환"""
        mock_kg = MagicMock()
        service = CategoryService(mock_kg)
        result = service.load_hierarchy(str(tmp_path / "nonexistent.json"))
        assert result == 0

    def test_load_hierarchy_empty_categories(self, tmp_path):
        """빈 카테고리 데이터 로드"""
        import json

        hierarchy_file = tmp_path / "category_hierarchy.json"
        hierarchy_file.write_text(json.dumps({"categories": {}}), encoding="utf-8")

        mock_kg = MagicMock()
        service = CategoryService(mock_kg)
        result = service.load_hierarchy(str(hierarchy_file))

        assert result == 0
        mock_kg.set_entity_metadata.assert_not_called()

    def test_load_hierarchy_single_root_category(self, tmp_path):
        """루트 카테고리 (부모 없음) 로드"""
        import json

        data = {
            "categories": {
                "beauty": {
                    "name": "Beauty & Personal Care",
                    "amazon_node_id": "beauty",
                    "level": 0,
                    "parent_id": None,
                    "path": ["beauty"],
                    "url": "https://amazon.com/beauty",
                    "children": ["skin_care"],
                }
            }
        }
        hierarchy_file = tmp_path / "hierarchy.json"
        hierarchy_file.write_text(json.dumps(data), encoding="utf-8")

        mock_kg = MagicMock()
        service = CategoryService(mock_kg)
        result = service.load_hierarchy(str(hierarchy_file))

        # Root has no parent -> no relations added
        assert result == 0
        # But metadata should be set
        mock_kg.set_entity_metadata.assert_called_once_with(
            "beauty",
            {
                "type": "category",
                "name": "Beauty & Personal Care",
                "amazon_node_id": "beauty",
                "level": 0,
                "parent_id": None,
                "path": ["beauty"],
                "url": "https://amazon.com/beauty",
                "children": ["skin_care"],
            },
        )

    def test_load_hierarchy_with_parent_child_relations(self, tmp_path):
        """부모-자식 관계가 있는 카테고리 로드"""
        import json

        data = {
            "categories": {
                "beauty": {
                    "name": "Beauty",
                    "amazon_node_id": "beauty",
                    "level": 0,
                    "parent_id": None,
                    "path": [],
                    "url": "",
                    "children": ["skin_care"],
                },
                "skin_care": {
                    "name": "Skin Care",
                    "amazon_node_id": "11060451",
                    "level": 1,
                    "parent_id": "beauty",
                    "path": ["beauty", "skin_care"],
                    "url": "",
                    "children": [],
                },
            }
        }
        hierarchy_file = tmp_path / "hierarchy.json"
        hierarchy_file.write_text(json.dumps(data), encoding="utf-8")

        mock_kg = MagicMock()
        mock_kg.add_relation.return_value = True
        service = CategoryService(mock_kg)
        result = service.load_hierarchy(str(hierarchy_file))

        # skin_care has parent=beauty -> 2 relations (PARENT_CATEGORY + HAS_SUBCATEGORY)
        assert result == 2
        assert mock_kg.add_relation.call_count == 2
        assert mock_kg.set_entity_metadata.call_count == 2

    def test_load_hierarchy_add_relation_fails_gracefully(self, tmp_path):
        """add_relation이 False 반환 시 카운트하지 않음"""
        import json

        data = {
            "categories": {
                "lip_care": {
                    "name": "Lip Care",
                    "amazon_node_id": "3761351",
                    "level": 2,
                    "parent_id": "skin_care",
                    "path": [],
                    "url": "",
                    "children": [],
                },
            }
        }
        hierarchy_file = tmp_path / "hierarchy.json"
        hierarchy_file.write_text(json.dumps(data), encoding="utf-8")

        mock_kg = MagicMock()
        mock_kg.add_relation.return_value = False  # Both fail
        service = CategoryService(mock_kg)
        result = service.load_hierarchy(str(hierarchy_file))

        assert result == 0

    def test_load_hierarchy_multiple_children(self, tmp_path):
        """여러 자식 카테고리가 있는 경우"""
        import json

        data = {
            "categories": {
                "beauty": {
                    "name": "Beauty",
                    "level": 0,
                    "parent_id": None,
                    "path": [],
                    "children": ["skin_care", "makeup"],
                },
                "skin_care": {
                    "name": "Skin Care",
                    "level": 1,
                    "parent_id": "beauty",
                    "path": [],
                    "children": [],
                },
                "makeup": {
                    "name": "Makeup",
                    "level": 1,
                    "parent_id": "beauty",
                    "path": [],
                    "children": [],
                },
            }
        }
        hierarchy_file = tmp_path / "hierarchy.json"
        hierarchy_file.write_text(json.dumps(data), encoding="utf-8")

        mock_kg = MagicMock()
        mock_kg.add_relation.return_value = True
        service = CategoryService(mock_kg)
        result = service.load_hierarchy(str(hierarchy_file))

        # 2 children x 2 relations each = 4
        assert result == 4


# =========================================================================
# Wave 3: get_hierarchy 심화 테스트
# =========================================================================


class TestCategoryServiceGetHierarchyExtended:
    """CategoryService.get_hierarchy 확장 테스트"""

    def test_get_hierarchy_with_ancestors(self):
        """상위 카테고리 탐색"""
        mock_kg = MagicMock()

        # lip_care metadata
        lip_care_meta = {"type": "category", "name": "Lip Care", "level": 2, "path": []}
        skin_care_meta = {"type": "category", "name": "Skin Care", "level": 1, "path": []}
        beauty_meta = {"type": "category", "name": "Beauty", "level": 0, "path": []}

        mock_kg.get_entity_metadata.side_effect = [lip_care_meta, skin_care_meta, beauty_meta]

        # Parent chain: lip_care -> skin_care -> beauty -> (no parent)
        parent_rel_1 = MagicMock()
        parent_rel_1.object = "skin_care"
        parent_rel_2 = MagicMock()
        parent_rel_2.object = "beauty"

        from src.domain.entities import RelationType

        def query_side_effect(subject=None, predicate=None):
            if predicate == RelationType.PARENT_CATEGORY:
                if subject == "lip_care":
                    return [parent_rel_1]
                elif subject == "skin_care":
                    return [parent_rel_2]
                else:
                    return []
            elif predicate == RelationType.HAS_SUBCATEGORY:
                return []
            return []

        mock_kg.query.side_effect = query_side_effect

        service = CategoryService(mock_kg)
        result = service.get_hierarchy("lip_care")

        assert len(result["ancestors"]) == 2
        assert result["ancestors"][0]["id"] == "skin_care"
        assert result["ancestors"][1]["id"] == "beauty"

    def test_get_hierarchy_with_descendants(self):
        """하위 카테고리 탐색"""
        mock_kg = MagicMock()

        skin_care_meta = {"type": "category", "name": "Skin Care", "level": 1, "path": []}
        lip_care_meta = {"type": "category", "name": "Lip Care", "level": 2, "path": []}

        mock_kg.get_entity_metadata.side_effect = [skin_care_meta, lip_care_meta]

        child_rel = MagicMock()
        child_rel.object = "lip_care"

        from src.domain.entities import RelationType

        def query_side_effect(subject=None, predicate=None):
            if predicate == RelationType.PARENT_CATEGORY:
                return []
            elif predicate == RelationType.HAS_SUBCATEGORY:
                if subject == "skin_care":
                    return [child_rel]
            return []

        mock_kg.query.side_effect = query_side_effect

        service = CategoryService(mock_kg)
        result = service.get_hierarchy("skin_care")

        assert len(result["descendants"]) == 1
        assert result["descendants"][0]["id"] == "lip_care"
        assert result["descendants"][0]["name"] == "Lip Care"

    def test_get_hierarchy_metadata_none_returns_error(self):
        """메타데이터가 None인 경우 에러 반환"""
        mock_kg = MagicMock()
        mock_kg.get_entity_metadata.return_value = None

        service = CategoryService(mock_kg)
        result = service.get_hierarchy("unknown")

        assert "error" in result

    def test_get_hierarchy_wrong_type_returns_error(self):
        """type이 category가 아닌 경우 에러 반환"""
        mock_kg = MagicMock()
        mock_kg.get_entity_metadata.return_value = {"type": "brand", "name": "LANEIGE"}

        service = CategoryService(mock_kg)
        result = service.get_hierarchy("laneige")

        assert "error" in result


# =========================================================================
# Wave 3: get_product_category_context 심화 테스트
# =========================================================================


class TestCategoryServiceProductContextExtended:
    """get_product_category_context 확장 테스트"""

    def test_product_with_multiple_categories(self):
        """여러 카테고리에 속한 제품"""
        mock_rel1 = MagicMock()
        mock_rel1.object = "lip_care"
        mock_rel1.properties = {"rank": 5}

        mock_rel2 = MagicMock()
        mock_rel2.object = "skin_care"
        mock_rel2.properties = {"rank": 15}

        mock_kg = MagicMock()

        from src.domain.entities import RelationType

        call_count = {"value": 0}

        def query_side_effect(subject=None, predicate=None, object_=None):
            if predicate == RelationType.RANKED_IN:
                return [mock_rel1, mock_rel2]
            # For hierarchy calls
            return []

        mock_kg.query.side_effect = query_side_effect
        mock_kg.get_entity_metadata.return_value = {
            "type": "category",
            "name": "Test Category",
            "level": 1,
            "path": [],
        }

        service = CategoryService(mock_kg)
        result = service.get_product_category_context("B0MULTI123")

        assert result["asin"] == "B0MULTI123"
        assert len(result["categories"]) == 2

    def test_product_context_with_missing_metadata(self):
        """카테고리 메타데이터가 없는 경우"""
        mock_rel = MagicMock()
        mock_rel.object = "unknown_cat"
        mock_rel.properties = {"rank": 50}

        mock_kg = MagicMock()

        from src.domain.entities import RelationType

        def query_side_effect(subject=None, predicate=None, object_=None):
            if predicate == RelationType.RANKED_IN:
                return [mock_rel]
            return []

        mock_kg.query.side_effect = query_side_effect
        mock_kg.get_entity_metadata.return_value = None

        service = CategoryService(mock_kg)
        result = service.get_product_category_context("B0NOMETA")

        assert result["asin"] == "B0NOMETA"
        assert len(result["categories"]) == 1
        # Name should be empty when metadata is None
        assert result["categories"][0]["name"] == ""

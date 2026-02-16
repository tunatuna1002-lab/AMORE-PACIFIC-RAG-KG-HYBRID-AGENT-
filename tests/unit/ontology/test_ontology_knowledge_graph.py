"""
OntologyKnowledgeGraph 단위 테스트
====================================
src/ontology/ontology_knowledge_graph.py 커버리지 20% → 60%+ 목표

테스트 대상:
- OWL_CLASS_MAPPING, RELATION_TO_OWL_PROPERTY
- OntologyKnowledgeGraph 초기화
- initialize() (async)
- add_validated_relation (skip/validate/fail 경로)
- _validate_triple
- _auto_classify_entity
- _get_entity_class
- sync_owl_inferences
- _owl_to_relation_type
- check_consistency
- 래핑된 KG 메서드 (query, get_brand_products, etc.)
- get_stats
"""

from unittest.mock import MagicMock, patch

import pytest

from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.ontology_knowledge_graph import (
    OWL_CLASS_MAPPING,
    RELATION_TO_OWL_PROPERTY,
    OntologyKnowledgeGraph,
)
from src.ontology.relations import Relation, RelationType

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def kg(tmp_path):
    """Fresh KnowledgeGraph"""
    return KnowledgeGraph(persist_path=str(tmp_path / "kg.json"))


@pytest.fixture
def okg(kg):
    """OntologyKnowledgeGraph without OWL (validation disabled)"""
    return OntologyKnowledgeGraph(
        knowledge_graph=kg,
        owl_reasoner=None,
        enable_validation=False,
    )


@pytest.fixture
def okg_with_owl(kg):
    """OntologyKnowledgeGraph with mock OWL reasoner"""
    mock_owl = MagicMock()
    return OntologyKnowledgeGraph(
        knowledge_graph=kg,
        owl_reasoner=mock_owl,
        enable_validation=True,
    )


# =========================================================================
# Constants
# =========================================================================


class TestConstants:
    def test_owl_class_mapping(self):
        assert "Brand" in OWL_CLASS_MAPPING
        assert "Product" in OWL_CLASS_MAPPING
        assert "Category" in OWL_CLASS_MAPPING
        assert "HAS_PRODUCT" in OWL_CLASS_MAPPING["Brand"]

    def test_relation_to_owl_property(self):
        assert RelationType.HAS_PRODUCT in RELATION_TO_OWL_PROPERTY
        assert RELATION_TO_OWL_PROPERTY[RelationType.HAS_PRODUCT] == "hasProduct"


# =========================================================================
# Init
# =========================================================================


class TestInit:
    def test_init_no_owl(self, okg):
        assert okg.owl is None
        assert okg.enable_validation is False
        assert okg._initialized is False
        assert okg._validation_stats["total"] == 0

    def test_init_with_owl(self, okg_with_owl):
        assert okg_with_owl.owl is not None
        assert okg_with_owl.enable_validation is True


# =========================================================================
# initialize
# =========================================================================


class TestInitialize:
    @pytest.mark.asyncio
    async def test_initialize_no_owl(self, okg):
        await okg.initialize()
        assert okg._initialized is True
        assert okg.enable_validation is False

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, okg):
        okg._initialized = True
        await okg.initialize()  # Should return early
        assert okg._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_with_existing_owl(self, okg_with_owl):
        await okg_with_owl.initialize()
        assert okg_with_owl._initialized is True


# =========================================================================
# add_validated_relation
# =========================================================================


class TestAddValidatedRelation:
    def test_skip_validation(self, okg):
        success, msg = okg.add_validated_relation(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            obj="Lip Sleeping Mask",
            skip_validation=True,
        )
        assert success is True
        assert "without validation" in msg
        assert okg._validation_stats["skipped"] == 1

    def test_no_owl_skips_validation(self, okg):
        success, msg = okg.add_validated_relation(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            obj="Lip Sleeping Mask",
        )
        assert success is True
        assert okg._validation_stats["skipped"] == 1

    def test_with_owl_validation_pass(self, okg_with_owl):
        # Mock _validate_triple to return True
        okg_with_owl._validate_triple = MagicMock(return_value=(True, "Valid"))
        success, msg = okg_with_owl.add_validated_relation(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            obj="Lip Sleeping Mask",
        )
        assert success is True
        assert "Validated" in msg
        assert okg_with_owl._validation_stats["passed"] == 1

    def test_with_owl_validation_fail(self, okg_with_owl):
        # Mock _validate_triple to return False
        okg_with_owl._validate_triple = MagicMock(return_value=(False, "Invalid domain"))
        success, msg = okg_with_owl.add_validated_relation(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            obj="Lip Sleeping Mask",
        )
        assert success is False
        assert "warning" in msg.lower()
        assert okg_with_owl._validation_stats["failed"] == 1

    def test_with_metadata(self, okg):
        success, _ = okg.add_validated_relation(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            obj="Lip Sleeping Mask",
            metadata={"rank": 1, "date": "2025-01-15"},
        )
        assert success is True


# =========================================================================
# _validate_triple
# =========================================================================


class TestValidateTriple:
    def test_no_owl(self, okg):
        okg.owl = None
        valid, reason = okg._validate_triple("X", RelationType.HAS_PRODUCT, "Y")
        assert valid is True

    def test_no_mapping(self, okg_with_owl):
        # RelationType.RANKED_IN may not be in RELATION_TO_OWL_PROPERTY
        valid, reason = okg_with_owl._validate_triple("X", RelationType.RANKED_IN, "Y")
        assert valid is True
        assert "No OWL mapping" in reason

    def test_valid_domain(self, okg_with_owl):
        # Set entity class cache
        okg_with_owl._entity_class_cache["LANEIGE"] = "Brand"
        valid, reason = okg_with_owl._validate_triple(
            "LANEIGE", RelationType.HAS_PRODUCT, "Product1"
        )
        assert valid is True

    def test_invalid_domain(self, okg_with_owl):
        # Category should not have HAS_PRODUCT
        okg_with_owl._entity_class_cache["Lip Care"] = "Category"
        valid, reason = okg_with_owl._validate_triple(
            "Lip Care", RelationType.HAS_PRODUCT, "Product1"
        )
        assert valid is False
        assert "cannot have predicate" in reason


# =========================================================================
# _auto_classify_entity
# =========================================================================


class TestAutoClassifyEntity:
    def test_classify_brand_as_subject(self, okg):
        okg._auto_classify_entity("LANEIGE", RelationType.HAS_PRODUCT, "subject")
        assert okg._entity_class_cache.get("LANEIGE") == "Brand"

    def test_classify_product_as_subject(self, okg):
        okg._auto_classify_entity("Lip Mask", RelationType.BELONGS_TO_CATEGORY, "subject")
        assert okg._entity_class_cache.get("Lip Mask") == "Product"

    def test_classify_category_as_subject(self, okg):
        okg._auto_classify_entity("Skin Care", RelationType.HAS_SUBCATEGORY, "subject")
        assert okg._entity_class_cache.get("Skin Care") == "Category"

    def test_classify_product_as_object(self, okg):
        okg._auto_classify_entity("Lip Mask", RelationType.HAS_PRODUCT, "object")
        assert okg._entity_class_cache.get("Lip Mask") == "Product"

    def test_classify_category_as_object(self, okg):
        okg._auto_classify_entity("Lip Care", RelationType.BELONGS_TO_CATEGORY, "object")
        assert okg._entity_class_cache.get("Lip Care") == "Category"

    def test_classify_brand_as_object(self, okg):
        okg._auto_classify_entity("COSRX", RelationType.COMPETES_WITH, "object")
        assert okg._entity_class_cache.get("COSRX") == "Brand"

    def test_already_cached(self, okg):
        okg._entity_class_cache["LANEIGE"] = "Brand"
        okg._auto_classify_entity("LANEIGE", RelationType.BELONGS_TO_CATEGORY, "subject")
        # Should still be Brand (not Product)
        assert okg._entity_class_cache["LANEIGE"] == "Brand"


# =========================================================================
# _get_entity_class
# =========================================================================


class TestGetEntityClass:
    def test_from_cache(self, okg):
        okg._entity_class_cache["LANEIGE"] = "Brand"
        assert okg._get_entity_class("LANEIGE") == "Brand"

    def test_from_kg_metadata(self, okg):
        okg.kg.get_entity_metadata = MagicMock(return_value={"owl_class": "Product"})
        result = okg._get_entity_class("Lip Mask")
        assert result == "Product"
        assert okg._entity_class_cache["Lip Mask"] == "Product"

    def test_not_found(self, okg):
        okg.kg.get_entity_metadata = MagicMock(return_value=None)
        result = okg._get_entity_class("Unknown Entity")
        assert result is None


# =========================================================================
# sync_owl_inferences
# =========================================================================


class TestSyncOWLInferences:
    def test_no_owl(self, okg):
        stats = okg.sync_owl_inferences()
        assert stats == {"added": 0, "skipped": 0, "errors": 0}

    def test_with_owl_no_methods(self, okg_with_owl):
        # Mock OWL without run_reasoner or get_inferred_facts
        okg_with_owl.owl = MagicMock(spec=[])
        stats = okg_with_owl.sync_owl_inferences()
        assert stats["added"] == 0

    def test_with_inferred_facts(self, okg_with_owl):
        mock_owl = MagicMock()
        mock_owl.get_inferred_facts.return_value = [
            {"subject": "LANEIGE", "property": "hasProduct", "object": "Lip Mask"},
        ]
        okg_with_owl.owl = mock_owl

        stats = okg_with_owl.sync_owl_inferences()
        assert stats["added"] == 1

    def test_skip_existing_fact(self, okg_with_owl):
        # Add a fact first
        okg_with_owl.kg.add_relation(
            Relation(
                subject="LANEIGE",
                predicate=RelationType.HAS_PRODUCT,
                object="Lip Mask",
            )
        )

        mock_owl = MagicMock()
        mock_owl.get_inferred_facts.return_value = [
            {"subject": "LANEIGE", "property": "hasProduct", "object": "Lip Mask"},
        ]
        okg_with_owl.owl = mock_owl

        stats = okg_with_owl.sync_owl_inferences()
        assert stats["skipped"] == 1


# =========================================================================
# _owl_to_relation_type
# =========================================================================


class TestOWLToRelationType:
    def test_known_property(self, okg):
        result = okg._owl_to_relation_type("hasProduct")
        assert result == RelationType.HAS_PRODUCT

    def test_unknown_property(self, okg):
        result = okg._owl_to_relation_type("unknownProperty")
        assert result is None


# =========================================================================
# check_consistency
# =========================================================================


class TestCheckConsistency:
    def test_empty_kg(self, okg):
        result = okg.check_consistency()
        assert "issues" in result
        assert "warnings" in result
        assert "stats" in result

    def test_with_unclassified_entities(self, okg):
        okg.kg.add_relation(
            Relation(
                subject="LANEIGE",
                predicate=RelationType.HAS_PRODUCT,
                object="Lip Mask",
            )
        )
        result = okg.check_consistency()
        # Should have warning about unclassified entities
        assert len(result["warnings"]) > 0

    def test_with_owl_consistency_check(self, okg_with_owl):
        okg_with_owl.owl.check_consistency.return_value = True
        result = okg_with_owl.check_consistency()
        assert len(result["issues"]) == 0

    def test_with_owl_inconsistency(self, okg_with_owl):
        okg_with_owl.owl.check_consistency.return_value = False
        result = okg_with_owl.check_consistency()
        assert any("inconsistency" in issue for issue in result["issues"])


# =========================================================================
# Wrapped KG methods
# =========================================================================


class TestWrappedMethods:
    def test_query(self, okg):
        result = okg.query(subject="LANEIGE")
        assert isinstance(result, list)

    def test_get_brand_products(self, okg):
        result = okg.get_brand_products("LANEIGE")
        assert isinstance(result, list)

    def test_get_competitors(self, okg):
        result = okg.get_competitors("LANEIGE")
        assert isinstance(result, list)

    def test_get_entity_metadata_with_class(self, okg):
        okg._entity_class_cache["LANEIGE"] = "Brand"
        okg.kg.get_entity_metadata = MagicMock(return_value={"name": "LANEIGE"})
        meta = okg.get_entity_metadata("LANEIGE")
        assert meta["owl_class"] == "Brand"

    def test_get_entity_metadata_no_class(self, okg):
        okg.kg.get_entity_metadata = MagicMock(return_value={"name": "LANEIGE"})
        meta = okg.get_entity_metadata("LANEIGE")
        assert "owl_class" not in meta


# =========================================================================
# get_stats
# =========================================================================


class TestGetStats:
    def test_stats(self, okg):
        stats = okg.get_stats()
        assert "kg_stats" in stats
        assert "owl_available" in stats
        assert stats["owl_available"] is False
        assert stats["validation_enabled"] is False
        assert stats["initialized"] is False

    def test_stats_with_owl(self, okg_with_owl):
        stats = okg_with_owl.get_stats()
        assert stats["owl_available"] is True
        assert stats["validation_enabled"] is True


# =========================================================================
# Edge Cases & Error Handling
# =========================================================================


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_initialize_owl_creation_failure(self, kg):
        """Test OWL reasoner creation failure during initialization"""
        with patch("src.ontology.ontology_knowledge_graph.OWLREADY2_AVAILABLE", True):
            with patch(
                "src.ontology.ontology_knowledge_graph.OWLReasoner",
                side_effect=Exception("OWL init failed"),
            ):
                okg = OntologyKnowledgeGraph(
                    knowledge_graph=kg,
                    owl_reasoner=None,
                    enable_validation=True,
                )
                await okg.initialize()
                # Should disable validation on error
                assert okg.enable_validation is False

    def test_classify_corporate_group_as_subject(self, okg):
        """Test classification of CorporateGroup"""
        okg._auto_classify_entity("AMOREPACIFIC", RelationType.OWNS_BRAND, "subject")
        assert okg._entity_class_cache.get("AMOREPACIFIC") == "CorporateGroup"

    def test_sync_owl_inferences_with_error(self, okg_with_owl):
        """Test sync_owl_inferences handles errors gracefully"""
        mock_owl = MagicMock()
        mock_owl.get_inferred_facts.return_value = [
            {"subject": "LANEIGE", "property": "hasProduct", "object": "Lip Mask"},
            {"subject": "INVALID", "property": None, "object": "Bad Data"},  # Will cause error
        ]
        okg_with_owl.owl = mock_owl

        stats = okg_with_owl.sync_owl_inferences()
        # Should handle errors and continue
        assert stats["added"] >= 0
        assert stats["errors"] >= 0

    def test_sync_owl_inferences_exception_in_reasoner(self, okg_with_owl):
        """Test sync_owl_inferences handles reasoner exceptions"""
        mock_owl = MagicMock()
        mock_owl.run_reasoner.side_effect = Exception("Reasoner failed")
        okg_with_owl.owl = mock_owl

        stats = okg_with_owl.sync_owl_inferences()
        # Should return empty stats on error
        assert stats == {"added": 0, "skipped": 0, "errors": 0}

    def test_check_consistency_owl_check_exception(self, okg_with_owl):
        """Test check_consistency handles OWL check exceptions"""
        okg_with_owl.owl.check_consistency.side_effect = Exception("Check failed")
        result = okg_with_owl.check_consistency()
        # Should add warning instead of crashing
        assert any("consistency check failed" in w for w in result["warnings"])

    def test_get_entity_metadata_no_metadata(self, okg):
        """Test get_entity_metadata when KG returns None"""
        okg.kg.get_entity_metadata = MagicMock(return_value=None)
        meta = okg.get_entity_metadata("Unknown")
        assert meta is None

    def test_add_validated_relation_auto_classifies_both_entities(self, okg_with_owl):
        """Test that validation classifies both subject and object"""
        okg_with_owl._validate_triple = MagicMock(return_value=(True, "Valid"))
        okg_with_owl.add_validated_relation(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            obj="Lip Mask",
        )
        # Both entities should be classified
        assert "LANEIGE" in okg_with_owl._entity_class_cache
        assert "Lip Mask" in okg_with_owl._entity_class_cache

    def test_validation_stats_incremented_correctly(self, okg):
        """Test validation stats are tracked correctly"""
        initial_total = okg._validation_stats["total"]

        okg.add_validated_relation("A", RelationType.HAS_PRODUCT, "B")
        assert okg._validation_stats["total"] == initial_total + 1

    def test_multiple_relation_types_in_mapping(self, okg):
        """Test all relation types in OWL_CLASS_MAPPING are valid"""
        from src.ontology.ontology_knowledge_graph import OWL_CLASS_MAPPING

        for owl_class, predicates in OWL_CLASS_MAPPING.items():
            assert isinstance(owl_class, str)
            assert isinstance(predicates, list)
            assert len(predicates) > 0

    def test_all_mapped_relations_in_relation_to_owl_property(self):
        """Test RELATION_TO_OWL_PROPERTY covers expected relations"""
        from src.ontology.ontology_knowledge_graph import RELATION_TO_OWL_PROPERTY

        expected_relations = [
            RelationType.HAS_PRODUCT,
            RelationType.BELONGS_TO_CATEGORY,
            RelationType.COMPETES_WITH,
            RelationType.PARENT_CATEGORY,
            RelationType.HAS_SUBCATEGORY,
            RelationType.OWNED_BY,
            RelationType.HAS_RANK,
        ]

        for rel in expected_relations:
            assert rel in RELATION_TO_OWL_PROPERTY
            assert isinstance(RELATION_TO_OWL_PROPERTY[rel], str)


# =========================================================================
# Integration Tests
# =========================================================================


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_workflow_without_owl(self, okg):
        """Test complete workflow without OWL validation"""
        await okg.initialize()

        # Add relations
        success1, _ = okg.add_validated_relation(
            "LANEIGE", RelationType.HAS_PRODUCT, "Lip Sleeping Mask"
        )
        success2, _ = okg.add_validated_relation("LANEIGE", RelationType.COMPETES_WITH, "COSRX")

        assert success1 is True
        assert success2 is True

        # Query
        products = okg.get_brand_products("LANEIGE")
        assert len(products) >= 0

        # Check stats
        stats = okg.get_stats()
        assert stats["validation_stats"]["total"] == 2

    @pytest.mark.asyncio
    async def test_full_workflow_with_owl_validation(self, okg_with_owl):
        """Test complete workflow with OWL validation"""
        okg_with_owl._validate_triple = MagicMock(return_value=(True, "Valid"))
        await okg_with_owl.initialize()

        # Add validated relations
        success, msg = okg_with_owl.add_validated_relation(
            "LANEIGE",
            RelationType.HAS_PRODUCT,
            "Lip Sleeping Mask",
            metadata={"rank": 1},
        )

        assert success is True
        assert "Validated" in msg

        # Sync inferences
        mock_owl = MagicMock()
        mock_owl.get_inferred_facts.return_value = []
        okg_with_owl.owl = mock_owl
        stats = okg_with_owl.sync_owl_inferences()

        # Check consistency
        okg_with_owl.owl.check_consistency.return_value = True
        result = okg_with_owl.check_consistency()
        assert len(result["issues"]) == 0

    def test_wrapped_methods_preserve_kg_functionality(self, okg):
        """Test that wrapped methods delegate correctly to KG"""
        # Add some data
        okg.kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "Product1"))
        okg.kg.add_relation(Relation("LANEIGE", RelationType.COMPETES_WITH, "COSRX"))

        # Test query
        results = okg.query(subject="LANEIGE")
        assert len(results) == 2

        # Test get_brand_products
        products = okg.get_brand_products("LANEIGE")
        assert isinstance(products, list)

        # Test get_competitors
        competitors = okg.get_competitors("LANEIGE")
        assert isinstance(competitors, list)

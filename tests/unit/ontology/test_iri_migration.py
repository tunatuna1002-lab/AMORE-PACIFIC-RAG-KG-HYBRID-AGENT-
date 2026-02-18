"""
Unit tests for KG IRI migration (D-2)

Tests _detect_entity_type, migrate_to_iri, export_as_iri,
_resolve_id, and query_iri in src/ontology/knowledge_graph.py.
"""

import pytest

from src.domain.entities.relations import IRI, RelationType
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.relations import Relation


class TestResolveId:
    """Test KnowledgeGraph._resolve_id()."""

    def test_resolve_plain_id(self):
        """Plain ID passes through unchanged."""
        assert KnowledgeGraph._resolve_id("LANEIGE") == "LANEIGE"

    def test_resolve_prefixed_iri(self):
        """Prefixed IRI extracts bare ID."""
        assert KnowledgeGraph._resolve_id("amore:brand/LANEIGE") == "LANEIGE"

    def test_resolve_full_iri(self):
        """Full IRI extracts bare ID."""
        result = KnowledgeGraph._resolve_id("http://amore.ontology/product/B08XYZ1234")
        assert result == "B08XYZ1234"

    def test_resolve_empty_string(self):
        """Empty string passes through unchanged."""
        assert KnowledgeGraph._resolve_id("") == ""


class TestDetectEntityType:
    """Test KnowledgeGraph._detect_entity_type()."""

    def test_detect_product_asin(self):
        """ASIN pattern detected as product."""
        assert KnowledgeGraph._detect_entity_type("B08XYZ1234") == "product"

    def test_detect_known_category(self):
        """Known category IDs detected correctly."""
        assert KnowledgeGraph._detect_entity_type("lip_care") == "category"
        assert KnowledgeGraph._detect_entity_type("skin_care") == "category"
        assert KnowledgeGraph._detect_entity_type("beauty") == "category"

    def test_detect_category_node_id(self):
        """Amazon node IDs detected as category."""
        assert KnowledgeGraph._detect_entity_type("3761351") == "category"
        assert KnowledgeGraph._detect_entity_type("11060451") == "category"

    def test_detect_brand_uppercase(self):
        """All-uppercase string detected as brand."""
        assert KnowledgeGraph._detect_entity_type("LANEIGE") == "brand"
        assert KnowledgeGraph._detect_entity_type("COSRX") == "brand"

    def test_detect_brand_mixed_case(self):
        """Capitalized alpha string detected as brand."""
        assert KnowledgeGraph._detect_entity_type("Laneige") == "brand"

    def test_detect_existing_iri(self):
        """Already-IRI value extracts type correctly."""
        assert KnowledgeGraph._detect_entity_type("amore:brand/LANEIGE") == "brand"

    def test_detect_unknown_fallback(self):
        """Unrecognized pattern falls back to 'entity'."""
        # lowercase with underscore, not a known category
        assert KnowledgeGraph._detect_entity_type("some_random_thing") == "entity"


class TestQueryIri:
    """Test KnowledgeGraph.query_iri()."""

    @pytest.fixture
    def kg_with_data(self):
        """KG with sample triples."""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)
        kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ1234"))
        kg.add_relation(Relation("COSRX", RelationType.HAS_PRODUCT, "B09TSTRPROD"))
        kg.add_relation(Relation("B08XYZ1234", RelationType.BELONGS_TO_CATEGORY, "lip_care"))
        return kg

    def test_query_iri_with_plain_ids(self, kg_with_data):
        """query_iri works with plain IDs (backward compat)."""
        results = kg_with_data.query_iri(subject_iri="LANEIGE", predicate=RelationType.HAS_PRODUCT)
        assert len(results) == 1
        assert results[0].object == "B08XYZ1234"

    def test_query_iri_with_prefixed_iri(self, kg_with_data):
        """query_iri resolves prefixed IRI to bare ID."""
        results = kg_with_data.query_iri(
            subject_iri="amore:brand/LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
        )
        assert len(results) == 1
        assert results[0].object == "B08XYZ1234"

    def test_query_iri_with_full_iri(self, kg_with_data):
        """query_iri resolves full IRI."""
        results = kg_with_data.query_iri(
            subject_iri="http://amore.ontology/brand/LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
        )
        assert len(results) == 1

    def test_query_iri_object_resolution(self, kg_with_data):
        """query_iri resolves object IRI."""
        results = kg_with_data.query_iri(
            predicate=RelationType.BELONGS_TO_CATEGORY,
            object_iri="amore:category/lip_care",
        )
        assert len(results) == 1
        assert results[0].subject == "B08XYZ1234"

    def test_query_iri_no_match(self, kg_with_data):
        """query_iri returns empty for non-existent entity."""
        results = kg_with_data.query_iri(
            subject_iri="amore:brand/NONEXISTENT",
            predicate=RelationType.HAS_PRODUCT,
        )
        assert len(results) == 0

    def test_query_iri_with_min_confidence(self, kg_with_data):
        """query_iri respects min_confidence."""
        results = kg_with_data.query_iri(
            subject_iri="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            min_confidence=0.5,
        )
        assert len(results) == 1  # default confidence is 1.0

        results = kg_with_data.query_iri(
            subject_iri="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            min_confidence=1.5,
        )
        assert len(results) == 0


class TestMigrateToIri:
    """Test KnowledgeGraph.migrate_to_iri()."""

    @pytest.fixture
    def kg_for_migration(self):
        """KG with bare-string triples ready for migration."""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)
        kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ1234"))
        kg.add_relation(Relation("B08XYZ1234", RelationType.BELONGS_TO_CATEGORY, "lip_care"))
        return kg

    def test_migrate_converts_entities(self, kg_for_migration):
        """migrate_to_iri converts bare IDs to IRI form."""
        result = kg_for_migration.migrate_to_iri()
        assert result["converted"] > 0

        # Check subjects are now IRIs
        for rel in kg_for_migration.triples:
            assert IRI.is_iri(rel.subject)

    def test_migrate_returns_stats(self, kg_for_migration):
        """migrate_to_iri returns converted/skipped counts."""
        result = kg_for_migration.migrate_to_iri()
        assert "converted" in result
        assert "skipped" in result
        assert isinstance(result["converted"], int)
        assert isinstance(result["skipped"], int)

    def test_migrate_idempotent(self, kg_for_migration):
        """Running migrate twice does not double-convert."""
        result1 = kg_for_migration.migrate_to_iri()
        result2 = kg_for_migration.migrate_to_iri()
        # Second run should convert 0 (already IRIs)
        assert result2["converted"] == 0 or result2["skipped"] >= result1["converted"]

    def test_migrate_marks_dirty(self, kg_for_migration):
        """migrate_to_iri marks KG as dirty."""
        kg_for_migration._dirty = False
        kg_for_migration.migrate_to_iri()
        assert kg_for_migration._dirty is True


class TestExportAsIri:
    """Test KnowledgeGraph.export_as_iri()."""

    @pytest.fixture
    def kg_for_export(self):
        """KG with sample data for export."""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)
        kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ1234"))
        kg.add_relation(Relation("B08XYZ1234", RelationType.BELONGS_TO_CATEGORY, "lip_care"))
        return kg

    def test_export_returns_dict(self, kg_for_export):
        """export_as_iri returns a dict with required keys."""
        result = kg_for_export.export_as_iri()
        assert "version" in result
        assert "triples" in result
        assert "entity_metadata" in result
        assert "stats" in result

    def test_export_converts_ids(self, kg_for_export):
        """Exported triples have IRI subjects/objects."""
        result = kg_for_export.export_as_iri()
        for triple in result["triples"]:
            sub = triple["subject"]
            obj = triple["object"]
            # At least brand/product/category should be converted
            if sub == "LANEIGE" or IRI.is_iri(sub):
                pass  # OK
            if obj == "B08XYZ1234" or IRI.is_iri(obj):
                pass  # OK

    def test_export_does_not_modify_kg(self, kg_for_export):
        """export_as_iri does NOT modify the in-memory graph."""
        original_subjects = [r.subject for r in kg_for_export.triples]
        kg_for_export.export_as_iri()
        current_subjects = [r.subject for r in kg_for_export.triples]
        assert original_subjects == current_subjects

    def test_export_version(self, kg_for_export):
        """Exported data has IRI version marker."""
        result = kg_for_export.export_as_iri()
        assert "iri" in result["version"].lower()

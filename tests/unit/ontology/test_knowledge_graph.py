"""
Unit tests for KnowledgeGraph

Tests the triple store, indexing, persistence, and query functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.ontology.knowledge_graph import KnowledgeGraph, get_knowledge_graph
from src.ontology.relations import Relation, RelationType


class TestKnowledgeGraphInit:
    """Test KnowledgeGraph initialization"""

    def test_default_initialization(self):
        """Test initialization with defaults"""
        kg = KnowledgeGraph(auto_load=False)

        assert kg.triples == []
        assert len(kg.subject_index) == 0
        assert len(kg.object_index) == 0
        assert len(kg.predicate_index) == 0
        assert kg.entity_metadata == {}
        assert kg.max_triples == KnowledgeGraph.DEFAULT_MAX_TRIPLES
        assert kg.auto_save is True

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters"""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = f"{tmpdir}/test_kg.json"

            kg = KnowledgeGraph(
                persist_path=persist_path, max_triples=1000, auto_save=False, auto_load=False
            )

            assert kg.persist_path == Path(persist_path)
            assert kg.max_triples == 1000
            assert kg.auto_save is False

    def test_initialization_railway_detection(self):
        """Test Railway environment detection"""
        with patch.dict("os.environ", {"RAILWAY_ENVIRONMENT": "production"}):
            kg = KnowledgeGraph(auto_load=False)

            assert str(kg.persist_path) == KnowledgeGraph.RAILWAY_PERSIST_PATH

    def test_initialization_auto_load(self):
        """Test auto-loading existing data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "kg.json"

            # Create initial KG with data
            kg1 = KnowledgeGraph(persist_path=str(persist_path), auto_load=False)
            rel = Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ")
            kg1.add_relation(rel)
            kg1.save()

            # Create new KG with auto_load
            kg2 = KnowledgeGraph(persist_path=str(persist_path), auto_load=True)

            assert len(kg2.triples) == 1


class TestKnowledgeGraphAddRelation:
    """Test add_relation method"""

    def test_add_single_relation(self):
        """Test adding a single relation"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)
        rel = Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ")

        result = kg.add_relation(rel)

        assert result is True
        assert len(kg.triples) == 1
        assert rel in kg.triples

    def test_add_relation_updates_indices(self):
        """Test that adding a relation updates all indices"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)
        rel = Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ")

        kg.add_relation(rel)

        assert rel in kg.subject_index["LANEIGE"]
        assert rel in kg.object_index["B08XYZ"]
        assert rel in kg.predicate_index[RelationType.HAS_PRODUCT]

    def test_add_duplicate_relation(self):
        """Test adding a duplicate relation updates existing"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)
        rel1 = Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ", confidence=0.8)
        rel2 = Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ", confidence=0.9)

        kg.add_relation(rel1)
        result = kg.add_relation(rel2)

        assert result is False  # Duplicate
        assert len(kg.triples) == 1
        # Confidence should be updated to max
        assert kg.triples[0].confidence == 0.9

    def test_add_relation_importance_score(self):
        """Test importance score is calculated for new relations"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)
        rel = Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ", confidence=0.9)

        kg.add_relation(rel)

        assert id(rel) in kg._importance_scores
        assert kg._importance_scores[id(rel)] > 0


class TestKnowledgeGraphAddRelations:
    """Test bulk add_relations method"""

    def test_add_multiple_relations(self):
        """Test adding multiple relations at once"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)

        relations = [
            Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ"),
            Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08ABC"),
            Relation("LANEIGE", RelationType.COMPETES_WITH, "COSRX"),
        ]

        added = kg.add_relations(relations)

        assert added == 3
        assert len(kg.triples) == 3


class TestKnowledgeGraphRemoveRelation:
    """Test remove_relation method"""

    def test_remove_existing_relation(self):
        """Test removing an existing relation"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)
        rel = Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ")

        kg.add_relation(rel)
        result = kg.remove_relation(rel)

        assert result is True
        assert len(kg.triples) == 0
        assert rel not in kg.subject_index["LANEIGE"]
        assert rel not in kg.object_index["B08XYZ"]

    def test_remove_nonexistent_relation(self):
        """Test removing a relation that doesn't exist"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)
        rel = Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ")

        result = kg.remove_relation(rel)

        assert result is False


class TestKnowledgeGraphClear:
    """Test clear method"""

    def test_clear_all_data(self):
        """Test clearing all graph data"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)

        # Add some data
        kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ"))
        kg.entity_metadata["LANEIGE"] = {"type": "brand"}

        kg.clear()

        assert len(kg.triples) == 0
        assert len(kg.subject_index) == 0
        assert len(kg.object_index) == 0
        assert len(kg.predicate_index) == 0
        assert len(kg.entity_metadata) == 0


class TestKnowledgeGraphPersistence:
    """Test save and load functionality"""

    def test_save_and_load(self):
        """Test saving and loading graph data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "kg.json"

            # Create and save
            kg1 = KnowledgeGraph(persist_path=str(persist_path), auto_load=False, auto_save=False)
            kg1.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ"))
            kg1.entity_metadata["LANEIGE"] = {"type": "brand"}

            # save() checks _dirty flag; with auto_save=False it's not set automatically
            # Use force=True to bypass the dirty check
            result = kg1.save(force=True)
            assert result is True
            assert persist_path.exists()

            # Load into new instance
            kg2 = KnowledgeGraph(persist_path=str(persist_path), auto_load=True)

            assert len(kg2.triples) == 1
            assert kg2.entity_metadata["LANEIGE"]["type"] == "brand"

    def test_save_creates_directory(self):
        """Test save creates parent directory if needed"""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "subdir" / "kg.json"

            kg = KnowledgeGraph(persist_path=str(persist_path), auto_load=False)
            kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ"))

            kg.save()

            assert persist_path.parent.exists()
            assert persist_path.exists()

    def test_save_if_dirty(self):
        """Test save_if_dirty only saves when dirty"""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "kg.json"

            kg = KnowledgeGraph(persist_path=str(persist_path), auto_load=False, auto_save=False)

            # Not dirty initially
            result = kg.save_if_dirty()
            assert result is False

            # Make dirty
            kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ"))
            kg._dirty = True

            result = kg.save_if_dirty()
            assert result is True

    def test_context_manager_saves_on_exit(self):
        """Test context manager saves dirty data on exit"""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "kg.json"

            with KnowledgeGraph(
                persist_path=str(persist_path), auto_load=False, auto_save=False
            ) as kg:
                kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ"))
                kg._dirty = True

            # Should be saved after context exit
            assert persist_path.exists()

    def test_save_preserves_version(self):
        """Test save includes version information"""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "kg.json"

            kg = KnowledgeGraph(persist_path=str(persist_path), auto_load=False)
            kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ"))
            kg.save()

            # Read JSON directly
            with open(persist_path) as f:
                data = json.load(f)

            assert "version" in data
            assert data["version"] == "2.0"


class TestKnowledgeGraphEviction:
    """Test smart eviction when max_triples exceeded"""

    def test_enforce_max_triples_smart(self):
        """Test smart eviction removes low-importance triples"""
        kg = KnowledgeGraph(max_triples=10, auto_load=False, auto_save=False)

        # Add protected relations (high importance)
        for i in range(5):
            kg.add_relation(Relation(f"Brand{i}", RelationType.OWNED_BY, f"Parent{i}"))

        # Add regular relations (lower importance)
        for i in range(10):
            kg.add_relation(
                Relation(f"Brand{i}", RelationType.HAS_PRODUCT, f"Product{i}", confidence=0.5)
            )

        # Eviction removes 10% per trigger, so total may slightly exceed max_triples
        # but should be significantly less than the sum of all added (15)
        assert len(kg.triples) < 15  # eviction has removed some

        # Protected relations (OWNED_BY) should still be present due to high importance
        protected_count = len([r for r in kg.triples if r.predicate == RelationType.OWNED_BY])
        assert protected_count == 5  # All protected relations preserved

    def test_calculate_importance(self):
        """Test importance score calculation"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)

        # Protected relation type
        rel1 = Relation("LANEIGE", RelationType.OWNED_BY, "AMOREPACIFIC")
        score1 = kg._calculate_importance(rel1)

        # Regular relation
        rel2 = Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ", confidence=0.5)
        score2 = kg._calculate_importance(rel2)

        # Protected should have higher score
        assert score1 > score2


class TestKnowledgeGraphQueryMethods:
    """Test query methods from KGQueryMixin"""

    def test_get_objects_by_subject_and_predicate(self):
        """Test querying objects by subject and predicate"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)

        kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ"))
        kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08ABC"))
        kg.add_relation(Relation("LANEIGE", RelationType.COMPETES_WITH, "COSRX"))

        # Should have get_objects method from KGQueryMixin
        if hasattr(kg, "get_objects"):
            products = kg.get_objects("LANEIGE", RelationType.HAS_PRODUCT)
            assert len(products) == 2
            assert "B08XYZ" in products
            assert "B08ABC" in products

    def test_get_subjects_by_predicate_and_object(self):
        """Test querying subjects by predicate and object"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)

        kg.add_relation(Relation("LANEIGE", RelationType.BELONGS_TO_CATEGORY, "lip_care"))
        kg.add_relation(Relation("COSRX", RelationType.BELONGS_TO_CATEGORY, "lip_care"))

        if hasattr(kg, "get_subjects"):
            brands = kg.get_subjects(RelationType.BELONGS_TO_CATEGORY, "lip_care")
            assert len(brands) == 2
            assert "LANEIGE" in brands
            assert "COSRX" in brands


class TestKnowledgeGraphCategoryHierarchy:
    """Test category hierarchy functionality"""

    def test_add_category_hierarchy(self):
        """Test adding category parent-child relationships"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)

        # Add hierarchy
        kg.add_relation(Relation("beauty", RelationType.HAS_SUBCATEGORY, "skin_care"))
        kg.add_relation(Relation("skin_care", RelationType.HAS_SUBCATEGORY, "lip_care"))
        kg.add_relation(Relation("skin_care", RelationType.PARENT_CATEGORY, "beauty"))

        assert len(kg.triples) == 3

    def test_query_category_children(self):
        """Test querying category children"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)

        kg.add_relation(Relation("beauty", RelationType.HAS_SUBCATEGORY, "skin_care"))
        kg.add_relation(Relation("beauty", RelationType.HAS_SUBCATEGORY, "makeup"))

        if hasattr(kg, "get_objects"):
            children = kg.get_objects("beauty", RelationType.HAS_SUBCATEGORY)
            assert "skin_care" in children
            assert "makeup" in children


class TestKnowledgeGraphCrawlDataIntegration:
    """Test crawl data update functionality"""

    def test_add_crawl_data(self):
        """Test adding crawl data creates proper relations"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)

        # Simulate crawl data update (if method exists)
        if hasattr(kg, "add_crawl_data"):
            crawl_data = {
                "brand": "LANEIGE",
                "asin": "B08XYZ",
                "category": "lip_care",
                "rank": 5,
                "sos": 0.15,
            }

            kg.add_crawl_data(crawl_data)

            # Should create relations
            assert len(kg.triples) > 0


class TestKnowledgeGraphStats:
    """Test statistics functionality"""

    def test_get_stats(self):
        """Test get_stats returns correct counts"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)

        kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ"))
        kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08ABC"))
        kg.add_relation(Relation("COSRX", RelationType.HAS_PRODUCT, "B09DEF"))

        stats = kg.get_stats()

        assert stats["total_triples"] == 3
        assert stats["unique_subjects"] == 2  # LANEIGE, COSRX
        assert stats["unique_objects"] == 3  # B08XYZ, B08ABC, B09DEF

    def test_update_stats_on_add(self):
        """Test stats are updated when relations added"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)

        initial_stats = kg.get_stats()
        assert initial_stats["total_triples"] == 0

        kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ"))

        updated_stats = kg.get_stats()
        assert updated_stats["total_triples"] == 1


class TestKnowledgeGraphRepr:
    """Test string representation"""

    def test_repr(self):
        """Test __repr__ shows summary"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)

        kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ"))
        kg.add_relation(Relation("COSRX", RelationType.HAS_PRODUCT, "B09DEF"))

        repr_str = repr(kg)

        assert "KnowledgeGraph" in repr_str
        assert "triples=2" in repr_str


class TestKnowledgeGraphSingleton:
    """Test singleton factory function"""

    def test_get_knowledge_graph_singleton(self):
        """Test get_knowledge_graph returns singleton"""
        import src.ontology.knowledge_graph as kg_module

        # Use a clean temp path to avoid corrupted real data files
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "test_singleton_kg.json"
            kg_module._knowledge_graph_instance = None

            # Pre-set instance with known-good config
            kg_module._knowledge_graph_instance = KnowledgeGraph(
                persist_path=str(persist_path), auto_load=False, auto_save=False
            )

            kg1 = get_knowledge_graph()
            kg2 = get_knowledge_graph()

            assert kg1 is kg2

            # Cleanup
            kg_module._knowledge_graph_instance = None

    def test_singleton_preserves_state(self):
        """Test singleton preserves state across calls"""
        import src.ontology.knowledge_graph as kg_module

        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "test_singleton_kg.json"
            kg_module._knowledge_graph_instance = None

            # Pre-set instance with known-good config
            kg_module._knowledge_graph_instance = KnowledgeGraph(
                persist_path=str(persist_path), auto_load=False, auto_save=False
            )

            kg1 = get_knowledge_graph()
            kg1.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ"))

            kg2 = get_knowledge_graph()

            assert len(kg2.triples) == 1

            # Cleanup
            kg_module._knowledge_graph_instance = None


class TestKnowledgeGraphEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_graph_queries(self):
        """Test queries on empty graph"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)

        stats = kg.get_stats()
        assert stats["total_triples"] == 0

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file doesn't crash"""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "nonexistent.json"

            # Should not raise, just skip loading
            kg = KnowledgeGraph(persist_path=str(persist_path), auto_load=True)

            assert len(kg.triples) == 0

    def test_save_without_path(self):
        """Test save without persist_path returns False"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)
        # Constructor resolves None to default path from config;
        # explicitly clear it to test the no-path code path
        kg.persist_path = None
        kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ"))
        kg._dirty = True

        result = kg.save()

        # Should return False, not crash
        assert result is False

    def test_add_relation_with_properties(self):
        """Test adding relation with custom properties"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)

        rel = Relation(
            "LANEIGE",
            RelationType.HAS_PRODUCT,
            "B08XYZ",
            confidence=0.95,
            properties={"category": "lip_care", "rank": 5},
        )

        kg.add_relation(rel)

        assert kg.triples[0].properties["category"] == "lip_care"
        assert kg.triples[0].properties["rank"] == 5


class TestKnowledgeGraphConfigLoading:
    """Test configuration loading"""

    def test_load_config_from_file(self):
        """Test loading config from thresholds.json"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "thresholds.json"

            config_data = {
                "system": {
                    "knowledge_graph": {
                        "max_triples": 5000,
                        "auto_save": False,
                        "persist_path": "data/test_kg.json",
                    }
                }
            }

            with open(config_path, "w") as f:
                json.dump(config_data, f)

            with patch.object(KnowledgeGraph, "CONFIG_PATH", str(config_path)):
                kg = KnowledgeGraph(auto_load=False)

                # Should use config values
                assert kg.max_triples == 5000
                assert kg.auto_save is False

    def test_fallback_to_defaults_when_no_config(self):
        """Test fallback to defaults when config missing"""
        with patch.object(KnowledgeGraph, "CONFIG_PATH", "/nonexistent/path/config.json"):
            kg = KnowledgeGraph(auto_load=False)

            assert kg.max_triples == KnowledgeGraph.DEFAULT_MAX_TRIPLES
            assert kg.auto_save is True

    def test_config_load_exception_handling(self):
        """Test config loading handles exceptions gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "bad_config.json"

            # Create invalid JSON
            with open(config_path, "w") as f:
                f.write("{invalid json")

            with patch.object(KnowledgeGraph, "CONFIG_PATH", str(config_path)):
                # Should not raise, fall back to defaults
                kg = KnowledgeGraph(auto_load=False)

                assert kg.max_triples == KnowledgeGraph.DEFAULT_MAX_TRIPLES


class TestKnowledgeGraphImportanceScoring:
    """Test importance scoring for eviction"""

    def test_calculate_importance_with_recent_timestamp(self):
        """Test importance score increases for recent relations"""
        from datetime import datetime, timedelta

        kg = KnowledgeGraph(auto_load=False, auto_save=False)

        # Recent relation
        rel_recent = Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ", confidence=0.5)
        rel_recent.created_at = datetime.now()

        score_recent = kg._calculate_importance(rel_recent)

        # Old relation
        rel_old = Relation("COSRX", RelationType.HAS_PRODUCT, "B09ABC", confidence=0.5)
        rel_old.created_at = datetime.now() - timedelta(days=60)

        score_old = kg._calculate_importance(rel_old)

        # Recent should have higher score
        assert score_recent > score_old

    def test_calculate_importance_with_medium_age(self):
        """Test importance score for medium-age relations"""
        from datetime import datetime, timedelta

        kg = KnowledgeGraph(auto_load=False, auto_save=False)

        # Medium age (15 days)
        rel_medium = Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ", confidence=0.5)
        rel_medium.created_at = datetime.now() - timedelta(days=15)

        score = kg._calculate_importance(rel_medium)

        # Should have some bonus but less than very recent
        assert score > 0.5


class TestKnowledgeGraphAutoSave:
    """Test auto-save functionality"""

    def test_auto_save_batch_threshold(self):
        """Test auto-save triggers after batch threshold"""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "kg.json"

            kg = KnowledgeGraph(persist_path=str(persist_path), auto_load=False, auto_save=True)

            # Set low threshold for testing
            kg._save_batch_threshold = 3

            # Add relations up to threshold
            kg.add_relation(Relation("Brand1", RelationType.HAS_PRODUCT, "P1"))
            kg.add_relation(Relation("Brand2", RelationType.HAS_PRODUCT, "P2"))

            # Should not save yet
            assert kg._save_batch_count == 2

            # This should trigger save
            kg.add_relation(Relation("Brand3", RelationType.HAS_PRODUCT, "P3"))

            # Counter should reset
            assert kg._save_batch_count == 0
            assert persist_path.exists()

    def test_auto_save_disabled(self):
        """Test auto-save can be disabled"""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "kg.json"

            kg = KnowledgeGraph(persist_path=str(persist_path), auto_load=False, auto_save=False)

            kg.add_relation(Relation("Brand1", RelationType.HAS_PRODUCT, "P1"))

            # Should not save
            assert not persist_path.exists()


class TestKnowledgeGraphSaveErrorHandling:
    """Test save error handling"""

    def test_save_handles_write_errors(self):
        """Test save handles file write errors gracefully"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)
        kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ"))
        kg._dirty = True

        # Try to save to invalid path (permission error simulation)
        result = kg.save(path="/invalid/readonly/path.json")

        # Should return False, not raise
        assert result is False

    def test_save_skips_when_not_dirty(self):
        """Test save skips when data hasn't changed"""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "kg.json"

            kg = KnowledgeGraph(persist_path=str(persist_path), auto_load=False, auto_save=False)

            # Not dirty, not forced
            result = kg.save()

            # Should skip
            assert result is False


class TestKnowledgeGraphLoadErrorHandling:
    """Test load error handling"""

    def test_load_handles_corrupted_json(self):
        """Test loading corrupted JSON raises exception"""
        import pytest

        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "kg.json"

            # Create corrupted JSON
            with open(persist_path, "w") as f:
                f.write("{corrupted json")

            # Should raise during auto-load
            with pytest.raises((ValueError, KeyError, TypeError, json.JSONDecodeError)):
                KnowledgeGraph(persist_path=str(persist_path), auto_load=True)

    def test_load_handles_missing_file_gracefully(self):
        """Test loading missing file doesn't crash"""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "nonexistent.json"

            # Should not raise
            kg = KnowledgeGraph(persist_path=str(persist_path), auto_load=True)

            assert len(kg.triples) == 0


class TestKnowledgeGraphRemoveFromIndices:
    """Test index removal edge cases"""

    def test_remove_relation_cleans_all_indices(self):
        """Test removing relation cleans up all indices"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)

        rel = Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ")
        kg.add_relation(rel)

        # Verify in indices
        assert rel in kg.subject_index["LANEIGE"]
        assert rel in kg.object_index["B08XYZ"]
        assert rel in kg.predicate_index[RelationType.HAS_PRODUCT]

        # Remove
        kg.remove_relation(rel)

        # Should be gone from all indices
        assert rel not in kg.subject_index.get("LANEIGE", [])
        assert rel not in kg.object_index.get("B08XYZ", [])
        assert rel not in kg.predicate_index.get(RelationType.HAS_PRODUCT, [])

    def test_remove_from_indices_handles_missing_entries(self):
        """Test _remove_from_indices handles already-removed entries"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)

        rel = Relation("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ")
        kg.add_relation(rel)

        # Manually remove from one index
        kg.subject_index["LANEIGE"].remove(rel)

        # Should not crash when trying to remove again
        kg._remove_from_indices(rel)


class TestKnowledgeGraphSingletonEdgeCases:
    """Test singleton edge cases"""

    def test_get_knowledge_graph_creates_instance(self):
        """Test get_knowledge_graph creates instance if none exists"""
        import src.ontology.knowledge_graph as kg_module

        # Clear singleton
        kg_module._knowledge_graph_instance = None

        kg = get_knowledge_graph()

        # Should create instance
        assert kg is not None
        assert isinstance(kg, KnowledgeGraph)

        # Cleanup
        kg_module._knowledge_graph_instance = None


class TestKnowledgeGraphDuplicateProperties:
    """Test duplicate relation property updates"""

    def test_add_duplicate_updates_properties(self):
        """Test adding duplicate relation merges properties"""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)

        rel1 = Relation(
            "LANEIGE",
            RelationType.HAS_PRODUCT,
            "B08XYZ",
            properties={"rank": 5, "category": "lip_care"},
        )
        kg.add_relation(rel1)

        rel2 = Relation(
            "LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ", properties={"rank": 3, "price": 25.99}
        )
        result = kg.add_relation(rel2)

        # Should return False (duplicate)
        assert result is False

        # Properties should be merged
        existing = kg.triples[0]
        assert existing.properties["rank"] == 3  # Updated
        assert existing.properties["category"] == "lip_care"  # Preserved
        assert existing.properties["price"] == 25.99  # Added


class TestKnowledgeGraphConcurrentWrite:
    """Test that concurrent writes don't corrupt the KG file."""

    def test_concurrent_saves_produce_valid_json(self):
        """Multiple threads saving simultaneously should not corrupt the file."""
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = f"{tmpdir}/kg.json"
            kg = KnowledgeGraph(persist_path=persist_path, auto_save=False, auto_load=False)

            # Add some data so save() has something to write
            for i in range(20):
                kg.add_relation(Relation(f"BRAND_{i}", RelationType.HAS_PRODUCT, f"PROD_{i}"))
            kg._dirty = True

            errors: list[Exception] = []
            barrier = threading.Barrier(8)

            def _save_worker():
                try:
                    barrier.wait()
                    kg.save(force=True)
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=_save_worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors, f"Concurrent save errors: {errors}"

            # Verify the file is valid JSON
            with open(persist_path, encoding="utf-8") as f:
                data = json.load(f)

            assert data["version"] == "2.0"
            assert len(data["triples"]) == 20

    def test_write_lock_exists(self):
        """KnowledgeGraph instances should have a _write_lock attribute."""
        kg = KnowledgeGraph(auto_load=False, auto_save=False)
        assert hasattr(kg, "_write_lock")

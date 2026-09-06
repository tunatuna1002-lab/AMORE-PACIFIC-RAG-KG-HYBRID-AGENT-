"""Bug D26 (dirty tracking with auto_save=False) + case-insensitive brand lookups."""

import pytest

from src.domain.entities.relations import Relation, RelationType
from src.ontology.knowledge_graph import KnowledgeGraph


@pytest.fixture
def kg(tmp_path) -> KnowledgeGraph:
    return KnowledgeGraph(persist_path=str(tmp_path / "kg.json"), auto_load=False, auto_save=False)


# --- D26 ----------------------------------------------------------------------


def test_add_relation_marks_dirty_without_auto_save(kg, tmp_path):
    assert kg.save() is False  # nothing to save yet
    kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B0LANE1"))
    assert kg._dirty is True
    assert kg.save() is True
    assert (tmp_path / "kg.json").exists()
    assert kg._dirty is False
    assert kg.save() is False  # clean again -> no-op


def test_set_entity_metadata_marks_dirty_without_auto_save(kg):
    kg.set_entity_metadata("LANEIGE", {"type": "brand"})
    assert kg._dirty is True
    assert kg.save() is True


def test_remove_relation_and_clear_mark_dirty(kg):
    rel = Relation("LANEIGE", RelationType.HAS_PRODUCT, "B0LANE1")
    kg.add_relation(rel)
    kg.save()
    kg.remove_relation(rel)
    assert kg._dirty is True
    kg.save()
    kg.clear()
    assert kg._dirty is True


# --- brand casing -------------------------------------------------------------


@pytest.fixture
def seeded(kg) -> KnowledgeGraph:
    kg.add_relation(
        Relation("LANEIGE", RelationType.HAS_PRODUCT, "B0LANE1", properties={"rank": 1})
    )
    kg.add_relation(
        Relation("LANEIGE", RelationType.DIRECT_COMPETITOR, "COSRX", properties={"category": "lip_care"})
    )
    return kg


def test_get_brand_products_is_case_insensitive(seeded):
    assert [p["asin"] for p in seeded.get_brand_products("LANEIGE")] == ["B0LANE1"]
    assert [p["asin"] for p in seeded.get_brand_products("laneige")] == ["B0LANE1"]
    assert [p["asin"] for p in seeded.get_brand_products("Laneige")] == ["B0LANE1"]
    assert seeded.get_brand_products("nope") == []


def test_get_competitors_is_case_insensitive(seeded):
    assert [c["brand"] for c in seeded.get_competitors("laneige")] == ["COSRX"]


def test_query_stays_case_sensitive(seeded):
    # Only the two brand helpers resolve casing; the raw triple query does not.
    assert seeded.query(subject="laneige") == []

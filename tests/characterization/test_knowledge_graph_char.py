"""Characterization tests: KnowledgeGraph (+ KGUpdaterMixin loaders).

Public entry points only: load_from_crawl_data / load_from_metrics_data / save / query /
get_brand_products / get_stats. Persistence goes to tmp_path; auto_load=False always.
"""

import pytest

from src.domain.entities.relations import RelationType
from src.ontology.knowledge_graph import KnowledgeGraph

from ._fixtures import build_lip_care_snapshot


def make_kg(tmp_path, name="kg.json", auto_load=False) -> KnowledgeGraph:
    return KnowledgeGraph(
        persist_path=str(tmp_path / name),
        auto_load=auto_load,
        auto_save=False,
    )


@pytest.fixture
def crawl_data() -> dict:
    # Top 20 of the snapshot: LANEIGE x3 (ranks 1,5,20), COSRX x10, 7 others -> 9 brands.
    return {"categories": {"lip_care": {"rank_records": build_lip_care_snapshot()[:20]}}}


def test_load_from_crawl_data_counts(tmp_path, crawl_data):
    kg = make_kg(tmp_path)
    added = kg.load_from_crawl_data(crawl_data)
    # 20 hasProduct + 20 belongsToCategory + 9 brands -> C(9,2)=36 pairs x2 directions = 72
    assert added == 112
    assert len(kg.triples) == 112
    stats = kg.get_stats()
    assert stats["relations_by_type"] == {
        "hasProduct": 20,
        "belongsToCategory": 20,
        "directCompetitor": 20,  # 5 "direct" brands -> C(5,2)=10 pairs x2
        "indirectCompetitor": 52,
    }


def test_load_from_crawl_data_is_idempotent(tmp_path, crawl_data):
    kg = make_kg(tmp_path)
    kg.load_from_crawl_data(crawl_data)
    first = len(kg.triples)
    added_again = kg.load_from_crawl_data(crawl_data)
    # Relation equality is (subject, predicate, object) -> duplicates are merged, not added.
    assert added_again == 0
    assert len(kg.triples) == first == 112


def test_save_then_reload_roundtrip(tmp_path, crawl_data):
    kg = make_kg(tmp_path)
    kg.load_from_crawl_data(crawl_data)
    # save() without force is a no-op when auto_save=False (nothing marked dirty)
    assert kg.save() is False
    assert kg.save(force=True) is True
    assert (tmp_path / "kg.json").exists()

    kg2 = make_kg(tmp_path, auto_load=True)
    assert len(kg2.triples) == len(kg.triples)
    # Relation hash/eq is (subject, predicate, object): the set of triples survives roundtrip
    assert set(kg2.triples) == set(kg.triples)
    assert kg2.get_stats() == kg.get_stats()


def test_get_stats_keys(tmp_path, crawl_data):
    kg = make_kg(tmp_path)
    assert kg.get_stats() == {
        "total_triples": 0,
        "unique_subjects": 0,
        "unique_objects": 0,
        "relations_by_type": {},
    }
    kg.load_from_crawl_data(crawl_data)
    stats = kg.get_stats()
    assert set(stats.keys()) == {
        "total_triples",
        "unique_subjects",
        "unique_objects",
        "relations_by_type",
    }
    assert stats["total_triples"] == 112
    # 9 brands + 20 ASINs as subjects; 20 ASINs + 9 brands + "lip_care" as objects
    assert stats["unique_subjects"] == 29
    assert stats["unique_objects"] == 30


def test_brand_casing_is_not_canonicalized(tmp_path, crawl_data):
    # PINS CURRENT BEHAVIOR (canonical-IRI issue): the KG stores the brand string verbatim
    # ("LANEIGE"), and query()/get_brand_products() match subjects case-sensitively, so a
    # lowercase lookup returns nothing. Expected to change if brand IRIs are canonicalized.
    kg = make_kg(tmp_path)
    kg.load_from_crawl_data(crawl_data)

    upper = kg.query(subject="LANEIGE")
    lower = kg.query(subject="laneige")
    # 3 hasProduct + 8 competitor edges (LANEIGE vs the other 8 brands)
    assert len(upper) == 11
    assert lower == []

    assert len(kg.query(subject="LANEIGE", predicate=RelationType.HAS_PRODUCT)) == 3
    assert sorted(kg.get_objects("LANEIGE", RelationType.HAS_PRODUCT)) == [
        "B000000001",
        "B000000005",
        "B000000020",
    ]

    assert len(kg.get_brand_products("LANEIGE")) == 3
    assert kg.get_brand_products("laneige") == []


def test_load_from_metrics_data_sets_metadata_and_alert_edges(tmp_path):
    kg = make_kg(tmp_path)
    added = kg.load_from_metrics_data(
        {
            "brand_metrics": [
                {
                    "brand_name": "LANEIGE",
                    "share_of_shelf": 3.0,
                    "avg_rank": 8.67,
                    "product_count": 3,
                    "is_laneige": True,
                    "category_id": "lip_care",
                }
            ],
            "product_metrics": [
                {"asin": "B000000001", "current_rank": 1, "category_id": "lip_care"}
            ],
            "alerts": [
                {
                    "asin": "B000000001",
                    "type": "rank_shock",
                    "severity": "warning",
                    "message": "m",
                }
            ],
        }
    )
    assert added == 1
    assert kg.get_entity_metadata("LANEIGE") == {
        "type": "brand",
        "sos": 3.0,
        "avg_rank": 8.67,
        "product_count": 3,
        "is_target": True,
        "category": "lip_care",
    }
    assert kg.get_entity_metadata("B000000001")["current_rank"] == 1
    assert kg.get_entity_metadata("B000000001")["rank_change_1d"] is None
    alerts = kg.query(subject="B000000001", predicate=RelationType.HAS_ALERT)
    assert [a.object for a in alerts] == ["rank_shock"]
    assert alerts[0].properties["severity"] == "warning"

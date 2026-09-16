"""F9-5: competency questions (CQ1..CQ12) on the ~30-record snapshot fixture.

Each CQ is answered from the materialized JSON KG (what the chat path reads), never by
calling owlready2 at question time.
"""

from __future__ import annotations

import pytest

from src.domain.entities.relations import RelationType
from src.ontology.builder import OntologyBuilder, canonical_brand
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.materializer import list_inferred, materialize

owlready2 = pytest.importorskip("owlready2")

SNAPSHOT_DATE = "2026-09-10"


@pytest.fixture(scope="module")
def world():
    from tests.unit.ontology.conftest import GROUPS, HIERARCHY, METRICS, RECORDS

    kg = KnowledgeGraph(persist_path=None, auto_load=False, auto_save=False)
    builder = OntologyBuilder()
    result = builder.from_snapshot(
        [dict(r) for r in RECORDS],
        METRICS,
        category_hierarchy=HIERARCHY,
        groups=GROUPS,
        kg=kg,
        snapshot_date=SNAPSHOT_DATE,
    )
    first = materialize(result.owl, kg)
    count_after_first = kg.get_stats()["total_triples"]
    second = materialize(result.owl, kg)
    count_after_second = kg.get_stats()["total_triples"]
    return {
        "kg": kg,
        "builder": builder,
        "result": result,
        "first": first,
        "second": second,
        "count_after_first": count_after_first,
        "count_after_second": count_after_second,
    }


def _objects(kg: KnowledgeGraph, subject: str, predicate: RelationType) -> set[str]:
    return {r.object for r in kg.query(subject, predicate)}


def test_cq1_group_of_laneige(world) -> None:
    assert _objects(world["kg"], "LANEIGE", RelationType.OWNED_BY_GROUP) == {"AMOREPACIFIC"}


def test_cq2_ancestors_of_lip_care(world) -> None:
    kg = world["kg"]
    rels = kg.query("lip_care", RelationType.PARENT_CATEGORY)
    ordered = sorted(rels, key=lambda r: r.properties.get("distance", 1))
    assert [r.object for r in ordered] == ["skin_care", "beauty"]


def test_cq3_brands_with_sos_at_least_030_in_lip_care(world) -> None:
    kg = world["kg"]
    hits = set()
    for rel in kg.query(predicate=RelationType.HAS_POSITION):
        sos = rel.properties.get("categories", {}).get("lip_care")
        if sos is not None and sos >= 0.30:
            hits.add(rel.subject)
    assert hits == {"LANEIGE"}
    # cross-check with the builder's own SoS table
    table = world["result"].stats["sos_by_category"]["lip_care"]
    assert {b for b, s in table.items() if s >= 0.30} == {"LANEIGE"}


def test_cq4_siblings_of_laneige(world) -> None:
    assert _objects(world["kg"], "LANEIGE", RelationType.SIBLING_BRAND) == {
        canonical_brand("INNISFREE")
    }


def test_cq5_products_case_insensitive(world) -> None:
    kg = world["kg"]
    lower = sorted(p["asin"] for p in kg.get_brand_products("laneige"))
    upper = sorted(p["asin"] for p in kg.get_brand_products("LANEIGE"))
    assert lower == upper
    assert set(lower) == {"B0LAN001", "B0LAN002", "B0LAN003", "B0LAN010"}


def test_cq6_percent_sos_is_a_consistency_error(snapshot_records) -> None:
    bad_metrics = {
        "brand_metrics": [
            {"brand_name": "LANEIGE", "category_id": "lip_care", "share_of_shelf": 150.0}
        ]
    }
    kg = KnowledgeGraph(persist_path=None, auto_load=False, auto_save=False)
    with pytest.raises(ValueError):
        OntologyBuilder().from_snapshot(snapshot_records, bad_metrics, kg=kg)


def test_cq7_materialize_is_idempotent(world) -> None:
    assert world["first"] > 0
    assert world["second"] == 0
    assert world["count_after_first"] == world["count_after_second"]


def test_cq8_every_inferred_triple_has_provenance(world) -> None:
    inferred = list_inferred(world["kg"])
    assert inferred
    for rel in inferred:
        assert rel.properties.get("provenance", "").startswith("owl:"), rel
        assert rel.properties.get("reasoner") in {"pellet", "hermit", "python"}


def test_cq9_laneige_market_position_with_owl_provenance(world) -> None:
    rels = world["kg"].query("LANEIGE", RelationType.HAS_POSITION)
    lip = [r for r in rels if "lip_care" in r.properties.get("categories", {})]
    assert len(lip) == 1
    assert lip[0].object == "DominantBrand"
    assert lip[0].properties["provenance"].startswith("owl:")


def test_cq10_competes_with_is_symmetric(world) -> None:
    kg = world["kg"]
    pairs = {(r.subject, r.object) for r in kg.query(predicate=RelationType.COMPETES_WITH)}
    assert pairs
    assert all((b, a) in pairs for a, b in pairs)
    assert ("LANEIGE", "COSRX") in pairs


def test_cq11_rank_at_most_10_is_top10_product(world) -> None:
    kg = world["kg"]
    top10 = {r.subject for r in kg.query(predicate=RelationType.HAS_STATE, object_="Top10Product")}
    # lip_care ranks 1..8 + skin_care 1,2,3,5,6,9 + beauty 1,2,4,7,9
    assert "B0LAN001" in top10 and "B0COS001" in top10 and "B0COS010" in top10
    assert "B0COS002" not in top10  # rank 15
    assert "B0TAT001" not in top10  # rank 20
    assert all(kg.get_entity_metadata(a) is not None for a in top10)


def test_cq12_first_seen_within_7_days_is_new_entrant(world) -> None:
    kg = world["kg"]
    new = {r.subject for r in kg.query(predicate=RelationType.HAS_STATE, object_="NewEntrant")}
    assert new == {"B0LAN002"}

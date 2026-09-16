"""F9-3: materialize(owl, kg) writes provenance-tagged inferred triples; idempotent."""

from __future__ import annotations

import pytest

from src.domain.entities.relations import Relation, RelationType
from src.ontology.builder import OntologyBuilder
from src.ontology.materializer import list_inferred, materialize

owlready2 = pytest.importorskip("owlready2")


@pytest.fixture(scope="module")
def built():
    from src.ontology.knowledge_graph import KnowledgeGraph
    from tests.unit.ontology.conftest import GROUPS, HIERARCHY, METRICS, RECORDS

    kg = KnowledgeGraph(persist_path=None, auto_load=False, auto_save=False)
    result = OntologyBuilder().from_snapshot(
        [dict(r) for r in RECORDS],
        METRICS,
        category_hierarchy=HIERARCHY,
        groups=GROUPS,
        kg=kg,
        snapshot_date="2026-09-10",
    )
    before = kg.get_stats()["total_triples"]
    first = materialize(result.owl, kg, reasoner="python")
    after_first = kg.get_stats()["total_triples"]
    second = materialize(result.owl, kg, reasoner="python")
    after_second = kg.get_stats()["total_triples"]
    return {
        "kg": kg,
        "owl": result.owl,
        "before": before,
        "first": first,
        "second": second,
        "after_first": after_first,
        "after_second": after_second,
    }


def test_first_run_adds_second_adds_zero(built) -> None:
    assert built["first"] > 0
    assert built["after_first"] == built["before"] + built["first"]
    assert built["second"] == 0
    assert built["after_second"] == built["after_first"]


def test_every_inferred_triple_has_owl_provenance(built) -> None:
    inferred = list_inferred(built["kg"])
    assert len(inferred) == built["first"]
    assert all(r.properties.get("provenance", "").startswith("owl:") for r in inferred)
    assert all(r.source == "owl" for r in inferred)


def test_market_position_triples(built) -> None:
    kg = built["kg"]
    rels = kg.query("LANEIGE", RelationType.HAS_POSITION)
    positions = {r.object: r for r in rels}
    # LANEIGE: lip_care 0.30 -> Dominant, skin_care 0.10 / beauty 0.10 -> Niche
    assert set(positions) == {"DominantBrand", "NicheBrand"}
    assert positions["DominantBrand"].properties["provenance"] == "owl:DominantBrand"
    assert positions["DominantBrand"].properties["categories"] == {"lip_care": pytest.approx(0.30)}
    assert set(positions["NicheBrand"].properties["categories"]) == {"skin_care", "beauty"}


def test_transitive_category_ancestors(built) -> None:
    kg = built["kg"]
    ancestors = {r.object: r for r in kg.query("lip_care", RelationType.PARENT_CATEGORY)}
    assert set(ancestors) == {"skin_care", "beauty"}
    assert ancestors["beauty"].properties["provenance"] == "owl:parentCategory.transitive"
    # products in lip_care belong to the ancestor categories too
    cats = {r.object for r in kg.query("B0LAN002", RelationType.BELONGS_TO_CATEGORY)}
    assert cats == {"lip_care", "skin_care", "beauty"}


def test_sibling_and_competes_symmetry(built) -> None:
    kg = built["kg"]
    innisfree = OntologyBuilder().canonical_brand("innisfree")
    sib = {r.object for r in kg.query("LANEIGE", RelationType.SIBLING_BRAND)}
    assert sib == {innisfree}
    assert {r.object for r in kg.query(innisfree, RelationType.SIBLING_BRAND)} == {"LANEIGE"}
    comp = {r.object for r in kg.query("LANEIGE", RelationType.COMPETES_WITH)}
    assert "COSRX" in comp
    for c in comp:
        assert "LANEIGE" in {r.object for r in kg.query(c, RelationType.COMPETES_WITH)}


def test_states_top10_and_new_entrant(built) -> None:
    kg = built["kg"]
    states = {r.object for r in kg.query("B0LAN002", RelationType.HAS_STATE)}
    assert states == {"Top10Product", "NewEntrant"}
    assert {r.object for r in kg.query("B0COS002", RelationType.HAS_STATE)} == set()


def test_stale_owl_triples_replaced(built) -> None:
    kg = built["kg"]
    stale = Relation(
        subject="LANEIGE",
        predicate=RelationType.HAS_POSITION,
        object="StrongBrand",
        properties={"provenance": "owl:StrongBrand"},
        source="owl",
    )
    kg.add_relation(stale)
    materialize(built["owl"], kg, reasoner="python")
    assert {r.object for r in kg.query("LANEIGE", RelationType.HAS_POSITION)} == {
        "DominantBrand",
        "NicheBrand",
    }


def test_reasoner_auto_matches_python(snapshot_records, snapshot_metrics, hierarchy, groups):
    """Pellet/HermiT (if runnable) must agree with the Python fallback."""
    from src.ontology.knowledge_graph import KnowledgeGraph

    def run(mode: str):
        kg = KnowledgeGraph(persist_path=None, auto_load=False, auto_save=False)
        r = OntologyBuilder().from_snapshot(
            [dict(x) for x in snapshot_records],
            snapshot_metrics,
            category_hierarchy=hierarchy,
            groups=groups,
            kg=kg,
            snapshot_date="2026-09-10",
            categories=["lip_care"],
        )
        materialize(r.owl, kg, reasoner=mode)
        return {(x.subject, x.predicate, x.object) for x in list_inferred(kg)}, list_inferred(kg)

    py_set, _ = run("python")
    auto_set, auto_rels = run("auto")
    assert py_set == auto_set
    engines = {r.properties.get("reasoner") for r in auto_rels}
    assert engines <= {"pellet", "hermit", "python"}

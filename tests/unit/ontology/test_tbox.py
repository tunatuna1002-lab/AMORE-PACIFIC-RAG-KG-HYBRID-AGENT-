"""F9-1: T-Box (src/ontology/tbox.py) axioms."""

from __future__ import annotations

import pytest

owlready2 = pytest.importorskip("owlready2")

from src.ontology.tbox import TBOX_IRI, define_tbox, new_world_ontology  # noqa: E402
from src.ontology.thresholds import Thresholds  # noqa: E402


@pytest.fixture
def onto():
    world, onto = new_world_ontology()
    define_tbox(onto)
    return onto


def test_classes_exist(onto) -> None:
    for name in ("Brand", "Product", "Category", "Group", "Trend"):
        assert getattr(onto, name) is not None, name
    for name in ("DominantBrand", "StrongBrand", "NicheBrand", "Top10Product", "NewEntrant"):
        assert getattr(onto, name) is not None, name
    assert onto.base_iri.startswith(TBOX_IRI.rsplit("#", 1)[0].rsplit("/", 1)[0])


def test_property_domains_ranges(onto) -> None:
    assert onto.hasBrand.domain == [onto.Product] and onto.hasBrand.range == [onto.Brand]
    assert onto.belongsToCategory.range == [onto.Category]
    assert onto.ownedByGroup.domain == [onto.Brand] and onto.ownedByGroup.range == [onto.Group]
    assert onto.parentCategory.domain == [onto.Category]
    assert onto.hasProduct.inverse_property == onto.hasBrand


def test_functional_and_transitive_symmetric(onto) -> None:
    from owlready2 import FunctionalProperty, SymmetricProperty, TransitiveProperty

    for name in ("shareOfShelf", "averageRank", "rank"):
        assert FunctionalProperty in getattr(onto, name).is_a, name
    assert TransitiveProperty in onto.parentCategory.is_a
    assert SymmetricProperty in onto.siblingBrand.is_a
    assert SymmetricProperty in onto.competesWith.is_a


def test_restrictions_use_thresholds_on_fraction_scale() -> None:
    world, onto = new_world_ontology()
    define_tbox(onto, Thresholds(owl_dominant_sos=0.42, owl_strong_sos=0.21, top_n=7))
    dom = str(onto.DominantBrand.equivalent_to[0])
    assert "min_inclusive" in dom and "0.42" in dom
    strong = str(onto.StrongBrand.equivalent_to[0])
    assert "0.21" in strong and "0.42" in strong
    niche = str(onto.NicheBrand.equivalent_to[0])
    assert "max_exclusive" in niche and "0.21" in niche
    top = str(onto.Top10Product.equivalent_to[0])
    assert "max_inclusive" in top and "7" in top


def test_disjoint_brand_product_category(onto) -> None:
    expected = {onto.Brand, onto.Product, onto.Category}
    assert any(set(d.entities) == expected for d in onto.disjoint_classes())


def test_define_tbox_is_idempotent(onto) -> None:
    define_tbox(onto)
    assert len(onto.DominantBrand.equivalent_to) == 1
    assert len(list(onto.disjoint_classes())) == 2


def test_transitive_parent_category_without_reasoner(onto) -> None:
    with onto:
        beauty = onto.Category("beauty")
        skin = onto.Category("skin_care")
        lip = onto.Category("lip_care")
        lip.parent_category = [skin]
        skin.parent_category = [beauty]
    assert set(lip.INDIRECT_parent_category) == {skin, beauty}


def test_owl_reasoner_uses_tbox() -> None:
    from src.ontology.owl_reasoner import OWLReasoner

    r = OWLReasoner()
    assert r.onto.Group is not None
    assert r.onto.Top10Product is not None
    assert r.onto.parentCategory is not None

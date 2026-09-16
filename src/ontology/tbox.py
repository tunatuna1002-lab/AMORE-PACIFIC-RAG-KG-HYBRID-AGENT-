"""
OWL T-Box (schema) of the AMORE brand ontology (F9-1).

The T-Box is code (version controlled); the A-Box is rebuilt from each snapshot by
``src.ontology.builder`` and reasoned over offline by ``src.ontology.materializer``.
The chat path never imports this module (it reads the materialized JSON KG).

Classes
    Brand, Product, Category, Group, Trend                (Brand ⊥ Product ⊥ Category)
    DominantBrand ≡ Brand ⊓ ∃shareOfShelf[≥ owl_dominant_sos]
    StrongBrand   ≡ Brand ⊓ ∃shareOfShelf[owl_strong_sos ≤ · < owl_dominant_sos]
    NicheBrand    ≡ Brand ⊓ ∃shareOfShelf[< owl_strong_sos]      (pairwise disjoint)
    Top10Product  ≡ Product ⊓ ∃rank[≤ top_n]
    NewEntrant    ≡ Product ⊓ ∃daysSinceFirstSeen[≤ new_entrant_days]
Object properties
    hasBrand (Product→Brand) ≡ hasProduct⁻ ; belongsToCategory (Product→Category, ≥1)
    parentCategory (Category→Category, transitive) ; hasSubcategory ≡ parentCategory⁻
    ownedByGroup (Brand→Group, functional) ≡ ownsBrand⁻ ; siblingBrand (Brand→Brand, symmetric)
    competesWith (Brand→Brand, symmetric) ; hasTrend (Brand→Trend)
Data properties (functional): shareOfShelf [0,1], averageRank, productCount, rank,
    daysSinceFirstSeen ; plus price, rating.

Thresholds come from ``src.ontology.thresholds`` (FRACTION scale) - never literals.
"""

from __future__ import annotations

import logging
from typing import Any

from .thresholds import Thresholds, get_thresholds

logger = logging.getLogger(__name__)

TBOX_IRI = "http://amorepacific.com/ontology/amore_brand.owl"

try:
    from owlready2 import (
        AllDisjoint,
        ConstrainedDatatype,
        DataProperty,
        FunctionalProperty,
        ObjectProperty,
        SymmetricProperty,
        Thing,
        TransitiveProperty,
        World,
        get_ontology,
    )

    OWLREADY2_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without owlready2
    OWLREADY2_AVAILABLE = False


def new_world_ontology(iri: str = TBOX_IRI) -> tuple[Any, Any]:
    """Create an isolated owlready2 ``World`` + ontology (one per snapshot build)."""
    if not OWLREADY2_AVAILABLE:
        raise RuntimeError("owlready2 is not installed")
    world = World()
    return world, world.get_ontology(iri)


def default_ontology(iri: str = TBOX_IRI) -> Any:
    """Ontology in owlready2's default world (used by the legacy ``OWLReasoner``)."""
    if not OWLREADY2_AVAILABLE:
        raise RuntimeError("owlready2 is not installed")
    return get_ontology(iri)


def _ensure_class(onto: Any, name: str, parent: Any) -> Any:
    existing = getattr(onto, name)
    if existing:
        return existing
    return type(name, (parent,), {"namespace": onto})


def _ensure_property(onto: Any, name: str, bases: tuple, **attrs: Any) -> Any:
    existing = getattr(onto, name)
    if existing:
        return existing
    return type(name, bases, {"namespace": onto, **attrs})


def define_tbox(onto: Any, thresholds: Thresholds | None = None) -> Any:
    """Define (idempotently) the T-Box axioms on ``onto`` and return it."""
    if not OWLREADY2_AVAILABLE or onto is None:
        return onto
    t = thresholds or get_thresholds()
    first_time = not getattr(onto, "Group")  # Group is new to this T-Box -> first definition

    with onto:
        # ===== Classes =====
        _ensure_class(onto, "Brand", Thing)
        _ensure_class(onto, "Product", Thing)
        _ensure_class(onto, "Category", Thing)
        _ensure_class(onto, "Group", Thing)
        _ensure_class(onto, "Trend", Thing)

    with onto:
        _ensure_class(onto, "DominantBrand", onto.Brand)
        _ensure_class(onto, "StrongBrand", onto.Brand)
        _ensure_class(onto, "NicheBrand", onto.Brand)
        _ensure_class(onto, "Top10Product", onto.Product)
        _ensure_class(onto, "NewEntrant", onto.Product)

        if first_time:
            # Brand subclasses are mutually exclusive market positions
            AllDisjoint([onto.DominantBrand, onto.StrongBrand, onto.NicheBrand])
            # Entity kinds never overlap
            AllDisjoint([onto.Brand, onto.Product, onto.Category])

        # ===== Object properties =====
        _ensure_property(
            onto,
            "hasBrand",
            (ObjectProperty,),
            domain=[onto.Product],
            range=[onto.Brand],
            python_name="has_brand",
        )
        _ensure_property(
            onto,
            "hasProduct",
            (ObjectProperty,),
            domain=[onto.Brand],
            range=[onto.Product],
            python_name="has_product",
        )
        _ensure_property(
            onto,
            "belongsToCategory",
            (ObjectProperty,),
            domain=[onto.Product],
            range=[onto.Category],
            python_name="belongs_to_category",
        )
        _ensure_property(
            onto,
            "parentCategory",
            (ObjectProperty, TransitiveProperty),
            domain=[onto.Category],
            range=[onto.Category],
            python_name="parent_category",
        )
        _ensure_property(
            onto,
            "hasSubcategory",
            (ObjectProperty,),
            domain=[onto.Category],
            range=[onto.Category],
            python_name="has_subcategory",
        )
        _ensure_property(
            onto,
            "ownedByGroup",
            (ObjectProperty, FunctionalProperty),
            domain=[onto.Brand],
            range=[onto.Group],
            python_name="owned_by_group",
        )
        _ensure_property(
            onto,
            "ownsBrand",
            (ObjectProperty,),
            domain=[onto.Group],
            range=[onto.Brand],
            python_name="owns_brand",
        )
        _ensure_property(
            onto,
            "siblingBrand",
            (ObjectProperty, SymmetricProperty),
            domain=[onto.Brand],
            range=[onto.Brand],
            python_name="sibling_brand",
        )
        _ensure_property(
            onto,
            "competesWith",
            (ObjectProperty, SymmetricProperty),
            domain=[onto.Brand],
            range=[onto.Brand],
            python_name="competes_with",
        )
        _ensure_property(
            onto,
            "hasTrend",
            (ObjectProperty,),
            domain=[onto.Brand],
            range=[onto.Trend],
            python_name="has_trend",
        )

        # Product must belong to at least one category (same ASIN may rank in several lists)
        if first_time:
            onto.Product.is_a.append(onto.belongsToCategory.min(1, onto.Category))

        # ===== Data properties =====
        _ensure_property(
            onto,
            "shareOfShelf",
            (DataProperty, FunctionalProperty),
            domain=[onto.Brand],
            range=[float],
            python_name="share_of_shelf",
        )
        _ensure_property(
            onto,
            "averageRank",
            (DataProperty, FunctionalProperty),
            domain=[onto.Brand],
            range=[float],
            python_name="average_rank",
        )
        _ensure_property(
            onto,
            "productCount",
            (DataProperty, FunctionalProperty),
            domain=[onto.Brand],
            range=[int],
            python_name="product_count",
        )
        _ensure_property(
            onto,
            "rank",
            (DataProperty, FunctionalProperty),
            domain=[onto.Product],
            range=[int],
            python_name="rank_value",
        )
        _ensure_property(
            onto,
            "daysSinceFirstSeen",
            (DataProperty, FunctionalProperty),
            domain=[onto.Product],
            range=[int],
            python_name="days_since_first_seen",
        )
        _ensure_property(
            onto,
            "price",
            (DataProperty,),
            domain=[onto.Product],
            range=[float],
            python_name="price_value",
        )
        _ensure_property(
            onto,
            "rating",
            (DataProperty,),
            domain=[onto.Product],
            range=[float],
            python_name="rating_value",
        )

    # ===== Restriction classes (FRACTION scale, thresholds.json) =====
    with onto:
        dominant, strong = float(t.owl_dominant_sos), float(t.owl_strong_sos)
        onto.DominantBrand.equivalent_to = [
            onto.Brand & onto.shareOfShelf.some(ConstrainedDatatype(float, min_inclusive=dominant))
        ]
        onto.StrongBrand.equivalent_to = [
            onto.Brand
            & onto.shareOfShelf.some(
                ConstrainedDatatype(float, min_inclusive=strong, max_exclusive=dominant)
            )
        ]
        onto.NicheBrand.equivalent_to = [
            onto.Brand & onto.shareOfShelf.some(ConstrainedDatatype(float, max_exclusive=strong))
        ]
        onto.Top10Product.equivalent_to = [
            onto.Product & onto.rank.some(ConstrainedDatatype(int, max_inclusive=int(t.top_n)))
        ]
        onto.NewEntrant.equivalent_to = [
            onto.Product
            & onto.daysSinceFirstSeen.some(
                ConstrainedDatatype(int, max_inclusive=int(t.new_entrant_days))
            )
        ]

    # ===== inverse properties =====
    with onto:
        if onto.hasProduct and onto.hasBrand:
            onto.hasProduct.inverse_property = onto.hasBrand
        if onto.ownsBrand and onto.ownedByGroup:
            onto.ownsBrand.inverse_property = onto.ownedByGroup
        if onto.hasSubcategory and onto.parentCategory:
            onto.hasSubcategory.inverse_property = onto.parentCategory

    logger.debug("OWL T-Box defined on %s", getattr(onto, "base_iri", onto))
    return onto


__all__ = [
    "TBOX_IRI",
    "OWLREADY2_AVAILABLE",
    "define_tbox",
    "new_world_ontology",
    "default_ontology",
]

"""
Materializer - run OWL reasoning offline and write the inferred facts into the JSON KG (F9-3).

``materialize(owl, kg)`` takes the ``OWLSnapshot`` produced by ``OntologyBuilder`` (or any
object with an owlready2 ``onto``), runs Pellet / HermiT when a working Java is available
(else a Python evaluation of the same axioms) and merges the inferred triples into the KG:

    HAS_POSITION          brand  -> DominantBrand | StrongBrand | NicheBrand   owl:<Class>
    PARENT_CATEGORY       cat    -> transitive ancestors                     owl:parentCategory.transitive
    BELONGS_TO_CATEGORY   asin   -> ancestor categories                      owl:belongsToCategory.parentCategory
    SIBLING_BRAND         brand  -> brand (same group, both directions)      owl:siblingBrand
    COMPETES_WITH         brand  -> brand (symmetric closure)                owl:competesWith.symmetric
    HAS_STATE             asin   -> Top10Product | NewEntrant                owl:<Class>

Every written triple carries ``properties["provenance"] = "owl:<axiom id>"`` and
``properties["reasoner"]`` (``pellet`` / ``hermit`` / ``python``) and ``source="owl"``.
The run is idempotent: facts already present are updated in place (count 0) and
owl-provenance triples that are no longer derivable are removed.

The chat path reads the result with ``list_inferred`` / ``inferred_facts`` and never imports
owlready2 (this module imports it lazily, only inside ``materialize``).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.domain.entities.relations import Relation, RelationType

from .thresholds import get_thresholds

logger = logging.getLogger(__name__)

PROVENANCE_PREFIX = "owl:"
SOURCE = "owl"
ENGINES = ("pellet", "hermit", "python")
ENV_REASONER = "AMORE_OWL_REASONER"

_POSITION_CLASSES = ("DominantBrand", "StrongBrand", "NicheBrand")
_STATE_CLASSES = ("Top10Product", "NewEntrant")

# engines that already failed to start in this process (Java missing / wrong version)
_unavailable_engines: set[str] = set()


# =========================================================================
# Reading materialized facts (chat path)
# =========================================================================


def is_inferred(relation: Relation) -> bool:
    return str(relation.properties.get("provenance", "")).startswith(PROVENANCE_PREFIX)


def list_inferred(kg: Any, subject: str | None = None) -> list[Relation]:
    """All owl-provenance triples in ``kg`` (optionally for one subject, case-insensitive)."""
    if subject is not None:
        resolve = getattr(kg, "_resolve_subject", None)
        subject = resolve(subject) if resolve else subject
        relations = kg.query(subject=subject)
    else:
        relations = list(getattr(kg, "triples", []))
    return [r for r in relations if is_inferred(r)]


def inferred_facts(kg: Any, subject: str | None = None) -> list[dict[str, Any]]:
    """Materialized facts as plain dicts (shape consumed by the retrieval strategy)."""
    facts: list[dict[str, Any]] = []
    for rel in list_inferred(kg, subject):
        base = {
            "subject": rel.subject,
            "provenance": rel.properties.get("provenance"),
            "reasoner": rel.properties.get("reasoner"),
        }
        if rel.predicate == RelationType.HAS_POSITION:
            categories = rel.properties.get("categories") or {}
            facts.append(
                {
                    **base,
                    "type": "market_position",
                    "position": rel.object,
                    "sos": max(categories.values()) if categories else rel.properties.get("sos"),
                    "categories": dict(categories),
                }
            )
        elif rel.predicate == RelationType.COMPETES_WITH:
            facts.append({**base, "type": "competition", "object": rel.object, "relation": "competesWith"})
        elif rel.predicate == RelationType.SIBLING_BRAND:
            facts.append({**base, "type": "sibling", "object": rel.object, "relation": "siblingBrand"})
        elif rel.predicate == RelationType.PARENT_CATEGORY:
            facts.append(
                {
                    **base,
                    "type": "category_ancestor",
                    "object": rel.object,
                    "distance": rel.properties.get("distance"),
                }
            )
        elif rel.predicate == RelationType.BELONGS_TO_CATEGORY:
            facts.append({**base, "type": "category_membership", "object": rel.object})
        elif rel.predicate == RelationType.HAS_STATE:
            facts.append(
                {
                    **base,
                    "type": "product_state",
                    "state": rel.object,
                    "categories": dict(rel.properties.get("categories") or {}),
                }
            )
    return facts


def brand_position(kg: Any, brand: str, category: str | None = None) -> dict[str, Any] | None:
    """Materialized market position of ``brand`` (optionally in ``category``)."""
    best: dict[str, Any] | None = None
    for fact in inferred_facts(kg, brand):
        if fact["type"] != "market_position":
            continue
        cats = fact.get("categories") or {}
        if category is not None:
            if category in cats:
                return {**fact, "sos": cats[category], "category": category}
            continue
        if best is None or (fact.get("sos") or 0) > (best.get("sos") or 0):
            best = fact
    return best


# =========================================================================
# Reasoning
# =========================================================================


def _requested_engine(reasoner: str | None) -> str:
    mode = (reasoner or os.environ.get(ENV_REASONER) or "auto").lower()
    if mode not in (*ENGINES, "auto"):
        raise ValueError(f"unknown reasoner {mode!r}; expected one of {ENGINES + ('auto',)}")
    return mode


def _run_reasoner(abox: Any, mode: str) -> str:
    """Run the requested OWL reasoner on ``abox``; return the engine that produced the facts."""
    if mode == "python":
        return "python"
    from owlready2 import (
        OwlReadyInconsistentOntologyError,
        sync_reasoner_hermit,
        sync_reasoner_pellet,
    )

    world = getattr(abox, "world", None)
    target = world if world is not None else abox.onto
    candidates = ["pellet", "hermit"] if mode == "auto" else [mode]
    for engine in candidates:
        if engine in _unavailable_engines:
            continue
        runner = sync_reasoner_pellet if engine == "pellet" else sync_reasoner_hermit
        try:
            with abox.onto:
                runner(target, infer_property_values=True, debug=0)
            return engine
        except OwlReadyInconsistentOntologyError as exc:
            # schema violation (e.g. functional/disjoint clash) must not be swallowed
            raise ValueError(f"OWL ontology inconsistent ({engine}): {exc}") from exc
        except Exception as exc:  # Java missing / wrong class version / reasoner crash
            _unavailable_engines.add(engine)
            logger.warning("OWL reasoner %s unavailable (%s); trying next engine", engine, exc)
    if mode != "auto":
        raise RuntimeError(f"OWL reasoner {mode!r} is not available in this environment")
    return "python"


# =========================================================================
# Fact extraction (works after a reasoner run and in pure-Python mode)
# =========================================================================


def _classify(value: float | None, is_a: list, engine: str, onto: Any, kind: str, t) -> str | None:
    """Class membership from the reasoner (``is_a``) or the Python evaluation of the axiom."""
    names = {getattr(c, "name", None) for c in is_a}
    if engine != "python":
        if kind == "position":
            for cls in _POSITION_CLASSES:
                if cls in names:
                    return cls
        elif kind in _STATE_CLASSES and kind in names:
            return kind
    if value is None:
        return None
    if kind == "position":
        if value >= t.owl_dominant_sos:
            return "DominantBrand"
        if value >= t.owl_strong_sos:
            return "StrongBrand"
        return "NicheBrand"
    if kind == "Top10Product":
        return kind if value <= t.top_n else None
    if kind == "NewEntrant":
        return kind if value <= t.new_entrant_days else None
    return None


def _ancestor_chain(cat_individual: Any) -> list[Any]:
    """Ancestors ordered by distance (direct parent first), via the transitive property."""
    chain: list[Any] = []
    seen: set = set()
    frontier = list(cat_individual.parent_category)
    while frontier:
        nxt: list[Any] = []
        for parent in frontier:
            if parent in seen:
                continue
            seen.add(parent)
            chain.append(parent)
            nxt.extend(parent.parent_category)
        frontier = nxt
    return chain


def _extract(abox: Any, engine: str, thresholds) -> dict[tuple, dict[str, Any]]:
    """Inferred facts of one A-Box keyed by ``(subject, predicate, object)``."""
    onto = abox.onto
    label = getattr(abox, "label", None) or (lambda ind: ind.name)
    category = getattr(abox, "category", None)
    facts: dict[tuple, dict[str, Any]] = {}

    def put(subject: str, predicate: RelationType, obj: str, provenance: str, **props: Any) -> dict:
        key = (subject, predicate, obj)
        fact = facts.setdefault(
            key, {"provenance": provenance, "reasoner": engine, **{k: v for k, v in props.items() if k != "categories"}}
        )
        if "categories" in props:
            fact.setdefault("categories", {}).update(props["categories"])
        return fact

    # --- market position (restriction classes on FRACTION SoS) ---
    for brand in onto.Brand.instances():
        sos = brand.share_of_shelf
        position = _classify(sos, brand.is_a, engine, onto, "position", thresholds)
        if position:
            put(
                label(brand),
                RelationType.HAS_POSITION,
                position,
                f"{PROVENANCE_PREFIX}{position}",
                categories={category: float(sos)} if category else {},
            )

    # --- transitive category ancestors + product membership in ancestors ---
    ancestors_of: dict[str, list[Any]] = {}
    for cat in onto.Category.instances():
        chain = _ancestor_chain(cat)
        ancestors_of[label(cat)] = chain
        for distance, anc in enumerate(chain, start=1):
            if distance == 1:
                continue  # asserted parent (source=config) is not an inference
            put(
                label(cat),
                RelationType.PARENT_CATEGORY,
                label(anc),
                f"{PROVENANCE_PREFIX}parentCategory.transitive",
                distance=distance,
            )
    for product in onto.Product.instances():
        for cat in product.belongs_to_category:
            for distance, anc in enumerate(ancestors_of.get(label(cat), []), start=1):
                put(
                    label(product),
                    RelationType.BELONGS_TO_CATEGORY,
                    label(anc),
                    f"{PROVENANCE_PREFIX}belongsToCategory.parentCategory",
                    via=label(cat),
                    distance=distance,
                )

    # --- siblingBrand: same group (ownedByGroup / ownsBrand inverse) ---
    for group in onto.Group.instances():
        members = sorted({label(b) for b in group.owns_brand})
        for a in members:
            for b in members:
                if a != b:
                    put(a, RelationType.SIBLING_BRAND, b, f"{PROVENANCE_PREFIX}siblingBrand", group=label(group))

    # --- competesWith symmetric closure ---
    for brand in onto.Brand.instances():
        for other in brand.competes_with:
            for a, b in ((label(brand), label(other)), (label(other), label(brand))):
                put(a, RelationType.COMPETES_WITH, b, f"{PROVENANCE_PREFIX}competesWith.symmetric", categories={category: True} if category else {})

    # --- product states ---
    for product in onto.Product.instances():
        rank = product.rank_value
        state = _classify(rank, product.is_a, engine, onto, "Top10Product", thresholds)
        if state:
            put(label(product), RelationType.HAS_STATE, state, f"{PROVENANCE_PREFIX}{state}", categories={category: rank} if category else {})
        days = product.days_since_first_seen
        state = _classify(days, product.is_a, engine, onto, "NewEntrant", thresholds)
        if state:
            put(label(product), RelationType.HAS_STATE, state, f"{PROVENANCE_PREFIX}{state}", categories={category: days} if category else {}, days_since_first_seen=days)
    return facts


# =========================================================================
# Entry point
# =========================================================================


def materialize(owl: Any, kg: Any, *, reasoner: str | None = None) -> int:
    """Run OWL reasoning over ``owl`` and merge the inferred triples into ``kg``.

    Args:
        owl: ``OWLSnapshot`` (per-category A-Boxes) or any object with an owlready2 ``onto``.
        kg: ``KnowledgeGraph`` to write into.
        reasoner: ``"auto"`` (Pellet -> HermiT -> Python), ``"pellet"``, ``"hermit"`` or
            ``"python"``; ``None`` reads ``$AMORE_OWL_REASONER`` then defaults to ``auto``.

    Returns:
        Number of NEW triples added (0 on an unchanged re-run).

    Raises:
        ValueError: the ontology is inconsistent (schema violation) or an unknown reasoner.
    """
    if owl is None:
        return 0
    mode = _requested_engine(reasoner)
    thresholds = getattr(owl, "thresholds", None) or get_thresholds()
    aboxes = list(owl.aboxes.values()) if hasattr(owl, "aboxes") else [owl]

    merged: dict[tuple, dict[str, Any]] = {}
    for abox in aboxes:
        engine = _run_reasoner(abox, mode)
        for key, props in _extract(abox, engine, thresholds).items():
            existing = merged.get(key)
            if existing is None:
                merged[key] = props
            elif "categories" in props:
                existing.setdefault("categories", {}).update(props["categories"])

    # stale owl-provenance triples (not derivable any more) -> remove
    removed = 0
    for rel in [r for r in list(getattr(kg, "triples", [])) if is_inferred(r)]:
        if (rel.subject, rel.predicate, rel.object) not in merged:
            if kg.remove_relation(rel):
                removed += 1

    added = 0
    for (subject, predicate, obj), props in merged.items():
        existing = kg.query(subject=subject, predicate=predicate, object_=obj)
        if existing:
            # asserted facts (crawl/config) stay asserted; only refresh earlier inferences
            if is_inferred(existing[0]):
                existing[0].properties.update(props)
                existing[0].source = SOURCE
            continue
        rel = Relation(subject=subject, predicate=predicate, object=obj, properties=props, source=SOURCE)
        if kg.add_relation(rel):
            added += 1

    logger.info(
        "materialized %d inferred triples (+%d new, -%d stale) from %d A-Box(es) [%s]",
        len(merged),
        added,
        removed,
        len(aboxes),
        mode,
    )
    return added


__all__ = [
    "materialize",
    "list_inferred",
    "inferred_facts",
    "brand_position",
    "is_inferred",
    "PROVENANCE_PREFIX",
    "ENV_REASONER",
]

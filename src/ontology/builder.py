"""
OntologyBuilder - the single writer of the A-Box (F9-2).

``OntologyBuilder.from_snapshot(records, metrics, ...)`` turns one crawl snapshot into

- the JSON ``KnowledgeGraph`` (via the ``kg_updater`` helpers) and
- OWL individuals (one isolated owlready2 ``World`` per category, T-Box from ``tbox.py``)

from the SAME normalised entities, so that both stores agree on names and units.

Invariants (docs/plans F9):
- one subject per brand: ``canonical_brand()`` is the only place brand names are normalised
  (case-fold + alias table from ``config/brands.json`` / ``config/competitors.json``)
- SoS is a FRACTION in [0, 1]; ``brand_metrics[].share_of_shelf`` (PERCENT) is converted once
  and a value outside the range raises ``ValueError`` (unit contract violation)
- group ownership comes from the ``groups`` argument or ``config/brands.json``
- the chat path never touches this module; reasoning happens in ``materializer.py``
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.domain.entities.relations import Relation, RelationType
from src.ontology.knowledge_graph import KnowledgeGraph
from src.shared.units import percent_to_fraction

from .thresholds import Thresholds, get_thresholds

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BRANDS_CONFIG = _PROJECT_ROOT / "config" / "brands.json"
_COMPETITORS_CONFIG = _PROJECT_ROOT / "config" / "competitors.json"
_HIERARCHY_CONFIG = _PROJECT_ROOT / "config" / "category_hierarchy.json"
DEFAULT_GROUP = "AMOREPACIFIC"
UNKNOWN_BRAND = "Unknown"


# =========================================================================
# Brand canonicalisation
# =========================================================================


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@lru_cache(maxsize=1)
def default_alias_table() -> dict[str, str]:
    """``casefold(alias) -> canonical name`` from config/brands.json + config/competitors.json."""
    table: dict[str, str] = {}

    def register(name: str | None, aliases: list[str] | None = None) -> None:
        if not name:
            return
        table.setdefault(name.strip().casefold(), name.strip())
        for alias in aliases or []:
            if alias:
                table.setdefault(alias.strip().casefold(), name.strip())

    brands = _read_json(_BRANDS_CONFIG)
    target = brands.get("target_brand") or {}
    register(target.get("name"), target.get("aliases"))
    for entry in brands.get("amorepacific_brands", []) + brands.get("competitor_brands", []):
        register(entry.get("name"), entry.get("aliases"))

    competitors = _read_json(_COMPETITORS_CONFIG)
    target = competitors.get("target_brand") or {}
    register(target.get("name"), target.get("aliases"))
    for entry in (competitors.get("fixed_competitors") or {}).get("brands", []):
        register(entry.get("name"), entry.get("aliases"))
    return table


@lru_cache(maxsize=1)
def default_groups() -> dict[str, list[str]]:
    """``{group: [canonical brand, ...]}`` from config/brands.json (amorepacific_brands)."""
    brands = _read_json(_BRANDS_CONFIG)
    members = [b.get("name") for b in brands.get("amorepacific_brands", []) if b.get("name")]
    return {DEFAULT_GROUP: members} if members else {}


def canonical_brand(name: str | None, alias_table: dict[str, str] | None = None) -> str:
    """Canonical spelling of a brand: alias table lookup, else the stripped input.

    ``None``/empty -> ``"Unknown"``. Stateless; ``OntologyBuilder.canonical_brand`` adds a
    per-build case-fold registry so unknown brands also collapse to one spelling.
    """
    if name is None:
        return UNKNOWN_BRAND
    stripped = str(name).strip()
    if not stripped:
        return UNKNOWN_BRAND
    table = default_alias_table() if alias_table is None else alias_table
    return table.get(stripped.casefold(), stripped)


# =========================================================================
# Normalised snapshot entities
# =========================================================================


@dataclass
class SnapshotRecord:
    category: str
    brand: str  # canonical
    asin: str
    name: str
    rank: int | None
    price: float | None = None
    rating: float | None = None
    first_seen: date | None = None


def _parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _individual_name(label: str) -> str:
    """owlready2-safe individual name (spaces and punctuation -> ``_``)."""
    return re.sub(r"[^\w\-.]", "_", label.strip()) or "_"


# =========================================================================
# OWL A-Box (per category world)
# =========================================================================


@dataclass
class OntologyABox:
    """OWL individuals of one category market, in an isolated owlready2 world."""

    category: str
    world: Any
    onto: Any
    brands: dict[str, Any] = field(default_factory=dict)  # canonical -> individual
    products: dict[str, Any] = field(default_factory=dict)  # asin -> individual
    categories: dict[str, Any] = field(default_factory=dict)  # category id -> individual
    groups: dict[str, Any] = field(default_factory=dict)  # group id -> individual
    labels: dict[str, str] = field(default_factory=dict)  # individual name -> label

    def brand(self, canonical: str) -> Any:
        return self.brands.get(canonical)

    def label(self, individual: Any) -> str:
        return self.labels.get(individual.name, individual.name)

    @property
    def individual_count(self) -> int:
        return len(self.brands) + len(self.products) + len(self.categories) + len(self.groups)


@dataclass
class OWLSnapshot:
    """All category A-Boxes of one snapshot (what ``materializer.materialize`` consumes)."""

    aboxes: dict[str, OntologyABox]
    snapshot_date: date
    thresholds: Thresholds

    @property
    def individual_count(self) -> int:
        return sum(a.individual_count for a in self.aboxes.values())


@dataclass
class BuildResult:
    kg: KnowledgeGraph
    owl: OWLSnapshot | None
    stats: dict[str, Any]


# =========================================================================
# Builder
# =========================================================================


class OntologyBuilder:
    """Build JSON KG + OWL A-Box from one snapshot (see module docstring)."""

    def __init__(
        self,
        alias_table: dict[str, str] | None = None,
        thresholds: Thresholds | None = None,
    ):
        self._alias_table = default_alias_table() if alias_table is None else dict(alias_table)
        self._registry: dict[str, str] = {}  # casefold -> first spelling seen (unknown brands)
        self._thresholds = thresholds

    # ----- canonicalisation -------------------------------------------------

    def canonical_brand(self, name: str | None) -> str:
        canonical = canonical_brand(name, self._alias_table)
        if canonical == UNKNOWN_BRAND:
            return canonical
        key = canonical.casefold()
        if key in self._alias_table:
            return self._alias_table[key]
        return self._registry.setdefault(key, canonical)

    # ----- input normalisation ---------------------------------------------

    def normalize_records(self, records: Any) -> list[SnapshotRecord]:
        """Accept a flat record list or a CrawlerAgent ``{"categories": {...}}`` payload."""
        flat: list[dict[str, Any]] = []
        if isinstance(records, dict):
            for cat_key, cat_data in (records.get("categories") or {}).items():
                for product in (cat_data or {}).get("rank_records") or []:
                    flat.append({**product, "category_id": product.get("category_id") or cat_key})
        else:
            flat = list(records or [])

        out: list[SnapshotRecord] = []
        for raw in flat:
            asin = raw.get("product_asin") or raw.get("asin") or ""
            category = raw.get("category_id") or raw.get("category") or ""
            if not asin or not category:
                continue
            out.append(
                SnapshotRecord(
                    category=str(category),
                    brand=self.canonical_brand(raw.get("brand")),
                    asin=str(asin),
                    name=str(raw.get("title") or raw.get("product_name") or ""),
                    rank=_to_int(raw.get("rank")),
                    price=_to_float(raw.get("price")),
                    rating=_to_float(raw.get("rating")),
                    first_seen=_parse_date(
                        raw.get("first_seen")
                        or raw.get("first_seen_date")
                        or raw.get("first_seen_at")
                    ),
                )
            )
        return out

    @staticmethod
    def _to_crawl_data(records: list[SnapshotRecord]) -> dict[str, Any]:
        categories: dict[str, dict[str, Any]] = {}
        for r in records:
            cat = categories.setdefault(r.category, {"rank_records": []})
            cat["rank_records"].append(
                {
                    "brand": r.brand,
                    "asin": r.asin,
                    "product_name": r.name,
                    "rank": r.rank,
                    "rating": r.rating,
                    "price": r.price,
                }
            )
        return {"categories": categories}

    # ----- hierarchy / groups ------------------------------------------------

    @staticmethod
    def _hierarchy_parents(category_hierarchy: Any) -> dict[str, str | None]:
        if isinstance(category_hierarchy, dict):
            cats = category_hierarchy.get("categories", category_hierarchy)
            return {cid: (c or {}).get("parent_id") for cid, c in cats.items()}
        path = Path(category_hierarchy) if category_hierarchy else _HIERARCHY_CONFIG
        data = _read_json(path)
        return {cid: (c or {}).get("parent_id") for cid, c in data.get("categories", {}).items()}

    @staticmethod
    def _load_hierarchy_into_kg(
        kg: KnowledgeGraph, category_hierarchy: Any, parents: dict[str, str | None]
    ) -> int:
        if isinstance(category_hierarchy, dict):
            added = 0
            cats = category_hierarchy.get("categories", category_hierarchy)
            for cat_id, cat_data in cats.items():
                cat_data = cat_data or {}
                kg.set_entity_metadata(
                    cat_id,
                    {
                        "type": "category",
                        "name": cat_data.get("name", ""),
                        "level": cat_data.get("level", 0),
                        "parent_id": cat_data.get("parent_id"),
                    },
                )
                parent_id = cat_data.get("parent_id")
                if not parent_id:
                    continue
                props = {"child_name": cat_data.get("name", ""), "child_level": cat_data.get("level", 0)}
                for rel in (
                    Relation(cat_id, RelationType.PARENT_CATEGORY, parent_id, dict(props), source="config"),
                    Relation(parent_id, RelationType.HAS_SUBCATEGORY, cat_id, dict(props), source="config"),
                ):
                    if kg.add_relation(rel):
                        added += 1
            return added
        path = str(category_hierarchy) if category_hierarchy else str(_HIERARCHY_CONFIG)
        return kg.load_category_hierarchy(path)

    def _resolve_groups(self, groups: dict[str, list[str]] | None) -> dict[str, list[str]]:
        raw = default_groups() if groups is None else groups
        return {g: sorted({self.canonical_brand(b) for b in members}) for g, members in raw.items()}

    @staticmethod
    def _load_groups_into_kg(kg: KnowledgeGraph, groups: dict[str, list[str]]) -> int:
        added = 0
        for group, members in groups.items():
            kg.set_entity_metadata(group, {"type": "corporate_group", "name": group})
            for brand in members:
                for rel in (
                    Relation(brand, RelationType.OWNED_BY_GROUP, group, source="config"),
                    Relation(group, RelationType.OWNS_BRAND, brand, source="config"),
                ):
                    if kg.add_relation(rel):
                        added += 1
        return added

    # ----- SoS -----------------------------------------------------------------

    def _sos_by_category(
        self, records: list[SnapshotRecord], metrics: dict[str, Any] | None
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, Any]]]]:
        """``{category: {brand: fraction}}`` (+ extra brand stats) from metrics, else records."""
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        ranks: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for r in records:
            counts[r.category][r.brand] += 1
            if r.rank is not None:
                ranks[r.category][r.brand].append(r.rank)

        sos: dict[str, dict[str, float]] = {}
        extra: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for cat, by_brand in counts.items():
            total = sum(by_brand.values()) or 1
            sos[cat] = {b: n / total for b, n in by_brand.items()}
            for b, n in by_brand.items():
                br = ranks[cat][b]
                extra[cat][b] = {
                    "product_count": n,
                    "avg_rank": (sum(br) / len(br)) if br else None,
                }

        for bm in (metrics or {}).get("brand_metrics") or []:
            brand = self.canonical_brand(bm.get("brand_name") or bm.get("brand"))
            cat = bm.get("category_id") or bm.get("category")
            if not cat or bm.get("share_of_shelf") is None:
                continue
            # brand_metrics.share_of_shelf is PERCENT -> FRACTION (single conversion point)
            fraction = percent_to_fraction(bm["share_of_shelf"])
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    f"share_of_shelf for {brand!r} in {cat!r} is {bm['share_of_shelf']!r} "
                    f"(fraction {fraction}); SoS must be a fraction in [0, 1]"
                )
            sos.setdefault(cat, {})[brand] = fraction
            info = extra[cat].setdefault(brand, {"product_count": None, "avg_rank": None})
            if bm.get("avg_rank") is not None:
                info["avg_rank"] = bm["avg_rank"]
            if bm.get("product_count") is not None:
                info["product_count"] = bm["product_count"]
        return sos, extra

    # ----- OWL A-Box -----------------------------------------------------------

    def _build_abox(
        self,
        category: str,
        records: list[SnapshotRecord],
        all_brands: set[str],
        sos: dict[str, float],
        extra: dict[str, dict[str, Any]],
        parents: dict[str, str | None],
        groups: dict[str, list[str]],
        snapshot_date: date,
        thresholds: Thresholds,
    ) -> OntologyABox:
        from .tbox import define_tbox, new_world_ontology

        world, onto = new_world_ontology()
        define_tbox(onto, thresholds)
        abox = OntologyABox(category=category, world=world, onto=onto)

        with onto:
            # categories: this one + ancestors, linked by parentCategory
            chain = [category]
            seen = {category}
            while True:
                parent = parents.get(chain[-1])
                if not parent or parent in seen:
                    break
                chain.append(parent)
                seen.add(parent)
            for cid in chain:
                abox.categories[cid] = onto.Category(_individual_name(cid))
                abox.labels[abox.categories[cid].name] = cid
            for child, parent in zip(chain, chain[1:]):
                abox.categories[child].parent_category = [abox.categories[parent]]

            # groups + brands (all snapshot brands so ownership/siblings are complete)
            for group in groups:
                abox.groups[group] = onto.Group(_individual_name(group))
                abox.labels[abox.groups[group].name] = group
            for brand in sorted(all_brands):
                ind = onto.Brand(_individual_name(brand))
                abox.brands[brand] = ind
                abox.labels[ind.name] = brand
                if brand in sos:
                    ind.share_of_shelf = float(sos[brand])
                    info = extra.get(brand, {})
                    if info.get("avg_rank") is not None:
                        ind.average_rank = float(info["avg_rank"])
                    if info.get("product_count") is not None:
                        ind.product_count = int(info["product_count"])
            for group, members in groups.items():
                for brand in members:
                    if brand in abox.brands:
                        abox.brands[brand].owned_by_group = abox.groups[group]

            # products of this category (same ASIN twice -> best rank)
            best: dict[str, SnapshotRecord] = {}
            for r in records:
                cur = best.get(r.asin)
                if cur is None or (r.rank is not None and (cur.rank is None or r.rank < cur.rank)):
                    best[r.asin] = r
            for asin, r in best.items():
                p = onto.Product(_individual_name(asin))
                abox.products[asin] = p
                abox.labels[p.name] = asin
                if r.rank is not None:
                    p.rank_value = int(r.rank)
                if r.price is not None:
                    p.price_value = [float(r.price)]
                if r.rating is not None:
                    p.rating_value = [float(r.rating)]
                if r.first_seen is not None:
                    p.days_since_first_seen = max(0, (snapshot_date - r.first_seen).days)
                p.has_brand = [abox.brands[r.brand]]
                p.belongs_to_category = [abox.categories[category]]

            # competition among the top brands of the category (rank order, like kg_updater)
            ordered: list[str] = []
            for r in sorted(records, key=lambda x: (x.rank if x.rank is not None else 10**6)):
                if r.brand not in ordered:
                    ordered.append(r.brand)
            top = ordered[:10]
            for i, b1 in enumerate(top):
                for b2 in top[i + 1 :]:
                    if abox.brands[b2] not in abox.brands[b1].competes_with:
                        abox.brands[b1].competes_with.append(abox.brands[b2])
        return abox

    # ----- entry point ---------------------------------------------------------

    def from_snapshot(
        self,
        records: Any,
        metrics: dict[str, Any] | None = None,
        category_hierarchy: Any = None,
        groups: dict[str, list[str]] | None = None,
        *,
        kg: KnowledgeGraph | None = None,
        snapshot_date: str | date | None = None,
        categories: list[str] | None = None,
        build_owl: bool = True,
    ) -> BuildResult:
        """Build the KG (and OWL A-Boxes) from one snapshot.

        Args:
            records: flat rank records or a CrawlerAgent ``{"categories": ...}`` payload.
            metrics: MetricsAgent payload (``brand_metrics[].share_of_shelf`` is PERCENT).
            category_hierarchy: dict (``{"categories": {id: {"parent_id": ...}}}``) or path;
                ``None`` -> ``config/category_hierarchy.json``.
            groups: ``{"AMOREPACIFIC": ["LANEIGE", ...]}``; ``None`` -> ``config/brands.json``.
            kg: KG to write into (``None`` -> a new in-memory KG).
            snapshot_date: date of the snapshot (for ``NewEntrant``); default today.
            categories: restrict the OWL A-Boxes to these categories (KG gets everything).
            build_owl: skip OWL individuals (JSON KG only) when False or owlready2 is missing.

        Raises:
            ValueError: SoS outside [0, 1] (percent value leaked into the fraction domain).
        """
        thresholds = self._thresholds or get_thresholds()
        kg = kg or KnowledgeGraph(persist_path=None, auto_load=False, auto_save=False)
        snap_date = _parse_date(snapshot_date) or date.today()

        norm = self.normalize_records(records)
        sos, extra = self._sos_by_category(norm, metrics)  # validates units first
        parents = self._hierarchy_parents(category_hierarchy)
        resolved_groups = self._resolve_groups(groups)

        relations_added = kg.load_from_crawl_data(self._to_crawl_data(norm))
        relations_added += self._load_hierarchy_into_kg(kg, category_hierarchy, parents)
        relations_added += self._load_groups_into_kg(kg, resolved_groups)

        # brand / product metadata on the KG (fraction SoS)
        for cat, by_brand in sos.items():
            for brand, fraction in by_brand.items():
                meta = kg.get_entity_metadata(brand) or {}
                by_cat = dict(meta.get("sos_by_category") or {})
                by_cat[cat] = fraction
                kg.set_entity_metadata(brand, {"type": "brand", "sos_by_category": by_cat})
        for r in norm:
            kg.set_entity_metadata(
                r.asin,
                {
                    "type": "product",
                    "name": r.name,
                    "brand": r.brand,
                    **({"first_seen": r.first_seen.isoformat()} if r.first_seen else {}),
                },
            )

        all_brands = {r.brand for r in norm}
        by_category: dict[str, list[SnapshotRecord]] = defaultdict(list)
        for r in norm:
            by_category[r.category].append(r)

        owl: OWLSnapshot | None = None
        if build_owl:
            from .tbox import OWLREADY2_AVAILABLE

            if OWLREADY2_AVAILABLE:
                wanted = [c for c in by_category if categories is None or c in categories]
                aboxes = {
                    cat: self._build_abox(
                        cat,
                        by_category[cat],
                        all_brands,
                        sos.get(cat, {}),
                        extra.get(cat, {}),
                        parents,
                        resolved_groups,
                        snap_date,
                        thresholds,
                    )
                    for cat in wanted
                }
                owl = OWLSnapshot(aboxes=aboxes, snapshot_date=snap_date, thresholds=thresholds)
            else:
                logger.warning("owlready2 not available: JSON KG built, OWL A-Box skipped")

        stats = {
            "records": len(norm),
            "brands": len(all_brands),
            "products": len({r.asin for r in norm}),
            "categories": sorted(by_category),
            "relations_added": relations_added,
            "sos_by_category": {cat: dict(v) for cat, v in sos.items()},
            "groups": resolved_groups,
            "snapshot_date": snap_date.isoformat(),
            "owl_individuals": owl.individual_count if owl else 0,
        }
        return BuildResult(kg=kg, owl=owl, stats=stats)


__all__ = [
    "BuildResult",
    "OWLSnapshot",
    "OntologyABox",
    "OntologyBuilder",
    "SnapshotRecord",
    "canonical_brand",
    "default_alias_table",
    "default_groups",
]

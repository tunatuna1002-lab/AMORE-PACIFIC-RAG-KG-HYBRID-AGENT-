"""Tests for the JSON single-source ontology loader (track O1, decisions OA-1..OA-4).

Source: config/ontology/schema.json + config/ontology/brands.json + config/category_hierarchy.json.
"""

import json
import shutil
import threading
from pathlib import Path

import pytest

from src.ontology.ontology import (
    Ontology,
    OntologyError,
    UnsupportedAxiomError,
    get_ontology,
    load_ontology,
    reset_ontology_cache,
)

REPO = Path(__file__).resolve().parents[3]
ONTOLOGY_DIR = REPO / "config" / "ontology"


@pytest.fixture(scope="module")
def onto() -> Ontology:
    return load_ontology()


def _copy_sources(tmp_path: Path) -> Path:
    target = tmp_path / "ontology"
    shutil.copytree(ONTOLOGY_DIR, target)
    return target


def _edit_json(path: Path, edit) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Loading, status, caching
# ---------------------------------------------------------------------------


class TestLoading:
    def test_status_fields(self, onto: Ontology) -> None:
        assert onto.version == "1.0.0"
        assert onto.as_of == "2026-09-18"
        assert onto.class_count == len(onto.classes) >= 10
        assert onto.brand_count == len(onto.instances_of("Brand"))
        assert onto.brand_count > 100

    def test_get_ontology_is_cached_and_resettable(self) -> None:
        reset_ontology_cache()
        first = get_ontology()
        assert get_ontology() is first
        reset_ontology_cache()
        assert get_ontology() is not first

    def test_get_ontology_thread_safe(self) -> None:
        reset_ontology_cache()
        results: list[Ontology] = []

        def worker() -> None:
            results.append(get_ontology())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len({id(r) for r in results}) == 1

    def test_deterministic(self, onto: Ontology) -> None:
        other = load_ontology()
        assert other.instances_of("Brand") == onto.instances_of("Brand")
        assert other.relations("siblingBrand") == onto.relations("siblingBrand")
        assert list(onto.instances_of("Brand")) == sorted(onto.instances_of("Brand"))


# ---------------------------------------------------------------------------
# Closure: subClassOf, defined classes, inverse, symmetric, transitive
# ---------------------------------------------------------------------------


class TestClosure:
    def test_subclass_transitive_membership(self, onto: Ontology) -> None:
        assert onto.is_a("laneige", "AmorepacificBrand")
        assert onto.is_a("laneige", "Brand")
        assert "Brand" in onto.superclasses("PremiumBrand")

    def test_defined_class_group(self, onto: Ontology) -> None:
        ap = onto.instances_of("AmorepacificBrand")
        assert "laneige" in ap
        assert "cosrx" in ap
        assert "tata_harper" in ap
        assert "elf" not in ap
        assert "tirtir" not in ap
        assert set(ap) == set(onto.brands_in_group("amorepacific"))
        assert len(ap) == 31

    def test_defined_class_origin(self, onto: Ontology) -> None:
        kb = onto.instances_of("KBeautyBrand")
        assert {"laneige", "cosrx", "tirtir", "beauty_of_joseon", "missha"} <= set(kb)
        assert "tata_harper" not in kb  # AP brand, origin USA
        assert "elf" not in kb
        # origin not stated in any source -> unknown, not K-Beauty
        assert onto.origin_of("iope") is None
        assert "iope" not in kb

    def test_defined_class_segment(self, onto: Ontology) -> None:
        assert onto.is_a("laneige", "PremiumBrand")
        assert onto.is_a("sulwhasoo", "LuxuryBrand")
        assert onto.is_a("elf", "AffordableBrand")
        assert onto.is_a("tirtir", "MidTierBrand")
        assert onto.is_a("cerave", "MassBrand")
        assert not onto.is_a("laneige", "LuxuryBrand")

    def test_inverse(self, onto: Ontology) -> None:
        assert "laneige" in onto.object_values("amorepacific", "ownsBrand")
        assert onto.object_values("laneige", "ownedByGroup") == ("amorepacific",)
        assert set(onto.object_values("skin_care", "hasSubCategory")) >= {"lip_care"}

    def test_symmetric_sibling(self, onto: Ontology) -> None:
        pairs = set(onto.relations("siblingBrand"))
        assert ("laneige", "cosrx") in pairs
        assert ("cosrx", "laneige") in pairs
        assert all((b, a) in pairs for a, b in pairs)
        assert all(a != b for a, b in pairs)

    def test_transitive_subcategory(self, onto: Ontology) -> None:
        assert "beauty" in onto.object_values("lip_care", "subCategoryOf")
        assert "lip_care" in onto.object_values("beauty", "hasSubCategory")
        assert "beauty" in onto.object_values("face_powder", "subCategoryOf")

    def test_domain_range_typing(self, onto: Ontology) -> None:
        assert onto.types_of("amorepacific") == ("CorporateGroup",)
        assert onto.types_of("south_korea") == ("Country",)
        assert onto.is_a("premium", "Segment")
        assert onto.is_a("lip_care", "Category")


# ---------------------------------------------------------------------------
# Group helpers and closed world (OE4)
# ---------------------------------------------------------------------------


class TestGroups:
    def test_group_of(self, onto: Ontology) -> None:
        assert onto.group_of("LANEIGE") == "amorepacific"
        assert onto.group_of("COSRX") == "amorepacific"
        assert onto.group_of("TIRTIR") is None
        assert onto.group_of("not a brand at all") is None

    def test_siblings(self, onto: Ontology) -> None:
        sib = onto.siblings("laneige")
        assert "laneige" not in sib
        assert "cosrx" in sib and "sulwhasoo" in sib
        assert len(sib) == 30
        assert list(sib) == sorted(sib)
        assert onto.siblings("tirtir") == ()

    def test_closed_world_rl015_rl016(self, onto: Ontology) -> None:
        # registry brands whose sources state no group -> definitely not AP (OE4)
        assert onto.closed_world_member("TIRTIR", "amorepacific") is False
        assert onto.closed_world_member("Beauty of Joseon", "AMOREPACIFIC") is False
        assert onto.closed_world_member("LANEIGE", "아모레퍼시픽") is True
        # brand outside the registry -> unknown
        assert onto.closed_world_member("Some New Indie Brand", "amorepacific") is None
        # placeholders are not brands -> unknown
        assert onto.closed_world_member("unknown", "amorepacific") is None

    def test_segment_origin_acquired(self, onto: Ontology) -> None:
        assert onto.segment_of("LANEIGE") == "premium"
        assert onto.segment_of("e.l.f.") == "affordable"
        assert onto.segment_of("abib") is None
        assert onto.origin_of("COSRX") == "south_korea"
        assert onto.origin_of("TATA HARPER") == "usa"
        assert onto.acquired_in("COSRX") == 2024
        assert onto.acquired_in("TATA HARPER") is None  # source says acquired=true, no year


# ---------------------------------------------------------------------------
# Normalization and placeholders
# ---------------------------------------------------------------------------


class TestNormalization:
    @pytest.mark.parametrize("text", ["e.l.f.", "ELF", "elf", "E.L.F.", " e.l.f "])
    def test_elf_variants(self, onto: Ontology, text: str) -> None:
        assert onto.normalize_brand(text) == "elf"

    @pytest.mark.parametrize("text", ["LANEIGE", "Laneige", "laneige", "라네즈"])
    def test_laneige_variants(self, onto: Ontology, text: str) -> None:
        assert onto.normalize_brand(text) == "laneige"

    def test_symbol_and_accent_variants(self, onto: Ontology) -> None:
        assert onto.normalize_brand("La Roche Posay") == "la_roche_posay"
        assert onto.normalize_brand("la roche-posay") == "la_roche_posay"
        assert onto.normalize_brand("Mise-en-scene") == "mise_en_scene"
        assert onto.normalize_brand("L’Oreal") == "loreal"
        assert onto.normalize_brand("beauty_of_joseon") == "beauty_of_joseon"

    def test_unknown_and_group_strings(self, onto: Ontology) -> None:
        assert onto.normalize_brand("") is None
        assert onto.normalize_brand("totally new brand") is None
        # the corporate group is not a brand; the 'Amore Pacific' brand line is
        assert onto.normalize_brand("AMOREPACIFIC") is None
        assert onto.normalize_group("AMOREPACIFIC") == "amorepacific"
        assert onto.normalize_group("아모레퍼시픽") == "amorepacific"
        assert onto.normalize_brand("Amore Pacific") == "amore_pacific"

    @pytest.mark.parametrize("text", ["unknown", "Fresh", "fresh", "chi"])
    def test_placeholders(self, onto: Ontology, text: str) -> None:
        bid = onto.normalize_brand(text)
        assert bid is not None
        assert onto.is_placeholder(bid)
        assert onto.is_placeholder(text)
        assert not onto.is_a(bid, "Brand")
        assert onto.is_a(bid, "PlaceholderBrand")
        assert bid not in onto.instances_of("Brand")

    def test_real_brand_not_placeholder(self, onto: Ontology) -> None:
        assert not onto.is_placeholder("laneige")
        assert not onto.is_placeholder("no such brand")


# ---------------------------------------------------------------------------
# Categories and OE3
# ---------------------------------------------------------------------------


class TestCategories:
    def test_ancestors(self, onto: Ontology) -> None:
        assert onto.category_ancestors("lip_care") == ("skin_care", "beauty")
        assert onto.category_ancestors("face_powder") == ("face_makeup", "makeup", "beauty")
        assert onto.category_ancestors("beauty") == ()

    def test_descendants(self, onto: Ontology) -> None:
        desc = onto.category_descendants("skin_care")
        assert "lip_care" in desc
        assert list(desc) == sorted(desc)
        assert "face_powder" in onto.category_descendants("beauty")
        assert "lip_makeup" not in desc  # Lip Makeup is under Makeup, not Skin Care

    def test_normalize_category(self, onto: Ontology) -> None:
        assert onto.normalize_category("Lip Care") == "lip_care"
        assert onto.normalize_category("lip_care") == "lip_care"
        assert onto.normalize_category("nope") is None

    def test_no_numeric_aggregation_api(self, onto: Ontology) -> None:
        """OE3: category inclusion never aggregates metrics."""
        forbidden = ("aggregate", "sum", "rollup", "roll_up", "total", "propagate")
        public = [n for n in dir(onto) if not n.startswith("_")]
        assert not [n for n in public if any(f in n.lower() for f in forbidden)]

    def test_numeric_predicates_not_transitive(self, onto: Ontology) -> None:
        for name in ("hasSoS", "hasHHI", "hasPricePosition"):
            spec = onto.predicate_spec(name)
            assert spec is not None
            assert spec.numeric and spec.requires_as_of
            assert not spec.transitive and not spec.static
        assert onto.numeric_propagation_allowed is False

    def test_numeric_propagation_flag_rejected(self, tmp_path: Path) -> None:
        src = _copy_sources(tmp_path)
        _edit_json(
            src / "schema.json",
            lambda d: d["category_inclusion"].__setitem__("numeric_propagation", True),
        )
        with pytest.raises(OntologyError, match="OE3"):
            load_ontology(src)


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


class TestPredicates:
    def test_canonical_predicate(self, onto: Ontology) -> None:
        assert onto.canonical_predicate("ownedBy") == "ownedByGroup"
        assert onto.canonical_predicate("ownedByGroup") == "ownedByGroup"
        assert onto.canonical_predicate("PRICE_POSITION") == "hasPricePosition"
        assert onto.canonical_predicate("parentCategory") == "subCategoryOf"
        assert onto.canonical_predicate("noSuchPredicate") is None
        # hasPosition is not one alias: it must be split by original_predicate
        assert onto.canonical_predicate("hasPosition") is None

    def test_resolve_kg_predicate(self, onto: Ontology) -> None:
        assert onto.resolve_kg_predicate("hasPosition", "hasSoS") == "hasSoS"
        assert onto.resolve_kg_predicate("hasPosition", "hasHHI") == "hasHHI"
        assert onto.resolve_kg_predicate("hasPosition", "PRICE_POSITION") == "hasPricePosition"
        assert onto.resolve_kg_predicate("hasPosition") is None
        assert onto.resolve_kg_predicate("belongsToCategory", "rankedIn") == "rankedIn"
        assert onto.resolve_kg_predicate("belongsToCategory") == "belongsToCategory"
        assert onto.resolve_kg_predicate("HAS_PRODUCT") == "hasProduct"

    def test_predicate_spec(self, onto: Ontology) -> None:
        spec = onto.predicate_spec("ownedBy")
        assert spec is not None and spec.name == "ownedByGroup"
        assert spec.inverse_of == "ownsBrand"
        assert onto.predicate_spec("siblingBrand").symmetric
        assert onto.predicate_spec("competesWith").symmetric
        assert onto.predicate_spec("subCategoryOf").transitive
        assert onto.predicate_spec("hasProduct").inverse_of == "hasBrand"
        assert onto.predicate_spec("belongsToCategory").domain == "Product"
        assert onto.predicate_spec("rankedIn").domain == "Brand"
        assert onto.predicate_spec("nope") is None

    def test_validate_triple_ok(self, onto: Ontology) -> None:
        assert onto.validate_triple("LANEIGE", "ownedBy", "AMOREPACIFIC") == []
        assert onto.validate_triple("laneige", "hasSoS", "lip_care", {"as_of": "2026-08-31"}) == []
        assert onto.validate_triple("B07XXPHQZK", "belongsToCategory", "lip_care") == []
        assert onto.validate_triple("lip_care", "subCategoryOf", "skin_care") == []

    def test_validate_triple_violations(self, onto: Ontology) -> None:
        assert any("unknown predicate" in v for v in onto.validate_triple("laneige", "likes", "x"))
        assert any(
            "original_predicate" in v for v in onto.validate_triple("laneige", "hasPosition", "x")
        )
        # domain mismatch: belongsToCategory is Product -> Category
        assert any(
            "domain" in v for v in onto.validate_triple("laneige", "belongsToCategory", "lip_care")
        )
        # range mismatch: ownedByGroup range CorporateGroup
        assert any("range" in v for v in onto.validate_triple("laneige", "ownedByGroup", "cosrx"))
        assert any(
            "placeholder" in v
            for v in onto.validate_triple("unknown", "hasSoS", "beauty", {"as_of": "2026-08-31"})
        )
        assert any("as_of" in v for v in onto.validate_triple("laneige", "hasSoS", "lip_care", {}))
        assert any("as_of" in v for v in onto.validate_triple("beauty", "hasHHI", "1312"))
        assert any(
            "allowed" in v
            for v in onto.validate_triple(
                "laneige", "hasPricePosition", "cheap", {"as_of": "2026-08-31"}
            )
        )
        assert any(
            "xsd:integer" in v for v in onto.validate_triple("tata_harper", "acquiredIn", "True")
        )

    def test_validate_triple_output_sorted(self, onto: Ontology) -> None:
        out = onto.validate_triple("unknown", "hasSoS", "cosrx", {})
        assert out == sorted(out)
        assert len(out) >= 3


# ---------------------------------------------------------------------------
# Registry self-consistency
# ---------------------------------------------------------------------------


class TestRegistryConsistency:
    def test_self_check_zero_violations(self, onto: Ontology) -> None:
        assert onto.self_check() == []

    def test_values_are_declared_individuals(self, onto: Ontology) -> None:
        reg = json.loads((ONTOLOGY_DIR / "brands.json").read_text(encoding="utf-8"))
        for b in reg["brands"]:
            if b["group"] is not None:
                assert onto.is_a(b["group"], "CorporateGroup"), b["id"]
            if b["segment"] is not None:
                assert onto.is_a(b["segment"], "Segment"), b["id"]
            if b["origin"] is not None:
                assert onto.is_a(b["origin"], "Country"), b["id"]

    def test_every_alias_normalizes_to_own_brand(self, onto: Ontology) -> None:
        reg = json.loads((ONTOLOGY_DIR / "brands.json").read_text(encoding="utf-8"))
        for b in reg["brands"]:
            for text in [b["id"], b["name"], *b["aliases"]]:
                assert onto.normalize_brand(text) == b["id"], (text, b["id"])

    def test_registry_sorted_and_meta_matches(self) -> None:
        reg = json.loads((ONTOLOGY_DIR / "brands.json").read_text(encoding="utf-8"))
        ids = [b["id"] for b in reg["brands"]]
        assert ids == sorted(ids)
        meta = reg["_meta"]
        brands = reg["brands"]
        assert meta["brands_total"] == len(brands)
        assert meta["with_group"] == sum(b["group"] is not None for b in brands)
        assert meta["with_segment"] == sum(b["segment"] is not None for b in brands)
        assert meta["with_origin"] == sum(b["origin"] is not None for b in brands)
        assert meta["placeholders"] == [b["id"] for b in brands if b["is_placeholder"]]
        assert set(meta["placeholders"]) == {"unknown", "fresh", "chi"}
        assert all(b["sources"] for b in brands)

    def test_duplicate_alias_rejected(self, tmp_path: Path) -> None:
        src = _copy_sources(tmp_path)

        def edit(d: dict) -> None:
            for b in d["brands"]:
                if b["id"] == "tirtir":
                    b["aliases"].append("Laneige")

        _edit_json(src / "brands.json", edit)
        with pytest.raises(OntologyError, match="alias"):
            load_ontology(src)

    def test_undeclared_value_rejected(self, tmp_path: Path) -> None:
        src = _copy_sources(tmp_path)

        def edit(d: dict) -> None:
            for b in d["brands"]:
                if b["id"] == "tirtir":
                    b["origin"] = "atlantis"

        _edit_json(src / "brands.json", edit)
        with pytest.raises(OntologyError, match="atlantis"):
            load_ontology(src)


# ---------------------------------------------------------------------------
# OA-3: unsupported axiom forms fail loudly
# ---------------------------------------------------------------------------


class TestUnsupportedAxioms:
    def test_union_of_rejected(self, tmp_path: Path) -> None:
        src = _copy_sources(tmp_path)
        _edit_json(
            src / "schema.json",
            lambda d: d["classes"]["Brand"].__setitem__("unionOf", ["Product", "Metric"]),
        )
        with pytest.raises(UnsupportedAxiomError, match="unionOf"):
            load_ontology(src)

    def test_property_chain_rejected(self, tmp_path: Path) -> None:
        src = _copy_sources(tmp_path)
        _edit_json(
            src / "schema.json",
            lambda d: d["predicates"]["siblingBrand"].__setitem__(
                "propertyChain", ["ownedByGroup", "ownsBrand"]
            ),
        )
        with pytest.raises(UnsupportedAxiomError, match="propertyChain"):
            load_ontology(src)

    def test_defined_by_extra_key_rejected(self, tmp_path: Path) -> None:
        src = _copy_sources(tmp_path)
        _edit_json(
            src / "schema.json",
            lambda d: d["classes"]["KBeautyBrand"]["defined_by"].__setitem__("some", True),
        )
        with pytest.raises(UnsupportedAxiomError, match="defined_by"):
            load_ontology(src)

    def test_unknown_top_level_key_rejected(self, tmp_path: Path) -> None:
        src = _copy_sources(tmp_path)
        _edit_json(src / "schema.json", lambda d: d.__setitem__("swrl_rules", []))
        with pytest.raises(UnsupportedAxiomError, match="swrl_rules"):
            load_ontology(src)

    def test_disjoint_violation_detected(self, tmp_path: Path) -> None:
        src = _copy_sources(tmp_path)

        def edit(d: dict) -> None:
            # a brand pointing at a category as its group -> range typing makes the
            # category a CorporateGroup, which is disjoint with Category
            d["groups"].append({"id": "lip_care", "name": "x", "aliases": [], "sources": ["t"]})

        _edit_json(src / "brands.json", edit)
        with pytest.raises(OntologyError):
            load_ontology(src)

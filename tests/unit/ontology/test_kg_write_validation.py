"""KG write validation (track O5, decision OA-7).

Mode flag ``kg.write_validation`` = ``off`` | ``warn`` (default) | ``enforce``.

- ``warn`` logs violations (aggregated) and stores exactly what ``off`` stores.
- ``enforce`` canonicalizes predicates/brands, blocks placeholder brands and undated
  numeric edges, and records entity types.

KnowledgeGraph instances here always use a tmp ``persist_path`` + ``auto_save=False`` so the
production KG file is never touched.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.domain.entities.relations import Relation, RelationType
from src.ontology.kg_enricher import KGEnricher
from src.ontology.kg_write_validation import (
    WriteValidationStats,
    canonical_brand_string,
    check_triple,
    get_write_validation_mode,
    normalize_triple,
)
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.ontology import get_ontology

ENV = "FF_KG_WRITE_VALIDATION"


@pytest.fixture
def onto():
    return get_ontology()


@pytest.fixture
def make_kg(tmp_path: Path):
    def _make(name: str = "kg.json") -> KnowledgeGraph:
        return KnowledgeGraph(persist_path=str(tmp_path / name), auto_save=False, auto_load=False)

    return _make


# ---------------------------------------------------------------------------
# Mode flag
# ---------------------------------------------------------------------------


class TestMode:
    def test_default_is_warn(self, monkeypatch):
        monkeypatch.delenv(ENV, raising=False)
        assert get_write_validation_mode() == "warn"

    @pytest.mark.parametrize("value", ["off", "warn", "enforce", " ENFORCE "])
    def test_env_values(self, monkeypatch, value):
        monkeypatch.setenv(ENV, value)
        assert get_write_validation_mode() == value.strip().lower()

    def test_invalid_value_falls_back_to_warn(self, monkeypatch):
        monkeypatch.setenv(ENV, "strict")
        assert get_write_validation_mode() == "warn"


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


class TestCanonicalBrandString:
    def test_lowercased_registry_name(self, onto):
        assert canonical_brand_string(onto, "LANEIGE") == "laneige"
        assert canonical_brand_string(onto, "laneige") == "laneige"
        assert canonical_brand_string(onto, "elf") == "e.l.f."
        assert canonical_brand_string(onto, "e.l.f.") == "e.l.f."

    def test_unregistered_brand_is_none(self, onto):
        assert canonical_brand_string(onto, "Totally Unknown Brand Xyz") is None

    def test_reader_variant_finds_canonical_form(self, onto):
        # hybrid_retriever._query_knowledge_graph tries (brand, lower, upper, title);
        # the entity linker emits registry-style names such as "LANEIGE" / "e.l.f.".
        for emitted in ("LANEIGE", "COSRX", "e.l.f.", "Beauty of Joseon", "La Roche-Posay"):
            canon = canonical_brand_string(onto, emitted)
            if canon is None:
                continue
            variants = {emitted, emitted.lower(), emitted.upper(), emitted.title()}
            assert canon in variants, (emitted, canon)


class TestCheckTriple:
    def test_legacy_sos_edge(self, onto):
        codes = check_triple(
            onto, "LANEIGE", "hasPosition", "lip_care", {"original_predicate": "hasSoS"}
        )
        assert "non_canonical_predicate" in codes
        assert "non_canonical_brand" in codes
        assert "missing_as_of" in codes

    def test_clean_edge_has_no_violation(self, onto):
        codes = check_triple(
            onto, "laneige", "hasSoS", "lip_care", {"sos_pct": 5.0, "as_of": "2026-09-18"}
        )
        assert codes == []

    def test_placeholder(self, onto):
        codes = check_triple(onto, "unknown", "competesWith", "laneige", {})
        assert "placeholder_brand" in codes

    def test_outside_ontology_is_informational(self, onto):
        assert check_triple(onto, "B0ABCDEFGH", "hasAlert", "rank_drop", {}) == ["outside_ontology"]

    def test_kept_category_alias_is_not_flagged(self, onto):
        assert "non_canonical_predicate" not in check_triple(
            onto, "lip_care", "parentCategory", "skin_care", {}
        )

    def test_unresolvable_legacy(self, onto):
        codes = check_triple(
            onto, "laneige", "hasPosition", "lip_care", {"original_predicate": "DOMINATES_CATEGORY"}
        )
        assert codes == ["unresolvable_legacy_predicate"]


class TestNormalizeTriple:
    def test_split_has_position(self, onto):
        r = normalize_triple(
            onto,
            "LANEIGE",
            "hasPosition",
            "lip_care",
            {"original_predicate": "hasSoS", "sos_pct": 5.0},
            as_of="2026-09-18",
        )
        assert r.blocked is None
        assert (r.subject, r.predicate, r.object) == ("laneige", "hasSoS", "lip_care")
        assert r.properties["original_predicate"] == "hasSoS"
        assert r.properties["as_of"] == "2026-09-18"
        assert "Brand" in r.types["laneige"]

    def test_hhi_and_price_position(self, onto):
        hhi = normalize_triple(
            onto, "lip_care", "hasPosition", "812.5", {"original_predicate": "hasHHI"}, as_of="d"
        )
        assert hhi.predicate == "hasHHI" and hhi.blocked is None
        price = normalize_triple(
            onto,
            "cosrx",
            "hasPosition",
            "budget",
            {"original_predicate": "PRICE_POSITION"},
            as_of="d",
        )
        assert price.predicate == "hasPricePosition" and price.blocked is None

    def test_ranked_in_split(self, onto):
        r = normalize_triple(
            onto, "laneige", "belongsToCategory", "lip_care", {"original_predicate": "rankedIn"}
        )
        assert r.predicate == "rankedIn"
        assert r.blocked is None

    def test_owned_by_alias(self, onto):
        r = normalize_triple(onto, "LANEIGE", "ownedBy", "AMOREPACIFIC", {})
        assert r.predicate == "ownedByGroup"
        assert r.properties["original_predicate"] == "ownedBy"
        assert r.subject == "laneige"

    def test_missing_as_of_blocks_by_default(self, onto):
        r = normalize_triple(
            onto, "laneige", "hasPosition", "lip_care", {"original_predicate": "hasSoS"}
        )
        assert r.blocked == "missing_as_of"

    def test_missing_as_of_keep_mode(self, onto):
        r = normalize_triple(
            onto,
            "laneige",
            "hasPosition",
            "lip_care",
            {"original_predicate": "hasSoS"},
            on_missing_as_of="keep",
        )
        assert r.blocked is None
        assert "undated_numeric" in r.notes

    def test_placeholder_blocked(self, onto):
        r = normalize_triple(onto, "laneige", "competesWith", "unknown", {})
        assert r.blocked == "placeholder_brand"

    def test_unresolvable_legacy_blocked(self, onto):
        r = normalize_triple(
            onto, "laneige", "hasPosition", "lip_care", {"original_predicate": "DOMINATES_CATEGORY"}
        )
        assert r.blocked == "unresolvable_legacy_predicate"

    def test_outside_ontology_passes_unchanged(self, onto):
        props = {"severity": "high"}
        r = normalize_triple(onto, "B0ABCDEFGH", "hasAlert", "rank_drop", props)
        assert r.blocked is None
        assert (r.subject, r.predicate, r.object, r.properties) == (
            "B0ABCDEFGH",
            "hasAlert",
            "rank_drop",
            props,
        )
        assert r.changes == []

    def test_input_properties_not_mutated(self, onto):
        props = {"original_predicate": "hasSoS"}
        normalize_triple(onto, "LANEIGE", "hasPosition", "lip_care", props, as_of="d")
        assert props == {"original_predicate": "hasSoS"}


class TestStats:
    def test_examples_are_capped(self):
        stats = WriteValidationStats(max_examples=2)
        first = [stats.record(["x"], f"t{i}") for i in range(10)]
        assert stats.counts["x"] == 10
        assert len(stats.examples["x"]) == 2
        assert sum(1 for new in first if new) == 2


# ---------------------------------------------------------------------------
# KnowledgeGraph write path
# ---------------------------------------------------------------------------


CRAWL = {
    "category": "lip_care",
    "products": [
        {"asin": "B0AAAAAAA1", "brand": "LANEIGE", "rank": 1, "price": 24.0, "title": "Mask"},
        {"asin": "B0AAAAAAA2", "brand": "LANEIGE", "rank": 3, "price": 22.0, "title": "Balm"},
        {"asin": "B0AAAAAAA3", "brand": "COSRX", "rank": 2, "price": 12.0, "title": "Sleep"},
        {"asin": "B0AAAAAAA4", "brand": "COSRX", "rank": 5, "price": 10.0, "title": "Butter"},
        {"asin": "B0AAAAAAA5", "brand": "Unknown", "rank": 4, "price": 5.0, "title": "X"},
        {"asin": "B0AAAAAAA6", "brand": "Unknown", "rank": 6, "price": 35.0, "title": "Y"},
    ],
}

UPDATER_CRAWL = {
    "categories": {
        "lip_care": {
            "rank_records": [
                {"brand": "LANEIGE", "asin": "B0AAAAAAA1", "title": "Mask", "rank": 1},
                {"brand": "COSRX", "asin": "B0AAAAAAA3", "title": "Sleep", "rank": 2},
                {"asin": "B0AAAAAAA5", "title": "no brand", "rank": 3},
            ]
        }
    }
}


def _run_writers(kg: KnowledgeGraph) -> None:
    KGEnricher(knowledge_graph=kg).enrich_and_store(CRAWL)
    kg.load_from_crawl_data(UPDATER_CRAWL)
    kg.load_brand_ownership()
    kg.load_category_hierarchy()


def _stable_dump(kg: KnowledgeGraph, path: Path) -> str:
    kg.save(path=str(path), force=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("saved_at", None)
    for t in data["triples"]:
        t.pop("created_at", None)  # wall clock (Relation default_factory=datetime.now)
    return json.dumps(data, ensure_ascii=False, indent=2)


class TestKnowledgeGraphModes:
    def test_warn_stores_exactly_what_off_stores(self, monkeypatch, tmp_path, make_kg):
        monkeypatch.setenv(ENV, "off")
        kg_off = make_kg("off.json")
        _run_writers(kg_off)
        monkeypatch.setenv(ENV, "warn")
        kg_warn = make_kg("warn.json")
        _run_writers(kg_warn)
        assert _stable_dump(kg_off, tmp_path / "off.json") == _stable_dump(
            kg_warn, tmp_path / "warn.json"
        )
        assert kg_warn.get_write_validation_summary()["counts"]  # violations were seen
        assert not kg_off.get_write_validation_summary()["counts"]

    def test_warn_logging_is_aggregated(self, monkeypatch, make_kg, caplog):
        monkeypatch.setenv(ENV, "warn")
        kg = make_kg()
        with caplog.at_level(logging.WARNING, logger="src.ontology.knowledge_graph"):
            for i in range(200):
                kg.add_relation(
                    Relation(
                        "LANEIGE",
                        RelationType.HAS_POSITION,
                        f"cat_{i}",
                        properties={"original_predicate": "hasSoS"},
                    )
                )
        summary = kg.get_write_validation_summary()
        assert summary["counts"]["non_canonical_brand"] == 200
        assert len(caplog.records) <= 3 * len(summary["counts"])
        assert len(kg.triples) == 200  # warn never drops

    def test_enforce_canonicalizes(self, monkeypatch, make_kg):
        monkeypatch.setenv(ENV, "enforce")
        kg = make_kg()
        added = kg.add_relation(
            Relation(
                "LANEIGE",
                RelationType.HAS_POSITION,
                "lip_care",
                properties={"original_predicate": "hasSoS", "sos_pct": 5.0, "as_of": "2026-09-18"},
            )
        )
        assert added
        (rel,) = kg.triples
        assert (rel.subject, rel.predicate, rel.object) == (
            "laneige",
            RelationType.HAS_SOS,
            "lip_care",
        )
        assert rel.properties["original_predicate"] == "hasSoS"
        assert kg.entity_metadata["laneige"]["type"] == "brand"
        assert "Brand" in kg.entity_metadata["laneige"]["ontology_types"]
        assert kg.entity_metadata["lip_care"]["type"] == "category"

    def test_enforce_blocks_and_counts(self, monkeypatch, make_kg):
        monkeypatch.setenv(ENV, "enforce")
        kg = make_kg()
        assert not kg.add_relation(Relation("unknown", RelationType.COMPETES_WITH, "laneige"))
        assert not kg.add_relation(
            Relation(
                "laneige",
                RelationType.HAS_POSITION,
                "lip_care",
                properties={"original_predicate": "hasSoS"},
            )
        )
        counts = kg.get_write_validation_summary()["counts"]
        assert counts["blocked:placeholder_brand"] == 1
        assert counts["blocked:missing_as_of"] == 1
        assert kg.triples == []

    def test_load_is_never_validated(self, monkeypatch, tmp_path):
        path = tmp_path / "legacy.json"
        monkeypatch.setenv(ENV, "off")
        kg = KnowledgeGraph(persist_path=str(path), auto_save=False, auto_load=False)
        kg.add_relation(Relation("unknown", RelationType.COMPETES_WITH, "LANEIGE"))
        kg.save(force=True)
        monkeypatch.setenv(ENV, "enforce")
        kg2 = KnowledgeGraph(persist_path=str(path), auto_save=False, auto_load=True)
        assert [(r.subject, r.object) for r in kg2.triples] == [("unknown", "LANEIGE")]

    def test_migrated_predicates_load(self, tmp_path):
        path = tmp_path / "kg.json"
        rel = Relation("laneige", RelationType.HAS_SOS, "lip_care", properties={"as_of": "d"})
        path.write_text(
            json.dumps({"version": "2.0", "triples": [rel.to_dict()], "entity_metadata": {}}),
            encoding="utf-8",
        )
        kg = KnowledgeGraph(persist_path=str(path), auto_save=False, auto_load=True)
        assert kg.triples[0].predicate is RelationType.HAS_SOS


class TestEnricherEnforce:
    def test_as_of_attached_and_placeholders_blocked(self, monkeypatch, make_kg):
        monkeypatch.setenv(ENV, "enforce")
        kg = make_kg()
        KGEnricher(knowledge_graph=kg).enrich_and_store({**CRAWL, "as_of": "2026-09-18"})
        preds = {(r.subject, r.predicate.value) for r in kg.triples}
        assert ("laneige", "hasSoS") in preds
        assert ("lip_care", "hasHHI") in preds
        assert ("laneige", "rankedIn") in preds
        assert all(r.subject != "unknown" and r.object != "unknown" for r in kg.triples)
        numeric = [r for r in kg.triples if r.predicate.value in {"hasSoS", "hasHHI"}]
        assert numeric and all(r.properties["as_of"] == "2026-09-18" for r in numeric)
        assert all(r.predicate is not RelationType.HAS_POSITION for r in kg.triples)

    def test_undated_numeric_blocked(self, monkeypatch, make_kg):
        monkeypatch.setenv(ENV, "enforce")
        kg = make_kg()
        KGEnricher(knowledge_graph=kg).enrich_and_store(CRAWL)
        assert not [r for r in kg.triples if r.predicate.value in {"hasSoS", "hasHHI"}]
        assert kg.get_write_validation_summary()["counts"]["blocked:missing_as_of"] > 0

    def test_warn_does_not_add_as_of(self, monkeypatch, make_kg):
        monkeypatch.setenv(ENV, "warn")
        kg = make_kg()
        KGEnricher(knowledge_graph=kg).enrich_and_store({**CRAWL, "as_of": "2026-09-18"})
        assert all("as_of" not in r.properties for r in kg.triples)

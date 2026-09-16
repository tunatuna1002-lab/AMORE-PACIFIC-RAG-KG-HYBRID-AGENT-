"""F9-2: OntologyBuilder.from_snapshot -> BuildResult(kg, owl, stats)."""

from __future__ import annotations

import pytest

from src.domain.entities.relations import RelationType
from src.ontology.builder import BuildResult, OntologyBuilder, canonical_brand


class TestCanonicalBrand:
    def test_case_variants_collapse(self) -> None:
        table = {"laneige": "LANEIGE"}
        assert canonical_brand("LANEIGE", table) == "LANEIGE"
        assert canonical_brand("laneige", table) == "LANEIGE"
        assert canonical_brand("  Laneige ", table) == "LANEIGE"

    def test_alias_table_from_config(self) -> None:
        # config/brands.json: target aliases + amorepacific brands + competitor aliases
        assert canonical_brand("라네즈") == "LANEIGE"
        assert canonical_brand("Laneige") == "LANEIGE"
        assert canonical_brand("cosrx") == "COSRX"
        assert canonical_brand("INNISFREE") == canonical_brand("innisfree")

    def test_unknown_brand_case_folds_to_first_spelling(self) -> None:
        b = OntologyBuilder(alias_table={})
        assert b.canonical_brand("Summer Fridays") == "Summer Fridays"
        assert b.canonical_brand("SUMMER FRIDAYS") == "Summer Fridays"

    def test_empty_or_none(self) -> None:
        assert canonical_brand("") == "Unknown"
        assert canonical_brand(None) == "Unknown"


class TestFromSnapshot:
    @pytest.fixture
    def result(self, snapshot_records, snapshot_metrics, hierarchy, groups, fresh_kg):
        return OntologyBuilder().from_snapshot(
            snapshot_records,
            snapshot_metrics,
            category_hierarchy=hierarchy,
            groups=groups,
            kg=fresh_kg,
            snapshot_date="2026-09-10",
            build_owl=False,
        )

    def test_returns_build_result(self, result, fresh_kg) -> None:
        assert isinstance(result, BuildResult)
        assert result.kg is fresh_kg
        assert result.owl is None
        assert result.stats["relations_added"] > 0
        assert result.stats["brands"] >= 10
        assert result.stats["products"] == 27  # unique ASINs in the fixture

    def test_single_subject_per_brand(self, result) -> None:
        kg = result.kg
        subjects = {s for s in kg.subject_index if s.lower() == "laneige"}
        assert subjects == {"LANEIGE"}
        laneige_products = {r.object for r in kg.query("LANEIGE", RelationType.HAS_PRODUCT)}
        assert laneige_products == {"B0LAN001", "B0LAN002", "B0LAN003", "B0LAN010"}

    def test_hierarchy_and_group_triples(self, result) -> None:
        kg = result.kg
        parents = {r.object for r in kg.query("lip_care", RelationType.PARENT_CATEGORY)}
        assert parents == {"skin_care"}
        owners = {r.object for r in kg.query("LANEIGE", RelationType.OWNED_BY_GROUP)}
        assert owners == {"AMOREPACIFIC"}
        owned = {r.object for r in kg.query("AMOREPACIFIC", RelationType.OWNS_BRAND)}
        assert canonical_brand("INNISFREE") in owned and "LANEIGE" in owned

    def test_brand_sos_is_fraction_from_metrics_and_records(self, result) -> None:
        sos = result.stats["sos_by_category"]
        assert sos["lip_care"]["LANEIGE"] == pytest.approx(0.30)
        assert sos["lip_care"]["COSRX"] == pytest.approx(0.20)
        # not in metrics -> computed from records (3 of 10)
        assert sos["skin_care"]["COSRX"] == pytest.approx(0.30)
        assert sos["beauty"]["LANEIGE"] == pytest.approx(0.10)

    def test_crawl_data_shape_accepted(self, hierarchy, fresh_kg) -> None:
        crawl = {
            "categories": {
                "lip_care": {
                    "rank_records": [
                        {"brand": "LANEIGE", "asin": "B1", "product_name": "x", "rank": 1},
                        {"brand": "laneige", "asin": "B2", "product_name": "y", "rank": 2},
                    ]
                }
            }
        }
        r = OntologyBuilder().from_snapshot(crawl, None, kg=fresh_kg, build_owl=False)
        assert {x.object for x in fresh_kg.query("LANEIGE", RelationType.HAS_PRODUCT)} == {
            "B1",
            "B2",
        }
        assert r.stats["sos_by_category"]["lip_care"]["LANEIGE"] == pytest.approx(1.0)

    def test_percent_scale_sos_rejected(self, snapshot_records, fresh_kg) -> None:
        bad = {
            "brand_metrics": [
                {"brand_name": "LANEIGE", "category_id": "lip_care", "share_of_shelf": 150.0}
            ]
        }
        with pytest.raises(ValueError):
            OntologyBuilder().from_snapshot(snapshot_records, bad, kg=fresh_kg, build_owl=False)

    def test_groups_default_from_config(self, snapshot_records, fresh_kg) -> None:
        r = OntologyBuilder().from_snapshot(snapshot_records, None, kg=fresh_kg, build_owl=False)
        owners = {x.object for x in fresh_kg.query("LANEIGE", RelationType.OWNED_BY_GROUP)}
        assert owners == {"AMOREPACIFIC"}
        assert "AMOREPACIFIC" in r.stats["groups"]


@pytest.mark.skipif(
    pytest.importorskip("owlready2", reason="owlready2 missing") is None, reason="no owlready2"
)
class TestOwlABox:
    def test_owl_individuals_from_same_entities(
        self, snapshot_records, snapshot_metrics, hierarchy, groups, fresh_kg
    ) -> None:
        r = OntologyBuilder().from_snapshot(
            snapshot_records,
            snapshot_metrics,
            category_hierarchy=hierarchy,
            groups=groups,
            kg=fresh_kg,
            snapshot_date="2026-09-10",
        )
        assert r.owl is not None
        assert set(r.owl.aboxes) == {"lip_care", "skin_care", "beauty"}
        lip = r.owl.aboxes["lip_care"]
        laneige = lip.brand("LANEIGE")
        assert laneige.share_of_shelf == pytest.approx(0.30)
        assert {b.name for b in laneige.owned_by_group.owns_brand} >= {"LANEIGE"}
        assert lip.onto.Product("B0LAN002").days_since_first_seen == 3
        assert lip.onto.Category("lip_care").parent_category[0].name == "skin_care"
        assert r.stats["owl_individuals"] > 0

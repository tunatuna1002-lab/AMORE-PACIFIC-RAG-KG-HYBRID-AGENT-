"""Tests for KGEnricher (4.4)"""

from unittest.mock import MagicMock

import pytest

from src.ontology.kg_enricher import KGEnricher, Triple


@pytest.fixture
def sample_crawl_data():
    return {
        "category": "lip_care",
        "products": [
            {
                "asin": "B001",
                "brand": "LANEIGE",
                "rank": 1,
                "price": 24.0,
                "title": "Lip Sleeping Mask",
            },
            {
                "asin": "B002",
                "brand": "LANEIGE",
                "rank": 3,
                "price": 22.0,
                "title": "Lip Glowy Balm",
            },
            {
                "asin": "B003",
                "brand": "COSRX",
                "rank": 2,
                "price": 12.0,
                "title": "Lip Sleep Balm",
            },
            {
                "asin": "B004",
                "brand": "COSRX",
                "rank": 5,
                "price": 10.0,
                "title": "Lip Butter",
            },
            {
                "asin": "B005",
                "brand": "Burt's Bees",
                "rank": 4,
                "price": 5.0,
                "title": "Lip Balm",
            },
            {
                "asin": "B006",
                "brand": "Rare Beauty",
                "rank": 6,
                "price": 35.0,
                "title": "Soft Lip",
            },
        ],
    }


@pytest.fixture
def mock_kg():
    kg = MagicMock()
    kg.add_relation.return_value = None
    return kg


class TestTriple:
    def test_create(self):
        t = Triple(subject="laneige", predicate="HAS_PRODUCT", object="B001")
        assert t.confidence == 1.0
        assert t.properties == {}

    def test_with_properties(self):
        t = Triple(
            subject="laneige",
            predicate="HAS_PRODUCT",
            object="B001",
            properties={"rank": 1},
        )
        assert t.properties["rank"] == 1


class TestBrandProductExtraction:
    def test_extracts_brand_product_relations(self, sample_crawl_data):
        enricher = KGEnricher()
        triples = enricher._extract_brand_product_relations(
            sample_crawl_data["products"], "lip_care"
        )
        has_product = [t for t in triples if t.predicate == "HAS_PRODUCT"]
        belongs_to = [t for t in triples if t.predicate == "BELONGS_TO_CATEGORY"]
        assert len(has_product) == 6
        assert len(belongs_to) == 6

    def test_deduplicates(self, sample_crawl_data):
        # Duplicate products
        products = sample_crawl_data["products"] + sample_crawl_data["products"]
        enricher = KGEnricher()
        triples = enricher._extract_brand_product_relations(products, "lip_care")
        has_product = [t for t in triples if t.predicate == "HAS_PRODUCT"]
        assert len(has_product) == 6  # No duplicates

    def test_normalizes_brand_case(self):
        enricher = KGEnricher()
        products = [{"brand": "LANEIGE", "asin": "B001"}]
        triples = enricher._extract_brand_product_relations(products, "test")
        has_product = [t for t in triples if t.predicate == "HAS_PRODUCT"]
        assert has_product[0].subject == "laneige"  # lowercased


class TestCompetitiveRelations:
    def test_extracts_competition(self, sample_crawl_data):
        enricher = KGEnricher()
        triples = enricher._extract_competitive_relations(sample_crawl_data["products"], "lip_care")
        # LANEIGE(2) and COSRX(2) both have >=2 products
        competes = [t for t in triples if t.predicate == "COMPETES_WITH"]
        assert len(competes) >= 1

    def test_no_competition_single_products(self):
        enricher = KGEnricher()
        products = [
            {"brand": "A", "asin": "1"},
            {"brand": "B", "asin": "2"},
        ]
        triples = enricher._extract_competitive_relations(products, "test")
        assert len(triples) == 0  # Single products don't compete

    def test_competition_confidence(self, sample_crawl_data):
        enricher = KGEnricher()
        triples = enricher._extract_competitive_relations(sample_crawl_data["products"], "lip_care")
        competes = [t for t in triples if t.predicate == "COMPETES_WITH"]
        if competes:
            assert competes[0].confidence == 0.8


class TestPricePositions:
    def test_extracts_positions(self, sample_crawl_data):
        enricher = KGEnricher()
        triples = enricher._extract_price_positions(sample_crawl_data["products"], "lip_care")
        positions = {t.subject: t.object for t in triples}
        # Rare Beauty ($35) should be premium, Burt's Bees ($5) should be budget
        assert "rare beauty" in positions
        assert "burt's bees" in positions

    def test_no_prices(self):
        enricher = KGEnricher()
        products = [{"brand": "A", "asin": "1"}]  # No price
        triples = enricher._extract_price_positions(products, "test")
        assert len(triples) == 0

    def test_price_thresholds(self):
        enricher = KGEnricher()
        products = [
            {"brand": "Premium", "asin": "1", "price": 100.0},
            {"brand": "Budget", "asin": "2", "price": 10.0},
            {"brand": "Mid", "asin": "3", "price": 50.0},
        ]
        triples = enricher._extract_price_positions(products, "test")
        positions = {t.subject: t.object for t in triples}
        # avg = 53.33, premium > 69.33, budget < 37.33
        assert positions["premium"] == "premium"
        assert positions["budget"] == "budget"


class TestCategoryDominance:
    def test_extracts_dominance(self, sample_crawl_data):
        enricher = KGEnricher()
        triples = enricher._extract_category_dominance(sample_crawl_data["products"], "lip_care")
        dominant = {t.subject: t.properties["share"] for t in triples}
        # LANEIGE has 2/6 = 33%, COSRX 2/6 = 33%
        assert "laneige" in dominant
        assert dominant["laneige"] > 0.1

    def test_filters_below_threshold(self):
        enricher = KGEnricher()
        products = [{"brand": "A", "asin": str(i)} for i in range(20)]
        products.append({"brand": "B", "asin": "21"})  # 1/21 = 4.8% < 10%
        triples = enricher._extract_category_dominance(products, "test")
        brands = {t.subject for t in triples}
        assert "a" in brands  # 20/21 = 95%
        assert "b" not in brands  # 1/21 < 10%


class TestEnrichFromCrawl:
    def test_full_enrichment(self, sample_crawl_data):
        enricher = KGEnricher()
        triples = enricher.enrich_from_crawl(sample_crawl_data)
        assert len(triples) > 0
        predicates = {t.predicate for t in triples}
        assert "HAS_PRODUCT" in predicates
        assert "BELONGS_TO_CATEGORY" in predicates

    def test_empty_products(self):
        enricher = KGEnricher()
        triples = enricher.enrich_from_crawl({"products": []})
        assert len(triples) == 0

    def test_stats_tracking(self, sample_crawl_data):
        enricher = KGEnricher()
        enricher.enrich_from_crawl(sample_crawl_data)
        stats = enricher.get_stats()
        assert stats["total_enrichments"] == 1
        assert stats["triples_extracted"] > 0


class TestStoreTriples:
    def test_stores_high_confidence(self, mock_kg):
        enricher = KGEnricher(knowledge_graph=mock_kg)
        triples = [
            Triple("a", "HAS_PRODUCT", "b", confidence=0.9),
            Triple("c", "HAS_PRODUCT", "d", confidence=0.8),
        ]
        stored = enricher.store_triples(triples)
        assert stored == 2
        assert mock_kg.add_relation.call_count == 2

    def test_filters_low_confidence(self, mock_kg):
        enricher = KGEnricher(knowledge_graph=mock_kg, min_confidence=0.7)
        triples = [
            Triple("a", "HAS_PRODUCT", "b", confidence=0.9),
            Triple("c", "HAS_PRODUCT", "d", confidence=0.5),  # Below threshold
        ]
        stored = enricher.store_triples(triples)
        assert stored == 1
        assert enricher.get_stats()["triples_filtered"] == 1

    def test_no_kg(self):
        enricher = KGEnricher(knowledge_graph=None)
        stored = enricher.store_triples([Triple("a", "HAS_PRODUCT", "b")])
        assert stored == 0

    def test_converts_triple_to_relation(self, mock_kg):
        enricher = KGEnricher(knowledge_graph=mock_kg)
        triple = Triple("laneige", "HAS_PRODUCT", "B001", properties={"rank": 1})
        enricher.store_triples([triple])

        # Check that add_relation was called with a Relation object
        assert mock_kg.add_relation.called
        call_args = mock_kg.add_relation.call_args[0][0]
        assert hasattr(call_args, "subject")
        assert hasattr(call_args, "predicate")
        assert hasattr(call_args, "object")


class TestEnrichAndStore:
    def test_enrich_and_store(self, sample_crawl_data, mock_kg):
        enricher = KGEnricher(knowledge_graph=mock_kg)
        result = enricher.enrich_and_store(sample_crawl_data)
        assert result["extracted"] > 0
        assert result["stored"] > 0
        assert result["extracted"] == result["stored"] + result["filtered"]


class TestStats:
    def test_stats_tracking(self, sample_crawl_data):
        enricher = KGEnricher()
        enricher.enrich_from_crawl(sample_crawl_data)
        stats = enricher.get_stats()
        assert stats["total_enrichments"] == 1
        assert stats["triples_extracted"] > 0

    def test_stats_accumulate(self, sample_crawl_data):
        enricher = KGEnricher()
        enricher.enrich_from_crawl(sample_crawl_data)
        enricher.enrich_from_crawl(sample_crawl_data)
        stats = enricher.get_stats()
        assert stats["total_enrichments"] == 2


class TestPredicateMapping:
    def test_predicate_to_relation_type(self, mock_kg):
        enricher = KGEnricher(knowledge_graph=mock_kg)
        triple = Triple("a", "HAS_PRODUCT", "b")
        relation = enricher._triple_to_relation(triple)

        from src.domain.entities.relations import RelationType

        assert relation.predicate == RelationType.HAS_PRODUCT

    def test_all_predicates_mapped(self, mock_kg):
        enricher = KGEnricher(knowledge_graph=mock_kg)
        predicates = [
            "HAS_PRODUCT",
            "BELONGS_TO_CATEGORY",
            "COMPETES_WITH",
            "PRICE_POSITION",
            "DOMINATES_CATEGORY",
        ]
        for pred in predicates:
            triple = Triple("a", pred, "b")
            relation = enricher._triple_to_relation(triple)
            assert relation is not None

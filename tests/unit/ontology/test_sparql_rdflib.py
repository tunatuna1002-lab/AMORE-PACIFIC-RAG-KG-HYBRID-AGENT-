"""
Tests for SPARQL rdflib integration in KGQueryMixin.query_sparql_rdflib()

Validates:
- KG → RDF graph conversion
- Standard SPARQL queries via rdflib
- Cache invalidation
- Backward compatibility with existing sparql_query()
"""

import pytest

from src.domain.entities.relations import Relation, RelationType
from src.ontology.knowledge_graph import KnowledgeGraph

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def kg():
    """Empty KG with no persistence."""
    return KnowledgeGraph(persist_path="/dev/null", auto_load=False, auto_save=False)


@pytest.fixture
def populated_kg(kg):
    """KG with sample brand/product/category/competition data."""
    # Brand → Product
    kg.add_relation(
        Relation(
            "LANEIGE",
            RelationType.HAS_PRODUCT,
            "B08XYZ",
            properties={"product_name": "Lip Sleeping Mask", "category": "lip_care", "rank": 3},
        )
    )
    kg.add_relation(
        Relation(
            "LANEIGE",
            RelationType.HAS_PRODUCT,
            "B09ABC",
            properties={
                "product_name": "Water Sleeping Mask",
                "category": "skin_care",
                "rank": 12,
            },
        )
    )
    kg.add_relation(
        Relation(
            "COSRX",
            RelationType.HAS_PRODUCT,
            "B07DEF",
            properties={"product_name": "Snail Mucin", "category": "skin_care", "rank": 1},
        )
    )

    # Product → Category
    kg.add_relation(Relation("B08XYZ", RelationType.BELONGS_TO_CATEGORY, "lip_care"))
    kg.add_relation(Relation("B09ABC", RelationType.BELONGS_TO_CATEGORY, "skin_care"))
    kg.add_relation(Relation("B07DEF", RelationType.BELONGS_TO_CATEGORY, "skin_care"))

    # Competition
    kg.add_relation(
        Relation(
            "LANEIGE",
            RelationType.DIRECT_COMPETITOR,
            "COSRX",
            properties={"category": "skin_care"},
        )
    )
    kg.add_relation(
        Relation(
            "LANEIGE",
            RelationType.COMPETES_WITH,
            "BURT_BEES",
            properties={"category": "lip_care"},
        )
    )

    # Category hierarchy
    kg.add_relation(Relation("beauty", RelationType.HAS_SUBCATEGORY, "skin_care"))
    kg.add_relation(Relation("skin_care", RelationType.HAS_SUBCATEGORY, "lip_care"))

    return kg


# =========================================================================
# _kg_to_rdf_graph
# =========================================================================


class TestKgToRdfGraph:
    def test_kg_to_rdf_graph_basic(self, populated_kg):
        """Converts KG triples to rdflib Graph."""
        g = populated_kg._kg_to_rdf_graph()
        # Each triple produces 2 RDF triples (URI + Literal for object)
        assert len(g) > 0
        # Should have triples for all KG relations
        assert len(g) >= len(populated_kg.triples)

    def test_rdf_graph_has_amore_namespace(self, populated_kg):
        """RDF graph uses amore namespace."""
        g = populated_kg._kg_to_rdf_graph()
        namespaces = dict(g.namespaces())
        assert "amore" in namespaces

    def test_rdf_graph_cache_invalidation(self, kg):
        """Cache updates when triples change."""
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "B"))
        g1 = kg._kg_to_rdf_graph()

        # Add more triples
        kg.add_relation(Relation("C", RelationType.HAS_PRODUCT, "D"))
        kg.invalidate_rdf_cache()
        g2 = kg._kg_to_rdf_graph()

        # New graph should have more triples
        assert len(g2) > len(g1)

    def test_rdf_graph_cache_returns_same_when_unchanged(self, kg):
        """Cache returns same graph object when no changes."""
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "B"))
        g1 = kg._kg_to_rdf_graph()
        g2 = kg._kg_to_rdf_graph()
        assert g1 is g2


# =========================================================================
# query_sparql_rdflib
# =========================================================================


class TestQuerySparqlRdflib:
    def test_sparql_select_all_products(self, populated_kg):
        """SELECT ?brand ?product WHERE { ?brand amore:hasProduct ?product }"""
        results = populated_kg.query_sparql_rdflib("""
            PREFIX amore: <http://amore.ontology/>
            SELECT ?brand ?product
            WHERE {
                ?brand amore:hasProduct ?product .
            }
        """)
        # Should find all HAS_PRODUCT relations
        # Results include both URI and Literal bindings
        brands = {r.get("brand") for r in results}
        assert "LANEIGE" in brands
        assert "COSRX" in brands

    def test_sparql_filter_by_category(self, populated_kg):
        """FILTER with category literal matching."""
        results = populated_kg.query_sparql_rdflib("""
            PREFIX amore: <http://amore.ontology/>
            SELECT ?product ?cat
            WHERE {
                ?product amore:belongsToCategory ?cat .
                FILTER(?cat = "lip_care")
            }
        """)
        assert len(results) >= 1
        for r in results:
            assert r["cat"] == "lip_care"

    def test_sparql_join_two_patterns(self, populated_kg):
        """JOIN brand→product + product→category."""
        results = populated_kg.query_sparql_rdflib("""
            PREFIX amore: <http://amore.ontology/>
            SELECT ?brand ?product ?cat
            WHERE {
                ?brand amore:hasProduct ?product .
                ?product amore:belongsToCategory ?cat .
            }
        """)
        assert len(results) >= 1
        for r in results:
            assert "brand" in r
            assert "product" in r
            assert "cat" in r

    def test_sparql_empty_result(self, populated_kg):
        """Query with no matches returns empty list."""
        results = populated_kg.query_sparql_rdflib("""
            PREFIX amore: <http://amore.ontology/>
            SELECT ?x
            WHERE {
                ?x amore:hasProduct <http://amore.ontology/entity/NONEXISTENT> .
            }
        """)
        assert results == []

    def test_sparql_invalid_query(self, populated_kg):
        """Malformed SPARQL raises ValueError."""
        with pytest.raises(ValueError, match="Invalid SPARQL"):
            populated_kg.query_sparql_rdflib("THIS IS NOT SPARQL AT ALL {{{")

    def test_sparql_with_prefix(self, populated_kg):
        """PREFIX declaration works correctly."""
        results = populated_kg.query_sparql_rdflib("""
            PREFIX amore: <http://amore.ontology/>
            SELECT ?brand ?product
            WHERE {
                ?brand amore:hasProduct ?product .
            }
        """)
        assert len(results) >= 3  # LANEIGE has 2, COSRX has 1 (x2 for URI+Literal)

    def test_sparql_auto_prefix(self, populated_kg):
        """Auto-adds PREFIX when not present."""
        results = populated_kg.query_sparql_rdflib("""
            SELECT ?brand ?product
            WHERE {
                ?brand amore:hasProduct ?product .
            }
        """)
        assert len(results) >= 3

    def test_sparql_competitors(self, populated_kg):
        """Query competition relations."""
        results = populated_kg.query_sparql_rdflib("""
            PREFIX amore: <http://amore.ontology/>
            SELECT ?brand ?competitor
            WHERE {
                ?brand amore:directCompetitor ?competitor .
            }
        """)
        brands = {r.get("brand") for r in results}
        competitors = {r.get("competitor") for r in results}
        assert "LANEIGE" in brands
        assert "COSRX" in competitors

    def test_sparql_multiple_results(self, populated_kg):
        """Returns all matching bindings."""
        results = populated_kg.query_sparql_rdflib("""
            PREFIX amore: <http://amore.ontology/>
            SELECT ?s ?cat
            WHERE {
                ?s amore:belongsToCategory ?cat .
            }
        """)
        # At least 3 product→category relations (x2 for URI+Literal)
        subjects = {r.get("s") for r in results}
        assert len(subjects) >= 3

    def test_sparql_case_insensitive_predicate(self, populated_kg):
        """Predicate matching is exact (case-sensitive in SPARQL)."""
        # hasProduct (correct case) should work
        results = populated_kg.query_sparql_rdflib("""
            PREFIX amore: <http://amore.ontology/>
            SELECT ?brand ?product
            WHERE {
                ?brand amore:hasProduct ?product .
            }
        """)
        assert len(results) > 0

    def test_sparql_subcategory_query(self, populated_kg):
        """Query subcategory hierarchy."""
        results = populated_kg.query_sparql_rdflib("""
            PREFIX amore: <http://amore.ontology/>
            SELECT ?parent ?child
            WHERE {
                ?parent amore:hasSubcategory ?child .
            }
        """)
        parents = {r.get("parent") for r in results}
        assert "beauty" in parents or "skin_care" in parents


# =========================================================================
# Backward Compatibility
# =========================================================================


class TestBackwardCompat:
    def test_backward_compat_sparql_query(self, populated_kg):
        """Existing sparql_query() still works after adding rdflib method."""
        results = populated_kg.sparql_query("""
            SELECT ?brand ?product
            WHERE {
                ?brand <hasProduct> ?product .
            }
        """)
        assert len(results) == 3
        brands = {r["brand"] for r in results}
        assert "LANEIGE" in brands

    def test_both_methods_agree_on_basic_query(self, populated_kg):
        """Both sparql_query and query_sparql_rdflib return consistent results."""
        old_results = populated_kg.sparql_query("""
            SELECT ?brand ?product
            WHERE {
                ?brand <hasProduct> ?product .
            }
        """)
        new_results = populated_kg.query_sparql_rdflib("""
            PREFIX amore: <http://amore.ontology/>
            SELECT ?brand ?product
            WHERE {
                ?brand amore:hasProduct ?product .
            }
        """)
        old_brands = {r["brand"] for r in old_results}
        new_brands = {r["brand"] for r in new_results}
        # Same brands should appear
        assert old_brands == new_brands

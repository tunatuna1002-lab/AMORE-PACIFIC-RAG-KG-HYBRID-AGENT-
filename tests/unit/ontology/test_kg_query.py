"""
Tests for src/ontology/kg_query.py (KGQueryMixin)

Coverage target: 49% → 75%+
Tests query, traversal, domain queries, SPARQL-like, metadata, and stats.
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
    """KG with sample brand/product/category data."""
    # Brand → Product relations
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
            properties={"product_name": "Water Sleeping Mask", "category": "skin_care", "rank": 12},
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
            properties={"category": "skin_care", "competition_type": "direct"},
        )
    )
    kg.add_relation(
        Relation(
            "LANEIGE",
            RelationType.INDIRECT_COMPETITOR,
            "NIVEA",
            properties={"category": "lip_care", "competition_type": "indirect"},
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
    kg.add_relation(Relation("skin_care", RelationType.PARENT_CATEGORY, "beauty"))
    kg.add_relation(Relation("skin_care", RelationType.HAS_SUBCATEGORY, "lip_care"))

    # Metadata
    kg.set_entity_metadata("LANEIGE", {"sos": 5.2, "avg_rank": 8})
    kg.set_entity_metadata("COSRX", {"sos": 12.0, "avg_rank": 3})

    return kg


# =========================================================================
# query()
# =========================================================================


class TestQuery:
    def test_query_by_subject(self, populated_kg):
        results = populated_kg.query(subject="LANEIGE")
        assert len(results) >= 2  # at least HAS_PRODUCT rels

    def test_query_by_predicate(self, populated_kg):
        results = populated_kg.query(predicate=RelationType.HAS_PRODUCT)
        assert len(results) == 3

    def test_query_by_object(self, populated_kg):
        results = populated_kg.query(object_="lip_care")
        assert len(results) >= 1

    def test_query_by_subject_and_predicate(self, populated_kg):
        results = populated_kg.query(subject="LANEIGE", predicate=RelationType.HAS_PRODUCT)
        assert len(results) == 2

    def test_query_min_confidence(self, kg):
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "B", confidence=0.9))
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "C", confidence=0.3))
        results = kg.query(subject="A", min_confidence=0.5)
        assert len(results) == 1
        assert results[0].object == "B"

    def test_query_no_match(self, populated_kg):
        results = populated_kg.query(subject="NONEXISTENT")
        assert results == []

    def test_query_all(self, populated_kg):
        results = populated_kg.query()
        assert len(results) == len(populated_kg.triples)

    def test_query_filters_predicate_from_subject_index(self, populated_kg):
        """Subject index returns all rels for subject; predicate filter narrows."""
        results = populated_kg.query(subject="LANEIGE", predicate=RelationType.DIRECT_COMPETITOR)
        assert len(results) == 1
        assert results[0].object == "COSRX"

    def test_query_filters_object_mismatch(self, populated_kg):
        results = populated_kg.query(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            object_="NONEXISTENT",
        )
        assert results == []


# =========================================================================
# get_subjects / get_objects / get_predicates / exists
# =========================================================================


class TestBasicAccessors:
    def test_get_subjects(self, populated_kg):
        subjects = populated_kg.get_subjects(RelationType.HAS_PRODUCT, "B08XYZ")
        assert "LANEIGE" in subjects

    def test_get_objects(self, populated_kg):
        objects = populated_kg.get_objects("LANEIGE", RelationType.HAS_PRODUCT)
        assert "B08XYZ" in objects
        assert "B09ABC" in objects

    def test_get_predicates(self, populated_kg):
        preds = populated_kg.get_predicates("LANEIGE", "COSRX")
        assert RelationType.DIRECT_COMPETITOR in preds

    def test_exists_true(self, populated_kg):
        assert populated_kg.exists("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ") is True

    def test_exists_false(self, populated_kg):
        assert populated_kg.exists("LANEIGE", RelationType.HAS_PRODUCT, "FAKE") is False


# =========================================================================
# get_neighbors
# =========================================================================


class TestGetNeighbors:
    def test_outgoing_neighbors(self, populated_kg):
        neighbors = populated_kg.get_neighbors("LANEIGE", direction="outgoing")
        assert len(neighbors["outgoing"]) > 0
        assert neighbors["incoming"] == []

    def test_incoming_neighbors(self, populated_kg):
        neighbors = populated_kg.get_neighbors("B08XYZ", direction="incoming")
        assert len(neighbors["incoming"]) > 0
        assert neighbors["outgoing"] == []

    def test_both_neighbors(self, populated_kg):
        neighbors = populated_kg.get_neighbors("skin_care", direction="both")
        # incoming: beauty HAS_SUBCATEGORY skin_care, products BELONGS_TO_CATEGORY skin_care
        assert len(neighbors["incoming"]) >= 1
        # outgoing: skin_care PARENT_CATEGORY beauty, skin_care HAS_SUBCATEGORY lip_care
        assert len(neighbors["outgoing"]) >= 1

    def test_neighbors_with_predicate_filter(self, populated_kg):
        neighbors = populated_kg.get_neighbors(
            "LANEIGE",
            direction="outgoing",
            predicate_filter=[RelationType.HAS_PRODUCT],
        )
        # Only HAS_PRODUCT relations, not DIRECT_COMPETITOR
        for pred, _ in neighbors["outgoing"]:
            assert pred == RelationType.HAS_PRODUCT

    def test_neighbors_incoming_with_filter(self, populated_kg):
        neighbors = populated_kg.get_neighbors(
            "lip_care",
            direction="incoming",
            predicate_filter=[RelationType.BELONGS_TO_CATEGORY],
        )
        assert len(neighbors["incoming"]) >= 1
        for pred, _ in neighbors["incoming"]:
            assert pred == RelationType.BELONGS_TO_CATEGORY


# =========================================================================
# bfs_traverse
# =========================================================================


class TestBfsTraverse:
    def test_bfs_basic(self, populated_kg):
        result = populated_kg.bfs_traverse("LANEIGE", max_depth=1, direction="outgoing")
        assert 0 in result
        assert "LANEIGE" in result[0]
        assert 1 in result
        # Depth 1 should include products and competitors
        assert len(result[1]) > 0

    def test_bfs_with_filter(self, populated_kg):
        result = populated_kg.bfs_traverse(
            "LANEIGE",
            max_depth=2,
            predicate_filter=[RelationType.HAS_PRODUCT, RelationType.BELONGS_TO_CATEGORY],
            direction="outgoing",
        )
        assert "LANEIGE" in result[0]

    def test_bfs_both_direction(self, populated_kg):
        result = populated_kg.bfs_traverse("skin_care", max_depth=1, direction="both")
        assert "skin_care" in result[0]
        # Should find both incoming and outgoing neighbors
        assert len(result.get(1, [])) > 0

    def test_bfs_max_depth_zero(self, populated_kg):
        result = populated_kg.bfs_traverse("LANEIGE", max_depth=0)
        assert result == {0: ["LANEIGE"]}

    def test_bfs_nonexistent_entity(self, populated_kg):
        result = populated_kg.bfs_traverse("GHOST", max_depth=2)
        assert result == {0: ["GHOST"]}


# =========================================================================
# find_path
# =========================================================================


class TestFindPath:
    def test_find_path_direct(self, populated_kg):
        path = populated_kg.find_path("LANEIGE", "B08XYZ")
        assert path is not None
        assert len(path) == 1
        assert path[0].subject == "LANEIGE"
        assert path[0].object == "B08XYZ"

    def test_find_path_multi_hop(self, populated_kg):
        # LANEIGE → B08XYZ → lip_care
        path = populated_kg.find_path("LANEIGE", "lip_care")
        assert path is not None
        assert len(path) == 2

    def test_find_path_same_entity(self, populated_kg):
        path = populated_kg.find_path("LANEIGE", "LANEIGE")
        assert path == []

    def test_find_path_no_path(self, populated_kg):
        path = populated_kg.find_path("lip_care", "LANEIGE", max_depth=5)
        # lip_care has no outgoing path to LANEIGE
        # (only incoming from products)
        assert path is None

    def test_find_path_max_depth_exceeded(self, populated_kg):
        path = populated_kg.find_path("LANEIGE", "lip_care", max_depth=0)
        assert path is None


# =========================================================================
# Domain: get_brand_products
# =========================================================================


class TestGetBrandProducts:
    def test_get_all_products(self, populated_kg):
        products = populated_kg.get_brand_products("LANEIGE")
        assert len(products) == 2
        asins = [p["asin"] for p in products]
        assert "B08XYZ" in asins
        assert "B09ABC" in asins

    def test_get_products_with_category_filter(self, populated_kg):
        products = populated_kg.get_brand_products("LANEIGE", category="lip_care")
        assert len(products) == 1
        assert products[0]["asin"] == "B08XYZ"
        assert products[0]["name"] == "Lip Sleeping Mask"

    def test_get_products_empty(self, populated_kg):
        products = populated_kg.get_brand_products("UNKNOWN_BRAND")
        assert products == []


# =========================================================================
# Domain: get_competitors
# =========================================================================


class TestGetCompetitors:
    def test_get_all_competitors(self, populated_kg):
        comps = populated_kg.get_competitors("LANEIGE")
        brands = [c["brand"] for c in comps]
        assert "COSRX" in brands
        assert "NIVEA" in brands
        assert "BURT_BEES" in brands

    def test_get_direct_competitors(self, populated_kg):
        comps = populated_kg.get_competitors("LANEIGE", competition_type="direct")
        brands = [c["brand"] for c in comps]
        assert "COSRX" in brands
        assert "NIVEA" not in brands

    def test_get_indirect_competitors(self, populated_kg):
        comps = populated_kg.get_competitors("LANEIGE", competition_type="indirect")
        brands = [c["brand"] for c in comps]
        assert "NIVEA" in brands
        assert "COSRX" not in brands

    def test_get_competitors_with_category(self, populated_kg):
        comps = populated_kg.get_competitors("LANEIGE", category="lip_care")
        brands = [c["brand"] for c in comps]
        assert "NIVEA" in brands
        assert "COSRX" not in brands  # COSRX is skin_care

    def test_get_competitors_deduplication(self, kg):
        """Duplicate competitor across types should be deduplicated."""
        kg.add_relation(
            Relation("A", RelationType.DIRECT_COMPETITOR, "B", properties={"category": "c1"})
        )
        kg.add_relation(
            Relation("A", RelationType.COMPETES_WITH, "B", properties={"category": "c1"})
        )
        comps = kg.get_competitors("A")
        assert len(comps) == 1


# =========================================================================
# Domain: get_category_brands
# =========================================================================


class TestGetCategoryBrands:
    def test_get_category_brands(self, populated_kg):
        brands = populated_kg.get_category_brands("skin_care")
        brand_names = [b["brand"] for b in brands]
        # COSRX has 1 product in skin_care, LANEIGE has 1
        assert "COSRX" in brand_names or "LANEIGE" in brand_names

    def test_get_category_brands_min_products(self, populated_kg):
        brands = populated_kg.get_category_brands("skin_care", min_products=5)
        assert brands == []

    def test_get_category_brands_sorted(self, kg):
        """Brands should be sorted by product_count descending."""
        for i in range(3):
            kg.add_relation(Relation("BRAND_A", RelationType.HAS_PRODUCT, f"P_A{i}"))
            kg.add_relation(Relation(f"P_A{i}", RelationType.BELONGS_TO_CATEGORY, "cat1"))
        kg.add_relation(Relation("BRAND_B", RelationType.HAS_PRODUCT, "P_B0"))
        kg.add_relation(Relation("P_B0", RelationType.BELONGS_TO_CATEGORY, "cat1"))

        brands = kg.get_category_brands("cat1")
        assert brands[0]["brand"] == "BRAND_A"
        assert brands[0]["product_count"] == 3


# =========================================================================
# get_entity_context
# =========================================================================


class TestGetEntityContext:
    def test_entity_context_basic(self, populated_kg):
        ctx = populated_kg.get_entity_context("LANEIGE", depth=1)
        assert ctx["entity"] == "LANEIGE"
        assert "metadata" in ctx
        assert ctx["metadata"]["sos"] == 5.2
        assert "relations" in ctx

    def test_entity_context_with_depth(self, populated_kg):
        ctx = populated_kg.get_entity_context("LANEIGE", depth=2)
        assert "connected" in ctx
        # Should have connected entities from neighbors
        assert len(ctx["connected"]) > 0

    def test_entity_context_depth_one_no_connected(self, populated_kg):
        ctx = populated_kg.get_entity_context("LANEIGE", depth=1)
        assert "connected" not in ctx

    def test_entity_context_unknown_entity(self, populated_kg):
        ctx = populated_kg.get_entity_context("GHOST", depth=1)
        assert ctx["entity"] == "GHOST"
        assert ctx["metadata"] == {}


# =========================================================================
# Entity Metadata
# =========================================================================


class TestEntityMetadata:
    def test_set_and_get_metadata(self, kg):
        kg.set_entity_metadata("BRAND_X", {"sos": 10.0, "rank": 5})
        meta = kg.get_entity_metadata("BRAND_X")
        assert meta["sos"] == 10.0
        assert meta["rank"] == 5

    def test_set_metadata_merge(self, kg):
        kg.set_entity_metadata("BRAND_X", {"sos": 10.0})
        kg.set_entity_metadata("BRAND_X", {"rank": 5})
        meta = kg.get_entity_metadata("BRAND_X")
        assert meta["sos"] == 10.0
        assert meta["rank"] == 5

    def test_get_metadata_missing(self, kg):
        assert kg.get_entity_metadata("NOBODY") == {}


# =========================================================================
# Stats
# =========================================================================


class TestStats:
    def test_get_stats(self, populated_kg):
        stats = populated_kg.get_stats()
        assert stats["total_triples"] == len(populated_kg.triples)
        assert stats["unique_subjects"] > 0
        assert stats["unique_objects"] > 0
        assert "relations_by_type" in stats

    def test_entity_degree(self, populated_kg):
        degree = populated_kg.get_entity_degree("LANEIGE")
        assert degree["out_degree"] > 0
        assert degree["total"] == degree["in_degree"] + degree["out_degree"]

    def test_entity_degree_nonexistent(self, kg):
        degree = kg.get_entity_degree("GHOST")
        assert degree == {"in_degree": 0, "out_degree": 0, "total": 0}

    def test_most_connected(self, populated_kg):
        top = populated_kg.get_most_connected(top_n=3)
        assert len(top) <= 3
        assert all(isinstance(t, tuple) and len(t) == 2 for t in top)
        # First should be highest degree
        if len(top) >= 2:
            assert top[0][1] >= top[1][1]


# =========================================================================
# SPARQL-like Query
# =========================================================================


class TestSparqlQuery:
    def test_sparql_basic_select(self, populated_kg):
        results = populated_kg.sparql_query("""
            SELECT ?brand ?product
            WHERE {
                ?brand <hasProduct> ?product .
            }
        """)
        assert len(results) == 3
        brands = {r["brand"] for r in results}
        assert "LANEIGE" in brands

    def test_sparql_with_literal(self, populated_kg):
        results = populated_kg.sparql_query("""
            SELECT ?product
            WHERE {
                ?product <belongsToCategory> "lip_care" .
            }
        """)
        assert len(results) >= 1
        assert results[0]["product"] == "B08XYZ"

    def test_sparql_join(self, populated_kg):
        results = populated_kg.sparql_query("""
            SELECT ?brand ?product
            WHERE {
                ?brand <hasProduct> ?product .
                ?product <belongsToCategory> "skin_care" .
            }
        """)
        assert len(results) >= 1
        for r in results:
            assert "brand" in r
            assert "product" in r

    def test_sparql_with_filter_numeric(self, kg):
        """Test FILTER with numeric comparison."""
        kg.add_relation(Relation("A", RelationType.HAS_RANK, "5"))
        kg.add_relation(Relation("B", RelationType.HAS_RANK, "15"))
        kg.add_relation(Relation("C", RelationType.HAS_RANK, "3"))

        results = kg.sparql_query("""
            SELECT ?entity ?rank
            WHERE {
                ?entity <hasRank> ?rank .
                FILTER (?rank <= 10)
            }
        """)
        entities = {r["entity"] for r in results}
        assert "A" in entities
        assert "C" in entities
        assert "B" not in entities

    def test_sparql_with_filter_string_eq(self, populated_kg):
        results = populated_kg.sparql_query("""
            SELECT ?product ?cat
            WHERE {
                ?product <belongsToCategory> ?cat .
                FILTER (?cat == lip_care)
            }
        """)
        assert len(results) >= 1
        assert all(r["cat"] == "lip_care" for r in results)

    def test_sparql_empty_result(self, populated_kg):
        results = populated_kg.sparql_query("""
            SELECT ?x
            WHERE {
                "NONEXISTENT" <hasProduct> ?x .
            }
        """)
        assert results == []

    def test_sparql_no_patterns(self, populated_kg):
        results = populated_kg.sparql_query("SELECT ?x WHERE { }")
        assert results == []

    def test_sparql_variable_predicate(self, populated_kg):
        results = populated_kg.sparql_query("""
            SELECT ?pred ?obj
            WHERE {
                "LANEIGE" ?pred ?obj .
            }
        """)
        assert len(results) > 0
        preds = {r["pred"] for r in results}
        assert RelationType.HAS_PRODUCT.value in preds

    def test_sparql_filter_not_equal(self, kg):
        kg.add_relation(Relation("A", RelationType.HAS_RANK, "5"))
        kg.add_relation(Relation("B", RelationType.HAS_RANK, "10"))

        results = kg.sparql_query("""
            SELECT ?entity ?rank
            WHERE {
                ?entity <hasRank> ?rank .
                FILTER (?rank != 5)
            }
        """)
        entities = {r["entity"] for r in results}
        assert "B" in entities
        assert "A" not in entities

    def test_sparql_filter_string_not_equal(self, populated_kg):
        results = populated_kg.sparql_query("""
            SELECT ?product ?cat
            WHERE {
                ?product <belongsToCategory> ?cat .
                FILTER (?cat != lip_care)
            }
        """)
        assert all(r["cat"] != "lip_care" for r in results)


class TestJoinBindings:
    def test_join_empty_left(self, kg):
        right = [{"x": "a"}, {"x": "b"}]
        result = kg._join_bindings([], right)
        assert result == right

    def test_join_empty_right(self, kg):
        left = [{"x": "a"}]
        result = kg._join_bindings(left, [])
        assert result == left

    def test_join_shared_variable(self, kg):
        left = [{"x": "a", "y": "1"}, {"x": "b", "y": "2"}]
        right = [{"x": "a", "z": "10"}, {"x": "c", "z": "30"}]
        result = kg._join_bindings(left, right)
        assert len(result) == 1
        assert result[0] == {"x": "a", "y": "1", "z": "10"}


class TestApplyFilter:
    def test_filter_invalid_condition(self, kg):
        bindings = [{"x": "1"}]
        result = kg._apply_filter(bindings, "invalid syntax")
        assert result == bindings

    def test_filter_missing_var(self, kg):
        bindings = [{"x": "1"}]
        result = kg._apply_filter(bindings, "?y > 0")
        assert result == []

    def test_filter_greater_than(self, kg):
        bindings = [{"rank": "5"}, {"rank": "15"}, {"rank": "10"}]
        result = kg._apply_filter(bindings, "?rank > 8")
        assert len(result) == 2

    def test_filter_less_than(self, kg):
        bindings = [{"rank": "5"}, {"rank": "15"}]
        result = kg._apply_filter(bindings, "?rank < 10")
        assert len(result) == 1

    def test_filter_string_comparison(self, kg):
        bindings = [{"cat": "apple"}, {"cat": "banana"}, {"cat": "cherry"}]
        result = kg._apply_filter(bindings, "?cat >= banana")
        cats = [r["cat"] for r in result]
        assert "apple" not in cats
        assert "banana" in cats
        assert "cherry" in cats

    def test_filter_string_less_than(self, kg):
        bindings = [{"cat": "apple"}, {"cat": "cherry"}]
        result = kg._apply_filter(bindings, "?cat < cherry")
        assert len(result) == 1
        assert result[0]["cat"] == "apple"

    def test_filter_string_less_equal(self, kg):
        bindings = [{"cat": "apple"}, {"cat": "banana"}]
        result = kg._apply_filter(bindings, "?cat <= banana")
        assert len(result) == 2

    def test_filter_string_greater_than(self, kg):
        bindings = [{"cat": "apple"}, {"cat": "cherry"}]
        result = kg._apply_filter(bindings, "?cat > apple")
        assert len(result) == 1
        assert result[0]["cat"] == "cherry"

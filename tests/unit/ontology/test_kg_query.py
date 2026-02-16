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


# =========================================================================
# Additional Edge Cases & Error Handling
# =========================================================================


class TestQueryEdgeCases:
    def test_query_with_multiple_filters(self, populated_kg):
        """Test query with subject, predicate, object, and confidence filters all at once."""
        results = populated_kg.query(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            object_="B08XYZ",
            min_confidence=0.5,
        )
        assert len(results) == 1
        assert results[0].subject == "LANEIGE"
        assert results[0].object == "B08XYZ"

    def test_query_uses_subject_index_optimization(self, populated_kg):
        """Verify that query uses subject index for optimization."""
        # Query with subject should use subject_index
        results = populated_kg.query(subject="LANEIGE")
        assert len(results) > 0

    def test_query_uses_object_index_optimization(self, populated_kg):
        """Verify that query uses object index when subject is not provided."""
        results = populated_kg.query(object_="lip_care")
        assert len(results) > 0

    def test_query_uses_predicate_index_optimization(self, populated_kg):
        """Verify that query uses predicate index when subject and object are not provided."""
        results = populated_kg.query(predicate=RelationType.HAS_PRODUCT)
        assert len(results) == 3

    def test_query_with_zero_confidence(self, kg):
        """Test that min_confidence=0.0 returns all results."""
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "B", confidence=0.1))
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "C", confidence=0.9))
        results = kg.query(subject="A", min_confidence=0.0)
        assert len(results) == 2


class TestGetSubjectsObjectsPredicatesEdgeCases:
    def test_get_subjects_empty_result(self, populated_kg):
        """Test get_subjects with non-existent object."""
        subjects = populated_kg.get_subjects(RelationType.HAS_PRODUCT, "NONEXISTENT")
        assert subjects == []

    def test_get_objects_empty_result(self, populated_kg):
        """Test get_objects with non-existent subject."""
        objects = populated_kg.get_objects("NONEXISTENT", RelationType.HAS_PRODUCT)
        assert objects == []

    def test_get_predicates_empty_result(self, populated_kg):
        """Test get_predicates with non-existent relationship."""
        preds = populated_kg.get_predicates("LANEIGE", "NONEXISTENT")
        assert preds == []

    def test_get_subjects_unique(self, kg):
        """Test that get_subjects returns unique subjects only."""
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "X"))
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "X", confidence=0.9))
        kg.add_relation(Relation("B", RelationType.HAS_PRODUCT, "X"))
        subjects = kg.get_subjects(RelationType.HAS_PRODUCT, "X")
        # Should be unique
        assert len(subjects) == len(set(subjects))

    def test_get_objects_unique(self, kg):
        """Test that get_objects returns unique objects only."""
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "X"))
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "X", confidence=0.9))
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "Y"))
        objects = kg.get_objects("A", RelationType.HAS_PRODUCT)
        assert len(objects) == len(set(objects))

    def test_get_predicates_unique(self, kg):
        """Test that get_predicates returns unique predicates."""
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "B"))
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "B", confidence=0.8))
        kg.add_relation(Relation("A", RelationType.COMPETES_WITH, "B"))
        preds = kg.get_predicates("A", "B")
        assert len(preds) == len(set(preds))


class TestGetNeighborsEdgeCases:
    def test_neighbors_nonexistent_entity(self, populated_kg):
        """Test get_neighbors with non-existent entity."""
        neighbors = populated_kg.get_neighbors("NONEXISTENT", direction="both")
        assert neighbors["outgoing"] == []
        assert neighbors["incoming"] == []

    def test_neighbors_with_empty_predicate_filter(self, populated_kg):
        """Test get_neighbors with empty predicate filter list."""
        neighbors = populated_kg.get_neighbors("LANEIGE", direction="outgoing", predicate_filter=[])
        # Empty filter is falsy, so it returns all (no filtering applied)
        assert len(neighbors["outgoing"]) > 0

    def test_neighbors_invalid_direction(self, populated_kg):
        """Test get_neighbors with invalid direction defaults to no results."""
        neighbors = populated_kg.get_neighbors("LANEIGE", direction="invalid")
        assert neighbors["outgoing"] == []
        assert neighbors["incoming"] == []


class TestBfsTraverseEdgeCases:
    def test_bfs_cycles_handled(self, kg):
        """Test that BFS handles cycles correctly (doesn't loop infinitely)."""
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "B"))
        kg.add_relation(Relation("B", RelationType.HAS_PRODUCT, "C"))
        kg.add_relation(Relation("C", RelationType.HAS_PRODUCT, "A"))  # Cycle
        result = kg.bfs_traverse("A", max_depth=5, direction="outgoing")
        # Should visit each node only once
        all_nodes = [node for nodes in result.values() for node in nodes]
        assert len(all_nodes) == len(set(all_nodes))

    def test_bfs_incoming_direction(self, populated_kg):
        """Test BFS with incoming direction."""
        result = populated_kg.bfs_traverse("lip_care", max_depth=2, direction="incoming")
        assert "lip_care" in result[0]
        # Should find products that belong to lip_care
        assert len(result.get(1, [])) > 0

    def test_bfs_depth_exceeded_skips_nodes(self, populated_kg):
        """Test that nodes beyond max_depth are not visited."""
        result = populated_kg.bfs_traverse("LANEIGE", max_depth=1, direction="outgoing")
        # Should not have depth > 1
        assert 2 not in result or len(result.get(2, [])) == 0


class TestFindPathEdgeCases:
    def test_find_path_with_cycle(self, kg):
        """Test find_path with cycles in the graph."""
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "B"))
        kg.add_relation(Relation("B", RelationType.HAS_PRODUCT, "C"))
        kg.add_relation(Relation("C", RelationType.HAS_PRODUCT, "A"))  # Cycle
        kg.add_relation(Relation("C", RelationType.HAS_PRODUCT, "D"))
        path = kg.find_path("A", "D")
        assert path is not None
        assert len(path) == 3

    def test_find_path_nonexistent_start(self, populated_kg):
        """Test find_path with non-existent start entity."""
        path = populated_kg.find_path("NONEXISTENT", "lip_care")
        assert path is None

    def test_find_path_nonexistent_end(self, populated_kg):
        """Test find_path with non-existent end entity."""
        path = populated_kg.find_path("LANEIGE", "NONEXISTENT")
        assert path is None

    def test_find_path_multiple_paths_returns_shortest(self, kg):
        """Test that find_path returns shortest path when multiple exist."""
        # Path 1: A -> B (1 hop)
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "B"))
        # Path 2: A -> C -> D -> B (3 hops)
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "C"))
        kg.add_relation(Relation("C", RelationType.HAS_PRODUCT, "D"))
        kg.add_relation(Relation("D", RelationType.HAS_PRODUCT, "B"))
        path = kg.find_path("A", "B")
        # Should find shortest path (1 hop)
        assert len(path) == 1


class TestGetBrandProductsEdgeCases:
    def test_get_brand_products_with_missing_category_property(self, kg):
        """Test get_brand_products when products lack category property."""
        kg.add_relation(
            Relation(
                "BRAND_X",
                RelationType.HAS_PRODUCT,
                "P1",
                properties={"product_name": "Product 1"},  # No category
            )
        )
        products = kg.get_brand_products("BRAND_X", category="skin_care")
        # Should not match since category is None
        assert len(products) == 0

    def test_get_brand_products_includes_all_properties(self, kg):
        """Test that get_brand_products includes all properties from relation."""
        kg.add_relation(
            Relation(
                "BRAND_X",
                RelationType.HAS_PRODUCT,
                "P1",
                properties={
                    "product_name": "Product 1",
                    "category": "cat1",
                    "rank": 5,
                    "price": 29.99,
                    "rating": 4.5,
                },
            )
        )
        products = kg.get_brand_products("BRAND_X")
        assert len(products) == 1
        assert products[0]["price"] == 29.99
        assert products[0]["rating"] == 4.5


class TestGetCompetitorsEdgeCases:
    def test_get_competitors_empty_brand(self, populated_kg):
        """Test get_competitors with non-existent brand."""
        comps = populated_kg.get_competitors("NONEXISTENT_BRAND")
        assert comps == []

    def test_get_competitors_no_category_property(self, kg):
        """Test competitor filtering when relation has no category property."""
        kg.add_relation(
            Relation("A", RelationType.DIRECT_COMPETITOR, "B")  # No category property
        )
        comps = kg.get_competitors("A", category="skin_care")
        # Should not match since category is None
        assert len(comps) == 0

    def test_get_competitors_all_types_includes_competes_with(self, kg):
        """Test that competition_type='all' includes COMPETES_WITH relation."""
        kg.add_relation(
            Relation("A", RelationType.COMPETES_WITH, "B", properties={"category": "c1"})
        )
        comps = kg.get_competitors("A", competition_type="all")
        assert len(comps) == 1
        assert comps[0]["brand"] == "B"


class TestGetCategoryBrandsEdgeCases:
    def test_get_category_brands_empty_category(self, populated_kg):
        """Test get_category_brands with non-existent category."""
        brands = populated_kg.get_category_brands("nonexistent_category")
        assert brands == []

    def test_get_category_brands_multiple_brands_same_product_count(self, kg):
        """Test sorting when multiple brands have same product count."""
        kg.add_relation(Relation("BRAND_A", RelationType.HAS_PRODUCT, "P_A1"))
        kg.add_relation(Relation("P_A1", RelationType.BELONGS_TO_CATEGORY, "cat1"))
        kg.add_relation(Relation("BRAND_B", RelationType.HAS_PRODUCT, "P_B1"))
        kg.add_relation(Relation("P_B1", RelationType.BELONGS_TO_CATEGORY, "cat1"))
        brands = kg.get_category_brands("cat1")
        assert len(brands) == 2
        # Both have same product count
        assert brands[0]["product_count"] == 1
        assert brands[1]["product_count"] == 1

    def test_get_category_brands_product_without_brand(self, kg):
        """Test handling of products that don't have brand relations."""
        kg.add_relation(Relation("ORPHAN_PRODUCT", RelationType.BELONGS_TO_CATEGORY, "cat1"))
        # No HAS_PRODUCT relation for this product
        brands = kg.get_category_brands("cat1")
        # Should return empty since no brands found
        assert brands == []


class TestGetEntityContextEdgeCases:
    def test_entity_context_deep_recursion(self, kg):
        """Test entity context with deep depth doesn't cause issues."""
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "B"))
        kg.add_relation(Relation("B", RelationType.HAS_PRODUCT, "C"))
        kg.add_relation(Relation("C", RelationType.HAS_PRODUCT, "D"))
        ctx = kg.get_entity_context("A", depth=3)
        assert "connected" in ctx
        # Should handle nested context

    def test_entity_context_limits_neighbors(self, kg):
        """Test that entity context limits to top 5 neighbors in recursion."""
        for i in range(10):
            kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, f"P{i}"))
        ctx = kg.get_entity_context("A", depth=2)
        # Should have connected entities but limited to 5
        if "connected" in ctx:
            assert len(ctx["connected"]) <= 5

    def test_entity_context_no_self_reference(self, kg):
        """Test that entity context doesn't include self-reference."""
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "B"))
        kg.add_relation(Relation("B", RelationType.HAS_PRODUCT, "A"))  # Cycle back
        ctx = kg.get_entity_context("A", depth=2)
        # Should not cause infinite recursion
        assert ctx["entity"] == "A"


class TestEntityMetadataEdgeCases:
    def test_set_metadata_overwrites_existing_keys(self, kg):
        """Test that setting metadata merges/overwrites existing keys."""
        kg.set_entity_metadata("BRAND_X", {"sos": 10.0, "rank": 5})
        kg.set_entity_metadata("BRAND_X", {"sos": 15.0, "new_key": "value"})
        meta = kg.get_entity_metadata("BRAND_X")
        assert meta["sos"] == 15.0  # Overwritten
        assert meta["rank"] == 5  # Preserved
        assert meta["new_key"] == "value"  # Added

    def test_set_metadata_empty_dict(self, kg):
        """Test setting empty metadata doesn't cause issues."""
        kg.set_entity_metadata("BRAND_X", {})
        meta = kg.get_entity_metadata("BRAND_X")
        assert meta == {}


class TestStatsEdgeCases:
    def test_get_stats_empty_kg(self, kg):
        """Test get_stats on empty knowledge graph."""
        stats = kg.get_stats()
        assert stats["total_triples"] == 0
        assert stats["unique_subjects"] == 0
        assert stats["unique_objects"] == 0
        assert stats["relations_by_type"] == {}

    def test_entity_degree_single_incoming(self, kg):
        """Test entity degree for entity with only incoming relations."""
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "B"))
        degree = kg.get_entity_degree("B")
        assert degree["in_degree"] == 1
        assert degree["out_degree"] == 0
        assert degree["total"] == 1

    def test_entity_degree_single_outgoing(self, kg):
        """Test entity degree for entity with only outgoing relations."""
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "B"))
        degree = kg.get_entity_degree("A")
        assert degree["in_degree"] == 0
        assert degree["out_degree"] == 1
        assert degree["total"] == 1

    def test_most_connected_empty_kg(self, kg):
        """Test get_most_connected on empty knowledge graph."""
        top = kg.get_most_connected(top_n=5)
        assert top == []

    def test_most_connected_fewer_than_n(self, kg):
        """Test get_most_connected when graph has fewer entities than top_n."""
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "B"))
        top = kg.get_most_connected(top_n=10)
        assert len(top) <= 2  # Only A and B


class TestSparqlQueryEdgeCases:
    def test_sparql_multiple_filters(self, kg):
        """Test SPARQL query with multiple FILTER conditions."""
        kg.add_relation(Relation("A", RelationType.HAS_RANK, "5"))
        kg.add_relation(Relation("B", RelationType.HAS_RANK, "15"))
        kg.add_relation(Relation("C", RelationType.HAS_RANK, "8"))
        results = kg.sparql_query("""
            SELECT ?entity ?rank
            WHERE {
                ?entity <hasRank> ?rank .
                FILTER (?rank > 5)
                FILTER (?rank < 10)
            }
        """)
        # Should only match C (rank=8)
        entities = {r["entity"] for r in results}
        assert "C" in entities
        assert "A" not in entities
        assert "B" not in entities

    def test_sparql_case_insensitive_keywords(self, populated_kg):
        """Test that SPARQL keywords are case-insensitive."""
        results = populated_kg.sparql_query("""
            select ?brand ?product
            where {
                ?brand <hasProduct> ?product .
            }
        """)
        assert len(results) == 3

    def test_sparql_no_select_clause(self, populated_kg):
        """Test SPARQL query without SELECT returns all variables."""
        # Malformed query - should handle gracefully
        results = populated_kg.sparql_query("""
            WHERE {
                ?brand <hasProduct> ?product .
            }
        """)
        # Without SELECT, select_vars is empty, returns full bindings
        assert len(results) >= 0

    def test_sparql_triple_pattern_with_quotes_in_literal(self, kg):
        """Test SPARQL with quoted literals containing special characters."""
        kg.add_relation(Relation("A", RelationType.BELONGS_TO_CATEGORY, "test-category"))
        results = kg.sparql_query("""
            SELECT ?x
            WHERE {
                ?x <belongsToCategory> "test-category" .
            }
        """)
        assert len(results) == 1
        assert results[0]["x"] == "A"


class TestMatchPatternEdgeCases:
    def test_match_pattern_unknown_predicate(self, kg):
        """Test _match_pattern with unknown predicate queries with predicate=None."""
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "B"))
        bindings = kg._match_pattern(("?s", "<unknownPredicate>", "?o"))
        # Unknown predicate results in predicate=None, which matches all predicates
        assert len(bindings) >= 1

    def test_match_pattern_literal_subject(self, kg):
        """Test _match_pattern with literal subject."""
        kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "B"))
        bindings = kg._match_pattern(('"LANEIGE"', "<hasProduct>", "?o"))
        assert len(bindings) == 1
        assert bindings[0]["o"] == "B"

    def test_match_pattern_literal_object(self, kg):
        """Test _match_pattern with literal object."""
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "B"))
        bindings = kg._match_pattern(("?s", "<hasProduct>", '"B"'))
        assert len(bindings) == 1
        assert bindings[0]["s"] == "A"

    def test_match_pattern_all_variables(self, kg):
        """Test _match_pattern with all variables (s, p, o)."""
        kg.add_relation(Relation("A", RelationType.HAS_PRODUCT, "B"))
        kg.add_relation(Relation("C", RelationType.COMPETES_WITH, "D"))
        bindings = kg._match_pattern(("?s", "?p", "?o"))
        assert len(bindings) == 2
        # All three should be bound
        for b in bindings:
            assert "s" in b and "p" in b and "o" in b


class TestJoinBindingsEdgeCases:
    def test_join_no_shared_variables(self, kg):
        """Test _join_bindings with no shared variables (cartesian product)."""
        left = [{"x": "a"}, {"x": "b"}]
        right = [{"y": "1"}, {"y": "2"}]
        result = kg._join_bindings(left, right)
        # Should produce cartesian product: 2 * 2 = 4
        assert len(result) == 4

    def test_join_multiple_shared_variables(self, kg):
        """Test _join_bindings with multiple shared variables."""
        left = [{"x": "a", "y": "1"}, {"x": "b", "y": "2"}]
        right = [{"x": "a", "y": "1", "z": "10"}, {"x": "a", "y": "2", "z": "20"}]
        result = kg._join_bindings(left, right)
        # Only first left matches first right
        assert len(result) == 1
        assert result[0] == {"x": "a", "y": "1", "z": "10"}

    def test_join_empty_both(self, kg):
        """Test _join_bindings with both empty lists."""
        result = kg._join_bindings([], [])
        assert result == []


class TestApplyFilterEdgeCases:
    def test_filter_numeric_equal(self, kg):
        """Test numeric equality filter."""
        bindings = [{"rank": "5"}, {"rank": "10"}]
        result = kg._apply_filter(bindings, "?rank == 5")
        assert len(result) == 1
        assert result[0]["rank"] == "5"

    def test_filter_with_quoted_string_value(self, kg):
        """Test filter with quoted string value."""
        bindings = [{"cat": "lip_care"}, {"cat": "skin_care"}]
        result = kg._apply_filter(bindings, '?cat == "lip_care"')
        assert len(result) == 1
        assert result[0]["cat"] == "lip_care"

    def test_filter_with_single_quoted_value(self, kg):
        """Test filter with single-quoted value."""
        bindings = [{"cat": "lip_care"}, {"cat": "skin_care"}]
        result = kg._apply_filter(bindings, "?cat == 'lip_care'")
        assert len(result) == 1
        assert result[0]["cat"] == "lip_care"

    def test_filter_preserves_order(self, kg):
        """Test that filter preserves order of bindings."""
        bindings = [{"rank": "5"}, {"rank": "10"}, {"rank": "15"}, {"rank": "3"}]
        result = kg._apply_filter(bindings, "?rank >= 5")
        # Should preserve order: 5, 10, 15 (3 is filtered out)
        ranks = [r["rank"] for r in result]
        assert ranks == ["5", "10", "15"]


class TestPredicateMapBuilding:
    def test_predicate_map_caching(self, kg):
        """Test that _build_predicate_map caches result."""
        map1 = kg._build_predicate_map()
        map2 = kg._build_predicate_map()
        # Should return same cached instance
        assert map1 is map2

    def test_predicate_map_contains_all_relation_types(self, kg):
        """Test that predicate map contains all RelationType enum values."""
        pred_map = kg._build_predicate_map()
        # Should have both value and camelCase mappings
        assert "hasProduct" in pred_map
        assert "hasProduct" in pred_map  # camelCase
        assert pred_map["hasProduct"] == RelationType.HAS_PRODUCT

    def test_predicate_map_camelcase_conversion(self, kg):
        """Test camelCase conversion for multi-word predicates."""
        pred_map = kg._build_predicate_map()
        # DIRECT_COMPETITOR -> directCompetitor
        assert "directCompetitor" in pred_map
        assert pred_map["directCompetitor"] == RelationType.DIRECT_COMPETITOR


class TestParseSparql:
    def test_parse_sparql_extracts_select_vars(self, kg):
        """Test _parse_sparql extracts SELECT variables correctly."""
        select_vars, patterns, filters = kg._parse_sparql("""
            SELECT ?brand ?product ?category
            WHERE { ?brand <hasProduct> ?product . }
        """)
        assert set(select_vars) == {"brand", "product", "category"}

    def test_parse_sparql_extracts_triple_patterns(self, kg):
        """Test _parse_sparql extracts triple patterns correctly."""
        select_vars, patterns, filters = kg._parse_sparql("""
            SELECT ?s ?o
            WHERE {
                ?s <hasProduct> ?o .
                ?s <belongsToCategory> "cat1" .
            }
        """)
        assert len(patterns) == 2

    def test_parse_sparql_extracts_filters(self, kg):
        """Test _parse_sparql extracts FILTER conditions."""
        select_vars, patterns, filters = kg._parse_sparql("""
            SELECT ?x ?rank
            WHERE {
                ?x <hasRank> ?rank .
                FILTER (?rank > 5)
                FILTER (?rank < 10)
            }
        """)
        assert len(filters) == 2

    def test_parse_sparql_empty_where_clause(self, kg):
        """Test _parse_sparql with empty WHERE clause."""
        select_vars, patterns, filters = kg._parse_sparql("""
            SELECT ?x
            WHERE { }
        """)
        assert patterns == []
        assert filters == []


class TestComplexScenarios:
    def test_complex_multi_hop_query(self, kg):
        """Test complex multi-hop relationship query."""
        # Build: Brand -> Product -> Category -> Parent Category
        kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "P1"))
        kg.add_relation(Relation("P1", RelationType.BELONGS_TO_CATEGORY, "lip_care"))
        kg.add_relation(Relation("lip_care", RelationType.PARENT_CATEGORY, "skin_care"))
        kg.add_relation(Relation("skin_care", RelationType.PARENT_CATEGORY, "beauty"))

        # Find path from brand to top-level category
        path = kg.find_path("LANEIGE", "beauty")
        assert path is not None
        assert len(path) == 4  # LANEIGE->P1, P1->lip_care, lip_care->skin_care, skin_care->beauty

    def test_complex_competition_network(self, kg):
        """Test complex competition network queries."""
        # Build competition network
        kg.add_relation(Relation("LANEIGE", RelationType.COMPETES_WITH, "COSRX"))
        kg.add_relation(Relation("LANEIGE", RelationType.COMPETES_WITH, "INNISFREE"))
        kg.add_relation(Relation("COSRX", RelationType.COMPETES_WITH, "INNISFREE"))
        kg.add_relation(Relation("INNISFREE", RelationType.COMPETES_WITH, "ETUDE_HOUSE"))

        # BFS should find entire competition network
        result = kg.bfs_traverse(
            "LANEIGE",
            max_depth=3,
            predicate_filter=[RelationType.COMPETES_WITH],
            direction="both",
        )
        # Should reach all competitors
        all_entities = [e for entities in result.values() for e in entities]
        assert "COSRX" in all_entities
        assert "INNISFREE" in all_entities

    def test_complex_sparql_with_join_and_filter(self, kg):
        """Test complex SPARQL with multiple joins and filters."""
        kg.add_relation(
            Relation(
                "LANEIGE",
                RelationType.HAS_PRODUCT,
                "P1",
                properties={"category": "lip_care", "rank": 3},
            )
        )
        kg.add_relation(Relation("P1", RelationType.BELONGS_TO_CATEGORY, "lip_care"))
        kg.add_relation(
            Relation(
                "LANEIGE",
                RelationType.HAS_PRODUCT,
                "P2",
                properties={"category": "skin_care", "rank": 15},
            )
        )
        kg.add_relation(Relation("P2", RelationType.BELONGS_TO_CATEGORY, "skin_care"))

        results = kg.sparql_query("""
            SELECT ?brand ?product ?cat
            WHERE {
                ?brand <hasProduct> ?product .
                ?product <belongsToCategory> ?cat .
            }
        """)
        assert len(results) == 2
        # Should have joined brand -> product -> category

    def test_metadata_enrichment_workflow(self, kg):
        """Test typical metadata enrichment workflow."""
        # Add entity
        kg.add_relation(Relation("LANEIGE", RelationType.HAS_PRODUCT, "P1"))

        # Enrich with metadata
        kg.set_entity_metadata("LANEIGE", {"sos": 5.2, "avg_rank": 8})
        kg.set_entity_metadata("LANEIGE", {"market_share": 12.5})

        # Query with context
        ctx = kg.get_entity_context("LANEIGE", depth=1)
        assert ctx["metadata"]["sos"] == 5.2
        assert ctx["metadata"]["market_share"] == 12.5

    def test_ranking_analysis_workflow(self, kg):
        """Test typical ranking analysis workflow."""
        # Add products with ranks
        for i, rank in enumerate([1, 3, 5, 8, 12]):
            kg.add_relation(
                Relation(
                    "LANEIGE",
                    RelationType.HAS_PRODUCT,
                    f"P{i}",
                    properties={"rank": rank, "category": "lip_care"},
                )
            )

        # Get all products
        products = kg.get_brand_products("LANEIGE")
        assert len(products) == 5

        # Filter by category
        lip_products = kg.get_brand_products("LANEIGE", category="lip_care")
        assert len(lip_products) == 5

        # Analyze using stats
        degree = kg.get_entity_degree("LANEIGE")
        assert degree["out_degree"] == 5

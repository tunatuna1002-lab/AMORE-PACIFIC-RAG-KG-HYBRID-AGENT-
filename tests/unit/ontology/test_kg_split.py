"""
KnowledgeGraph 분할 테스트
==========================
knowledge_graph.py → kg_query.py + kg_updater.py 분할 후 통합 테스트

테스트 범위:
1. KGQueryMixin — 쿼리, 그래프 탐색, 도메인 쿼리
2. KGUpdaterMixin — 데이터 로딩, 감성, 브랜드 소유권
3. KnowledgeGraph 통합 — 모든 기능이 단일 인스턴스에서 동작
"""

import pytest

from src.domain.entities.relations import Relation, RelationType
from src.ontology.kg_query import KGQueryMixin
from src.ontology.kg_updater import KGUpdaterMixin
from src.ontology.knowledge_graph import KnowledgeGraph, get_knowledge_graph


@pytest.fixture
def kg():
    """테스트용 KnowledgeGraph (auto_load=False)"""
    return KnowledgeGraph(auto_load=False, auto_save=False)


@pytest.fixture
def populated_kg(kg):
    """데이터가 미리 로드된 KG"""
    # 브랜드-제품 관계
    kg.add_relation(
        Relation(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            object="B08XYZ",
            properties={"product_name": "Lip Sleeping Mask", "category": "lip_care", "rank": 5},
        )
    )
    kg.add_relation(
        Relation(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            object="B09ABC",
            properties={"product_name": "Water Bank", "category": "skin_care", "rank": 15},
        )
    )
    kg.add_relation(
        Relation(
            subject="COSRX",
            predicate=RelationType.HAS_PRODUCT,
            object="B0ADEF",
            properties={"product_name": "Snail Mucin", "category": "skin_care", "rank": 3},
        )
    )

    # 제품-카테고리 관계
    kg.add_relation(
        Relation(
            subject="B08XYZ",
            predicate=RelationType.BELONGS_TO_CATEGORY,
            object="lip_care",
            properties={"rank": 5},
        )
    )
    kg.add_relation(
        Relation(
            subject="B09ABC",
            predicate=RelationType.BELONGS_TO_CATEGORY,
            object="skin_care",
            properties={"rank": 15},
        )
    )
    kg.add_relation(
        Relation(
            subject="B0ADEF",
            predicate=RelationType.BELONGS_TO_CATEGORY,
            object="skin_care",
            properties={"rank": 3},
        )
    )

    # 경쟁 관계
    kg.add_relation(
        Relation(
            subject="LANEIGE",
            predicate=RelationType.COMPETES_WITH,
            object="COSRX",
            properties={"category": "skin_care"},
        )
    )
    kg.add_relation(
        Relation(
            subject="COSRX",
            predicate=RelationType.COMPETES_WITH,
            object="LANEIGE",
            properties={"category": "skin_care"},
        )
    )

    # 엔티티 메타데이터
    kg.set_entity_metadata("LANEIGE", {"type": "brand", "sos": 0.12, "avg_rank": 10})
    kg.set_entity_metadata("COSRX", {"type": "brand", "sos": 0.18, "avg_rank": 8})

    return kg


# =============================================================================
# 1. Mixin Import 테스트
# =============================================================================


class TestMixinImports:
    """Mixin 모듈 임포트 테스트"""

    def test_kg_query_mixin_importable(self):
        """KGQueryMixin 임포트 가능"""
        from src.ontology.kg_query import KGQueryMixin

        assert KGQueryMixin is not None

    def test_kg_updater_mixin_importable(self):
        """KGUpdaterMixin 임포트 가능"""
        from src.ontology.kg_updater import KGUpdaterMixin

        assert KGUpdaterMixin is not None

    def test_knowledge_graph_inherits_mixins(self):
        """KnowledgeGraph가 두 Mixin을 상속"""
        assert issubclass(KnowledgeGraph, KGQueryMixin)
        assert issubclass(KnowledgeGraph, KGUpdaterMixin)


# =============================================================================
# 2. KGQueryMixin 테스트 (쿼리 & 탐색)
# =============================================================================


class TestKGQueryMixin:
    """KGQueryMixin 메서드 테스트"""

    def test_query_by_subject(self, populated_kg):
        """주체별 쿼리"""
        results = populated_kg.query(subject="LANEIGE")
        assert len(results) >= 2  # HAS_PRODUCT x2 + COMPETES_WITH

    def test_query_by_predicate(self, populated_kg):
        """관계 유형별 쿼리"""
        results = populated_kg.query(predicate=RelationType.HAS_PRODUCT)
        assert len(results) == 3

    def test_query_by_object(self, populated_kg):
        """객체별 쿼리"""
        results = populated_kg.query(object_="lip_care")
        assert len(results) == 1

    def test_query_combined(self, populated_kg):
        """복합 조건 쿼리"""
        results = populated_kg.query(subject="LANEIGE", predicate=RelationType.HAS_PRODUCT)
        assert len(results) == 2

    def test_get_subjects(self, populated_kg):
        """특정 관계/객체의 주체 조회"""
        subjects = populated_kg.get_subjects(RelationType.BELONGS_TO_CATEGORY, "skin_care")
        assert set(subjects) == {"B09ABC", "B0ADEF"}

    def test_get_objects(self, populated_kg):
        """특정 주체/관계의 객체 조회"""
        objects = populated_kg.get_objects("LANEIGE", RelationType.HAS_PRODUCT)
        assert set(objects) == {"B08XYZ", "B09ABC"}

    def test_exists(self, populated_kg):
        """관계 존재 여부 확인"""
        assert populated_kg.exists("LANEIGE", RelationType.HAS_PRODUCT, "B08XYZ")
        assert not populated_kg.exists("LANEIGE", RelationType.HAS_PRODUCT, "NONEXIST")

    def test_get_neighbors_outgoing(self, populated_kg):
        """이웃 조회 - 나가는 방향"""
        neighbors = populated_kg.get_neighbors("LANEIGE", direction="outgoing")
        assert len(neighbors["outgoing"]) >= 3  # 2 products + 1 competes
        assert len(neighbors["incoming"]) == 0

    def test_get_neighbors_both(self, populated_kg):
        """이웃 조회 - 양방향"""
        neighbors = populated_kg.get_neighbors("LANEIGE", direction="both")
        assert len(neighbors["outgoing"]) >= 3
        assert len(neighbors["incoming"]) >= 1  # COSRX COMPETES_WITH

    def test_bfs_traverse(self, populated_kg):
        """BFS 탐색"""
        levels = populated_kg.bfs_traverse("LANEIGE", max_depth=2)
        assert 0 in levels
        assert "LANEIGE" in levels[0]
        assert len(levels) >= 2

    def test_find_path(self, populated_kg):
        """경로 탐색"""
        path = populated_kg.find_path("LANEIGE", "lip_care")
        assert path is not None
        assert len(path) == 2  # LANEIGE -> B08XYZ -> lip_care

    def test_find_path_no_path(self, populated_kg):
        """경로 없음"""
        path = populated_kg.find_path("LANEIGE", "NONEXISTENT")
        assert path is None

    def test_get_brand_products(self, populated_kg):
        """브랜드 제품 조회"""
        products = populated_kg.get_brand_products("LANEIGE")
        assert len(products) == 2
        asins = {p["asin"] for p in products}
        assert asins == {"B08XYZ", "B09ABC"}

    def test_get_brand_products_with_category(self, populated_kg):
        """브랜드 제품 조회 - 카테고리 필터"""
        products = populated_kg.get_brand_products("LANEIGE", category="lip_care")
        assert len(products) == 1
        assert products[0]["asin"] == "B08XYZ"

    def test_get_competitors(self, populated_kg):
        """경쟁사 조회"""
        competitors = populated_kg.get_competitors("LANEIGE")
        brands = {c["brand"] for c in competitors}
        assert "COSRX" in brands

    def test_entity_metadata(self, populated_kg):
        """엔티티 메타데이터 설정/조회"""
        meta = populated_kg.get_entity_metadata("LANEIGE")
        assert meta["type"] == "brand"
        assert meta["sos"] == 0.12

    def test_get_stats(self, populated_kg):
        """통계 조회"""
        stats = populated_kg.get_stats()
        assert stats["total_triples"] == 8
        assert stats["unique_subjects"] > 0

    def test_get_entity_degree(self, populated_kg):
        """엔티티 차수 계산"""
        degree = populated_kg.get_entity_degree("LANEIGE")
        assert degree["out_degree"] >= 3
        assert degree["total"] >= 4

    def test_get_most_connected(self, populated_kg):
        """가장 연결된 엔티티"""
        most = populated_kg.get_most_connected(top_n=3)
        assert len(most) > 0
        # LANEIGE should be among most connected
        names = [name for name, _ in most]
        assert "LANEIGE" in names

    def test_get_entity_context(self, populated_kg):
        """엔티티 컨텍스트 조회"""
        ctx = populated_kg.get_entity_context("LANEIGE", depth=1)
        assert ctx["entity"] == "LANEIGE"
        assert "relations" in ctx
        assert "metadata" in ctx


# =============================================================================
# 3. KGUpdaterMixin 테스트 (데이터 로딩)
# =============================================================================


class TestKGUpdaterMixin:
    """KGUpdaterMixin 메서드 테스트"""

    def test_load_from_crawl_data(self, kg):
        """크롤링 데이터 로딩"""
        crawl_data = {
            "categories": {
                "lip_care": {
                    "rank_records": [
                        {
                            "brand": "LANEIGE",
                            "product_asin": "B08XYZ",
                            "title": "Lip Mask",
                            "rank": 5,
                        },
                        {
                            "brand": "COSRX",
                            "product_asin": "B09ABC",
                            "title": "Lip Balm",
                            "rank": 10,
                        },
                    ]
                }
            }
        }
        added = kg.load_from_crawl_data(crawl_data)
        assert added > 0
        assert len(kg.triples) > 0

    def test_load_from_metrics_data(self, kg):
        """지표 데이터 로딩"""
        metrics_data = {
            "brand_metrics": [
                {
                    "brand_name": "LANEIGE",
                    "share_of_shelf": 0.12,
                    "avg_rank": 10,
                    "product_count": 5,
                },
            ],
            "product_metrics": [
                {"asin": "B08XYZ", "current_rank": 5, "rank_change_1d": -2},
            ],
            "alerts": [],
        }
        added = kg.load_from_metrics_data(metrics_data)
        meta = kg.get_entity_metadata("LANEIGE")
        assert meta["type"] == "brand"
        assert meta["sos"] == 0.12

    def test_load_from_sentiment_data(self, populated_kg):
        """감성 데이터 로딩"""
        sentiment_data = {
            "products": {
                "B08XYZ": {
                    "asin": "B08XYZ",
                    "ai_customers_say": "Customers love the moisturizing effect",
                    "sentiment_tags": ["Moisturizing", "Value for money"],
                    "success": True,
                }
            },
            "collected_at": "2026-01-15",
        }
        added = populated_kg.load_from_sentiment_data(sentiment_data)
        assert added > 0

    def test_get_product_sentiments(self, populated_kg):
        """제품 감성 조회 (빈 결과)"""
        sentiments = populated_kg.get_product_sentiments("B08XYZ")
        assert sentiments["asin"] == "B08XYZ"
        assert isinstance(sentiments["sentiment_tags"], list)

    def test_get_brand_sentiment_profile(self, populated_kg):
        """브랜드 감성 프로필"""
        profile = populated_kg.get_brand_sentiment_profile("LANEIGE")
        assert profile["brand"] == "LANEIGE"
        assert profile["product_count"] == 2

    def test_compare_product_sentiments(self, populated_kg):
        """제품 감성 비교"""
        comparison = populated_kg.compare_product_sentiments("B08XYZ", "B0ADEF")
        assert "product1" in comparison
        assert "product2" in comparison
        assert "common_tags" in comparison

    def test_get_category_hierarchy_not_found(self, kg):
        """카테고리 계층 조회 - 미존재"""
        result = kg.get_category_hierarchy("nonexistent")
        assert "error" in result

    def test_get_product_category_context(self, populated_kg):
        """제품 카테고리 컨텍스트"""
        ctx = populated_kg.get_product_category_context("B08XYZ")
        assert ctx["product"] == "B08XYZ"
        assert len(ctx["categories"]) >= 1


# =============================================================================
# 4. Core CRUD 테스트
# =============================================================================


class TestKGCoreCRUD:
    """KnowledgeGraph 핵심 CRUD 테스트"""

    def test_add_relation(self, kg):
        """관계 추가"""
        rel = Relation(subject="A", predicate=RelationType.HAS_PRODUCT, object="B")
        assert kg.add_relation(rel) is True
        assert len(kg.triples) == 1

    def test_add_duplicate_returns_false(self, kg):
        """중복 관계 추가 시 False"""
        rel = Relation(subject="A", predicate=RelationType.HAS_PRODUCT, object="B")
        kg.add_relation(rel)
        assert kg.add_relation(rel) is False
        assert len(kg.triples) == 1

    def test_remove_relation(self, kg):
        """관계 삭제"""
        rel = Relation(subject="A", predicate=RelationType.HAS_PRODUCT, object="B")
        kg.add_relation(rel)
        assert kg.remove_relation(rel) is True
        assert len(kg.triples) == 0

    def test_clear(self, populated_kg):
        """전체 초기화"""
        assert len(populated_kg.triples) > 0
        populated_kg.clear()
        assert len(populated_kg.triples) == 0
        assert len(populated_kg.entity_metadata) == 0

    def test_add_relations_batch(self, kg):
        """일괄 관계 추가"""
        rels = [
            Relation(subject="A", predicate=RelationType.HAS_PRODUCT, object="B1"),
            Relation(subject="A", predicate=RelationType.HAS_PRODUCT, object="B2"),
            Relation(subject="A", predicate=RelationType.HAS_PRODUCT, object="B3"),
        ]
        added = kg.add_relations(rels)
        assert added == 3

    def test_context_manager(self):
        """컨텍스트 매니저"""
        with KnowledgeGraph(auto_load=False, auto_save=False) as kg:
            kg.add_relation(Relation(subject="A", predicate=RelationType.HAS_PRODUCT, object="B"))
            assert len(kg.triples) == 1


# =============================================================================
# 5. 싱글톤 패턴 테스트
# =============================================================================


class TestSingleton:
    """싱글톤 패턴 테스트"""

    def test_singleton_import(self):
        """get_knowledge_graph 임포트"""
        assert callable(get_knowledge_graph)

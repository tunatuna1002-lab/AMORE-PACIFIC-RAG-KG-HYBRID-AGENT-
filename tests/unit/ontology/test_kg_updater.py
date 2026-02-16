"""
KGUpdaterMixin 단위 테스트
==========================
src/ontology/kg_updater.py 커버리지 38% → 85%+ 목표

테스트 대상 메서드 (14개):
1. load_from_crawl_data
2. load_from_metrics_data
3. load_category_hierarchy
4. get_category_hierarchy
5. get_product_category_context
6. load_from_sentiment_data
7. get_product_sentiments
8. get_brand_sentiment_profile
9. compare_product_sentiments
10. find_products_by_sentiment
11. load_brand_ownership
12. get_brand_ownership
13. is_amorepacific_brand
14. get_amorepacific_brands
"""

import json

import pytest

from src.domain.entities.relations import Relation, RelationType
from src.ontology.knowledge_graph import KnowledgeGraph

# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def kg():
    """빈 KnowledgeGraph 인스턴스"""
    return KnowledgeGraph(persist_path="/dev/null", auto_load=False, auto_save=False)


@pytest.fixture
def crawl_data_single():
    """단일 카테고리, 2개 브랜드, 3개 제품"""
    return {
        "categories": {
            "lip_care": {
                "rank_records": [
                    {
                        "brand": "LANEIGE",
                        "product_asin": "B08XYZ001",
                        "title": "Lip Sleeping Mask",
                        "rank": 1,
                        "rating": 4.7,
                        "price": 24.0,
                    },
                    {
                        "brand": "LANEIGE",
                        "product_asin": "B08XYZ002",
                        "title": "Lip Glowy Balm",
                        "rank": 5,
                        "rating": 4.5,
                        "price": 18.0,
                    },
                    {
                        "brand": "COSRX",
                        "product_asin": "B09ABC001",
                        "title": "Lip Sleep Ceramide",
                        "rank": 3,
                        "rating": 4.3,
                        "price": 12.0,
                    },
                ]
            }
        }
    }


@pytest.fixture
def crawl_data_multi_brand():
    """12개 브랜드 — 경쟁 관계 top-10 로직 테스트"""
    brands = [f"Brand_{i}" for i in range(12)]
    records = []
    for i, brand in enumerate(brands):
        records.append(
            {
                "brand": brand,
                "product_asin": f"ASIN_{i:03d}",
                "title": f"Product {i}",
                "rank": i + 1,
            }
        )
    return {"categories": {"beauty": {"rank_records": records}}}


@pytest.fixture
def metrics_data():
    """메트릭 데이터 샘플"""
    return {
        "brand_metrics": [
            {
                "brand_name": "LANEIGE",
                "share_of_shelf": 15.0,
                "avg_rank": 3.2,
                "product_count": 5,
                "is_laneige": True,
                "category_id": "lip_care",
            },
            {
                "brand_name": "COSRX",
                "share_of_shelf": 8.0,
                "avg_rank": 7.5,
                "product_count": 3,
                "is_laneige": False,
                "category_id": "lip_care",
            },
        ],
        "product_metrics": [
            {
                "asin": "B08XYZ001",
                "current_rank": 1,
                "rank_change_1d": -2,
                "rank_change_7d": 5,
                "rank_volatility": 1.2,
                "streak_days": 10,
                "rating": 4.7,
                "category_id": "lip_care",
            }
        ],
        "alerts": [
            {
                "asin": "B08XYZ001",
                "type": "rank_drop",
                "severity": "high",
                "message": "Rank dropped 5 positions",
                "details": {"from": 1, "to": 6},
            }
        ],
    }


@pytest.fixture
def sentiment_data():
    """감성 데이터 샘플"""
    return {
        "collected_at": "2025-01-15",
        "products": {
            "B08XYZ001": {
                "asin": "B08XYZ001",
                "ai_customers_say": "Customers like the moisturizing effect",
                "sentiment_tags": ["Moisturizing", "Value for money", "Nice scent"],
                "success": True,
            },
            "B09ABC001": {
                "asin": "B09ABC001",
                "ai_customers_say": "Gentle and effective for dry lips",
                "sentiment_tags": ["Moisturizing", "Gentle"],
                "success": True,
            },
            "B00FAIL01": {
                "asin": "B00FAIL01",
                "ai_customers_say": None,
                "sentiment_tags": [],
                "success": False,
            },
        },
    }


@pytest.fixture
def category_hierarchy_file(tmp_path):
    """임시 카테고리 계층 JSON"""
    data = {
        "categories": {
            "beauty": {
                "name": "Beauty & Personal Care",
                "amazon_node_id": "3760911",
                "level": 0,
                "parent_id": None,
                "path": ["beauty"],
                "url": "https://amazon.com/beauty",
                "children": ["skin_care", "makeup"],
            },
            "skin_care": {
                "name": "Skin Care",
                "amazon_node_id": "11060451",
                "level": 1,
                "parent_id": "beauty",
                "path": ["beauty", "skin_care"],
                "url": "",
                "children": ["lip_care"],
            },
            "lip_care": {
                "name": "Lip Care",
                "amazon_node_id": "3761351",
                "level": 2,
                "parent_id": "skin_care",
                "path": ["beauty", "skin_care", "lip_care"],
                "url": "",
                "children": [],
            },
        }
    }
    path = tmp_path / "category_hierarchy.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


@pytest.fixture
def brands_config_file(tmp_path):
    """임시 브랜드 소유권 JSON"""
    data = {
        "amorepacific_brands": [
            {
                "name": "LANEIGE",
                "segment": "Premium",
                "category": "Skincare",
                "acquired": None,
                "country": "Korea",
                "aliases": ["라네즈"],
            },
            {
                "name": "Sulwhasoo",
                "segment": "Luxury",
                "category": "Skincare",
                "acquired": None,
                "country": "Korea",
                "aliases": ["설화수"],
            },
            {
                "name": "COSRX",
                "segment": "K-Beauty",
                "category": "Skincare",
                "acquired": 2024,
                "country": "Korea",
                "aliases": [],
            },
        ],
        "brand_ownership": {
            "COSRX": {
                "note": "COSRX is a Korean brand acquired by Amorepacific in 2024",
                "evidence": ["IR 2025 Q1-Q3"],
                "country_of_origin": "Korea",
            }
        },
    }
    path = tmp_path / "brands.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


# =========================================================================
# 1. load_from_crawl_data
# =========================================================================


class TestLoadFromCrawlData:
    """크롤링 데이터 로드 테스트"""

    def test_basic_loading(self, kg, crawl_data_single):
        """기본 크롤링 데이터 로드 — 관계 수 확인"""
        added = kg.load_from_crawl_data(crawl_data_single)
        # 3 products × (brand_product + product_category) = 6
        # 2 brands × 1 pair (LANEIGE↔COSRX) = 2 competition relations
        assert added >= 6

    def test_brand_product_relations(self, kg, crawl_data_single):
        """Brand → Product 관계 생성 확인"""
        kg.load_from_crawl_data(crawl_data_single)
        rels = kg.query(subject="LANEIGE", predicate=RelationType.HAS_PRODUCT)
        asins = {r.object for r in rels}
        assert "B08XYZ001" in asins
        assert "B08XYZ002" in asins

    def test_product_category_relations(self, kg, crawl_data_single):
        """Product → Category 관계 생성 확인"""
        kg.load_from_crawl_data(crawl_data_single)
        rels = kg.query(subject="B08XYZ001", predicate=RelationType.BELONGS_TO_CATEGORY)
        assert len(rels) >= 1
        assert rels[0].object == "lip_care"

    def test_competition_relations(self, kg, crawl_data_single):
        """브랜드 간 경쟁 관계 생성 확인"""
        kg.load_from_crawl_data(crawl_data_single)
        rels = kg.query(subject="LANEIGE", predicate=RelationType.DIRECT_COMPETITOR)
        if not rels:
            # indirect일 수도 있음
            rels = kg.query(subject="LANEIGE", predicate=RelationType.INDIRECT_COMPETITOR)
        assert len(rels) >= 1

    def test_competition_bidirectional(self, kg, crawl_data_single):
        """경쟁 관계가 양방향으로 생성되는지 확인"""
        kg.load_from_crawl_data(crawl_data_single)
        fwd = kg.query(subject="LANEIGE")
        rev = kg.query(subject="COSRX")
        fwd_targets = {r.object for r in fwd}
        rev_targets = {r.object for r in rev}
        assert "COSRX" in fwd_targets
        assert "LANEIGE" in rev_targets

    def test_empty_crawl_data(self, kg):
        """빈 데이터 — 관계 0"""
        assert kg.load_from_crawl_data({}) == 0
        assert kg.load_from_crawl_data({"categories": {}}) == 0

    def test_product_without_asin_skipped(self, kg):
        """ASIN 없는 제품은 건너뛰기"""
        data = {
            "categories": {
                "lip_care": {
                    "rank_records": [{"brand": "LANEIGE", "title": "No ASIN Product", "rank": 1}]
                }
            }
        }
        added = kg.load_from_crawl_data(data)
        assert added == 0

    def test_alternative_asin_key(self, kg):
        """product_asin이 없으면 asin 키 사용"""
        data = {
            "categories": {
                "lip_care": {
                    "rank_records": [
                        {
                            "brand": "LANEIGE",
                            "asin": "ALT_ASIN_01",
                            "product_name": "Alt Product",
                            "rank": 1,
                        }
                    ]
                }
            }
        }
        added = kg.load_from_crawl_data(data)
        assert added >= 1
        rels = kg.query(subject="LANEIGE", predicate=RelationType.HAS_PRODUCT)
        assert any(r.object == "ALT_ASIN_01" for r in rels)

    def test_many_brands_competition_top10(self, kg, crawl_data_multi_brand):
        """12개 브랜드 중 Top 10만 경쟁 관계 생성"""
        added = kg.load_from_crawl_data(crawl_data_multi_brand)
        # Top 5 이내 = direct, 그 외 = indirect
        brand_10 = kg.query(subject="Brand_10")
        brand_11 = kg.query(subject="Brand_11")
        # Brand_10, Brand_11은 index 10, 11이므로 경쟁 관계에 포함되지 않음
        comp_rels_10 = [
            r
            for r in brand_10
            if r.predicate in (RelationType.DIRECT_COMPETITOR, RelationType.INDIRECT_COMPETITOR)
        ]
        comp_rels_11 = [
            r
            for r in brand_11
            if r.predicate in (RelationType.DIRECT_COMPETITOR, RelationType.INDIRECT_COMPETITOR)
        ]
        assert len(comp_rels_10) == 0
        assert len(comp_rels_11) == 0

    def test_direct_vs_indirect_competition(self, kg, crawl_data_multi_brand):
        """Top 5 내 = direct, 그 외 = indirect"""
        kg.load_from_crawl_data(crawl_data_multi_brand)
        # Brand_0 (rank 1) vs Brand_1 (rank 2) → both top 5 → direct
        direct = kg.query(subject="Brand_0", predicate=RelationType.DIRECT_COMPETITOR)
        direct_targets = {r.object for r in direct}
        assert "Brand_1" in direct_targets

        # Brand_0 vs Brand_6 → Brand_6 is index 6 (not top 5) → indirect
        indirect = kg.query(subject="Brand_0", predicate=RelationType.INDIRECT_COMPETITOR)
        indirect_targets = {r.object for r in indirect}
        assert "Brand_6" in indirect_targets


# =========================================================================
# 2. load_from_metrics_data
# =========================================================================


class TestLoadFromMetricsData:
    """지표 데이터 로드 테스트"""

    def test_brand_metadata_set(self, kg, metrics_data):
        """브랜드 메타데이터가 올바르게 설정되는지"""
        kg.load_from_metrics_data(metrics_data)
        meta = kg.get_entity_metadata("LANEIGE")
        assert meta["type"] == "brand"
        assert meta["sos"] == 15.0
        assert meta["avg_rank"] == 3.2
        assert meta["is_target"] is True

    def test_product_metadata_set(self, kg, metrics_data):
        """제품 메타데이터 설정 확인"""
        kg.load_from_metrics_data(metrics_data)
        meta = kg.get_entity_metadata("B08XYZ001")
        assert meta["type"] == "product"
        assert meta["current_rank"] == 1
        assert meta["rank_change_1d"] == -2
        assert meta["rank_change_7d"] == 5
        assert meta["streak_days"] == 10

    def test_alert_relations(self, kg, metrics_data):
        """알림 관계 생성 확인"""
        added = kg.load_from_metrics_data(metrics_data)
        assert added >= 1
        rels = kg.query(subject="B08XYZ001", predicate=RelationType.HAS_ALERT)
        assert len(rels) == 1
        assert rels[0].object == "rank_drop"
        assert rels[0].properties["severity"] == "high"

    def test_empty_metrics(self, kg):
        """빈 메트릭 데이터"""
        assert kg.load_from_metrics_data({}) == 0

    def test_brand_without_name_skipped(self, kg):
        """브랜드명 없는 메트릭은 건너뛰기"""
        data = {"brand_metrics": [{"share_of_shelf": 10.0}]}
        kg.load_from_metrics_data(data)
        # No exception, no metadata set for None

    def test_product_without_asin_skipped(self, kg):
        """ASIN 없는 제품 메트릭은 건너뛰기"""
        data = {"product_metrics": [{"current_rank": 5}]}
        kg.load_from_metrics_data(data)
        # No exception

    def test_alert_without_asin_skipped(self, kg):
        """ASIN 없는 알림은 건너뛰기"""
        data = {"alerts": [{"type": "rank_drop", "severity": "low"}]}
        added = kg.load_from_metrics_data(data)
        assert added == 0

    def test_multiple_alerts(self, kg):
        """여러 알림 관계 생성"""
        data = {
            "alerts": [
                {"asin": "A001", "type": "rank_drop", "severity": "high", "message": "drop"},
                {"asin": "A002", "type": "new_entry", "severity": "info", "message": "new"},
            ]
        }
        added = kg.load_from_metrics_data(data)
        assert added == 2


# =========================================================================
# 3. load_category_hierarchy
# =========================================================================


class TestLoadCategoryHierarchy:
    """카테고리 계층 로드 테스트"""

    def test_basic_loading(self, kg, category_hierarchy_file):
        """기본 카테고리 계층 로드"""
        added = kg.load_category_hierarchy(category_hierarchy_file)
        # beauty (no parent) → 0 rels
        # skin_care (parent=beauty) → 2 rels (parent + subcategory)
        # lip_care (parent=skin_care) → 2 rels
        assert added == 4

    def test_category_metadata(self, kg, category_hierarchy_file):
        """카테고리 메타데이터 확인"""
        kg.load_category_hierarchy(category_hierarchy_file)
        meta = kg.get_entity_metadata("lip_care")
        assert meta["type"] == "category"
        assert meta["name"] == "Lip Care"
        assert meta["level"] == 2
        assert meta["parent_id"] == "skin_care"

    def test_parent_child_relations(self, kg, category_hierarchy_file):
        """부모-자식 관계 확인"""
        kg.load_category_hierarchy(category_hierarchy_file)
        # skin_care → beauty (parent)
        parent_rels = kg.query(subject="skin_care", predicate=RelationType.PARENT_CATEGORY)
        assert len(parent_rels) == 1
        assert parent_rels[0].object == "beauty"

        # beauty → skin_care (subcategory)
        child_rels = kg.query(subject="beauty", predicate=RelationType.HAS_SUBCATEGORY)
        child_objects = {r.object for r in child_rels}
        assert "skin_care" in child_objects

    def test_file_not_found(self, kg):
        """존재하지 않는 파일 — 0 반환"""
        added = kg.load_category_hierarchy("/nonexistent/path/hierarchy.json")
        assert added == 0

    def test_root_category_no_parent(self, kg, category_hierarchy_file):
        """루트 카테고리(beauty)는 부모 관계 없음"""
        kg.load_category_hierarchy(category_hierarchy_file)
        parent_rels = kg.query(subject="beauty", predicate=RelationType.PARENT_CATEGORY)
        assert len(parent_rels) == 0


# =========================================================================
# 4. get_category_hierarchy
# =========================================================================


class TestGetCategoryHierarchy:
    """카테고리 계층 조회 테스트"""

    def test_leaf_category(self, kg, category_hierarchy_file):
        """리프 카테고리(lip_care) 조회"""
        kg.load_category_hierarchy(category_hierarchy_file)
        result = kg.get_category_hierarchy("lip_care")
        assert result["category"] == "lip_care"
        assert result["name"] == "Lip Care"
        assert result["level"] == 2
        # 상위: skin_care → beauty
        assert len(result["ancestors"]) == 2
        ancestor_ids = [a["id"] for a in result["ancestors"]]
        assert "skin_care" in ancestor_ids
        assert "beauty" in ancestor_ids
        # 하위: 없음
        assert len(result["descendants"]) == 0

    def test_middle_category(self, kg, category_hierarchy_file):
        """중간 카테고리(skin_care) 조회"""
        kg.load_category_hierarchy(category_hierarchy_file)
        result = kg.get_category_hierarchy("skin_care")
        assert result["name"] == "Skin Care"
        assert len(result["ancestors"]) == 1
        assert result["ancestors"][0]["id"] == "beauty"
        assert len(result["descendants"]) >= 1

    def test_root_category(self, kg, category_hierarchy_file):
        """루트 카테고리(beauty) 조회"""
        kg.load_category_hierarchy(category_hierarchy_file)
        result = kg.get_category_hierarchy("beauty")
        assert result["name"] == "Beauty & Personal Care"
        assert len(result["ancestors"]) == 0
        assert len(result["descendants"]) >= 1

    def test_nonexistent_category(self, kg):
        """존재하지 않는 카테고리"""
        result = kg.get_category_hierarchy("nonexistent")
        assert "error" in result

    def test_non_category_entity(self, kg):
        """type이 category가 아닌 엔티티"""
        kg.set_entity_metadata("LANEIGE", {"type": "brand"})
        result = kg.get_category_hierarchy("LANEIGE")
        assert "error" in result


# =========================================================================
# 5. get_product_category_context
# =========================================================================


class TestGetProductCategoryContext:
    """제품 카테고리 컨텍스트 조회 테스트"""

    def test_product_with_categories(self, kg, crawl_data_single, category_hierarchy_file):
        """제품의 카테고리 컨텍스트 조회"""
        kg.load_from_crawl_data(crawl_data_single)
        kg.load_category_hierarchy(category_hierarchy_file)
        result = kg.get_product_category_context("B08XYZ001")
        assert result["product"] == "B08XYZ001"
        assert len(result["categories"]) >= 1
        cat = result["categories"][0]
        assert cat["category_id"] == "lip_care"
        assert "hierarchy" in cat

    def test_product_no_categories(self, kg):
        """카테고리 없는 제품"""
        result = kg.get_product_category_context("UNKNOWN_ASIN")
        assert result["product"] == "UNKNOWN_ASIN"
        assert result["categories"] == []


# =========================================================================
# 6. load_from_sentiment_data
# =========================================================================


class TestLoadFromSentimentData:
    """감성 데이터 로드 테스트"""

    def test_basic_loading(self, kg, sentiment_data):
        """기본 감성 데이터 로드"""
        added = kg.load_from_sentiment_data(sentiment_data)
        # B08XYZ001: 1 ai_summary + 3 sentiment tags = 4
        # B09ABC001: 1 ai_summary + 2 sentiment tags = 3
        # B00FAIL01: success=False → skip
        assert added >= 7

    def test_ai_summary_relation(self, kg, sentiment_data):
        """AI Summary 관계 확인"""
        kg.load_from_sentiment_data(sentiment_data)
        rels = kg.query(subject="B08XYZ001", predicate=RelationType.HAS_AI_SUMMARY)
        assert len(rels) == 1
        assert "moisturizing" in rels[0].properties["summary_text"].lower()

    def test_sentiment_tag_relations(self, kg, sentiment_data):
        """감성 태그 관계 확인"""
        kg.load_from_sentiment_data(sentiment_data)
        rels = kg.query(subject="B08XYZ001", predicate=RelationType.HAS_SENTIMENT)
        tags = {r.object for r in rels}
        assert "Moisturizing" in tags
        assert "Value for money" in tags

    def test_failed_product_skipped(self, kg, sentiment_data):
        """success=False 제품은 건너뛰기"""
        kg.load_from_sentiment_data(sentiment_data)
        rels = kg.query(subject="B00FAIL01")
        assert len(rels) == 0

    def test_metadata_updated(self, kg, sentiment_data):
        """AI Summary 메타데이터 업데이트"""
        kg.load_from_sentiment_data(sentiment_data)
        meta = kg.get_entity_metadata("B08XYZ001")
        assert meta.get("ai_summary") is not None
        assert meta.get("ai_summary_collected_at") == "2025-01-15"

    def test_empty_sentiment_data(self, kg):
        """빈 감성 데이터"""
        assert kg.load_from_sentiment_data({}) == 0
        assert kg.load_from_sentiment_data({"products": {}}) == 0

    def test_product_without_ai_summary(self, kg):
        """AI Summary 없는 제품"""
        data = {
            "products": {
                "ASIN_X": {
                    "asin": "ASIN_X",
                    "ai_customers_say": None,
                    "sentiment_tags": ["Moisturizing"],
                    "success": True,
                }
            }
        }
        added = kg.load_from_sentiment_data(data)
        assert added >= 1  # sentiment tag만 추가
        rels = kg.query(subject="ASIN_X", predicate=RelationType.HAS_AI_SUMMARY)
        assert len(rels) == 0


# =========================================================================
# 7. get_product_sentiments
# =========================================================================


class TestGetProductSentiments:
    """제품 감성 정보 조회 테스트"""

    def test_with_data(self, kg, sentiment_data):
        """감성 데이터가 있는 제품"""
        kg.load_from_sentiment_data(sentiment_data)
        result = kg.get_product_sentiments("B08XYZ001")
        assert result["asin"] == "B08XYZ001"
        assert result["ai_summary"] is not None
        assert "Moisturizing" in result["sentiment_tags"]
        assert len(result["sentiment_clusters"]) >= 1

    def test_without_data(self, kg):
        """감성 데이터 없는 제품"""
        result = kg.get_product_sentiments("UNKNOWN")
        assert result["asin"] == "UNKNOWN"
        assert result["ai_summary"] is None
        assert result["sentiment_tags"] == []
        assert result["sentiment_clusters"] == {}

    def test_clusters_grouped(self, kg, sentiment_data):
        """클러스터별 그룹핑 확인"""
        kg.load_from_sentiment_data(sentiment_data)
        result = kg.get_product_sentiments("B08XYZ001")
        # Moisturizing → Hydration, Value for money → Pricing, Nice scent → Sensory
        clusters = result["sentiment_clusters"]
        assert "Hydration" in clusters or "Pricing" in clusters or "Sensory" in clusters


# =========================================================================
# 8. get_brand_sentiment_profile
# =========================================================================


class TestGetBrandSentimentProfile:
    """브랜드 감성 프로필 테스트"""

    def test_brand_profile(self, kg, crawl_data_single, sentiment_data):
        """브랜드 감성 프로필 생성"""
        kg.load_from_crawl_data(crawl_data_single)
        kg.load_from_sentiment_data(sentiment_data)
        result = kg.get_brand_sentiment_profile("LANEIGE")
        assert result["brand"] == "LANEIGE"
        assert result["product_count"] >= 1
        assert len(result["all_tags"]) >= 1

    def test_dominant_sentiment(self, kg, crawl_data_single, sentiment_data):
        """dominant_sentiment이 올바르게 설정되는지"""
        kg.load_from_crawl_data(crawl_data_single)
        kg.load_from_sentiment_data(sentiment_data)
        result = kg.get_brand_sentiment_profile("LANEIGE")
        # dominant_sentiment은 가장 많이 언급된 태그
        assert result["dominant_sentiment"] is not None

    def test_brand_no_products(self, kg):
        """제품이 없는 브랜드"""
        result = kg.get_brand_sentiment_profile("UNKNOWN_BRAND")
        assert result["product_count"] == 0
        assert result["all_tags"] == []
        assert result["dominant_sentiment"] is None

    def test_brand_with_product_no_sentiments(self, kg):
        """제품은 있지만 감성 데이터 없는 브랜드"""
        rel = Relation(
            subject="TestBrand",
            predicate=RelationType.HAS_PRODUCT,
            object="VALID_ASIN",
            properties={"product_name": "Valid"},
            source="test",
        )
        kg.add_relation(rel)
        result = kg.get_brand_sentiment_profile("TestBrand")
        assert result["product_count"] >= 1
        assert result["all_tags"] == []
        assert result["dominant_sentiment"] is None


# =========================================================================
# 9. compare_product_sentiments
# =========================================================================


class TestCompareProductSentiments:
    """두 제품 감성 비교 테스트"""

    def test_comparison(self, kg, sentiment_data):
        """두 제품의 감성 비교"""
        kg.load_from_sentiment_data(sentiment_data)
        result = kg.compare_product_sentiments("B08XYZ001", "B09ABC001")
        assert "product1" in result
        assert "product2" in result
        # Moisturizing은 공통
        assert "Moisturizing" in result["common_tags"]
        # Nice scent은 B08XYZ001만
        assert "Nice scent" in result["unique_to_1"]
        # Gentle은 B09ABC001만
        assert "Gentle" in result["unique_to_2"]

    def test_comparison_no_data(self, kg):
        """데이터 없는 제품 비교"""
        result = kg.compare_product_sentiments("A", "B")
        assert result["common_tags"] == []
        assert result["unique_to_1"] == []
        assert result["unique_to_2"] == []

    def test_cluster_comparison(self, kg, sentiment_data):
        """클러스터 비교 포함 확인"""
        kg.load_from_sentiment_data(sentiment_data)
        result = kg.compare_product_sentiments("B08XYZ001", "B09ABC001")
        assert "cluster_comparison" in result
        assert "product1_clusters" in result["cluster_comparison"]
        assert "product2_clusters" in result["cluster_comparison"]


# =========================================================================
# 10. find_products_by_sentiment
# =========================================================================


class TestFindProductsBySentiment:
    """감성 태그로 제품 검색 테스트"""

    def test_find_by_tag(self, kg, sentiment_data):
        """특정 감성 태그로 검색"""
        kg.load_from_sentiment_data(sentiment_data)
        asins = kg.find_products_by_sentiment("Moisturizing")
        assert "B08XYZ001" in asins
        assert "B09ABC001" in asins

    def test_find_unique_tag(self, kg, sentiment_data):
        """고유한 감성 태그로 검색"""
        kg.load_from_sentiment_data(sentiment_data)
        asins = kg.find_products_by_sentiment("Nice scent")
        assert "B08XYZ001" in asins
        assert "B09ABC001" not in asins

    def test_brand_filter(self, kg, crawl_data_single, sentiment_data):
        """브랜드 필터 적용"""
        kg.load_from_crawl_data(crawl_data_single)
        kg.load_from_sentiment_data(sentiment_data)
        # B08XYZ001의 메타데이터에 brand 설정
        kg.set_entity_metadata("B08XYZ001", {"brand": "LANEIGE"})
        kg.set_entity_metadata("B09ABC001", {"brand": "COSRX"})

        asins = kg.find_products_by_sentiment("Moisturizing", brand_filter="LANEIGE")
        assert "B08XYZ001" in asins
        assert "B09ABC001" not in asins

    def test_brand_filter_case_insensitive(self, kg, sentiment_data):
        """브랜드 필터 대소문자 무시"""
        kg.load_from_sentiment_data(sentiment_data)
        kg.set_entity_metadata("B08XYZ001", {"brand": "LANEIGE"})

        asins = kg.find_products_by_sentiment("Moisturizing", brand_filter="laneige")
        assert "B08XYZ001" in asins

    def test_no_results(self, kg, sentiment_data):
        """결과 없는 검색"""
        kg.load_from_sentiment_data(sentiment_data)
        asins = kg.find_products_by_sentiment("NonexistentTag")
        assert asins == []


# =========================================================================
# 11. load_brand_ownership
# =========================================================================


class TestLoadBrandOwnership:
    """브랜드 소유권 데이터 로드 테스트"""

    def test_basic_loading(self, kg, brands_config_file):
        """기본 소유권 로드"""
        added = kg.load_brand_ownership(brands_config_file)
        # 3 brands × (OWNED_BY_GROUP + OWNS_BRAND + HAS_SEGMENT + ORIGINATES_FROM) = 12
        # COSRX has acquired=2024 → +1 ACQUIRED_IN
        # 3 brands → 3 pairs of siblings → 6 sibling relations
        assert added >= 12

    def test_corporate_group_metadata(self, kg, brands_config_file):
        """AMOREPACIFIC 메타데이터 확인"""
        kg.load_brand_ownership(brands_config_file)
        meta = kg.get_entity_metadata("AMOREPACIFIC")
        assert meta["type"] == "corporate_group"
        assert meta["country"] == "Korea"

    def test_ownership_relations(self, kg, brands_config_file):
        """Brand → CorporateGroup 관계"""
        kg.load_brand_ownership(brands_config_file)
        rels = kg.query(subject="LANEIGE", predicate=RelationType.OWNED_BY_GROUP)
        assert len(rels) == 1
        assert rels[0].object == "AMOREPACIFIC"

    def test_reverse_ownership(self, kg, brands_config_file):
        """CorporateGroup → Brand 역관계"""
        kg.load_brand_ownership(brands_config_file)
        rels = kg.query(subject="AMOREPACIFIC", predicate=RelationType.OWNS_BRAND)
        brands = {r.object for r in rels}
        assert "LANEIGE" in brands
        assert "Sulwhasoo" in brands
        assert "COSRX" in brands

    def test_segment_relations(self, kg, brands_config_file):
        """Brand → Segment 관계"""
        kg.load_brand_ownership(brands_config_file)
        rels = kg.query(subject="LANEIGE", predicate=RelationType.HAS_SEGMENT)
        assert len(rels) == 1
        assert rels[0].object == "Premium"

    def test_origin_relations(self, kg, brands_config_file):
        """Brand → Country 관계"""
        kg.load_brand_ownership(brands_config_file)
        rels = kg.query(subject="COSRX", predicate=RelationType.ORIGINATES_FROM)
        assert len(rels) == 1
        assert rels[0].object == "Korea"

    def test_acquired_brand(self, kg, brands_config_file):
        """인수 브랜드 관계 (COSRX → 2024)"""
        kg.load_brand_ownership(brands_config_file)
        rels = kg.query(subject="COSRX", predicate=RelationType.ACQUIRED_IN)
        assert len(rels) == 1
        assert rels[0].object == "2024"

    def test_non_acquired_brand(self, kg, brands_config_file):
        """비인수 브랜드는 ACQUIRED_IN 없음"""
        kg.load_brand_ownership(brands_config_file)
        rels = kg.query(subject="LANEIGE", predicate=RelationType.ACQUIRED_IN)
        assert len(rels) == 0

    def test_brand_metadata(self, kg, brands_config_file):
        """브랜드 메타데이터 확인"""
        kg.load_brand_ownership(brands_config_file)
        meta = kg.get_entity_metadata("LANEIGE")
        assert meta["segment"] == "Premium"
        assert meta["parent_group"] == "AMOREPACIFIC"
        assert meta["country"] == "Korea"
        assert "라네즈" in meta.get("aliases", [])

    def test_ownership_note(self, kg, brands_config_file):
        """brand_ownership 상세 정보 (COSRX)"""
        kg.load_brand_ownership(brands_config_file)
        meta = kg.get_entity_metadata("COSRX")
        assert "Korean brand" in meta.get("ownership_note", "")
        assert meta.get("country_of_origin") == "Korea"

    def test_sibling_brand_relations(self, kg, brands_config_file):
        """자매 브랜드 관계 (대칭)"""
        kg.load_brand_ownership(brands_config_file)
        rels = kg.query(subject="LANEIGE", predicate=RelationType.SIBLING_BRAND)
        siblings = {r.object for r in rels}
        assert "Sulwhasoo" in siblings
        assert "COSRX" in siblings

        # 역방향도 확인
        rev_rels = kg.query(subject="COSRX", predicate=RelationType.SIBLING_BRAND)
        rev_siblings = {r.object for r in rev_rels}
        assert "LANEIGE" in rev_siblings

    def test_file_not_found(self, kg):
        """존재하지 않는 파일"""
        added = kg.load_brand_ownership("/nonexistent/brands.json")
        assert added == 0

    def test_brand_without_name_skipped(self, kg, tmp_path):
        """이름 없는 브랜드는 건너뛰기"""
        data = {
            "amorepacific_brands": [{"segment": "Premium"}],
            "brand_ownership": {},
        }
        path = tmp_path / "brands_no_name.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        added = kg.load_brand_ownership(str(path))
        # AMOREPACIFIC corporate metadata만 설정되고 관계는 추가 안됨
        assert added == 0


# =========================================================================
# 12. get_brand_ownership
# =========================================================================


class TestGetBrandOwnership:
    """브랜드 소유권 조회 테스트"""

    def test_amorepacific_brand(self, kg, brands_config_file):
        """아모레퍼시픽 브랜드 소유권"""
        kg.load_brand_ownership(brands_config_file)
        result = kg.get_brand_ownership("LANEIGE")
        assert result["brand"] == "LANEIGE"
        assert result["parent_group"] == "AMOREPACIFIC"
        assert result["segment"] == "Premium"
        assert result["country_of_origin"] == "Korea"
        assert result["acquired"] is None
        assert "Sulwhasoo" in result["sibling_brands"]

    def test_acquired_brand_ownership(self, kg, brands_config_file):
        """인수 브랜드 소유권 (COSRX)"""
        kg.load_brand_ownership(brands_config_file)
        result = kg.get_brand_ownership("COSRX")
        assert result["parent_group"] == "AMOREPACIFIC"
        assert result["acquired"] == 2024
        assert result["note"] is not None

    def test_unknown_brand(self, kg):
        """알 수 없는 브랜드"""
        result = kg.get_brand_ownership("UNKNOWN")
        assert result["parent_group"] is None
        assert result["segment"] is None
        assert result["sibling_brands"] == []

    def test_country_from_metadata_fallback(self, kg):
        """origin 관계 없을 때 메타데이터에서 country 읽기"""
        kg.set_entity_metadata("TestBrand", {"country": "Japan"})
        result = kg.get_brand_ownership("TestBrand")
        assert result["country_of_origin"] == "Japan"

    def test_country_from_metadata_country_of_origin(self, kg):
        """메타데이터 country_of_origin 우선"""
        kg.set_entity_metadata("TestBrand", {"country_of_origin": "France", "country": "Korea"})
        result = kg.get_brand_ownership("TestBrand")
        assert result["country_of_origin"] == "France"


# =========================================================================
# 13. is_amorepacific_brand
# =========================================================================


class TestIsAmorepacificBrand:
    """아모레퍼시픽 브랜드 확인 테스트"""

    def test_true_for_owned_brand(self, kg, brands_config_file):
        """아모레퍼시픽 소속 브랜드"""
        kg.load_brand_ownership(brands_config_file)
        assert kg.is_amorepacific_brand("LANEIGE") is True
        assert kg.is_amorepacific_brand("COSRX") is True

    def test_false_for_other_brand(self, kg, brands_config_file):
        """아모레퍼시픽 미소속 브랜드"""
        kg.load_brand_ownership(brands_config_file)
        assert kg.is_amorepacific_brand("L'Oreal") is False

    def test_false_for_unknown(self, kg):
        """데이터 없는 브랜드"""
        assert kg.is_amorepacific_brand("UNKNOWN") is False


# =========================================================================
# 14. get_amorepacific_brands
# =========================================================================


class TestGetAmorepacificBrands:
    """아모레퍼시픽 브랜드 목록 조회 테스트"""

    def test_all_brands(self, kg, brands_config_file):
        """전체 브랜드 목록"""
        kg.load_brand_ownership(brands_config_file)
        brands = kg.get_amorepacific_brands()
        brand_names = {b["brand"] for b in brands}
        assert "LANEIGE" in brand_names
        assert "Sulwhasoo" in brand_names
        assert "COSRX" in brand_names

    def test_segment_filter(self, kg, brands_config_file):
        """세그먼트 필터"""
        kg.load_brand_ownership(brands_config_file)
        luxury = kg.get_amorepacific_brands(segment_filter="Luxury")
        assert len(luxury) == 1
        assert luxury[0]["brand"] == "Sulwhasoo"

    def test_segment_filter_no_match(self, kg, brands_config_file):
        """세그먼트 필터 — 매칭 없음"""
        kg.load_brand_ownership(brands_config_file)
        result = kg.get_amorepacific_brands(segment_filter="MassMarket")
        assert result == []

    def test_empty_kg(self, kg):
        """빈 KG에서 조회"""
        result = kg.get_amorepacific_brands()
        assert result == []

    def test_brand_info_includes_category(self, kg, brands_config_file):
        """브랜드 정보에 category 포함"""
        kg.load_brand_ownership(brands_config_file)
        brands = kg.get_amorepacific_brands()
        for b in brands:
            assert "segment" in b
            assert "category" in b


# =========================================================================
# 통합 시나리오
# =========================================================================


class TestIntegrationScenarios:
    """여러 메서드를 조합한 통합 테스트"""

    def test_full_pipeline(
        self, kg, crawl_data_single, metrics_data, sentiment_data, category_hierarchy_file
    ):
        """전체 데이터 로딩 → 조회 파이프라인"""
        # 1. 크롤링 데이터 로드
        c1 = kg.load_from_crawl_data(crawl_data_single)
        assert c1 > 0

        # 2. 메트릭 데이터 로드
        c2 = kg.load_from_metrics_data(metrics_data)
        assert c2 > 0

        # 3. 카테고리 계층 로드
        c3 = kg.load_category_hierarchy(category_hierarchy_file)
        assert c3 > 0

        # 4. 감성 데이터 로드
        c4 = kg.load_from_sentiment_data(sentiment_data)
        assert c4 > 0

        # 5. 제품 카테고리 컨텍스트 조회
        ctx = kg.get_product_category_context("B08XYZ001")
        assert len(ctx["categories"]) >= 1

        # 6. 제품 감성 조회
        sent = kg.get_product_sentiments("B08XYZ001")
        assert sent["ai_summary"] is not None

    def test_crawl_then_sentiment(self, kg, crawl_data_single, sentiment_data):
        """크롤링 후 감성 데이터 로드 → 브랜드 프로필"""
        kg.load_from_crawl_data(crawl_data_single)
        kg.load_from_sentiment_data(sentiment_data)

        profile = kg.get_brand_sentiment_profile("LANEIGE")
        assert profile["product_count"] >= 1
        assert len(profile["all_tags"]) >= 1

    def test_ownership_then_sentiment(
        self, kg, brands_config_file, sentiment_data, crawl_data_single
    ):
        """소유권 + 감성 → 아모레퍼시픽 브랜드 감성 분석"""
        kg.load_brand_ownership(brands_config_file)
        kg.load_from_crawl_data(crawl_data_single)
        kg.load_from_sentiment_data(sentiment_data)

        # LANEIGE는 아모레퍼시픽
        assert kg.is_amorepacific_brand("LANEIGE")

        # 감성 조회
        profile = kg.get_brand_sentiment_profile("LANEIGE")
        assert profile["product_count"] >= 1

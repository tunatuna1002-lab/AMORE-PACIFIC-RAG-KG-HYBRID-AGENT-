"""EntityLinker 단위 테스트"""

from unittest.mock import patch

import pytest

import src.rag.entity_linker as entity_linker_module
from src.rag.entity_linker import EntityLinker, LinkedEntity, get_entity_linker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def linker():
    """Basic EntityLinker with rule-based NER (no spaCy)."""
    return EntityLinker(use_spacy=False)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton before and after each test."""
    entity_linker_module._linker_instance = None
    yield
    entity_linker_module._linker_instance = None


@pytest.fixture(autouse=True)
def reset_config_cache():
    """Clear the class-level config cache before each test."""
    EntityLinker._config_cache = None
    EntityLinker._config_loaded_at = None
    yield
    EntityLinker._config_cache = None
    EntityLinker._config_loaded_at = None


# ===========================================================================
# TestLinkedEntity
# ===========================================================================


class TestLinkedEntity:
    def test_to_dict(self):
        entity = LinkedEntity(
            text="LANEIGE",
            entity_type="brand",
            concept_uri="http://example.com/Brand/LANEIGE",
            concept_label="LANEIGE",
            confidence=1.0,
            context={"matched_key": "laneige"},
        )
        result = entity.to_dict()
        assert result["text"] == "LANEIGE"
        assert result["entity_type"] == "brand"
        assert result["concept_uri"] == "http://example.com/Brand/LANEIGE"
        assert result["concept_label"] == "LANEIGE"
        assert result["confidence"] == 1.0
        assert result["context"] == {"matched_key": "laneige"}

    def test_default_context(self):
        entity = LinkedEntity(
            text="Peptide",
            entity_type="ingredient",
            concept_uri="http://example.com/Ingredient/Peptide",
            concept_label="펩타이드",
            confidence=1.0,
        )
        assert entity.context == {}


# ===========================================================================
# TestEntityLinkerLink
# ===========================================================================


class TestEntityLinkerLink:
    def test_link_brand_exact(self, linker):
        entities = linker.link("LANEIGE")
        brands = [e for e in entities if e.entity_type == "brand"]
        assert len(brands) >= 1
        laneige = next((e for e in brands if "LANEIGE" in e.concept_label), None)
        assert laneige is not None
        assert laneige.confidence == 1.0

    def test_link_brand_korean(self, linker):
        entities = linker.link("라네즈 제품 분석")
        brands = [e for e in entities if e.entity_type == "brand"]
        assert len(brands) >= 1
        assert any("LANEIGE" in e.concept_label for e in brands)

    def test_link_category(self, linker):
        entities = linker.link("Lip Care 제품 현황")
        categories = [e for e in entities if e.entity_type == "category"]
        assert len(categories) >= 1
        assert any("Lip Care" in e.concept_label for e in categories)
        assert all(e.confidence >= 0.5 for e in categories)

    def test_link_metric(self, linker):
        entities = linker.link("SoS 지표 분석")
        metrics = [e for e in entities if e.entity_type == "metric"]
        assert len(metrics) >= 1
        assert any("Share of Shelf" in e.concept_label for e in metrics)

    def test_link_ingredient(self, linker):
        entities = linker.link("Peptide 성분 트렌드")
        ingredients = [e for e in entities if e.entity_type == "ingredient"]
        assert len(ingredients) >= 1
        assert any("펩타이드" in e.concept_label or "Peptide" in e.concept_uri for e in ingredients)

    def test_link_trend(self, linker):
        entities = linker.link("tiktok 바이럴 현황")
        trends = [e for e in entities if e.entity_type == "trend"]
        assert len(trends) >= 1
        assert any("TikTok" in e.concept_uri for e in trends)

    def test_link_product_asin(self, linker):
        entities = linker.link("B0BSHRYY1S 제품 정보")
        products = [e for e in entities if e.entity_type == "product"]
        assert len(products) >= 1
        assert products[0].text == "B0BSHRYY1S"
        assert products[0].confidence == 1.0

    def test_link_multiple_entities(self, linker):
        entities = linker.link("LANEIGE Lip Care SoS")
        types_found = {e.entity_type for e in entities}
        assert "brand" in types_found
        assert "category" in types_found
        assert "metric" in types_found

    def test_link_entity_type_filter(self, linker):
        entities = linker.link("LANEIGE Lip Care SoS", entity_types=["brand"])
        assert all(e.entity_type == "brand" for e in entities)
        assert len(entities) >= 1

    def test_link_min_confidence_filter(self, linker):
        entities = linker.link("LANEIGE Lip Care", min_confidence=0.9)
        assert all(e.confidence >= 0.9 for e in entities)

    def test_link_empty_query(self, linker):
        entities = linker.link("")
        assert entities == []


# ===========================================================================
# TestEntityLinkerExtractEntities
# ===========================================================================


class TestEntityLinkerExtractEntities:
    def test_extract_brands(self, linker):
        result = linker.extract_entities("LANEIGE 제품 분석")
        assert isinstance(result["brands"], list)
        assert len(result["brands"]) >= 1

    def test_extract_brands_korean_alias(self, linker):
        result = linker.extract_entities("라네즈 비교 분석")
        brands = result["brands"]
        assert len(brands) >= 1
        assert any("laneige" in b.lower() for b in brands)

    def test_extract_categories(self, linker):
        result = linker.extract_entities("lip care 현황")
        assert "lip_care" in result["categories"]

    def test_extract_indicators(self, linker):
        result = linker.extract_entities("sos 분석해줘")
        assert "sos" in result["indicators"]

    def test_extract_time_range(self, linker):
        result = linker.extract_entities("오늘 기준 데이터")
        assert "today" in result["time_range"]

    def test_extract_sentiments(self, linker):
        result = linker.extract_entities("보습 효과 리뷰")
        assert len(result["sentiments"]) >= 1
        assert len(result["sentiment_clusters"]) >= 1

    def test_extract_asin(self, linker):
        result = linker.extract_entities("B0BSHRYY1S 제품 정보")
        assert "B0BSHRYY1S" in result["products"]

    def test_extract_multiple(self, linker):
        result = linker.extract_entities("LANEIGE lip care sos 오늘 보습")
        assert len(result["brands"]) >= 1
        assert len(result["categories"]) >= 1
        assert len(result["indicators"]) >= 1
        assert len(result["time_range"]) >= 1
        assert len(result["sentiments"]) >= 1

    def test_extract_empty(self, linker):
        result = linker.extract_entities("")
        assert result["brands"] == []
        assert result["categories"] == []
        assert result["indicators"] == []
        assert result["time_range"] == []
        assert result["products"] == []
        assert result["sentiments"] == []
        assert result["sentiment_clusters"] == []


# ===========================================================================
# TestEntityLinkerMergedMaps
# ===========================================================================


class TestEntityLinkerMergedMaps:
    def test_get_merged_brands(self, linker):
        """Class-level KNOWN_BRANDS are reflected in the merged map."""
        merged = linker._get_merged_brands()
        assert isinstance(merged, dict)
        # laneige should always be present
        assert "laneige" in merged

    def test_get_merged_brands_config(self, linker):
        """Config brands are merged on top of class-level brands."""
        fake_config = {"known_brands": [{"name": "TestBrand", "aliases": ["테스트브랜드"]}]}
        with patch.object(EntityLinker, "_load_entity_config", return_value=fake_config):
            merged = linker._get_merged_brands()
        assert "testbrand" in merged
        assert "테스트브랜드" in merged

    def test_get_merged_categories(self, linker):
        merged = linker._get_merged_categories()
        assert isinstance(merged, dict)
        assert "lip care" in merged
        assert merged["lip care"] == "lip_care"

    def test_get_merged_categories_config(self, linker):
        fake_config = {"category_map": {"new_category": "new_cat_id"}}
        with patch.object(EntityLinker, "_load_entity_config", return_value=fake_config):
            merged = linker._get_merged_categories()
        assert "new_category" in merged
        assert merged["new_category"] == "new_cat_id"

    def test_get_merged_indicators(self, linker):
        merged = linker._get_merged_indicators()
        assert isinstance(merged, dict)
        assert "sos" in merged
        assert merged["sos"] == "sos"

    def test_get_merged_indicators_config(self, linker):
        fake_config = {"indicator_map": {"custom_metric": "custom_id"}}
        with patch.object(EntityLinker, "_load_entity_config", return_value=fake_config):
            merged = linker._get_merged_indicators()
        assert "custom_metric" in merged

    def test_get_merged_time_ranges(self, linker):
        merged = linker._get_merged_time_ranges()
        assert isinstance(merged, dict)
        assert "오늘" in merged
        assert merged["오늘"] == "today"
        assert "today" in merged

    def test_get_merged_sentiments(self, linker):
        merged = linker._get_merged_sentiments()
        assert isinstance(merged, dict)
        assert "보습" in merged
        assert merged["보습"] == "Hydration"


# ===========================================================================
# TestOntologyFilters
# ===========================================================================


class TestOntologyFilters:
    def _make_entity(self, entity_type, concept_label, text=None, concept_uri=None, context=None):
        return LinkedEntity(
            text=text or concept_label,
            entity_type=entity_type,
            concept_uri=concept_uri or f"http://example.com/{entity_type}/{concept_label}",
            concept_label=concept_label,
            confidence=1.0,
            context=context or {},
        )

    def test_get_ontology_filters_single(self, linker):
        entities = [self._make_entity("brand", "LANEIGE")]
        result = linker.get_ontology_filters(entities)
        assert result == {"brand": "LANEIGE"}

    def test_get_ontology_filters_multiple(self, linker):
        entities = [
            self._make_entity("brand", "LANEIGE"),
            self._make_entity(
                "category",
                "Lip Care",
                context={"matched_key": "lip_care"},
            ),
        ]
        result = linker.get_ontology_filters(entities)
        assert "$or" in result
        assert len(result["$or"]) == 2

    def test_get_ontology_filters_empty(self, linker):
        result = linker.get_ontology_filters([])
        assert result == {}


# ===========================================================================
# TestGetEntityLinker
# ===========================================================================


class TestGetEntityLinker:
    def test_singleton(self):
        a = get_entity_linker(use_spacy=False)
        b = get_entity_linker(use_spacy=False)
        assert a is b

    def test_reset_singleton(self):
        first = get_entity_linker(use_spacy=False)
        entity_linker_module._linker_instance = None
        second = get_entity_linker(use_spacy=False)
        assert first is not second


# ===========================================================================
# TestEntityLinkerStats
# ===========================================================================


class TestEntityLinkerStats:
    def test_stats_tracking(self, linker):
        initial = linker.get_stats()
        assert initial["total_links"] == 0

        linker.link("LANEIGE Lip Care SoS")

        after = linker.get_stats()
        assert after["total_links"] > 0

    def test_stats_exact_match_increments(self, linker):
        linker.link("LANEIGE")
        stats = linker.get_stats()
        # LANEIGE exact-matches → exact_matches should be >= 1
        assert stats["exact_matches"] >= 1

    def test_stats_returns_copy(self, linker):
        stats = linker.get_stats()
        stats["total_links"] = 9999
        assert linker.get_stats()["total_links"] != 9999


# ===========================================================================
# 브랜드 오탐 방지 + 제품명 역링크 (2026-08-30 사이클 4)
# ===========================================================================


class _StubRelation:
    def __init__(self, subject, predicate, obj, properties):
        self.subject = subject
        self.predicate = predicate
        self.object = obj
        self.properties = properties


class _StubKG:
    """hasProduct 트리플만 돌려주는 최소 KG 스텁."""

    def __init__(self, relations):
        self._relations = relations

    def query(self, *args, **kwargs):
        return self._relations


class TestBrandMatchingPrecision:
    def test_shelf_does_not_match_elf_brand(self, linker):
        """'share of shelf'의 shelf가 브랜드 e.l.f.로 오탐되지 않아야 한다."""
        result = linker.extract_entities("SoS(Share of Shelf)란 무엇인가요?")
        assert result["brands"] == []

    def test_standalone_elf_still_matches(self, linker):
        result = linker.extract_entities("elf 브랜드 순위는?")
        assert "e.l.f." in result["brands"]

    def test_korean_particle_does_not_block_match(self, linker):
        """한글은 조사가 붙어도 매칭돼야 한다 (경계 조건 미적용)."""
        result = linker.extract_entities("라네즈의 점유율은?")
        assert any("laneige" in b.lower() for b in result["brands"])


class TestProductNameBackLink:
    def test_product_name_links_brand_and_product(self, linker):
        """질의가 제품명만 언급해도 KG 타이틀로 브랜드를 역링크한다."""
        kg = _StubKG(
            [
                _StubRelation(
                    "laneige",
                    "hasProduct",
                    "B07XXPHQZK",
                    {
                        "title": "LANEIGE Lip Sleeping Mask: Korean Overnight Treatment",
                        "rank": 8,
                        "category": "lip_care",
                    },
                )
            ]
        )

        result = linker.extract_entities("Lip Sleeping Mask 순위 변화는?", knowledge_graph=kg)

        assert "laneige" in result["brands"]
        assert "lip_sleeping_mask" in result["products"]

    def test_unrelated_query_gets_no_product(self, linker):
        kg = _StubKG(
            [
                _StubRelation(
                    "laneige",
                    "hasProduct",
                    "B07XXPHQZK",
                    {"title": "LANEIGE Lip Sleeping Mask", "rank": 8, "category": "lip_care"},
                )
            ]
        )

        result = linker.extract_entities("HHI가 무엇인가요?", knowledge_graph=kg)

        assert result["products"] == []
        assert result["brands"] == []


class TestConceptMatchingPrecision:
    """개념 매칭도 라틴 키워드에 단어 경계를 요구한다 (2026-08-30 사이클 7).

    'rating'이 "ope|rating| profit"에 걸려 review_rating을 오탐하고 있었다 —
    사이클 4에서 브랜드에만 적용했던 경계 규칙의 사각지대.
    """

    def test_operating_profit_does_not_match_rating_concept(self, linker):
        assert "review_rating" not in linker.extract_concepts("AMOREPACIFIC 1Q25 operating profit")

    def test_standalone_rating_still_matches(self, linker):
        assert "review_rating" in linker.extract_concepts("LANEIGE rating trend")

    def test_korean_concept_keywords_still_match(self, linker):
        assert "sos" in linker.extract_concepts("SoS 점유율 알려줘")


class TestParentCompanyEntity:
    """모기업 엔티티 링킹 (사이클 7).

    KG에 `LANEIGE -ownedByGroup-> AMOREPACIFIC`으로 존재하고 IR 문항의 주어인데
    엔티티 사전에 없어 링킹되지 않았다.
    """

    def test_korean_parent_company_links(self, linker):
        assert "amorepacific" in linker.extract_entities("아모레퍼시픽 1분기 매출은?")["brands"]

    def test_english_parent_company_links(self, linker):
        assert "amorepacific" in linker.extract_entities("AMOREPACIFIC 1Q25 revenue")["brands"]

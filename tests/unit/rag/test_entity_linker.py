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

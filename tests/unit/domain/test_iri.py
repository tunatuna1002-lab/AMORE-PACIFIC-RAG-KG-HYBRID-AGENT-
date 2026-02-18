"""
Unit tests for IRI scheme (D-1)

Tests the IRI utility class in src/domain/entities/relations.py.
"""

import pytest

from src.domain.entities.relations import (
    AMORE_NS,
    AMORE_PREFIX,
    IRI,
    IRI_TYPE_MAP,
)


class TestIRIToIRI:
    """Test IRI.to_iri() conversion."""

    def test_to_iri_brand(self):
        result = IRI.to_iri("brand", "LANEIGE")
        assert result == "amore:brand/LANEIGE"

    def test_to_iri_product(self):
        result = IRI.to_iri("product", "B08XYZ1234")
        assert result == "amore:product/B08XYZ1234"

    def test_to_iri_category(self):
        result = IRI.to_iri("category", "lip_care")
        assert result == "amore:category/lip_care"

    def test_to_iri_metric(self):
        result = IRI.to_iri("metric", "sos_lip_care")
        assert result == "amore:metric/sos_lip_care"

    def test_to_iri_trend(self):
        result = IRI.to_iri("trend", "hydration")
        assert result == "amore:trend/hydration"

    def test_to_iri_sentiment(self):
        result = IRI.to_iri("sentiment", "moisturizing")
        assert result == "amore:sentiment/moisturizing"

    def test_to_iri_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown entity type"):
            IRI.to_iri("unknown_type", "test")


class TestIRIFromIRI:
    """Test IRI.from_iri() parsing."""

    def test_from_iri_brand(self):
        entity_type, entity_id = IRI.from_iri("amore:brand/LANEIGE")
        assert entity_type == "brand"
        assert entity_id == "LANEIGE"

    def test_from_iri_product(self):
        entity_type, entity_id = IRI.from_iri("amore:product/B08XYZ1234")
        assert entity_type == "product"
        assert entity_id == "B08XYZ1234"

    def test_from_iri_full_uri(self):
        entity_type, entity_id = IRI.from_iri("http://amore.ontology/brand/COSRX")
        assert entity_type == "brand"
        assert entity_id == "COSRX"

    def test_from_iri_category(self):
        entity_type, entity_id = IRI.from_iri("amore:category/lip_care")
        assert entity_type == "category"
        assert entity_id == "lip_care"

    def test_from_iri_invalid_raises(self):
        with pytest.raises(ValueError, match="Not a valid AMORE IRI"):
            IRI.from_iri("http://other.ontology/brand/X")

    def test_from_iri_unknown_path_raises(self):
        with pytest.raises(ValueError, match="Cannot parse entity type"):
            IRI.from_iri("amore:unknown_prefix/foo")


class TestIRIIsIRI:
    """Test IRI.is_iri() check."""

    def test_is_iri_true_prefixed(self):
        assert IRI.is_iri("amore:brand/LANEIGE") is True

    def test_is_iri_true_full(self):
        assert IRI.is_iri("http://amore.ontology/brand/LANEIGE") is True

    def test_is_iri_false_plain(self):
        assert IRI.is_iri("LANEIGE") is False

    def test_is_iri_false_other_prefix(self):
        assert IRI.is_iri("http://other.com/brand/X") is False

    def test_is_iri_false_empty(self):
        assert IRI.is_iri("") is False


class TestIRIConversions:
    """Test full/prefixed IRI conversions."""

    def test_to_full_iri(self):
        result = IRI.to_full_iri("amore:brand/LANEIGE")
        assert result == "http://amore.ontology/brand/LANEIGE"

    def test_to_full_iri_already_full(self):
        full = "http://amore.ontology/brand/LANEIGE"
        assert IRI.to_full_iri(full) == full

    def test_to_full_iri_invalid_raises(self):
        with pytest.raises(ValueError, match="Not a prefixed AMORE IRI"):
            IRI.to_full_iri("LANEIGE")

    def test_to_prefixed(self):
        result = IRI.to_prefixed("http://amore.ontology/brand/LANEIGE")
        assert result == "amore:brand/LANEIGE"

    def test_to_prefixed_already_prefixed(self):
        prefixed = "amore:brand/LANEIGE"
        assert IRI.to_prefixed(prefixed) == prefixed

    def test_to_prefixed_invalid_raises(self):
        with pytest.raises(ValueError, match="Not a full AMORE IRI"):
            IRI.to_prefixed("LANEIGE")


class TestIRIExtractID:
    """Test IRI.extract_id()."""

    def test_extract_id_from_prefixed(self):
        assert IRI.extract_id("amore:brand/LANEIGE") == "LANEIGE"

    def test_extract_id_from_full(self):
        assert IRI.extract_id("http://amore.ontology/product/B08XYZ1234") == "B08XYZ1234"

    def test_extract_id_category(self):
        assert IRI.extract_id("amore:category/lip_care") == "lip_care"


class TestIRIRoundtrip:
    """Test roundtrip conversion."""

    def test_roundtrip_conversion(self):
        """to_iri -> from_iri -> to_iri should be identity."""
        original = IRI.to_iri("brand", "LANEIGE")
        entity_type, entity_id = IRI.from_iri(original)
        reconstructed = IRI.to_iri(entity_type, entity_id)
        assert reconstructed == original

    def test_roundtrip_full_prefixed(self):
        """to_full_iri -> to_prefixed -> to_full_iri should be identity."""
        prefixed = "amore:product/B08XYZ1234"
        full = IRI.to_full_iri(prefixed)
        back = IRI.to_prefixed(full)
        assert back == prefixed

    def test_all_types_roundtrip(self):
        """All entity types should roundtrip correctly."""
        for entity_type in IRI_TYPE_MAP:
            iri = IRI.to_iri(entity_type, "test_id")
            parsed_type, parsed_id = IRI.from_iri(iri)
            assert parsed_type == entity_type
            assert parsed_id == "test_id"


class TestIRIConstants:
    """Test IRI constants."""

    def test_amore_ns(self):
        assert AMORE_NS == "http://amore.ontology/"

    def test_amore_prefix(self):
        assert AMORE_PREFIX == "amore:"

    def test_iri_type_map_has_core_types(self):
        assert "brand" in IRI_TYPE_MAP
        assert "product" in IRI_TYPE_MAP
        assert "category" in IRI_TYPE_MAP

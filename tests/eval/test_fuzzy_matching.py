"""Tests for fuzzy matching utilities."""

from eval.metrics.base import (
    BRAND_ALIASES,
    CATEGORY_ALIASES,
    MetricCalculator,
    fuzzy_match,
    fuzzy_set_match,
    normalize_entity,
)


class TestNormalizeEntity:
    """Tests for normalize_entity function."""

    def test_normalize_korean_brand(self):
        """Test Korean brand alias normalization."""
        assert normalize_entity("라네즈", BRAND_ALIASES) == "laneige"
        assert normalize_entity("코스알엑스", BRAND_ALIASES) == "cosrx"

    def test_normalize_english_brand(self):
        """Test English brand normalization."""
        assert normalize_entity("LANEIGE", BRAND_ALIASES) == "laneige"
        assert normalize_entity("laneige", BRAND_ALIASES) == "laneige"

    def test_normalize_unknown_entity(self):
        """Test unknown entity returns lowercase."""
        assert normalize_entity("UnknownBrand", BRAND_ALIASES) == "unknownbrand"

    def test_normalize_category(self):
        """Test category normalization."""
        assert normalize_entity("립케어", CATEGORY_ALIASES) == "lip_care"
        assert normalize_entity("Lip Care", CATEGORY_ALIASES) == "lip_care"

    def test_normalize_without_alias_map(self):
        """Test normalization without explicit alias map."""
        # Should try all maps
        assert normalize_entity("라네즈") == "laneige"
        assert normalize_entity("립케어") == "lip_care"
        assert normalize_entity("sos") == "sos"


class TestFuzzyMatch:
    """Tests for fuzzy_match function."""

    def test_exact_match(self):
        """Test exact match returns perfect score."""
        match, score = fuzzy_match("laneige", ["laneige", "cosrx"])
        assert match == "laneige"
        assert score == 1.0

    def test_near_match(self):
        """Test near match with typo."""
        match, score = fuzzy_match("laneig", ["laneige", "cosrx"])
        assert match == "laneige"
        assert score > 0.8

    def test_no_match_below_threshold(self):
        """Test no match when similarity below threshold."""
        match, score = fuzzy_match("xyz", ["laneige", "cosrx"], threshold=0.8)
        assert match is None
        assert score == 0.0

    def test_case_insensitive(self):
        """Test matching is case insensitive."""
        match, score = fuzzy_match("LANEIGE", ["laneige", "cosrx"])
        assert match == "laneige"
        assert score == 1.0


class TestFuzzySetMatch:
    """Tests for fuzzy_set_match function."""

    def test_exact_set_match(self):
        """Test exact set matching."""
        pred = {"laneige", "cosrx"}
        gold = {"laneige", "cosrx"}

        matched_pred, matched_gold, match_map = fuzzy_set_match(pred, gold)

        assert matched_pred == pred
        assert matched_gold == gold
        assert len(match_map) == 2

    def test_korean_alias_matching(self):
        """Test Korean to English alias matching."""
        pred = {"라네즈", "cosrx"}
        gold = {"laneige", "cosrx"}

        matched_pred, matched_gold, match_map = fuzzy_set_match(pred, gold, alias_map=BRAND_ALIASES)

        assert matched_pred == pred
        assert matched_gold == gold
        assert match_map["라네즈"] == "laneige"

    def test_partial_match(self):
        """Test partial set matching."""
        pred = {"laneige", "innisfree"}
        gold = {"laneige", "cosrx"}

        matched_pred, matched_gold, match_map = fuzzy_set_match(pred, gold)

        assert "laneige" in matched_pred
        assert "innisfree" not in matched_pred
        assert len(match_map) == 1

    def test_empty_sets(self):
        """Test empty set matching."""
        matched_pred, matched_gold, match_map = fuzzy_set_match(set(), set())
        assert matched_pred == set()
        assert matched_gold == set()


class TestSetF1Fuzzy:
    """Tests for set_f1_fuzzy method."""

    def test_perfect_match_with_aliases(self):
        """Test perfect match using Korean aliases."""
        pred = {"라네즈", "코스알엑스"}
        gold = {"laneige", "cosrx"}

        f1 = MetricCalculator.set_f1_fuzzy(pred, gold, alias_map=BRAND_ALIASES)
        assert f1 == 1.0

    def test_exact_f1_gives_zero_for_aliases(self):
        """Test that exact F1 gives 0 for Korean aliases (no fuzzy)."""
        pred = {"라네즈", "코스알엑스"}
        gold = {"laneige", "cosrx"}

        f1 = MetricCalculator.set_f1(pred, gold)
        assert f1 == 0.0  # Exact match doesn't understand aliases

    def test_partial_match_f1(self):
        """Test partial match F1 score."""
        pred = {"라네즈"}  # Only one entity
        gold = {"laneige", "cosrx"}

        f1 = MetricCalculator.set_f1_fuzzy(pred, gold, alias_map=BRAND_ALIASES)
        # Precision: 1/1 = 1.0, Recall: 1/2 = 0.5
        # F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 0.667
        assert 0.66 < f1 < 0.67

    def test_empty_both_returns_one(self):
        """Test both empty returns 1.0."""
        f1 = MetricCalculator.set_f1_fuzzy(set(), set())
        assert f1 == 1.0

    def test_empty_pred_returns_zero(self):
        """Test empty predicted returns 0.0."""
        f1 = MetricCalculator.set_f1_fuzzy(set(), {"laneige"})
        assert f1 == 0.0

    def test_mixed_language_entities(self):
        """Test mixed Korean and English entities."""
        pred = {"라네즈", "cosrx", "티르티르"}
        gold = {"laneige", "cosrx", "tirtir"}

        f1 = MetricCalculator.set_f1_fuzzy(pred, gold, alias_map=BRAND_ALIASES)
        assert f1 == 1.0


class TestL1MetricsWithFuzzy:
    """Integration tests for L1 metrics with fuzzy matching."""

    def test_entity_link_f1_fuzzy(self):
        """Test entity linking F1 with fuzzy matching."""
        from eval.metrics.l1_query import entity_link_f1
        from eval.schemas import EntityLinkingTrace, GoldEvidence

        trace = EntityLinkingTrace(
            extracted_brands=["라네즈"],  # Korean
            extracted_categories=[],
            extracted_indicators=[],
            extracted_products=[],
        )
        gold = GoldEvidence(kg_entities=["laneige"])  # English

        # Without fuzzy - should be 0
        f1_exact = entity_link_f1(trace, gold, use_fuzzy=False)
        assert f1_exact == 0.0

        # With fuzzy - should be 1.0
        f1_fuzzy = entity_link_f1(trace, gold, use_fuzzy=True)
        assert f1_fuzzy == 1.0

    def test_concept_map_f1_fuzzy(self):
        """Test concept mapping F1 with fuzzy matching."""
        from eval.metrics.l1_query import concept_map_f1
        from eval.schemas import EntityLinkingTrace, GoldEvidence

        trace = EntityLinkingTrace(
            extracted_brands=[],
            extracted_categories=["립케어"],  # Korean
            extracted_indicators=[],
            extracted_products=[],
        )
        gold = GoldEvidence(concepts=["lip_care"])  # Normalized

        # Without fuzzy - should be 0
        f1_exact = concept_map_f1(trace, gold, use_fuzzy=False)
        assert f1_exact == 0.0

        # With fuzzy - should be 1.0
        f1_fuzzy = concept_map_f1(trace, gold, use_fuzzy=True)
        assert f1_fuzzy == 1.0

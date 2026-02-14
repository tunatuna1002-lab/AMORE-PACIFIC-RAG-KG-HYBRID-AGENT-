"""Tests for domain-specific metrics."""

import pytest

from eval.metrics.base import BRAND_ALIASES, CATEGORY_ALIASES
from eval.metrics.domain_specific import (
    BrandEntityRecall,
    CategoryCorrectness,
    KPINumericalAccuracy,
)


class TestKPINumericalAccuracy:
    """Tests for KPI numerical accuracy calculator."""

    @pytest.fixture
    def calculator(self):
        """Create KPI accuracy calculator with 5% tolerance."""
        return KPINumericalAccuracy(tolerance=0.05)

    def test_exact_match(self, calculator):
        """Test exact numeric match."""
        predicted = "SoS is 23.5%, HHI is 1250"
        gold = "SoS: 23.5%, HHI: 1250"

        accuracy = calculator.compute(predicted, gold)
        assert accuracy == 1.0

    def test_within_tolerance(self, calculator):
        """Test numbers within tolerance."""
        predicted = "SoS is 23.5%"
        gold = "SoS: 24%"

        accuracy = calculator.compute(predicted, gold)
        assert accuracy == 1.0  # 23.5 vs 24 within 5%

    def test_outside_tolerance(self, calculator):
        """Test numbers outside tolerance."""
        predicted = "SoS is 20%"
        gold = "SoS: 24%"

        accuracy = calculator.compute(predicted, gold)
        assert accuracy == 0.0  # 20 vs 24 is 16.7% difference

    def test_multiple_kpis(self, calculator):
        """Test multiple KPI values."""
        predicted = "SoS: 24%, HHI: 1250, CPI: 98"
        gold = "SoS: 24.5%, HHI: 1250, CPI: 100"

        accuracy = calculator.compute(predicted, gold)
        # 24 vs 24.5 (OK), 1250 vs 1250 (OK), 98 vs 100 (OK: 2% diff)
        assert accuracy == 1.0

    def test_partial_match(self, calculator):
        """Test partial KPI match."""
        predicted = "SoS: 24%, HHI: 1250"
        gold = "SoS: 24%, HHI: 1250, CPI: 100"

        accuracy = calculator.compute(predicted, gold)
        # 2 of 3 gold numbers matched
        assert abs(accuracy - 0.667) < 0.01

    def test_no_gold_numbers(self, calculator):
        """Test when gold has no numbers."""
        predicted = "SoS is high"
        gold = "Market is competitive"

        accuracy = calculator.compute(predicted, gold)
        assert accuracy == 1.0  # No gold numbers = trivially correct

    def test_no_predicted_numbers(self, calculator):
        """Test when predicted has no numbers but gold does."""
        predicted = "SoS is high"
        gold = "SoS: 24%"

        accuracy = calculator.compute(predicted, gold)
        assert accuracy == 0.0  # Missing prediction

    def test_extract_from_kpi_patterns(self, calculator):
        """Test extraction from KPI-specific patterns."""
        numbers = calculator._extract_numbers("Share of Shelf: 23.5%, 허핀달: 1250")
        assert 23.5 in numbers
        assert 1250 in numbers

    def test_extract_from_general_numbers(self, calculator):
        """Test extraction from general numeric patterns."""
        numbers = calculator._extract_numbers("The value is 42.7 and 100")
        assert 42.7 in numbers
        assert 100 in numbers

    def test_negative_numbers(self, calculator):
        """Test extraction of negative numbers (growth rates)."""
        numbers = calculator._extract_numbers("Growth: -5.2%")
        assert -5.2 in numbers

    def test_rank_extraction(self, calculator):
        """Test extraction of rank numbers."""
        numbers = calculator._extract_numbers("Rank: #3, Position 5")
        assert 3 in numbers or 5 in numbers  # At least one rank extracted

    def test_is_close_exact(self, calculator):
        """Test _is_close with exact match."""
        assert calculator._is_close(100.0, 100.0) is True

    def test_is_close_within_tolerance(self, calculator):
        """Test _is_close within tolerance."""
        assert calculator._is_close(24.0, 24.5) is True  # ~2% diff
        assert calculator._is_close(99.0, 100.0) is True  # 1% diff

    def test_is_close_outside_tolerance(self, calculator):
        """Test _is_close outside tolerance."""
        assert calculator._is_close(20.0, 24.0) is False  # 20% diff
        assert calculator._is_close(90.0, 100.0) is False  # 10% diff

    def test_is_close_small_numbers(self, calculator):
        """Test _is_close with very small numbers (absolute tolerance)."""
        # For numbers < 0.001, uses absolute tolerance of 0.001
        assert calculator._is_close(0.0001, 0.0005) is True  # Both < 0.001, diff < 0.001
        assert calculator._is_close(0.00001, 0.00002) is True  # Both tiny, diff < 0.001


class TestBrandEntityRecall:
    """Tests for brand entity recall calculator."""

    @pytest.fixture
    def calculator(self):
        """Create brand recall calculator."""
        return BrandEntityRecall(alias_map=BRAND_ALIASES)

    def test_exact_brand_match(self, calculator):
        """Test exact brand name match."""
        predicted = "LANEIGE and COSRX are popular"
        gold = ["laneige", "cosrx"]

        recall = calculator.compute(predicted, gold)
        assert recall == 1.0

    def test_korean_alias_match(self, calculator):
        """Test Korean brand alias matching."""
        predicted = "라네즈 is a Korean brand"
        gold = ["laneige"]

        recall = calculator.compute(predicted, gold)
        assert recall == 1.0  # 라네즈 → laneige via alias

    def test_mixed_language_brands(self, calculator):
        """Test mixed Korean/English brands."""
        predicted = "Popular brands: 라네즈, COSRX, 티르티르"
        gold = ["laneige", "cosrx", "tirtir"]

        recall = calculator.compute(predicted, gold)
        assert recall == 1.0

    def test_partial_recall(self, calculator):
        """Test partial brand recall."""
        predicted = "LANEIGE is mentioned"
        gold = ["laneige", "cosrx", "tirtir"]

        recall = calculator.compute(predicted, gold)
        assert abs(recall - 0.333) < 0.01  # 1 of 3 found

    def test_no_brands_found(self, calculator):
        """Test when no brands found."""
        predicted = "Some random text"
        gold = ["laneige", "cosrx"]

        recall = calculator.compute(predicted, gold)
        assert recall == 0.0

    def test_empty_gold(self, calculator):
        """Test with empty gold brands."""
        predicted = "LANEIGE is mentioned"
        gold = []

        recall = calculator.compute(predicted, gold)
        assert recall == 1.0  # No gold = trivially correct

    def test_case_insensitive(self, calculator):
        """Test case insensitive matching."""
        predicted = "laneige and LANEIGE and LaNeiGe"
        gold = ["laneige"]

        recall = calculator.compute(predicted, gold)
        assert recall == 1.0

    def test_word_boundary_matching(self, calculator):
        """Test word boundary matching (avoid partial matches)."""
        predicted = "laneigee is not a real brand"
        gold = ["laneige"]

        recall = calculator.compute(predicted, gold)
        # Should NOT match "laneigee" as "laneige"
        assert recall == 0.0

    def test_extract_brands(self, calculator):
        """Test _extract_brands method."""
        text = "LANEIGE, 코스알엑스, and Rare Beauty"
        brands = calculator._extract_brands(text)

        assert "laneige" in brands
        assert "cosrx" in brands  # 코스알엑스 → cosrx
        assert "rare_beauty" in brands

    def test_normalize_brand(self, calculator):
        """Test _normalize_brand method."""
        assert calculator._normalize_brand("라네즈") == "laneige"
        assert calculator._normalize_brand("LANEIGE") == "laneige"
        assert calculator._normalize_brand("Unknown") == "unknown"


class TestCategoryCorrectness:
    """Tests for category correctness calculator."""

    @pytest.fixture
    def calculator(self):
        """Create category correctness calculator."""
        return CategoryCorrectness(alias_map=CATEGORY_ALIASES)

    def test_correct_category(self, calculator):
        """Test correct category mention."""
        predicted = "This is a Lip Care product"
        gold = "lip_care"

        score = calculator.compute(predicted, gold)
        assert score == 1.0

    def test_correct_category_korean(self, calculator):
        """Test correct category with Korean alias."""
        predicted = "이것은 립케어 제품입니다"
        gold = "lip_care"

        score = calculator.compute(predicted, gold)
        assert score == 1.0

    def test_confusion_lip_care_vs_lip_makeup(self, calculator):
        """Test Lip Care vs Lip Makeup confusion detection."""
        # Wrong: says Lip Makeup when it should be Lip Care
        predicted = "This is a Lip Makeup product"
        gold = "lip_care"

        score = calculator.compute(predicted, gold)
        assert score == 0.0  # Confusion detected

    def test_confusion_both_mentioned(self, calculator):
        """Test confusion when both categories mentioned."""
        predicted = "This is both Lip Care and Lip Makeup"
        gold = "lip_care"

        score = calculator.compute(predicted, gold)
        assert score == 0.0  # Confusion: both mentioned

    def test_no_confusion_correct_only(self, calculator):
        """Test no confusion when only correct category mentioned."""
        predicted = "LANEIGE Lip Sleeping Mask is a Lip Care product"
        gold = "lip_care"

        score = calculator.compute(predicted, gold)
        assert score == 1.0

    def test_missing_category(self, calculator):
        """Test when category is not mentioned."""
        predicted = "This is a product"
        gold = "lip_care"

        score = calculator.compute(predicted, gold)
        assert score == 0.0  # Category not mentioned

    def test_skin_care_vs_makeup_confusion(self, calculator):
        """Test Skin Care vs Makeup confusion."""
        predicted = "This is a Makeup product"
        gold = "skin_care"

        score = calculator.compute(predicted, gold)
        assert score == 0.0  # Confusion

    def test_no_confusion_different_categories(self, calculator):
        """Test no confusion for non-conflicting categories."""
        predicted = "This is a Face Powder in the Beauty category"
        gold = "face_powder"

        score = calculator.compute(predicted, gold)
        assert score == 1.0  # No confusion (beauty and face_powder not conflicting)

    def test_extract_categories(self, calculator):
        """Test _extract_categories method."""
        text = "Lip Care and 스킨케어 products"
        categories = calculator._extract_categories(text)

        assert "lip_care" in categories
        assert "skin_care" in categories

    def test_normalize_category(self, calculator):
        """Test _normalize_category method."""
        assert calculator._normalize_category("립케어") == "lip_care"
        assert calculator._normalize_category("Lip Care") == "lip_care"
        assert calculator._normalize_category("Unknown") == "unknown"

    def test_case_insensitive(self, calculator):
        """Test case insensitive matching."""
        predicted = "LIP CARE product"
        gold = "lip_care"

        score = calculator.compute(predicted, gold)
        assert score == 1.0


class TestDomainSpecificEdgeCases:
    """Tests for edge cases across domain-specific metrics."""

    def test_kpi_with_custom_tolerance(self):
        """Test KPI calculator with custom tolerance."""
        calc = KPINumericalAccuracy(tolerance=0.10)  # 10% tolerance

        predicted = "SoS: 22%"
        gold = "SoS: 24%"

        accuracy = calc.compute(predicted, gold)
        assert accuracy == 1.0  # Within 10%

    def test_brand_recall_with_custom_aliases(self):
        """Test brand recall with custom alias map."""
        custom_aliases = {"mybrand": "mybrand", "마이브랜드": "mybrand"}
        calc = BrandEntityRecall(alias_map=custom_aliases)

        predicted = "마이브랜드 is popular"
        gold = ["mybrand"]

        recall = calc.compute(predicted, gold)
        assert recall == 1.0

    def test_category_with_custom_aliases(self):
        """Test category correctness with custom alias map."""
        custom_aliases = {"mycat": "mycat", "내카테고리": "mycat"}
        calc = CategoryCorrectness(alias_map=custom_aliases)

        predicted = "This is 내카테고리"
        gold = "mycat"

        score = calc.compute(predicted, gold)
        assert score == 1.0

    def test_kpi_empty_strings(self):
        """Test KPI calculator with empty strings."""
        calc = KPINumericalAccuracy()

        assert calc.compute("", "") == 1.0  # Both empty
        assert calc.compute("", "SoS: 24%") == 0.0  # Empty pred, non-empty gold
        assert calc.compute("SoS: 24%", "") == 1.0  # Non-empty pred, empty gold

    def test_brand_recall_empty_strings(self):
        """Test brand recall with empty strings."""
        calc = BrandEntityRecall()

        assert calc.compute("", []) == 1.0  # Empty gold
        assert calc.compute("", ["laneige"]) == 0.0  # Empty pred, non-empty gold

    def test_category_empty_strings(self):
        """Test category correctness with empty strings."""
        calc = CategoryCorrectness()

        assert calc.compute("", "lip_care") == 0.0  # Category not found

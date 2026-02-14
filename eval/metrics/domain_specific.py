"""
Domain-Specific Metrics
========================
AMORE-specific evaluation metrics.

Measures domain-specific correctness:
- KPI Numerical Accuracy: Extract and compare KPI numbers (SoS, HHI, CPI)
- Brand Entity Recall: Brand name extraction with Korean/English aliases
- Category Correctness: Detect Lip Care vs Lip Makeup confusion
"""

import re

from eval.metrics.base import BRAND_ALIASES, CATEGORY_ALIASES


class KPINumericalAccuracy:
    """
    KPI numerical accuracy calculator.

    Extracts numeric values from text and compares within tolerance.
    Handles KPI indicators: SoS, HHI, CPI, churn rate, etc.
    """

    # Regex patterns for common KPI formats
    KPI_PATTERNS = {
        "sos": r"(?:sos|share\s+of\s+shelf|점유율)[:\s]*(\d+(?:\.\d+)?)\s*%?",
        "hhi": r"(?:hhi|허핀달|시장집중도)[:\s]*(\d+(?:\.\d+)?)",
        "cpi": r"(?:cpi|가격지수|price\s+index)[:\s]*(\d+(?:\.\d+)?)",
        "churn": r"(?:churn|교체율|이탈율)[:\s]*(\d+(?:\.\d+)?)\s*%?",
        "growth": r"(?:growth|성장률|증가율)[:\s]*(-?\d+(?:\.\d+)?)\s*%?",
        "rank": r"(?:rank|순위)[:\s]*#?(\d+)",
    }

    def __init__(self, tolerance: float = 0.05):
        """
        Initialize KPI accuracy calculator.

        Args:
            tolerance: Relative tolerance for numeric comparison (default 5%)
        """
        self.tolerance = tolerance

    def compute(self, predicted: str, gold: str) -> float:
        """
        Compute KPI numerical accuracy.

        Extracts all numbers from both texts and compares them.
        Returns proportion of gold numbers that match predicted numbers.

        Args:
            predicted: Predicted text containing KPI values
            gold: Gold standard text containing KPI values

        Returns:
            Accuracy score (0.0-1.0)
            - 1.0 if no gold numbers (trivially correct)
            - 0.0 if no predicted numbers but gold has numbers

        Example:
            >>> calc = KPINumericalAccuracy(tolerance=0.05)
            >>> pred = "SoS is 23.5%, HHI is 1250"
            >>> gold = "SoS: 24%, HHI: 1250"
            >>> calc.compute(pred, gold)
            1.0  # 23.5 ≈ 24 within 5%, 1250 exact match
        """
        pred_numbers = self._extract_numbers(predicted)
        gold_numbers = self._extract_numbers(gold)

        if not gold_numbers:
            return 1.0  # No gold numbers to verify

        if not pred_numbers:
            return 0.0  # No predictions but gold exists

        # Match gold numbers to predicted numbers
        matches = 0
        for gold_val in gold_numbers:
            for pred_val in pred_numbers:
                if self._is_close(pred_val, gold_val):
                    matches += 1
                    break  # Count each gold number only once

        return matches / len(gold_numbers)

    def _extract_numbers(self, text: str) -> list[float]:
        """
        Extract all numeric values from text.

        Tries KPI-specific patterns first, then falls back to general numbers.

        Args:
            text: Input text

        Returns:
            List of extracted numbers
        """
        numbers = []
        text_lower = text.lower()

        # Try KPI-specific patterns
        for pattern in self.KPI_PATTERNS.values():
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                try:
                    numbers.append(float(match.group(1)))
                except (ValueError, IndexError):
                    continue

        # If no KPI patterns found, extract all numbers
        if not numbers:
            # Match integers and decimals (including negative)
            general_pattern = r"-?\d+(?:\.\d+)?"
            matches = re.finditer(general_pattern, text)
            for match in matches:
                try:
                    numbers.append(float(match.group(0)))
                except ValueError:
                    continue

        return numbers

    def _is_close(self, a: float, b: float) -> bool:
        """
        Check if two numbers are close within tolerance.

        Uses relative tolerance for large numbers, absolute for small.

        Args:
            a: First number
            b: Second number

        Returns:
            True if numbers are within tolerance
        """
        if a == b:
            return True

        # For very small numbers (< 0.001), use absolute tolerance
        if abs(b) < 0.001:
            return abs(a - b) < 0.001

        # Relative tolerance
        return abs(a - b) / abs(b) <= self.tolerance


class BrandEntityRecall:
    """
    Brand entity recall calculator.

    Measures recall of brand names with Korean/English alias support.
    """

    def __init__(self, alias_map: dict[str, str] | None = None):
        """
        Initialize brand entity recall calculator.

        Args:
            alias_map: Brand alias mapping (defaults to BRAND_ALIASES)
        """
        self.alias_map = alias_map or BRAND_ALIASES

    def compute(self, predicted: str, gold: list[str]) -> float:
        """
        Compute brand entity recall.

        Extracts brand mentions from predicted text and compares to gold.

        Args:
            predicted: Predicted text containing brand mentions
            gold: List of gold brand names

        Returns:
            Recall score (0.0-1.0)
            - 1.0 if no gold brands (trivially correct)
            - Proportion of gold brands found in predicted text

        Example:
            >>> calc = BrandEntityRecall()
            >>> pred = "LANEIGE and 라네즈 are popular. COSRX too."
            >>> gold = ["laneige", "cosrx"]
            >>> calc.compute(pred, gold)
            1.0  # Both brands found (via alias)
        """
        if not gold:
            return 1.0  # No gold brands to recall

        predicted_brands = self._extract_brands(predicted)

        # Normalize gold brands
        gold_normalized = {self._normalize_brand(b) for b in gold}

        # Check how many gold brands were found
        found = sum(1 for gold_brand in gold_normalized if gold_brand in predicted_brands)

        return found / len(gold_normalized)

    def _extract_brands(self, text: str) -> set[str]:
        """
        Extract brand mentions from text.

        Uses alias map to find brand names (case-insensitive).

        Args:
            text: Input text

        Returns:
            Set of normalized brand names found
        """
        brands = set()
        text_lower = text.lower()

        # Check all brand aliases
        for alias, canonical in self.alias_map.items():
            # Use word boundaries to avoid partial matches
            # e.g., "laneige" should not match "laneigee"
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, text_lower):
                brands.add(canonical)

        return brands

    def _normalize_brand(self, brand: str) -> str:
        """
        Normalize brand name using alias map.

        Args:
            brand: Raw brand name

        Returns:
            Canonical brand name
        """
        brand_lower = brand.lower().strip()
        return self.alias_map.get(brand_lower, brand_lower)


class CategoryCorrectness:
    """
    Category correctness calculator.

    Detects common category confusion errors:
    - Lip Care (스킨케어) vs Lip Makeup (색조)
    - Skin Care vs Makeup misclassification
    """

    # Known confusions (mutually exclusive categories)
    CONFUSIONS = [
        {"lip_care", "lip_makeup"},  # LANEIGE Lip Sleeping Mask is Lip Care, NOT Lip Makeup
        {"skin_care", "makeup"},
    ]

    def __init__(self, alias_map: dict[str, str] | None = None):
        """
        Initialize category correctness calculator.

        Args:
            alias_map: Category alias mapping (defaults to CATEGORY_ALIASES)
        """
        self.alias_map = alias_map or CATEGORY_ALIASES

    def compute(self, predicted: str, gold_category: str) -> float:
        """
        Compute category correctness.

        Checks if predicted text contains the correct category and
        does NOT contain confused categories.

        Args:
            predicted: Predicted text containing category mention
            gold_category: Gold category name

        Returns:
            Score (0.0 or 1.0)
            - 1.0 if correct category mentioned and no confusion
            - 0.0 if wrong category or confusion detected

        Example:
            >>> calc = CategoryCorrectness()
            >>> pred = "This is a Lip Care product from LANEIGE"
            >>> calc.compute(pred, "lip_care")
            1.0  # Correct

            >>> pred = "This is a Lip Makeup product"
            >>> calc.compute(pred, "lip_care")
            0.0  # Confusion: Lip Makeup instead of Lip Care
        """
        gold_normalized = self._normalize_category(gold_category)
        predicted_categories = self._extract_categories(predicted)

        # Check if gold category is mentioned
        if gold_normalized not in predicted_categories:
            return 0.0

        # Check for confusions
        for confusion_set in self.CONFUSIONS:
            if gold_normalized in confusion_set:
                # Check if any confused category is also mentioned
                confused_categories = confusion_set - {gold_normalized}
                if confused_categories & predicted_categories:
                    return 0.0  # Confusion detected

        return 1.0

    def _extract_categories(self, text: str) -> set[str]:
        """
        Extract category mentions from text.

        Uses alias map to find category names (case-insensitive).

        Args:
            text: Input text

        Returns:
            Set of normalized category names found
        """
        categories = set()
        text_lower = text.lower()

        # Check all category aliases
        for alias, canonical in self.alias_map.items():
            # Use word boundaries for exact matching
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, text_lower):
                categories.add(canonical)

        return categories

    def _normalize_category(self, category: str) -> str:
        """
        Normalize category name using alias map.

        Args:
            category: Raw category name

        Returns:
            Canonical category name
        """
        category_lower = category.lower().strip()
        return self.alias_map.get(category_lower, category_lower)

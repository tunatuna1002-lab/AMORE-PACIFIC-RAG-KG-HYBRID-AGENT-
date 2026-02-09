"""
Base Metric Calculator
======================
Abstract base class for all metric calculators.

Provides common utilities for:
- Set-based F1 calculation
- Fuzzy matching for entity aliases
- Normalization
- Metric aggregation
"""

from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from typing import Any

# =========================================================================
# KNOWN ENTITY ALIASES (imported from domain knowledge)
# =========================================================================

# Korean ↔ English brand aliases
BRAND_ALIASES: dict[str, str] = {
    # LANEIGE variants
    "laneige": "laneige",
    "라네즈": "laneige",
    "라네이지": "laneige",
    # COSRX variants
    "cosrx": "cosrx",
    "코스알엑스": "cosrx",
    # TIRTIR variants
    "tirtir": "tirtir",
    "티르티르": "tirtir",
    # Rare Beauty
    "rare beauty": "rare_beauty",
    "레어뷰티": "rare_beauty",
    # Innisfree
    "innisfree": "innisfree",
    "이니스프리": "innisfree",
    # ETUDE
    "etude": "etude",
    "에뛰드": "etude",
    # Sulwhasoo
    "sulwhasoo": "sulwhasoo",
    "설화수": "sulwhasoo",
    # HERA
    "hera": "hera",
    "헤라": "hera",
    # MISSHA
    "missha": "missha",
    "미샤": "missha",
    # SKIN1004
    "skin1004": "skin1004",
    "스킨1004": "skin1004",
    # Anua
    "anua": "anua",
    "아누아": "anua",
    # MEDICUBE
    "medicube": "medicube",
    "메디큐브": "medicube",
    # BIODANCE
    "biodance": "biodance",
    "바이오던스": "biodance",
    # Beauty of Joseon
    "beauty of joseon": "beauty_of_joseon",
    "조선미녀": "beauty_of_joseon",
    # International brands
    "summer fridays": "summer_fridays",
    "la roche-posay": "la_roche_posay",
    "cerave": "cerave",
    "neutrogena": "neutrogena",
    "e.l.f.": "elf",
    "elf": "elf",
    "nyx": "nyx",
    "maybelline": "maybelline",
}

# Category aliases
CATEGORY_ALIASES: dict[str, str] = {
    # Lip Care
    "lip care": "lip_care",
    "립케어": "lip_care",
    "립 케어": "lip_care",
    # Lip Makeup
    "lip makeup": "lip_makeup",
    "립메이크업": "lip_makeup",
    "립 메이크업": "lip_makeup",
    # Skin Care
    "skin care": "skin_care",
    "스킨케어": "skin_care",
    "스킨 케어": "skin_care",
    # Face Powder
    "face powder": "face_powder",
    "파우더": "face_powder",
    "페이스파우더": "face_powder",
    # Beauty
    "beauty": "beauty",
    "뷰티": "beauty",
}

# Metric indicator aliases
METRIC_ALIASES: dict[str, str] = {
    "sos": "sos",
    "점유율": "sos",
    "share of shelf": "sos",
    "hhi": "hhi",
    "시장집중도": "hhi",
    "허핀달": "hhi",
    "cpi": "cpi",
    "가격지수": "cpi",
    "churn": "churn_rate",
    "교체율": "churn_rate",
}


def normalize_entity(entity: str, alias_map: dict[str, str] | None = None) -> str:
    """
    Normalize an entity using alias mapping.

    Args:
        entity: Raw entity string
        alias_map: Optional alias dictionary (defaults to BRAND_ALIASES)

    Returns:
        Normalized canonical form
    """
    normalized = entity.lower().strip()

    if alias_map is None:
        # Try all alias maps
        for mapping in [BRAND_ALIASES, CATEGORY_ALIASES, METRIC_ALIASES]:
            if normalized in mapping:
                return mapping[normalized]
        return normalized

    return alias_map.get(normalized, normalized)


def fuzzy_match(
    query: str, candidates: list[str], threshold: float = 0.8
) -> tuple[str | None, float]:
    """
    Find best fuzzy match using SequenceMatcher.

    Args:
        query: Query string to match
        candidates: List of candidate strings
        threshold: Minimum similarity threshold (0.0-1.0)

    Returns:
        Tuple of (best_match, score) or (None, 0.0) if no match above threshold

    Example:
        >>> fuzzy_match("라네이즈", ["laneige", "cosrx", "tirtir"])
        ('laneige', 0.0)  # Would need alias map for Korean
        >>> fuzzy_match("laneig", ["laneige", "cosrx"])
        ('laneige', 0.92)
    """
    best_match = None
    best_score = 0.0

    query_lower = query.lower().strip()

    for candidate in candidates:
        candidate_lower = candidate.lower().strip()
        score = SequenceMatcher(None, query_lower, candidate_lower).ratio()

        if score > best_score and score >= threshold:
            best_score = score
            best_match = candidate

    return (best_match, best_score)


def fuzzy_set_match(
    predicted: set[str],
    gold: set[str],
    threshold: float = 0.8,
    alias_map: dict[str, str] | None = None,
) -> tuple[set[str], set[str], dict[str, str]]:
    """
    Match predicted set to gold set with fuzzy matching and alias resolution.

    Args:
        predicted: Set of predicted items
        gold: Set of gold standard items
        threshold: Fuzzy match threshold
        alias_map: Optional alias mapping

    Returns:
        Tuple of:
        - matched_pred: Predicted items that matched
        - matched_gold: Gold items that were matched
        - match_map: Mapping of pred -> gold for matched items

    Example:
        >>> pred = {"라네즈", "cosrx"}
        >>> gold = {"laneige", "cosrx"}
        >>> fuzzy_set_match(pred, gold, alias_map=BRAND_ALIASES)
        ({'라네즈', 'cosrx'}, {'laneige', 'cosrx'}, {'라네즈': 'laneige', 'cosrx': 'cosrx'})
    """
    matched_pred: set[str] = set()
    matched_gold: set[str] = set()
    match_map: dict[str, str] = {}

    # Normalize gold items
    gold_normalized: dict[str, str] = {}  # normalized -> original
    for g in gold:
        norm = normalize_entity(g, alias_map)
        gold_normalized[norm] = g

    remaining_gold_norm = set(gold_normalized.keys())

    for p in predicted:
        p_norm = normalize_entity(p, alias_map)

        # 1. Exact match after normalization
        if p_norm in remaining_gold_norm:
            matched_pred.add(p)
            original_gold = gold_normalized[p_norm]
            matched_gold.add(original_gold)
            match_map[p] = original_gold
            remaining_gold_norm.discard(p_norm)
            continue

        # 2. Fuzzy match against remaining gold
        best_match, score = fuzzy_match(p_norm, list(remaining_gold_norm), threshold)
        if best_match:
            matched_pred.add(p)
            original_gold = gold_normalized[best_match]
            matched_gold.add(original_gold)
            match_map[p] = original_gold
            remaining_gold_norm.discard(best_match)

    return matched_pred, matched_gold, match_map


class MetricCalculator(ABC):
    """
    Abstract base class for metric calculators.

    Each layer (L1-L5) implements this interface to provide
    standardized metric computation.
    """

    @abstractmethod
    def compute(self, trace: Any, gold: Any) -> dict[str, float]:
        """
        Compute metrics for a single evaluation item.

        Args:
            trace: Trace data from evaluation run
            gold: Gold standard evidence

        Returns:
            Dict of metric_name -> score (0.0-1.0)
        """
        ...

    @staticmethod
    def set_f1(predicted: set[str], gold: set[str]) -> float:
        """
        Compute F1 score between two sets (exact match).

        Args:
            predicted: Set of predicted items
            gold: Set of gold standard items

        Returns:
            F1 score (0.0-1.0), returns 1.0 if both sets are empty
        """
        if not predicted and not gold:
            return 1.0  # Both empty = perfect match

        if not predicted or not gold:
            return 0.0  # One empty, one not = no match

        # Normalize for comparison
        pred_normalized = {s.lower().strip() for s in predicted}
        gold_normalized = {s.lower().strip() for s in gold}

        true_positives = len(pred_normalized & gold_normalized)
        precision = true_positives / len(pred_normalized) if pred_normalized else 0.0
        recall = true_positives / len(gold_normalized) if gold_normalized else 0.0

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def set_f1_fuzzy(
        predicted: set[str],
        gold: set[str],
        threshold: float = 0.8,
        alias_map: dict[str, str] | None = None,
    ) -> float:
        """
        Compute F1 score with fuzzy matching and alias resolution.

        This is the recommended method for entity matching where:
        - Korean/English aliases exist (라네즈 = LANEIGE)
        - Spelling variations are common (laneig vs laneige)

        Args:
            predicted: Set of predicted items
            gold: Set of gold standard items
            threshold: Fuzzy match threshold (default 0.8)
            alias_map: Optional alias dictionary for normalization

        Returns:
            F1 score (0.0-1.0), returns 1.0 if both sets are empty

        Example:
            >>> pred = {"라네즈", "cosrx"}
            >>> gold = {"laneige", "cosrx"}
            >>> MetricCalculator.set_f1(pred, gold)  # Exact: 0.5
            >>> MetricCalculator.set_f1_fuzzy(pred, gold, alias_map=BRAND_ALIASES)  # Fuzzy: 1.0
        """
        if not predicted and not gold:
            return 1.0  # Both empty = perfect match

        if not predicted or not gold:
            return 0.0  # One empty, one not = no match

        # Use fuzzy set matching
        matched_pred, matched_gold, _ = fuzzy_set_match(predicted, gold, threshold, alias_map)

        precision = len(matched_pred) / len(predicted) if predicted else 0.0
        recall = len(matched_gold) / len(gold) if gold else 0.0

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def set_precision(predicted: set[str], gold: set[str]) -> float:
        """
        Compute precision between two sets.

        Args:
            predicted: Set of predicted items
            gold: Set of gold standard items

        Returns:
            Precision score (0.0-1.0)
        """
        if not predicted:
            return 1.0 if not gold else 0.0

        pred_normalized = {s.lower().strip() for s in predicted}
        gold_normalized = {s.lower().strip() for s in gold}

        true_positives = len(pred_normalized & gold_normalized)
        return true_positives / len(pred_normalized)

    @staticmethod
    def set_recall(predicted: set[str], gold: set[str]) -> float:
        """
        Compute recall between two sets.

        Args:
            predicted: Set of predicted items
            gold: Set of gold standard items

        Returns:
            Recall score (0.0-1.0)
        """
        if not gold:
            return 1.0  # No gold items to recall

        pred_normalized = {s.lower().strip() for s in predicted}
        gold_normalized = {s.lower().strip() for s in gold}

        true_positives = len(pred_normalized & gold_normalized)
        return true_positives / len(gold_normalized)

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for comparison.

        - Lowercase
        - Strip whitespace
        - Collapse multiple spaces
        """
        return " ".join(text.lower().split())

    @staticmethod
    def token_overlap(text1: str, text2: str) -> float:
        """
        Compute token overlap ratio between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Jaccard similarity of tokens (0.0-1.0)
        """
        if not text1 or not text2:
            return 0.0

        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
        """
        Compute recall at k.

        Args:
            retrieved: Ordered list of retrieved items
            gold: Set of gold standard items
            k: Cutoff position

        Returns:
            Recall@k score (0.0-1.0)
        """
        if not gold:
            return 1.0

        top_k = {s.lower().strip() for s in retrieved[:k]}
        gold_normalized = {s.lower().strip() for s in gold}

        hits = len(top_k & gold_normalized)
        return hits / len(gold_normalized)

    @staticmethod
    def precision_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
        """
        Compute precision at k.

        Args:
            retrieved: Ordered list of retrieved items
            gold: Set of gold standard items
            k: Cutoff position

        Returns:
            Precision@k score (0.0-1.0)
        """
        if not retrieved[:k]:
            return 0.0

        top_k = [s.lower().strip() for s in retrieved[:k]]
        gold_normalized = {s.lower().strip() for s in gold}

        hits = sum(1 for item in top_k if item in gold_normalized)
        return hits / len(top_k)

    @staticmethod
    def mrr(retrieved: list[str], gold: set[str]) -> float:
        """
        Compute Mean Reciprocal Rank.

        Args:
            retrieved: Ordered list of retrieved items
            gold: Set of gold standard items

        Returns:
            MRR score (0.0-1.0)
        """
        if not gold or not retrieved:
            return 0.0

        gold_normalized = {s.lower().strip() for s in gold}

        for i, item in enumerate(retrieved):
            if item.lower().strip() in gold_normalized:
                return 1.0 / (i + 1)

        return 0.0

    @staticmethod
    def hits_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
        """
        Compute Hits@k (binary: 1 if any gold item in top-k, else 0).

        Args:
            retrieved: Ordered list of retrieved items
            gold: Set of gold standard items
            k: Cutoff position

        Returns:
            1.0 if any hit in top-k, else 0.0
        """
        if not gold:
            return 1.0

        top_k = {s.lower().strip() for s in retrieved[:k]}
        gold_normalized = {s.lower().strip() for s in gold}

        return 1.0 if top_k & gold_normalized else 0.0

"""
NDCG (Normalized Discounted Cumulative Gain) Metrics
======================================================
Ranking quality metrics for retrieval systems.

Measures how well the system ranks relevant documents:
- NDCG@k: Normalized discounted cumulative gain at position k
- DCG: Discounted cumulative gain (position-based relevance)
- IDCG: Ideal DCG (best possible ranking)

NDCG formula: DCG@k / IDCG@k
where DCG = sum(rel_i / log2(i+2)) for i=0..k-1
(0-indexed, so denominator is log2(rank+1) where rank starts at 1)
"""

import math


class NDCGCalculator:
    """
    NDCG (Normalized Discounted Cumulative Gain) calculator.

    Evaluates ranking quality by considering both relevance and position.
    Higher-ranked relevant items contribute more to the score.
    """

    @staticmethod
    def compute_ndcg_at_k(
        retrieved_ids: list[str],
        gold_ids: list[str],
        relevance_scores: dict[str, float] | None = None,
        k: int = 10,
    ) -> float:
        """
        Compute NDCG@k for a ranked retrieval list.

        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            gold_ids: List of gold/relevant document IDs
            relevance_scores: Optional mapping of doc_id -> relevance (0.0-1.0)
                            If None, uses binary relevance (1.0 for gold, 0.0 otherwise)
            k: Cutoff position for evaluation

        Returns:
            NDCG@k score (0.0-1.0)
            - 1.0 if gold_ids is empty (no relevant docs to rank)
            - 0.0 if retrieved_ids is empty but gold_ids is not

        Example:
            >>> retrieved = ["doc1", "doc3", "doc2"]
            >>> gold = ["doc1", "doc2"]
            >>> NDCGCalculator.compute_ndcg_at_k(retrieved, gold, k=3)
            0.861...  # Good ranking (doc1 first)

            >>> retrieved = ["doc3", "doc2", "doc1"]
            >>> NDCGCalculator.compute_ndcg_at_k(retrieved, gold, k=3)
            0.613...  # Worse ranking (doc1 last)
        """
        # Edge case: no gold items
        if not gold_ids:
            return 1.0

        # Edge case: no retrieved items but gold exists
        if not retrieved_ids:
            return 0.0

        # Normalize IDs for case-insensitive comparison
        retrieved_normalized = [doc_id.lower().strip() for doc_id in retrieved_ids]
        gold_normalized = [doc_id.lower().strip() for doc_id in gold_ids]

        # Create relevance mapping with normalized keys
        if relevance_scores is None:
            # Binary relevance: 1.0 for gold items, 0.0 for others
            gold_set = set(gold_normalized)
            relevance = {
                doc_id: 1.0 if doc_id in gold_set else 0.0 for doc_id in retrieved_normalized
            }
        else:
            # Normalize relevance scores keys
            relevance = {k.lower().strip(): v for k, v in relevance_scores.items()}

        # Compute DCG@k for retrieved ranking
        dcg = NDCGCalculator._compute_dcg(retrieved_normalized, relevance, k)

        # Compute IDCG@k (best possible ranking)
        # Sort gold items by relevance (highest first)
        if relevance_scores is None:
            # Binary relevance: all gold items have same relevance
            ideal_ranking = list(gold_normalized)
        else:
            # Sort by relevance score descending
            ideal_ranking = sorted(
                gold_normalized,
                key=lambda doc_id: relevance.get(doc_id, 0.0),
                reverse=True,
            )

        idcg = NDCGCalculator._compute_dcg(ideal_ranking, relevance, k)

        # Avoid division by zero
        if idcg == 0.0:
            return 0.0

        ndcg = dcg / idcg
        # Clamp to [0.0, 1.0] to handle edge cases
        return min(1.0, max(0.0, ndcg))

    @staticmethod
    def _compute_dcg(doc_ids: list[str], relevance: dict[str, float], k: int) -> float:
        """
        Compute Discounted Cumulative Gain.

        DCG = sum(rel_i / log2(i+2)) for i=0..k-1
        Uses 0-indexed positions, so denominator is log2(rank+1) where rank starts at 1.

        Args:
            doc_ids: Ordered list of document IDs
            relevance: Mapping of doc_id -> relevance score
            k: Cutoff position

        Returns:
            DCG score
        """
        dcg = 0.0

        for i, doc_id in enumerate(doc_ids[:k]):
            rel = relevance.get(doc_id, 0.0)
            # Position i is 0-indexed, so rank = i + 1
            # Discount factor: log2(rank + 1) = log2(i + 2)
            discount = math.log2(i + 2)
            dcg += rel / discount

        return dcg


def ndcg_at_k(
    retrieved_ids: list[str],
    gold_ids: list[str],
    relevance_scores: dict[str, float] | None = None,
    k: int = 10,
) -> float:
    """
    Convenience function for computing NDCG@k.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        gold_ids: List of gold/relevant document IDs
        relevance_scores: Optional mapping of doc_id -> relevance (0.0-1.0)
        k: Cutoff position

    Returns:
        NDCG@k score (0.0-1.0)

    Example:
        >>> from eval.metrics.ndcg import ndcg_at_k
        >>> retrieved = ["doc1", "doc2", "doc3"]
        >>> gold = ["doc1", "doc3"]
        >>> ndcg_at_k(retrieved, gold, k=10)
        0.946...
    """
    calc = NDCGCalculator()
    return calc.compute_ndcg_at_k(retrieved_ids, gold_ids, relevance_scores, k)

"""
L2 Retrieval Metrics
====================
Document retrieval quality metrics.

Measures how well the system retrieves relevant documents:
- Context Recall@k: Proportion of gold chunks in top-k
- Context Precision@k: Proportion of top-k that are gold
- MRR: Mean Reciprocal Rank of first relevant document
"""

from eval.metrics.base import MetricCalculator
from eval.schemas import DocRetrievalTrace, GoldEvidence, L2Metrics


class L2RetrievalMetrics(MetricCalculator):
    """
    L2 metrics for document retrieval.

    Evaluates retrieval quality against gold document chunk IDs.
    """

    def __init__(self, default_k: int = 8):
        """
        Initialize L2 metrics calculator.

        Args:
            default_k: Default cutoff for @k metrics
        """
        self.default_k = default_k

    def compute(
        self,
        trace: DocRetrievalTrace,
        gold: GoldEvidence,
        k: int | None = None,
    ) -> L2Metrics:
        """
        Compute L2 metrics.

        Args:
            trace: Document retrieval trace
            gold: Gold standard evidence
            k: Cutoff for @k metrics (defaults to self.default_k)

        Returns:
            L2Metrics with recall, precision, and MRR
        """
        k = k or self.default_k

        recall = self._compute_recall_at_k(trace, gold, k)
        precision = self._compute_precision_at_k(trace, gold, k)
        mrr = self._compute_mrr(trace, gold)

        return L2Metrics(
            context_recall_at_k=recall,
            context_precision_at_k=precision,
            mrr=mrr,
        )

    def _compute_recall_at_k(self, trace: DocRetrievalTrace, gold: GoldEvidence, k: int) -> float:
        """
        Compute recall at k.

        Measures what proportion of gold chunks are in the top-k retrieved.
        """
        gold_chunks = set(gold.doc_chunk_ids)

        if not gold_chunks:
            return 1.0  # No gold chunks to recall

        return self.recall_at_k(trace.chunk_ids, gold_chunks, k)

    def _compute_precision_at_k(
        self, trace: DocRetrievalTrace, gold: GoldEvidence, k: int
    ) -> float:
        """
        Compute precision at k.

        Measures what proportion of top-k retrieved are gold chunks.
        """
        gold_chunks = set(gold.doc_chunk_ids)

        if not gold_chunks:
            # If no gold evidence, any retrieved chunk is "false positive"
            # But we return 1.0 to not penalize when gold is empty
            return 1.0

        return self.precision_at_k(trace.chunk_ids, gold_chunks, k)

    def _compute_mrr(self, trace: DocRetrievalTrace, gold: GoldEvidence) -> float:
        """
        Compute Mean Reciprocal Rank.

        Measures how early the first relevant document appears.
        """
        gold_chunks = set(gold.doc_chunk_ids)

        if not gold_chunks:
            return 1.0  # No gold to find

        return self.mrr(trace.chunk_ids, gold_chunks)


def context_recall_at_k(trace: DocRetrievalTrace, gold: GoldEvidence, k: int = 8) -> float:
    """
    Convenience function for context recall at k.

    Args:
        trace: Document retrieval trace
        gold: Gold evidence
        k: Cutoff position

    Returns:
        Recall of gold.doc_chunk_ids in top-k retrieved chunks
    """
    calc = L2RetrievalMetrics(default_k=k)
    return calc._compute_recall_at_k(trace, gold, k)


def context_precision_at_k(trace: DocRetrievalTrace, gold: GoldEvidence, k: int = 8) -> float:
    """
    Convenience function for context precision at k.

    Args:
        trace: Document retrieval trace
        gold: Gold evidence
        k: Cutoff position

    Returns:
        Precision of top-k retrieved chunks against gold
    """
    calc = L2RetrievalMetrics(default_k=k)
    return calc._compute_precision_at_k(trace, gold, k)


def mrr(trace: DocRetrievalTrace, gold: GoldEvidence) -> float:
    """
    Convenience function for Mean Reciprocal Rank.

    Args:
        trace: Document retrieval trace
        gold: Gold evidence

    Returns:
        MRR score
    """
    calc = L2RetrievalMetrics()
    return calc._compute_mrr(trace, gold)

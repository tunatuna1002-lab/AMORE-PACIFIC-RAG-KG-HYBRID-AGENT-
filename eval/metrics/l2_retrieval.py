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
        doc_recall = self._compute_doc_recall_at_k(trace, gold, k)

        return L2Metrics(
            context_recall_at_k=recall,
            context_precision_at_k=precision,
            mrr=mrr,
            context_recall_at_k_doc=doc_recall,
        )

    @staticmethod
    def _source_doc(chunk_id: str) -> str:
        """청크 ID(`{doc_id}_{section}_{part}`) → 출처 문서 ID."""
        parts = str(chunk_id).rsplit("_", 2)
        return parts[0] if len(parts) == 3 else str(chunk_id)

    def _compute_doc_recall_at_k(
        self, trace: DocRetrievalTrace, gold: GoldEvidence, k: int
    ) -> float:
        """출처 문서 단위 recall@k.

        골드 `doc_chunk_ids`는 원래 **문서 수준** 가상 ID(market_trend_01,
        competitor_analysis_01 등)였고, 2026-08-30 재매핑에서 각 개념을 대표
        청크 1개로 1:1 치환했다(`scripts/remap_golden_chunk_ids.py`). 그 결과
        160문항이 서로 다른 청크 7개만 가리키고, 그중 119건이 단 2개 청크
        (`laneige_strategy_2026`의 291자·379자 문단)에 몰려 있다. 즉 청크 단위
        recall은 "38개 청크를 가진 문서에서 하필 그 문단이 top-k에 들었는가"를
        묻는 셈이라 검색 품질보다 라벨 입도를 측정한다.
        실제로 v5.0에서 청크 단위 0.102 vs 문서 단위 0.555로 벌어졌다.
        라벨이 실제로 가진 입도(문서)에 맞춘 지표를 함께 보고하고 게이트는
        이쪽을 쓴다. 청크 단위 지표는 연속성을 위해 계속 보고한다.
        """
        gold_docs = {self._source_doc(c) for c in gold.doc_chunk_ids}
        if not gold_docs:
            return 1.0
        retrieved_docs = {self._source_doc(c) for c in trace.chunk_ids[:k]}
        return len(gold_docs & retrieved_docs) / len(gold_docs)

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

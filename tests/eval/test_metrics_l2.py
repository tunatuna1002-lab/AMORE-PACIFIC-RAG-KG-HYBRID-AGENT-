"""Tests for L2 retrieval metrics."""

import pytest

from eval.metrics.l2_retrieval import (
    L2RetrievalMetrics,
    context_precision_at_k,
    context_recall_at_k,
    mrr,
)
from eval.schemas import DocRetrievalTrace, GoldEvidence


class TestL2RetrievalMetrics:
    """Tests for L2 retrieval metrics."""

    @pytest.fixture
    def calculator(self):
        """Create L2 metrics calculator."""
        return L2RetrievalMetrics(default_k=5)

    def test_recall_at_k_all_hits(self, calculator):
        """Test recall@k with all gold chunks retrieved."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2", "c3", "c4", "c5"],
            snippets=[""] * 5,
            scores=[0.9, 0.8, 0.7, 0.6, 0.5],
        )
        gold = GoldEvidence(doc_chunk_ids=["c1", "c2"])

        recall = calculator._compute_recall_at_k(trace, gold, k=5)
        assert recall == 1.0

    def test_recall_at_k_partial_hits(self, calculator):
        """Test recall@k with partial hits."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c3", "c5"],
            snippets=[""] * 3,
            scores=[0.9, 0.8, 0.7],
        )
        gold = GoldEvidence(doc_chunk_ids=["c1", "c2"])

        recall = calculator._compute_recall_at_k(trace, gold, k=3)
        assert recall == 0.5  # 1 of 2 gold chunks found

    def test_recall_at_k_no_hits(self, calculator):
        """Test recall@k with no hits."""
        trace = DocRetrievalTrace(
            chunk_ids=["c3", "c4", "c5"],
            snippets=[""] * 3,
            scores=[0.9, 0.8, 0.7],
        )
        gold = GoldEvidence(doc_chunk_ids=["c1", "c2"])

        recall = calculator._compute_recall_at_k(trace, gold, k=3)
        assert recall == 0.0

    def test_recall_at_k_empty_gold(self, calculator):
        """Test recall@k with empty gold."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2"],
            snippets=[""] * 2,
            scores=[0.9, 0.8],
        )
        gold = GoldEvidence(doc_chunk_ids=[])

        recall = calculator._compute_recall_at_k(trace, gold, k=5)
        assert recall == 1.0  # No gold = perfect recall

    def test_precision_at_k_all_relevant(self, calculator):
        """Test precision@k when all retrieved are relevant."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2"],
            snippets=[""] * 2,
            scores=[0.9, 0.8],
        )
        gold = GoldEvidence(doc_chunk_ids=["c1", "c2", "c3"])

        precision = calculator._compute_precision_at_k(trace, gold, k=2)
        assert precision == 1.0

    def test_precision_at_k_partial_relevant(self, calculator):
        """Test precision@k with partial relevance."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2", "c3", "c4"],
            snippets=[""] * 4,
            scores=[0.9, 0.8, 0.7, 0.6],
        )
        gold = GoldEvidence(doc_chunk_ids=["c1", "c3"])

        precision = calculator._compute_precision_at_k(trace, gold, k=4)
        assert precision == 0.5  # 2 of 4 are relevant

    def test_mrr_first_position(self, calculator):
        """Test MRR when first result is relevant."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2", "c3"],
            snippets=[""] * 3,
            scores=[0.9, 0.8, 0.7],
        )
        gold = GoldEvidence(doc_chunk_ids=["c1"])

        mrr_score = calculator._compute_mrr(trace, gold)
        assert mrr_score == 1.0  # 1/(1) = 1.0

    def test_mrr_second_position(self, calculator):
        """Test MRR when second result is relevant."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2", "c3"],
            snippets=[""] * 3,
            scores=[0.9, 0.8, 0.7],
        )
        gold = GoldEvidence(doc_chunk_ids=["c2"])

        mrr_score = calculator._compute_mrr(trace, gold)
        assert mrr_score == 0.5  # 1/(2) = 0.5

    def test_mrr_third_position(self, calculator):
        """Test MRR when third result is relevant."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2", "c3"],
            snippets=[""] * 3,
            scores=[0.9, 0.8, 0.7],
        )
        gold = GoldEvidence(doc_chunk_ids=["c3"])

        mrr_score = calculator._compute_mrr(trace, gold)
        assert abs(mrr_score - 0.333) < 0.01  # 1/(3) ≈ 0.333

    def test_mrr_no_relevant(self, calculator):
        """Test MRR when no relevant results."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2", "c3"],
            snippets=[""] * 3,
            scores=[0.9, 0.8, 0.7],
        )
        gold = GoldEvidence(doc_chunk_ids=["c4", "c5"])

        mrr_score = calculator._compute_mrr(trace, gold)
        assert mrr_score == 0.0

    def test_compute_full_metrics(self, calculator):
        """Test full L2 metrics computation."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2", "c3", "c4", "c5"],
            snippets=[""] * 5,
            scores=[0.9, 0.8, 0.7, 0.6, 0.5],
        )
        gold = GoldEvidence(doc_chunk_ids=["c1", "c3"])

        metrics = calculator.compute(trace, gold, k=5)

        assert metrics.context_recall_at_k == 1.0  # Both found in top 5
        assert metrics.context_precision_at_k == 0.4  # 2/5 relevant
        assert metrics.mrr == 1.0  # c1 is first

    def test_case_insensitive(self, calculator):
        """Test metrics are case insensitive."""
        trace = DocRetrievalTrace(
            chunk_ids=["CHUNK1", "chunk2"],
            snippets=[""] * 2,
            scores=[0.9, 0.8],
        )
        gold = GoldEvidence(doc_chunk_ids=["chunk1", "CHUNK2"])

        metrics = calculator.compute(trace, gold, k=2)
        assert metrics.context_recall_at_k == 1.0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_context_recall_at_k_function(self):
        """Test context_recall_at_k convenience function."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2"],
            snippets=[""] * 2,
            scores=[0.9, 0.8],
        )
        gold = GoldEvidence(doc_chunk_ids=["c1"])

        recall = context_recall_at_k(trace, gold, k=2)
        assert recall == 1.0

    def test_context_precision_at_k_function(self):
        """Test context_precision_at_k convenience function."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2"],
            snippets=[""] * 2,
            scores=[0.9, 0.8],
        )
        gold = GoldEvidence(doc_chunk_ids=["c1"])

        precision = context_precision_at_k(trace, gold, k=2)
        assert precision == 0.5

    def test_mrr_function(self):
        """Test mrr convenience function."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2"],
            snippets=[""] * 2,
            scores=[0.9, 0.8],
        )
        gold = GoldEvidence(doc_chunk_ids=["c2"])

        mrr_score = mrr(trace, gold)
        assert mrr_score == 0.5


class TestDocumentLevelRecall:
    """출처 문서 단위 recall (2026-08-30 사이클 5 신설).

    골드 doc_chunk_ids는 문서 수준 가상 ID를 대표 청크 1개로 치환한 결과라
    청크 단위 판정이 라벨 입도를 측정하는 문제가 있었다.
    """

    @pytest.fixture
    def calculator(self):
        return L2RetrievalMetrics(default_k=8)

    def test_same_document_different_chunk_counts_as_hit(self, calculator):
        trace = DocRetrievalTrace(chunk_ids=["laneige_strategy_2026_5_0"], snippets=[], scores=[])
        gold = GoldEvidence(doc_chunk_ids=["laneige_strategy_2026_3_1"])

        metrics = calculator.compute(trace, gold)
        assert metrics.context_recall_at_k_doc == 1.0
        # 청크 단위는 여전히 0 — 두 지표를 함께 보고하는 이유
        assert metrics.context_recall_at_k == 0.0

    def test_different_document_is_a_miss(self, calculator):
        trace = DocRetrievalTrace(chunk_ids=["strategic_indicators_1_0"], snippets=[], scores=[])
        gold = GoldEvidence(doc_chunk_ids=["laneige_strategy_2026_3_1"])

        assert calculator._compute_doc_recall_at_k(trace, gold, 8) == 0.0

    def test_empty_gold_is_one(self, calculator):
        trace = DocRetrievalTrace(chunk_ids=["strategic_indicators_1_0"], snippets=[], scores=[])
        gold = GoldEvidence(doc_chunk_ids=[])

        assert calculator._compute_doc_recall_at_k(trace, gold, 8) == 1.0

    def test_respects_k_cutoff(self, calculator):
        trace = DocRetrievalTrace(
            chunk_ids=[f"other_doc_{i}_0" for i in range(8)] + ["laneige_strategy_2026_1_0"],
            snippets=[],
            scores=[],
        )
        gold = GoldEvidence(doc_chunk_ids=["laneige_strategy_2026_3_1"])

        assert calculator._compute_doc_recall_at_k(trace, gold, 8) == 0.0
        assert calculator._compute_doc_recall_at_k(trace, gold, 9) == 1.0


class TestConceptLevelRecall:
    """개념 단위 recall (2026-08-30 사이클 6 신설).

    골드 `doc_chunk_groups`는 개념마다 그 개념이 서술된 절의 청크 집합을 담는다.
    한 개념의 근거는 문단 하나가 아니라 절 전체이므로, 집합 중 하나라도 검색되면
    그 개념의 근거를 찾은 것으로 본다.
    """

    @pytest.fixture
    def calculator(self):
        return L2RetrievalMetrics(default_k=8)

    def test_any_chunk_in_group_counts_as_hit(self, calculator):
        trace = DocRetrievalTrace(chunk_ids=["strategic_indicators_1_1"], snippets=[], scores=[])
        gold = GoldEvidence(
            doc_chunk_ids=["strategic_indicators_1_0", "strategic_indicators_1_1"],
            doc_chunk_groups=[["strategic_indicators_1_0", "strategic_indicators_1_1"]],
        )

        assert calculator.compute(trace, gold).context_recall_at_k_concept == 1.0

    def test_partial_concept_coverage_is_penalised(self, calculator):
        trace = DocRetrievalTrace(chunk_ids=["a_1_0"], snippets=[], scores=[])
        gold = GoldEvidence(
            doc_chunk_ids=["a_1_0", "b_1_0"],
            doc_chunk_groups=[["a_1_0"], ["b_1_0"]],
        )

        assert calculator.compute(trace, gold).context_recall_at_k_concept == 0.5

    def test_no_group_hit_is_zero(self, calculator):
        trace = DocRetrievalTrace(chunk_ids=["other_1_0"], snippets=[], scores=[])
        gold = GoldEvidence(doc_chunk_ids=["a_1_0"], doc_chunk_groups=[["a_1_0", "a_1_1"]])

        assert calculator._compute_concept_recall_at_k(trace, gold, 8) == 0.0

    def test_falls_back_to_chunk_recall_without_groups(self, calculator):
        """구 골드(그룹 없음)는 평면 청크 recall로 폴백해 호환을 유지한다."""
        trace = DocRetrievalTrace(chunk_ids=["a_1_0"], snippets=[], scores=[])
        gold = GoldEvidence(doc_chunk_ids=["a_1_0", "b_1_0"])

        assert calculator._compute_concept_recall_at_k(trace, gold, 8) == 0.5

    def test_respects_k_cutoff(self, calculator):
        trace = DocRetrievalTrace(
            chunk_ids=[f"filler_{i}_0" for i in range(8)] + ["a_1_0"], snippets=[], scores=[]
        )
        gold = GoldEvidence(doc_chunk_ids=["a_1_0"], doc_chunk_groups=[["a_1_0"]])

        assert calculator._compute_concept_recall_at_k(trace, gold, 8) == 0.0
        assert calculator._compute_concept_recall_at_k(trace, gold, 9) == 1.0


class TestGoldenSetChunkGroups:
    """재설계된 골든셋 라벨의 무결성 (사이클 6)."""

    def test_groups_flatten_to_doc_chunk_ids(self):
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "eval/data/golden/laneige_golden_v2.jsonl"
        checked = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            gold = json.loads(line)["gold"]
            groups = gold.get("doc_chunk_groups") or []
            if not groups:
                continue
            flat = {cid for group in groups for cid in group}
            assert flat == set(gold["doc_chunk_ids"])
            assert all(len(group) >= 2 for group in groups), "개념 집합은 2청크 이상이어야 한다"
            checked += 1
        assert checked >= 130, f"그룹이 있는 문항이 너무 적다: {checked}"

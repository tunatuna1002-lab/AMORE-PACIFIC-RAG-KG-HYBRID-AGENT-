"""평가 하네스 기술 부채 4건의 회귀 테스트 (2026-09-06)

근거: docs/eval/rag-eval-review-2026-09-06.md §1-c, §3-4.
이 결함들이 남아 있으면 이후 사이클의 측정을 신뢰할 수 없다.

a. LLM 호출에 타임아웃이 없어 실행이 무기한 정지 → 문항당 상한, 실패는 0점이 아니라 분리
b. report.json이 metadata.requires_kg를 잃어 리포트만으로 게이팅 재현 불가
c. CostTracker를 러너가 호출하지 않아 모든 baseline의 비용이 0
d. 종합 점수 공식과 게이트가 서로 다른 지표를 사용
"""

import asyncio
import json

import pytest

from eval.metrics.aggregator import MetricAggregator
from eval.report import ReportGenerator
from eval.runner import EvalRunner
from eval.schemas import (
    EvalConfig,
    EvalItem,
    GoldEvidence,
    ItemMetadata,
    L1Metrics,
    L2Metrics,
    L3Metrics,
    L4Metrics,
    L5Metrics,
)


class _FakeAgent:
    """chat()만 가진 최소 에이전트."""

    model = "gpt-4.1-mini"

    def __init__(self, delay: float = 0.0, raises: Exception | None = None, usage=None):
        self.delay = delay
        self.raises = raises
        self.usage = usage

    async def chat(self, question: str) -> dict:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raises:
            raise self.raises
        result = {"response": f"{question}에 대한 답변입니다. SoS는 2.0%입니다.", "sources": []}
        if self.usage:
            result["llm_usage"] = self.usage
        return result


class _FakeJudge:
    """on_usage 훅으로 토큰을 보고하는 judge (LLMJudge와 같은 계약)."""

    on_usage = None

    def __init__(self, tokens=(120, 30)):
        self.tokens = tokens

    def _report(self):
        if self.on_usage is not None:
            self.on_usage(*self.tokens)

    async def score_groundedness(self, answer: str, context: str) -> float:
        self._report()
        return 0.9

    async def score_relevance(self, answer: str, question: str) -> float:
        self._report()
        return 0.9

    async def score_factuality(self, answer: str, facts: list[str]):
        self._report()
        return 1.0, []


def _item(item_id: str = "t001", **meta) -> EvalItem:
    return EvalItem(
        id=item_id,
        question="LANEIGE Lip Care SoS는?",
        gold=GoldEvidence(answer="LANEIGE의 Lip Care SoS는 2.0%입니다."),
        metadata=ItemMetadata(**meta) if meta else ItemMetadata(),
    )


# =============================================================================
# a. 타임아웃과 인프라 실패 분리
# =============================================================================


class TestTimeoutAndErrorSeparation:
    @pytest.mark.asyncio
    async def test_agent_timeout_is_excluded_not_scored_zero(self):
        runner = EvalRunner(
            agent=_FakeAgent(delay=5.0),
            config=EvalConfig(item_timeout_seconds=0.05),
        )

        result = await runner.run_item(_item())

        assert result.trace is not None
        assert result.trace.error.startswith("agent_timeout")
        assert result.passed is False
        # 인프라 실패를 모델 실패 사유로 태깅하지 않는다
        assert result.fail_reason_tags == []

    @pytest.mark.asyncio
    async def test_api_error_is_recorded_as_infrastructure_failure(self):
        runner = EvalRunner(
            agent=_FakeAgent(raises=RuntimeError("rate limit")),
            config=EvalConfig(item_timeout_seconds=5),
        )

        result = await runner.run_item(_item())

        assert result.trace.error.startswith("agent_error")
        assert "rate limit" in result.trace.error

    @pytest.mark.asyncio
    async def test_errored_items_leave_the_aggregate_denominator(self, tmp_path):
        ok_runner = EvalRunner(agent=_FakeAgent(), config=EvalConfig())
        bad_runner = EvalRunner(
            agent=_FakeAgent(delay=5.0), config=EvalConfig(item_timeout_seconds=0.05)
        )
        results = [
            await ok_runner.run_item(_item("ok001")),
            await bad_runner.run_item(_item("bad001")),
        ]

        report = ReportGenerator().generate_report(results, tmp_path)

        assert report.aggregates.total == 1  # 채점된 문항만 분모
        assert report.aggregates.errored == 1
        assert report.aggregates.error_item_ids == ["bad001"]
        # 0점짜리 실패 문항이 평균을 끌어내리지 않는다
        assert report.aggregates.avg_overall_score == pytest.approx(results[0].overall_score)


# =============================================================================
# b. requires_kg 직렬화 보존
# =============================================================================


class TestMetadataSurvivesSerialization:
    @pytest.mark.asyncio
    async def test_requires_kg_and_domain_reach_report_json(self, tmp_path):
        runner = EvalRunner(agent=_FakeAgent(), config=EvalConfig())
        item = _item("lg201", requires_kg=False, domain="ir", difficulty="hard")

        result = await runner.run_item(item)
        assert result.metadata.requires_kg is False

        ReportGenerator().generate_report([result], tmp_path)
        written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))

        meta = written["items"][0]["metadata"]
        assert meta["requires_kg"] is False
        assert meta["domain"] == "ir"
        assert meta["difficulty"] == "hard"
        # 도메인 breakdown도 더 이상 전부 "general"이 아니다
        assert set(written["aggregates"]["by_domain"]) == {"ir"}
        assert written["items"][0]["question"] == item.question


# =============================================================================
# c. 비용 추적
# =============================================================================


class TestCostTracking:
    @pytest.mark.asyncio
    async def test_answer_and_judge_tokens_reach_the_report(self, tmp_path):
        runner = EvalRunner(
            agent=_FakeAgent(usage={"prompt_tokens": 1000, "completion_tokens": 200}),
            config=EvalConfig(use_judge=True),
            judge=_FakeJudge(tokens=(120, 30)),
        )

        result = await runner.run_item(_item())

        assert result.trace.cost.l5_tokens == 1200
        # groundedness·relevance·factuality 3회 호출
        assert result.trace.cost.judge_tokens == 3 * 150

        report = ReportGenerator().generate_report([result], tmp_path)
        assert report.aggregates.total_tokens == 1200 + 450
        assert report.aggregates.total_cost_usd > 0

    @pytest.mark.asyncio
    async def test_item_costs_are_not_cumulative_across_items(self):
        """문항 트레이스에 실행 누계가 들어가면 리포트가 중복 집계된다."""
        runner = EvalRunner(
            agent=_FakeAgent(usage={"prompt_tokens": 100, "completion_tokens": 10}),
            config=EvalConfig(use_judge=True),
            judge=_FakeJudge(tokens=(10, 5)),
        )

        first = await runner.run_item(_item("a"))
        second = await runner.run_item(_item("b"))

        assert first.trace.cost.l5_tokens == second.trace.cost.l5_tokens == 110
        assert first.trace.cost.judge_tokens == second.trace.cost.judge_tokens == 45

    @pytest.mark.asyncio
    async def test_usage_is_not_estimated_when_absent(self):
        """usage가 없으면 0으로 남겨 미계측임이 드러나야 한다 (추정 금지)."""
        runner = EvalRunner(agent=_FakeAgent(), config=EvalConfig())

        result = await runner.run_item(_item())

        assert result.trace.cost.l5_tokens == 0


# =============================================================================
# d. 종합 점수와 게이트의 지표 정합
# =============================================================================


class TestOverallScoreUsesGateMetrics:
    @staticmethod
    def _score(**l2l3):
        return MetricAggregator().compute_overall_score(
            L1Metrics(entity_link_f1=1.0, concept_map_f1=1.0, constraint_extraction_f1=1.0),
            L2Metrics(
                context_recall_at_k=l2l3.get("chunk_recall", 0.0),
                context_precision_at_k=1.0,
                mrr=1.0,
                context_recall_at_k_concept=l2l3.get("concept_recall", 0.0),
            ),
            L3Metrics(
                hits_at_k=1.0,
                kg_edge_f1=l2l3.get("edge_f1", 0.0),
                kg_edge_recall=l2l3.get("edge_recall", 0.0),
            ),
            L4Metrics(constraint_violation_rate=0.0, type_consistency_rate=1.0),
            L5Metrics(answer_exact_match=1.0, answer_f1=1.0),
            ItemMetadata(requires_kg=l2l3.get("requires_kg", True)),
        )

    def test_concept_recall_moves_the_score(self):
        assert self._score(concept_recall=1.0) > self._score(concept_recall=0.0)

    def test_edge_recall_moves_the_score(self):
        assert self._score(edge_recall=1.0) > self._score(edge_recall=0.0)

    def test_chunk_recall_and_edge_f1_no_longer_move_the_score(self):
        """게이트가 보지 않는 지표는 종합 점수도 보지 않는다 (이중 기준 해소)."""
        base = self._score(concept_recall=0.5, edge_recall=0.5)
        assert self._score(concept_recall=0.5, edge_recall=0.5, chunk_recall=1.0) == base
        assert self._score(concept_recall=0.5, edge_recall=0.5, edge_f1=1.0) == base

    def test_concept_recall_used_for_non_kg_items_too(self):
        assert self._score(concept_recall=1.0, requires_kg=False) > self._score(
            concept_recall=0.0, requires_kg=False
        )


# =============================================================================
# 사이클 10: 크롤 DB 수치 사실과 데이터 시점 고정
# =============================================================================


class TestDataFactsAndAsOf:
    @pytest.mark.asyncio
    async def test_data_facts_reach_the_judge_context(self):
        from types import SimpleNamespace

        fact = {"type": "category_market", "category": "lip_care", "hhi": 0.0681}
        hybrid = SimpleNamespace(
            entities={}, rag_chunks=[], ontology_facts=[], inferences=[], metric_facts=[fact]
        )
        runner = EvalRunner(agent=_FakeAgent(), config=EvalConfig())

        trace = await runner._capture_trace(
            "lg049", {"response": "HHI는 0.0681입니다.", "hybrid_context": hybrid}, 0.0
        )

        assert trace.data_facts == [fact]
        assert "0.0681" in runner._build_context_string(trace)
        # KG 사실과 분리 — L3 엔티티 추출을 오염시키지 않는다
        assert trace.l3_kg_query.kg_entities_found == []

    def test_as_of_comes_from_snapshot_items(self):
        from eval.cli import resolve_data_as_of

        items = [
            _item("a", gold_source="snapshot", as_of="2026-08-31"),
            _item("b", gold_source="document"),
        ]
        assert resolve_data_as_of(items) == "2026-08-31"
        assert resolve_data_as_of(items, "2026-09-11") == "2026-09-11"
        assert resolve_data_as_of([_item("c")]) is None

    def test_conflicting_as_of_requires_explicit_choice(self):
        from eval.cli import resolve_data_as_of

        items = [
            _item("a", gold_source="snapshot", as_of="2026-08-31"),
            _item("b", gold_source="snapshot", as_of="2026-09-11"),
        ]
        with pytest.raises(ValueError):
            resolve_data_as_of(items)

"""규칙 추론 관측(트랙 3-B)이 평가 러너 → 스키마 → 리포트까지 이어지는지 검증 (트랙 3-C).

배경: 트랙 3-B가 검색기에서 `HybridContext.metadata["rule_evaluation"]`에 규칙 엔진
평가 결과({"combinations":..., "evaluated": int, "fired": [...], "non_fire_top":
[[label, count], ...], "non_fire_counts_by_kind": {...}})를 남긴다. 유형별 시험지의
rule 문항은 `metadata.rule_gold = {"rule_ids": [...], "expected_conclusion":
{"fires": bool}, ...}`를 가지지만, 예전에는 `ItemMetadata`가 이 키를 보존하지 않아
리포트만으로 규칙 정답을 알 수 없었다 (scripts/typed_eval_summary.py가 시험지
파일을 따로 읽어야 했다).

이 테스트는 실제 `EvalRunner` + `ReportGenerator`와 최소 가짜 에이전트(LLM 호출
없음)로 다음을 확인한다:
  1) `ItemMetadata.rule_gold`가 로더·러너를 거쳐 보존되는지
  2) `EvalTrace.rule_evaluation`이 `hybrid_context.metadata["rule_evaluation"]`을
     그대로 캡처하는지 (키가 없으면 None)
  3) `ItemResult.rule_agreement`가 규칙 정답과 실제 발동 규칙을 비교해 계산되는지
     (rule_gold가 없으면 None)
  4) `AggregateMetrics.rule_agreement_rate`/`rule_agreement_items`/
     `non_fire_reason_top`이 채점된 문항만 대상으로 집계되는지
  5) 마크다운 요약에 "## Rules" 섹션이 실리는지
  6) rule_evaluation/rule_gold가 없는 구형 report.json도 하위 호환으로 로딩되는지
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from eval.report import ReportGenerator
from eval.runner import EvalRunner
from eval.schemas import EvalConfig, EvalItem, EvalReport, ItemMetadata

RULE_GOLD_FIRES = {
    "rule_ids": ["low_sos_warning"],
    "expected_conclusion": {"fires": True},
}


class _FakeAgent:
    """inferences·hybrid_context.metadata만 그대로 돌려주는 최소 가짜 에이전트 (LLM 호출 없음)."""

    model = "gpt-4.1-mini"

    def __init__(
        self,
        inferences: list[dict[str, Any]] | None = None,
        hybrid_metadata: dict[str, Any] | None = None,
        should_raise: bool = False,
    ) -> None:
        self._inferences = inferences or []
        self._hybrid_metadata = hybrid_metadata
        self._should_raise = should_raise

    async def chat(self, question: str) -> dict[str, Any]:
        if self._should_raise:
            raise RuntimeError("agent_error probe")
        hybrid_context = (
            SimpleNamespace(metadata=self._hybrid_metadata)
            if self._hybrid_metadata is not None
            else None
        )
        return {
            "response": "테스트 답변",
            "inferences": self._inferences,
            "hybrid_context": hybrid_context,
        }


async def _run(agent: _FakeAgent, item_id: str, rule_gold: dict[str, Any] | None = None):
    runner = EvalRunner(agent=agent, config=EvalConfig())
    metadata = ItemMetadata(rule_gold=rule_gold) if rule_gold is not None else ItemMetadata()
    item = EvalItem(id=item_id, question="테스트 질문", metadata=metadata)
    return await runner.run_item(item)


@pytest.mark.asyncio
class TestItemMetadataRuleGoldPreserved:
    async def test_rule_gold_survives_run_item(self):
        result = await _run(
            _FakeAgent(inferences=[{"rule_name": "low_sos_warning"}]),
            "meta1",
            rule_gold=RULE_GOLD_FIRES,
        )
        assert result.metadata.rule_gold == RULE_GOLD_FIRES


@pytest.mark.asyncio
class TestEvalTraceRuleEvaluationCapture:
    async def test_rule_evaluation_captured_from_hybrid_context_metadata(self):
        rule_eval = {"evaluated": 3, "fired": ["low_sos_warning"], "non_fire_top": []}
        result = await _run(_FakeAgent(hybrid_metadata={"rule_evaluation": rule_eval}), "t1")

        assert result.trace is not None
        assert result.trace.rule_evaluation == rule_eval

    async def test_rule_evaluation_none_without_hybrid_context(self):
        """hybrid_context 자체가 없는 구형 에이전트 — 조용히 None."""
        result = await _run(_FakeAgent(), "t2")

        assert result.trace is not None
        assert result.trace.rule_evaluation is None

    async def test_rule_evaluation_none_when_key_missing_from_metadata(self):
        """hybrid_context는 있지만 metadata에 rule_evaluation 키가 없는 경우
        (트랙 3-B 미병합) — 기존 리포트 동작이 그대로 유지돼야 한다."""
        result = await _run(_FakeAgent(hybrid_metadata={}), "t3")

        assert result.trace is not None
        assert result.trace.rule_evaluation is None


@pytest.mark.asyncio
class TestItemResultRuleAgreement:
    async def test_agreement_when_gold_fires_and_rule_applied(self):
        result = await _run(
            _FakeAgent(inferences=[{"rule_name": "low_sos_warning"}]),
            "a1",
            rule_gold=RULE_GOLD_FIRES,
        )
        assert result.rule_agreement is True

    async def test_disagreement_when_gold_fires_but_no_rule_applied(self):
        result = await _run(
            _FakeAgent(
                inferences=[],
                hybrid_metadata={
                    "rule_evaluation": {"non_fire_top": [["conditions_not_met:hhi_below_0.15", 2]]}
                },
            ),
            "b1",
            rule_gold=RULE_GOLD_FIRES,
        )
        assert result.rule_agreement is False

    async def test_rule_agreement_none_without_rule_gold(self):
        result = await _run(
            _FakeAgent(inferences=[{"rule_name": "low_sos_warning"}]),
            "c1",
        )
        assert result.rule_agreement is None

    async def test_applied_rules_fallback_to_rule_evaluation_fired(self):
        """l4_ontology.applied_rules가 비어도 trace.rule_evaluation.fired로 판정한다."""
        result = await _run(
            _FakeAgent(
                inferences=[],
                hybrid_metadata={"rule_evaluation": {"fired": ["low_sos_warning"]}},
            ),
            "d1",
            rule_gold=RULE_GOLD_FIRES,
        )
        assert result.rule_agreement is True


@pytest.mark.asyncio
class TestReportRuleAgreementAggregate:
    async def test_rate_and_items_and_non_fire_reasons(self, tmp_path):
        agree = await _run(
            _FakeAgent(inferences=[{"rule_name": "low_sos_warning"}]),
            "ra1",
            rule_gold=RULE_GOLD_FIRES,
        )
        disagree = await _run(
            _FakeAgent(
                inferences=[],
                hybrid_metadata={
                    "rule_evaluation": {"non_fire_top": [["conditions_not_met:hhi_below_0.15", 2]]}
                },
            ),
            "ra2",
            rule_gold=RULE_GOLD_FIRES,
        )
        no_gold = await _run(
            _FakeAgent(inferences=[{"rule_name": "some_other_rule"}]),
            "ra3",
        )

        report = ReportGenerator().generate_report([agree, disagree, no_gold], tmp_path / "out")

        assert report.aggregates.rule_agreement_rate == 0.5
        assert report.aggregates.rule_agreement_items == 2
        assert report.aggregates.non_fire_reason_top == [("conditions_not_met:hhi_below_0.15", 2)]

    async def test_errored_item_excluded_from_rule_aggregates(self, tmp_path):
        agree = await _run(
            _FakeAgent(inferences=[{"rule_name": "low_sos_warning"}]),
            "ok1",
            rule_gold=RULE_GOLD_FIRES,
        )
        errored = await _run(
            _FakeAgent(should_raise=True),
            "bad1",
            rule_gold=RULE_GOLD_FIRES,
        )
        assert errored.trace is not None
        assert errored.trace.error is not None

        report = ReportGenerator().generate_report([agree, errored], tmp_path / "out")

        # 인프라 실패 문항은 rule_agreement_items/rate 어디에도 들어가지 않는다
        assert report.aggregates.rule_agreement_items == 1
        assert report.aggregates.rule_agreement_rate == 1.0
        assert report.aggregates.errored == 1

    async def test_markdown_summary_includes_rules_section(self, tmp_path):
        agree = await _run(
            _FakeAgent(inferences=[{"rule_name": "low_sos_warning"}]),
            "md1",
            rule_gold=RULE_GOLD_FIRES,
        )
        disagree = await _run(
            _FakeAgent(
                inferences=[],
                hybrid_metadata={
                    "rule_evaluation": {"non_fire_top": [["conditions_not_met:hhi_below_0.15", 2]]}
                },
            ),
            "md2",
            rule_gold=RULE_GOLD_FIRES,
        )

        ReportGenerator().generate_report([agree, disagree], tmp_path / "out")

        content = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
        assert "## Rules" in content
        assert "Rule agreement rate" in content
        assert "50.0%" in content
        assert "conditions_not_met:hhi_below_0.15" in content

    async def test_no_rules_section_when_no_rule_data(self, tmp_path):
        plain = await _run(_FakeAgent(), "plain1")

        ReportGenerator().generate_report([plain], tmp_path / "out")

        content = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
        assert "## Rules" not in content


class TestLegacyReportJsonCompat:
    def test_report_without_rule_observation_fields_still_loads(self):
        """rule_gold/rule_evaluation/rule_agreement 필드가 없는 구형 report.json도
        기본값(None/0/[])으로 로딩된다."""
        legacy = {
            "timestamp": "2026-01-01T00:00:00",
            "config": {},
            "aggregates": {
                "total": 1,
                "passed": 1,
                "failed": 0,
                "pass_rate": 1.0,
                "avg_overall_score": 0.9,
            },
            "items": [
                {
                    "item_id": "legacy1",
                    "question": "구형 문항",
                    "passed": True,
                    "overall_score": 0.9,
                    "trace": {"item_id": "legacy1"},
                }
            ],
        }

        report = EvalReport.model_validate(legacy)

        assert report.aggregates.rule_agreement_rate is None
        assert report.aggregates.rule_agreement_items == 0
        assert report.aggregates.non_fire_reason_top == []
        assert report.items[0].rule_agreement is None
        assert report.items[0].metadata.rule_gold is None
        assert report.items[0].trace is not None
        assert report.items[0].trace.rule_evaluation is None

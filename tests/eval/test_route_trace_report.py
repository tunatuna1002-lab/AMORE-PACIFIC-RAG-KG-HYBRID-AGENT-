"""route_trace(커밋 31040bf) 관측이 평가 러너 → 스키마 → 리포트까지 이어지는지 검증.

배경 (트랙 0-D2): `QueryGraph._finalize_route_trace`가 응답 `metadata["route_trace"]`에
남기는 문항별 경로 관측(route/confidence_level/...)을 `eval/brain_adapter.py`의
`BrainEvalAdapter.chat()`이 결과 dict에 동봉하게 됐다. 이 테스트는 실제 `EvalRunner`와
최소 가짜 에이전트(route_trace/inferences만 흉내)로:
  1) 러너가 `EvalTrace.route_trace`로 캡처하는지
  2) 리포트가 `route_counts`/`confidence_level_counts`/`react_items`/
     `rule_fired_items`/`rule_inference_total`로 집계하는지 (인프라 실패 문항은 제외)
  3) 마크다운 요약에 Route/Confidence 표가 실리는지
  4) route_trace 필드가 없는 구형 report.json도 하위 호환으로 로딩되는지
를 확인한다.
"""

from __future__ import annotations

from typing import Any

import pytest

from eval.report import ReportGenerator
from eval.runner import EvalRunner
from eval.schemas import EvalConfig, EvalItem, EvalReport


class _FakeAgent:
    """route_trace·inferences만 그대로 돌려주는 최소 가짜 에이전트 (LLM 호출 없음)."""

    model = "gpt-4.1-mini"

    def __init__(
        self,
        route_trace: dict[str, Any] | None,
        inferences: list[dict[str, Any]] | None = None,
        should_raise: bool = False,
    ) -> None:
        self._route_trace = route_trace
        self._inferences = inferences or []
        self._should_raise = should_raise

    async def chat(self, question: str) -> dict[str, Any]:
        if self._should_raise:
            raise RuntimeError("agent_error probe")
        return {
            "response": "테스트 답변",
            "route_trace": self._route_trace,
            "inferences": self._inferences,
        }


def _route_trace(route: str, confidence_level: str) -> dict[str, Any]:
    return {
        "route": route,
        "confidence_level": confidence_level,
        "confidence_score": 0.9,
        "confidence_components": {},
        "tools_used": [],
        "decision_tool": None,
        "is_complex": False,
    }


async def _run(agent: _FakeAgent, item_id: str):
    runner = EvalRunner(agent=agent, config=EvalConfig())
    return await runner.run_item(EvalItem(id=item_id, question="테스트 질문"))


@pytest.mark.asyncio
class TestRouteTraceCapture:
    async def test_runner_captures_route_trace_on_eval_trace(self):
        trace = _route_trace("direct", "high")
        result = await _run(_FakeAgent(trace), "t1")

        assert result.trace is not None
        assert result.trace.route_trace == trace

    async def test_runner_leaves_route_trace_none_without_v4_adapter(self):
        """v1 경로처럼 route_trace 키가 없는(None인) 에이전트는 None으로 남는다."""
        result = await _run(_FakeAgent(route_trace=None), "t2")

        assert result.trace is not None
        assert result.trace.route_trace is None


@pytest.mark.asyncio
class TestReportRouteAndConfidenceDistribution:
    async def test_route_counts_and_confidence_distribution(self, tmp_path):
        r1 = await _run(_FakeAgent(_route_trace("direct", "high")), "d1")
        r2 = await _run(_FakeAgent(_route_trace("direct", "high")), "d2")
        r3 = await _run(_FakeAgent(_route_trace("decide", "medium")), "dec1")

        report = ReportGenerator().generate_report([r1, r2, r3], tmp_path / "out")

        assert report.aggregates.route_counts == {"direct": 2, "decide": 1}
        assert report.aggregates.confidence_level_counts == {"high": 2, "medium": 1}
        assert report.aggregates.react_items == 0

    async def test_react_route_counted_in_react_items(self, tmp_path):
        r1 = await _run(_FakeAgent(_route_trace("react", "low")), "r1")

        report = ReportGenerator().generate_report([r1], tmp_path / "out")

        assert report.aggregates.route_counts == {"react": 1}
        assert report.aggregates.react_items == 1

    async def test_errored_item_excluded_from_route_distribution(self, tmp_path):
        scored = await _run(_FakeAgent(_route_trace("direct", "high")), "ok1")
        errored = await _run(_FakeAgent(route_trace=None, should_raise=True), "bad1")

        assert errored.trace is not None
        assert errored.trace.error is not None

        report = ReportGenerator().generate_report([scored, errored], tmp_path / "out")

        assert report.aggregates.route_counts == {"direct": 1}
        assert report.aggregates.errored == 1
        assert report.aggregates.error_item_ids == ["bad1"]

    async def test_rule_fired_items_and_inference_total(self, tmp_path):
        fired = await _run(
            _FakeAgent(
                _route_trace("decide", "medium"),
                inferences=[{"rule_name": "low_sos_warning"}, {"rule_name": "growth_alert"}],
            ),
            "rule1",
        )
        not_fired = await _run(_FakeAgent(_route_trace("direct", "high")), "rule2")

        report = ReportGenerator().generate_report([fired, not_fired], tmp_path / "out")

        assert report.aggregates.rule_fired_items == 1
        assert report.aggregates.rule_inference_total == 2

    async def test_markdown_summary_includes_route_confidence_table(self, tmp_path):
        r1 = await _run(_FakeAgent(_route_trace("direct", "high")), "md1")

        ReportGenerator().generate_report([r1], tmp_path / "out")

        content = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
        assert "## Route / Confidence" in content
        assert "| direct | 1 |" in content
        assert "| high | 1 |" in content


class TestLegacyReportJsonCompat:
    def test_report_without_route_trace_fields_still_loads(self):
        """route_trace/route_counts 등 새 필드가 없는 구형 report.json도 기본값으로 로딩된다."""
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
                    "trace": {
                        "item_id": "legacy1",
                    },
                }
            ],
        }

        report = EvalReport.model_validate(legacy)

        assert report.aggregates.route_counts == {}
        assert report.aggregates.confidence_level_counts == {}
        assert report.aggregates.react_items == 0
        assert report.aggregates.rule_fired_items == 0
        assert report.aggregates.rule_inference_total == 0
        assert report.items[0].trace is not None
        assert report.items[0].trace.route_trace is None

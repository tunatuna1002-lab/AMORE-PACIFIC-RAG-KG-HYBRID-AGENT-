"""답변 수치 검증(트랙 2-D) 관측이 평가 러너 → 스키마 → 리포트까지 이어지는지 검증.

배경: `ResponsePipeline`이 응답 `metadata["numeric_verification"]`에 수치 검증 결과
(verified/mismatch/no_citation/unknown_card/replaced 등)를 남긴다. `BrainEvalAdapter.chat()`이
이 값을 결과 dict에 동봉하고, 러너가 `EvalTrace.numeric_verification`으로 복사하며,
리포트가 채점된 문항만 합산한다. 가짜로 두는 것은 에이전트(LLM 경로)뿐이다.
"""

from __future__ import annotations

from typing import Any

import pytest

from eval.report import ReportGenerator
from eval.runner import EvalRunner
from eval.schemas import EvalConfig, EvalItem, EvalReport


class _FakeAgent:
    """numeric_verification만 그대로 돌려주는 최소 가짜 에이전트 (LLM 호출 없음)."""

    model = "gpt-4.1-mini"

    def __init__(self, numeric_verification: Any, should_raise: bool = False) -> None:
        self._nv = numeric_verification
        self._should_raise = should_raise

    async def chat(self, question: str) -> dict[str, Any]:
        if self._should_raise:
            raise RuntimeError("agent_error probe")
        result: dict[str, Any] = {"response": "테스트 답변"}
        if self._nv is not None:
            result["numeric_verification"] = self._nv
        return result


def _nv(
    verified: int = 0,
    mismatch: int = 0,
    no_citation: int = 0,
    unknown_card: int = 0,
    replaced: int = 0,
    found_in_other_cards: int = 0,
    skipped: str | None = None,
    mode: str = "annotate",
) -> dict[str, Any]:
    return {
        "mode": mode,
        "skipped": skipped,
        "checked": verified + mismatch + no_citation + unknown_card,
        "verified": verified,
        "mismatch": mismatch,
        "no_citation": no_citation,
        "unknown_card": unknown_card,
        "found_in_other_cards": found_in_other_cards,
        "replaced": replaced,
        "details": [],
    }


async def _run(agent: _FakeAgent, item_id: str):
    runner = EvalRunner(agent=agent, config=EvalConfig())
    return await runner.run_item(EvalItem(id=item_id, question="테스트 질문"))


@pytest.mark.asyncio
class TestNumericVerificationCapture:
    async def test_runner_copies_numeric_verification_to_trace(self):
        nv = _nv(verified=2, mismatch=1)
        result = await _run(_FakeAgent(nv), "n1")
        assert result.trace is not None
        assert result.trace.numeric_verification == nv

    async def test_missing_or_non_dict_is_none(self):
        r1 = await _run(_FakeAgent(None), "n2")
        r2 = await _run(_FakeAgent("bad"), "n3")
        assert r1.trace.numeric_verification is None
        assert r2.trace.numeric_verification is None


@pytest.mark.asyncio
class TestNumericVerificationAggregate:
    async def test_counts_summed_over_scored_items(self, tmp_path):
        r1 = await _run(_FakeAgent(_nv(verified=3, mismatch=1, found_in_other_cards=1)), "a1")
        r2 = await _run(_FakeAgent(_nv(verified=1, no_citation=2, unknown_card=1)), "a2")
        r3 = await _run(_FakeAgent(_nv(skipped="no_evidence")), "a3")
        r4 = await _run(_FakeAgent(None), "a4")
        report = ReportGenerator().generate_report([r1, r2, r3, r4], tmp_path / "out")
        agg = report.aggregates
        assert agg.numeric_verification_items == 3
        assert agg.numeric_verification_counts == {
            "checked": 8,
            "verified": 4,
            "mismatch": 1,
            "no_citation": 2,
            "unknown_card": 1,
            "replaced": 0,
            "found_in_other_cards": 1,
        }
        assert agg.numeric_verification_skipped == {"no_evidence": 1}
        assert agg.numeric_verification_items_with_unverified == 2

    async def test_errored_item_excluded(self, tmp_path):
        scored = await _run(_FakeAgent(_nv(verified=1)), "e1")
        errored = await _run(_FakeAgent(_nv(mismatch=5), should_raise=True), "e2")
        report = ReportGenerator().generate_report([scored, errored], tmp_path / "out")
        assert report.aggregates.numeric_verification_counts["mismatch"] == 0
        assert report.aggregates.numeric_verification_items == 1

    async def test_markdown_section_and_json_roundtrip(self, tmp_path):
        r1 = await _run(_FakeAgent(_nv(verified=2, mismatch=1, replaced=1, mode="enforce")), "m1")
        out = tmp_path / "out"
        report = ReportGenerator().generate_report([r1], out)
        md = (out / "summary.md").read_text(encoding="utf-8")
        assert "## Numeric Verification" in md
        assert "| mismatch | 1 |" in md
        loaded = EvalReport.model_validate_json((out / "report.json").read_text(encoding="utf-8"))
        assert loaded.aggregates.numeric_verification_counts["replaced"] == 1

    async def test_old_report_without_fields_loads(self, tmp_path):
        r1 = await _run(_FakeAgent(None), "o1")
        report = ReportGenerator().generate_report([r1], tmp_path / "out")
        data = report.model_dump(mode="json")
        for key in (
            "numeric_verification_items",
            "numeric_verification_counts",
            "numeric_verification_skipped",
            "numeric_verification_items_with_unverified",
        ):
            data["aggregates"].pop(key, None)
        for item in data["items"]:
            item["trace"].pop("numeric_verification", None)
        loaded = EvalReport.model_validate(data)
        assert loaded.aggregates.numeric_verification_items == 0
        assert loaded.items[0].trace.numeric_verification is None

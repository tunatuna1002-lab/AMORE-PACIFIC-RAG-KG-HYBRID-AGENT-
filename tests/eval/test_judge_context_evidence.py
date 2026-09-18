"""judge 근거성 컨텍스트를 증거 카드로 만드는 트랙 2-C 검증.

배경: `eval/runner.py::_build_context_string`은 judge(근거성 채점) 컨텍스트를 문서
스니펫 + `ontology_facts` 전부 + `data_facts`(DB 수치 사실) 전부로 만들었다. 그런데 v4
답변 프롬프트에는 DB 수치가 없고 KG 사실도 일부만 실린다 — judge가 답변 모델이 보지
못한 근거로 근거성을 채점하는 불일치였다 (2026-09 KG 실험: KG 사실을 컨텍스트에서만
빼도 근거성이 0.03~0.13 떨어지는 채점 효과가 확인됐다).

확정 설계 E1: 프롬프트 조립·출처·평가 트레이스·judge 컨텍스트가 증거 카드만 읽는다.
judge 컨텍스트는 답변 프롬프트에 실제로 실린 카드(`prompt_evidence`)만
`render_for_judge`로 렌더한다 — `render_for_prompt`와 정확히 같은 집합·같은 내용.

검증 대상:
1. `hybrid_context.prompt_evidence`가 있으면(트랙 2-B 병합 후) judge가 받는 컨텍스트
   문자열 == `render_for_judge(prompt_evidence)`, `trace.evidence`의 id 목록 ==
   prompt_evidence id 목록, `judge_context_source == "evidence"`.
2. `prompt_evidence` 속성이 아예 없는 에이전트(2-B 미병합, v1 구형)는 예전 문자열
   그대로(특성화 테스트), `judge_context_source == "legacy"`.
3. v4 어댑터 `_build_trace`: ReAct 관찰이 있는 스텝만 observation 카드로
   `prompt_evidence` 끝에 추가되고, 예전처럼 `metric_facts`에 dict로 끼워 넣지 않는다.
4. `evidence`/`evidence_all_count`/`judge_context_source` 필드가 없는 구형 report.json도
   기본값으로 로딩된다.

실제 `EvalRunner`·`evidence_renderer`·`Evidence` 모델을 쓰고, judge와 에이전트만
가짜로 둔다 (실제 API 호출 없음).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from eval.brain_adapter import BrainEvalAdapter
from eval.runner import EvalRunner
from eval.schemas import EvalConfig, EvalItem, EvalReport, GoldEvidence
from src.domain.entities.evidence import Evidence, EvidenceKind, EvidenceUnit
from src.rag.evidence_renderer import render_for_judge


def _metric_card(subject: str, value: float) -> Evidence:
    return Evidence.create(
        kind=EvidenceKind.METRIC,
        subject=subject,
        predicate="sos",
        object="lip_care",
        value=value,
        unit=EvidenceUnit.RATIO,
        as_of="2026-09-01",
        source="sqlite:brand_metrics",
        text="unused for metric cards",
    )


def _relation_card(subject: str, obj: str) -> Evidence:
    return Evidence.create(
        kind=EvidenceKind.RELATION,
        subject=subject,
        predicate="competesWith",
        object=obj,
        source="kg:competesWith",
        text=f"{subject}는 {obj}와 경쟁한다",
    )


def _document_card(chunk_id: str) -> Evidence:
    return Evidence.create(
        kind=EvidenceKind.DOCUMENT,
        subject=chunk_id,
        predicate="states",
        source="rag:playbook",
        text="SoS는 카테고리 내 판매 점유율이다",
        detail="SoS(Share of Shelf)는 카테고리 Top100 안에서 브랜드가 차지하는 비중이다.",
    )


def _inference_card(subject: str) -> Evidence:
    return Evidence.create(
        kind=EvidenceKind.INFERENCE,
        subject=subject,
        predicate="marketPosition",
        source="rule:low_sos_check",
        text=f"{subject}는 저점유 위험군이다",
    )


class _RecordingJudge:
    """score_groundedness가 실제로 받은 context를 기록하는 가짜 judge (API 호출 없음)."""

    on_usage = None

    def __init__(self) -> None:
        self.groundedness_calls: list[str] = []

    async def score_groundedness(self, answer: str, context: str) -> float:
        self.groundedness_calls.append(context)
        return 0.9

    async def score_relevance(self, answer: str, question: str) -> float:
        return 0.9

    async def score_factuality(self, answer: str, facts: list[str]):
        return 0.9, []


def _hybrid_ctx(**overrides) -> SimpleNamespace:
    base = {
        "entities": {"brands": ["laneige"], "categories": ["lip_care"]},
        "rag_chunks": [],
        "ontology_facts": [],
        "inferences": [],
        "metadata": {},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _EvidenceAgent:
    """hybrid_context를 그대로 동봉해 돌려주는 최소 가짜 에이전트 (LLM 호출 없음)."""

    model = "gpt-4.1-mini"

    def __init__(self, hybrid_context) -> None:
        self._ctx = hybrid_context

    async def chat(self, question: str) -> dict:
        return {
            "response": "LANEIGE Lip Care SoS는 3.2%다.",
            "hybrid_context": self._ctx,
            "llm_usage": {},
        }


def _item(item_id: str = "q1") -> EvalItem:
    return EvalItem(id=item_id, question="LANEIGE Lip Care SoS는?", gold=GoldEvidence())


@pytest.mark.asyncio
async def test_judge_receives_exactly_the_prompt_evidence_rendering():
    prompt_cards = [
        _metric_card("laneige", 0.032),
        _relation_card("laneige", "cosrx"),
        _document_card("doc::sos-def"),
    ]
    all_cards = [*prompt_cards, _inference_card("laneige")]  # 선별에서 빠진 4번째 카드
    ctx = _hybrid_ctx(evidence=all_cards, prompt_evidence=prompt_cards)

    judge = _RecordingJudge()
    runner = EvalRunner(
        agent=_EvidenceAgent(ctx), config=EvalConfig(target="v4", use_judge=True), judge=judge
    )

    result = await runner.run_item(_item())

    assert judge.groundedness_calls == [render_for_judge(prompt_cards)]
    assert result.l5.groundedness_score == 0.9
    assert result.trace.judge_context_source == "evidence"
    assert [d["id"] for d in result.trace.evidence] == [c.id for c in prompt_cards]
    assert result.trace.evidence_all_count == 4


@pytest.mark.asyncio
async def test_no_prompt_evidence_attribute_falls_back_to_legacy_context():
    """prompt_evidence 속성이 아예 없는 에이전트(트랙 2-B 미병합, v1 구형)는 예전 문자열 그대로."""
    legacy_ctx = SimpleNamespace(
        entities={"brands": ["laneige"], "categories": ["lip_care"]},
        rag_chunks=[{"id": "chunk1", "text": "SoS 정의 스니펫"}],
        ontology_facts=[{"subject": "laneige", "predicate": "competesWith", "object": "cosrx"}],
        inferences=[],
        metadata={},
        metric_facts=[{"type": "category_market", "hhi": 0.2}],
    )
    assert not hasattr(legacy_ctx, "prompt_evidence")
    assert not hasattr(legacy_ctx, "evidence")

    judge = _RecordingJudge()
    runner = EvalRunner(
        agent=_EvidenceAgent(legacy_ctx),
        config=EvalConfig(target="v4", use_judge=True),
        judge=judge,
    )

    result = await runner.run_item(_item("q2"))

    # 변경 전 _build_context_string과 동등한 특성화 문자열 — 스니펫 + KG 사실 + data_facts
    expected = "\n\n".join(
        [
            "SoS 정의 스니펫",
            str({"subject": "laneige", "predicate": "competesWith", "object": "cosrx"}),
            str({"type": "category_market", "hhi": 0.2}),
        ]
    )
    assert judge.groundedness_calls == [expected]
    assert result.l5.groundedness_score == 0.9
    assert result.trace.judge_context_source == "legacy"
    assert result.trace.evidence == []
    assert result.trace.evidence_all_count == 0


def test_react_observations_appended_to_prompt_evidence_tail_not_metric_facts():
    """v4 어댑터 `_build_trace`: 관찰이 있는 ReAct 스텝만 observation 카드로 만들어
    `prompt_evidence` 끝에 붙는다. 예전처럼 `metric_facts`에 dict를 끼워 넣지 않는다."""
    existing_card = _metric_card("laneige", 0.05)
    ctx = _hybrid_ctx(
        metric_facts=[{"type": "category_market", "hhi": 0.2}],
        evidence=[existing_card],
        prompt_evidence=[existing_card],
    )
    holder = {
        "hybrid_context": ctx,
        "react_steps": [
            {
                "action": "query_knowledge_graph",
                "action_input": {"entity": "laneige"},
                "observation": "라네즈는 코스알엑스와 경쟁한다",
            },
            {
                "action": "get_metric",
                "action_input": {"metric": "sos"},
                "observation": "SoS 3.2%",
            },
            {
                # final_answer는 관찰이 아니라 최종 답 — 카드로 만들지 않는다
                "action": "final_answer",
                "action_input": {"answer": "답"},
                "observation": "",
            },
        ],
    }

    trace = BrainEvalAdapter._build_trace("질문", holder)

    assert [f for f in trace.metric_facts if f.get("type") == "react_observation"] == []
    assert trace.metric_facts == [{"type": "category_market", "hhi": 0.2}]
    assert trace.prompt_evidence[0] is existing_card
    observation_cards = trace.prompt_evidence[1:]
    assert len(observation_cards) == 2
    assert [c.kind for c in observation_cards] == [
        EvidenceKind.OBSERVATION,
        EvidenceKind.OBSERVATION,
    ]
    assert [c.subject for c in observation_cards] == ["query_knowledge_graph", "get_metric"]
    # evidence(검색이 만든 전체 집합)에는 도구 관찰을 넣지 않는다 — 출처가 다르다
    assert trace.evidence == [existing_card]


def test_legacy_report_json_without_evidence_fields_still_loads():
    """evidence/evidence_all_count/judge_context_source가 없는 구형 report.json도
    기본값으로 로딩된다 (트랙 0-D2의 route_trace 하위 호환 테스트와 같은 패턴)."""
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

    trace = report.items[0].trace
    assert trace is not None
    assert trace.evidence == []
    assert trace.evidence_all_count == 0
    assert trace.judge_context_source == "legacy"

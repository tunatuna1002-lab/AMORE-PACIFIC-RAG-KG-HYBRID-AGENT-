"""
v4 Brain 평가 어댑터 검증 (결정 D3)

- 동시 실행에서도 문항별 검색 트레이스가 섞이지 않는다
- litellm 사용량이 문항별로 적립된다 (mock_response — 네트워크 없음)
- ReAct 도구 관찰이 observation 증거 카드로 만들어져 prompt_evidence로 넘어간다 (트랙 2-C)
- CLI --target 인자와 리포트 설정 기록

실제 UnifiedBrain·QueryGraph·EvalRunner를 쓰고, 검색 백엔드와 LLM 호출 지점만 가짜로 둔다.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import litellm
import pytest

from eval.brain_adapter import BrainEvalAdapter
from eval.runner import EvalRunner
from eval.schemas import EvalConfig
from src.core.models import Decision, Response
from src.domain.entities.evidence import EvidenceKind
from src.infrastructure.feature_flags import FeatureFlags
from src.rag.hybrid_retriever import HybridContext, HybridRetriever

QUESTIONS = [
    "LANEIGE Lip Care SoS는?",
    "LANEIGE Skin Care 순위는?",
    "LANEIGE Face Powder HHI는?",
]


@pytest.fixture
def flags(monkeypatch, tmp_path):
    data_path = tmp_path / "dashboard_data.json"
    data_path.write_text(json.dumps({"brand": {"competitors": []}}), encoding="utf-8")
    monkeypatch.setenv("DASHBOARD_DATA_PATH", str(data_path))

    def _set(react: bool = False) -> None:
        monkeypatch.setenv("FF_AGENTS_USE_REACT_AGENT", "true" if react else "false")
        monkeypatch.setenv("FF_RETRIEVER_USE_OWL_STRATEGY", "false")
        FeatureFlags.reset_instance()

    yield _set
    FeatureFlags.reset_instance()


async def _adapter() -> BrainEvalAdapter:
    from src.core.brain import UnifiedBrain

    brain = UnifiedBrain()
    with patch.object(HybridRetriever, "initialize", AsyncMock()):
        await brain.initialize()
    return BrainEvalAdapter(brain)


def _fake_retrieve(delays: dict[str, float]):
    async def retrieve(query: str, **kwargs):
        await asyncio.sleep(delays[query])
        return HybridContext(
            query=query,
            entities={"brands": ["laneige"], "categories": [query]},
            rag_chunks=[{"id": f"chunk::{query}", "content": query}],
            metric_facts=[{"type": "probe", "query": query}],
        )

    return retrieve


@pytest.mark.asyncio
async def test_concurrent_items_keep_their_own_trace_and_usage(flags):
    flags(react=False)
    adapter = await _adapter()
    brain = adapter.brain
    retriever = brain._context_gatherer.retriever
    # 먼저 들어온 문항이 가장 늦게 끝나게 해서 공유 상태였다면 덮어써지도록 한다
    delays = {q: 0.03 * (len(QUESTIONS) - i) for i, q in enumerate(QUESTIONS)}
    retriever.retrieve = _fake_retrieve(delays)

    async def generate(query, context, decision, tool_result=None):
        # 실제 litellm 경로를 타되 네트워크 없이 — 사용량 콜백 검증용
        await litellm.acompletion(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": query}],
            mock_response=f"answer:{query}",
        )
        return Response(text=f"answer:{query}", confidence_score=0.5)

    decide = AsyncMock(return_value=Decision(tool="direct_answer", tool_params={}))
    with (
        patch.object(brain._response_pipeline, "generate", side_effect=generate),
        patch.object(brain.decision_maker, "decide", decide),
    ):
        await adapter.initialize()
        results = await asyncio.gather(*(adapter.chat(q) for q in QUESTIONS))

    for question, result in zip(QUESTIONS, results, strict=True):
        assert result["response"] == f"answer:{question}"
        ctx = result["hybrid_context"]
        assert ctx.rag_chunks[0]["id"] == f"chunk::{question}"
        assert ctx.metric_facts == [{"type": "probe", "query": question}]
        assert ctx.retriever_type == "legacy"
        assert result["llm_usage"]["calls"] == 1
        assert result["llm_usage"]["prompt_tokens"] > 0


@pytest.mark.asyncio
async def test_runner_scores_v4_trace_with_same_schema(flags):
    flags(react=False)
    adapter = await _adapter()
    brain = adapter.brain
    brain._context_gatherer.retriever.retrieve = _fake_retrieve({QUESTIONS[0]: 0})

    with (
        patch.object(
            brain._response_pipeline,
            "generate",
            AsyncMock(return_value=Response(text="LANEIGE SoS 3%", confidence_score=0.5)),
        ),
        patch.object(
            brain.decision_maker,
            "decide",
            AsyncMock(return_value=Decision(tool="direct_answer", tool_params={})),
        ),
    ):
        await adapter.initialize()
        runner = EvalRunner(agent=adapter, config=EvalConfig(target="v4"))
        result = await adapter.chat(QUESTIONS[0])
        trace = await runner._capture_trace("lgX", result, start_time=0.0)

    assert trace.l2_doc_retrieval.chunk_ids == [f"chunk::{QUESTIONS[0]}"]
    assert trace.l1_entity_linking.extracted_brands == ["laneige"]
    assert trace.data_facts == [{"type": "probe", "query": QUESTIONS[0]}]
    assert trace.l5_answer.final_answer == "LANEIGE SoS 3%"


@pytest.mark.asyncio
async def test_react_observations_become_prompt_evidence_cards(flags):
    flags(react=True)
    adapter = await _adapter()
    brain = adapter.brain
    question = "LANEIGE 경쟁사와 비교해서 점유율이 왜 달라졌는지 분석해줘"

    def reply(payload):
        content = json.dumps(payload, ensure_ascii=False)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    llm = AsyncMock(
        side_effect=[
            reply(
                {
                    "thought": "경쟁사 확인",
                    "action": "query_knowledge_graph",
                    "action_input": {"entity": "LANEIGE", "relation": "competitors"},
                }
            ),
            reply({"thought": "끝", "action": "final_answer", "action_input": {"answer": "답"}}),
            reply({"quality_score": 0.8, "needs_improvement": False}),
        ]
    )
    thin = HybridContext(query=question, entities={"brands": ["laneige"]})
    leaked = AssertionError("ReAct 경로를 타지 않았다")
    with (
        patch.object(brain._context_gatherer.retriever, "retrieve", AsyncMock(return_value=thin)),
        patch("src.core.react_agent.acompletion", llm),
        patch.object(brain.decision_maker, "decide", AsyncMock(side_effect=leaked)),
        patch.object(brain._response_pipeline, "generate", AsyncMock(side_effect=leaked)),
    ):
        await adapter.initialize()
        result = await adapter.chat(question)

    assert result["query_type"] == "react"
    assert result["response"] == "답"
    trace = result["hybrid_context"]
    # 옛 dict 삽입은 더 이상 없다 — observation은 증거 카드로만 실린다 (트랙 2-C)
    assert [f for f in trace.metric_facts if f.get("type") == "react_observation"] == []
    observation_cards = [c for c in trace.prompt_evidence if c.kind == EvidenceKind.OBSERVATION]
    assert [c.subject for c in observation_cards] == ["query_knowledge_graph"]
    assert "LANEIGE" in observation_cards[0].text


def test_cli_target_argument_defaults_to_v1():
    from eval.cli import parse_args

    assert parse_args(["run", "--dataset", "x.jsonl"]).target == "v1"
    assert parse_args(["run", "--dataset", "x.jsonl", "--target", "v4"]).target == "v4"


def test_eval_config_records_target_and_commit():
    config = EvalConfig(target="v4", git_commit="abc1234")
    dumped = config.model_dump()
    assert dumped["target"] == "v4"
    assert dumped["git_commit"] == "abc1234"

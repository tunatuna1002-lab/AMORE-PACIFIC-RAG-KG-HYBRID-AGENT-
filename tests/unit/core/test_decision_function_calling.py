"""
DecisionMaker 네이티브 function calling → 레지스트리 도구 실행 → 답변 프롬프트 카드 (트랙 4-A)

가짜는 LLM 호출뿐이다: 결정 LLM(``src.core.decision_maker.acompletion``)과 답변·환각 점검 LLM
(``litellm.acompletion``). 레지스트리·ToolCoordinator·QueryGraph 노드·ResponsePipeline·
SQLite·KG·규칙 추론기는 실제 객체다 (문서 검색기만 ``FakeDocRetriever``).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.core.cache import ResponseCache
from src.core.confidence import ConfidenceAssessor
from src.core.decision_maker import DecisionMaker
from src.core.graph_state import QueryState
from src.core.models import ConfidenceLevel, Context, SystemState
from src.core.query_graph import QueryGraph
from src.core.response_pipeline import ResponsePipeline
from src.core.tool_coordinator import ToolCoordinator
from src.core.tool_registry import TOOL_NAMES, ToolRegistry, function_schemas, tool_evidence
from src.domain.entities.evidence import EvidenceKind
from src.infrastructure.feature_flags import FeatureFlags
from src.rag.evidence_renderer import CITATION_INSTRUCTION
from tests.unit.rag.evidence_pipeline_fixtures import AS_OF, make_retriever

QUERY = "LANEIGE 립케어 점유율 알려줘"


def _tool_call_reply(name: str, arguments: str | dict, content: str | None = None):
    args = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    call = SimpleNamespace(
        id="call_1", type="function", function=SimpleNamespace(name=name, arguments=args)
    )
    message = SimpleNamespace(content=content, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def _text_reply(content: str):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def _system_state(tools: list[str]) -> dict:
    return {
        "data_status": "최신",
        "mode": "responding",
        "available_tools": tools,
        "failed_tools": [],
    }


@pytest.fixture(autouse=True)
def isolated_flags(monkeypatch):
    monkeypatch.delenv("AMORE_DATA_AS_OF", raising=False)
    FeatureFlags.reset_instance()
    yield
    FeatureFlags.reset_instance()


@pytest.fixture
def registry(tmp_path):
    return ToolRegistry(make_retriever(tmp_path))


def _context() -> Context:
    return Context(query=QUERY, entities={}, system_state=SystemState())


# ── DecisionMaker ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_decide_uses_native_tools_and_executes_registry_tool(registry):
    llm = AsyncMock(return_value=_tool_call_reply("get_metrics", {"brand": "LANEIGE"}))
    available = registry.get_available_tools()

    with patch("src.core.decision_maker.acompletion", llm):
        decision = await DecisionMaker().decide(QUERY, _context(), _system_state(available), "low")

    kwargs = llm.await_args.kwargs
    assert kwargs["tools"] == function_schemas(available)
    assert kwargs["tool_choice"] == "auto"
    assert decision.tool == "get_metrics"
    assert decision.tool_params == {"brand": "LANEIGE"}
    assert decision.requires_tool()

    coordinator = ToolCoordinator(tool_executor=registry, cache=ResponseCache())
    result = await coordinator.execute(decision.tool, decision.tool_params)

    assert result.success, result.error
    cards = tool_evidence(result)
    assert cards and {c.kind for c in cards} == {EvidenceKind.METRIC}
    assert {c.as_of for c in cards} == {AS_OF}


@pytest.mark.asyncio
async def test_decide_prompt_no_longer_requests_json_decision():
    llm = AsyncMock(return_value=_text_reply("- 컨텍스트로 충분"))

    with patch("src.core.decision_maker.acompletion", llm):
        await DecisionMaker().decide(QUERY, _context(), _system_state(list(TOOL_NAMES)), "medium")

    prompt = "\n".join(m["content"] for m in llm.await_args.kwargs["messages"])
    assert '"tool"' not in prompt
    assert "JSON" not in prompt


@pytest.mark.asyncio
async def test_decide_without_tool_call_is_direct_answer_with_key_points():
    llm = AsyncMock(return_value=_text_reply("근거가 충분합니다.\n- SoS 2%\n- HHI 0.0681"))

    with patch("src.core.decision_maker.acompletion", llm):
        decision = await DecisionMaker().decide(
            QUERY, _context(), _system_state(list(TOOL_NAMES)), "high"
        )

    assert decision.tool == "direct_answer"
    assert not decision.requires_tool()
    assert decision.key_points == ["SoS 2%", "HHI 0.0681"]
    assert "근거가 충분합니다." in decision.reason


@pytest.mark.asyncio
async def test_decide_without_available_tools_does_not_send_tools():
    llm = AsyncMock(return_value=_text_reply("도구 없음"))

    with patch("src.core.decision_maker.acompletion", llm):
        decision = await DecisionMaker().decide(QUERY, _context(), _system_state([]), "low")

    assert "tools" not in llm.await_args.kwargs
    assert "tool_choice" not in llm.await_args.kwargs
    assert decision.tool == "direct_answer"


@pytest.mark.asyncio
async def test_decide_rejects_tool_that_was_not_offered():
    llm = AsyncMock(return_value=_tool_call_reply("apply_rules", {"brand": "LANEIGE"}))

    with patch("src.core.decision_maker.acompletion", llm):
        decision = await DecisionMaker().decide(
            QUERY, _context(), _system_state(["get_metrics"]), "low"
        )

    assert decision.tool == "direct_answer"
    assert "apply_rules" in decision.reason


@pytest.mark.asyncio
async def test_decide_with_malformed_arguments_falls_back():
    llm = AsyncMock(return_value=_tool_call_reply("get_metrics", "{brand: LANEIGE"))

    with patch("src.core.decision_maker.acompletion", llm):
        decision = await DecisionMaker().decide(
            QUERY, _context(), _system_state(list(TOOL_NAMES)), "low"
        )

    assert decision.tool == "direct_answer"
    assert decision.tool_params == {}


@pytest.mark.asyncio
async def test_decide_llm_error_falls_back():
    llm = AsyncMock(side_effect=RuntimeError("API down"))

    with patch("src.core.decision_maker.acompletion", llm):
        decision = await DecisionMaker().decide(
            QUERY, _context(), _system_state(list(TOOL_NAMES)), "low"
        )

    assert decision.tool == "direct_answer"
    assert "API down" in decision.reason


# ── QueryGraph decide 경로 → 답변 프롬프트 카드 ─────────────────────


class _AnswerLLM:
    """litellm.acompletion 대역 (답변·환각 점검): 호출 메시지를 기록한다."""

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs["messages"])
        return _text_reply("LANEIGE의 Lip Care SoS는 2%입니다.")


@pytest.mark.asyncio
async def test_decide_path_tool_cards_reach_prompt_sources_and_route_trace(registry, monkeypatch):
    import litellm

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    answer_llm = _AnswerLLM()
    monkeypatch.setattr(litellm, "acompletion", answer_llm)
    coordinator = ToolCoordinator(tool_executor=registry, cache=ResponseCache())
    graph = QueryGraph(
        cache=ResponseCache(),
        context_gatherer=None,
        confidence_assessor=ConfidenceAssessor(),
        decision_maker=DecisionMaker(),
        tool_coordinator=coordinator,
        response_pipeline=ResponsePipeline(),
    )
    state = QueryState(query=QUERY, system_state=_system_state(coordinator.get_available_tools()))
    state.context = _context()  # 검색 카드 없음 — 출처는 도구 카드에서만 나올 수 있다
    state.confidence_level = ConfidenceLevel.LOW
    decision_llm = AsyncMock(
        return_value=_tool_call_reply("get_metrics", {"brand": "LANEIGE", "category": "lip_care"})
    )

    with patch("src.core.decision_maker.acompletion", decision_llm):
        state = await graph._node_decide(state)
    assert graph._route_after_decide(state) == "execute_tool"
    state = await graph._node_execute_tool(state)
    state = await graph._node_generate_response(state)
    state = graph._finalize_route_trace(state, "decide")

    cards = tool_evidence(state.tool_result)
    assert cards
    answer_calls = [
        messages
        for messages in answer_llm.calls
        if any("[도구 실행 결과]" in m["content"] for m in messages)
    ]
    assert len(answer_calls) == 1
    prompt = "\n".join(m["content"] for m in answer_calls[0])
    for card in cards:
        assert f"[{card.id}]" in prompt
    assert prompt.count(CITATION_INSTRUCTION) == 1
    response = state.response
    assert response.tools_called == ["get_metrics"]
    assert f"sqlite:brand_metrics ({AS_OF})" in response.sources
    trace = response.metadata["route_trace"]
    assert trace["decision_tool"] == "get_metrics"
    assert trace["tools_used"] == [{"tool": "get_metrics", "executed": True}]
    # 원래 컨텍스트 객체는 바꾸지 않는다
    assert state.context.prompt_evidence == []

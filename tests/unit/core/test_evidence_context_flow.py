"""증거 카드가 v4 답변 경로 끝까지 전달된다 (트랙 2-B).

retrieve_unified → ContextGatherer.gather → ResponsePipeline 메시지·출처.
검색기·KG·SQLite·어댑터·렌더러·규칙 추론기는 실제 객체이고, 가짜는 문서 검색기와
LLM 호출(litellm.acompletion)뿐이다. v1 챗봇 프롬프트 조립부도 같은 카드를 쓰는지 본다.
"""

from types import SimpleNamespace

from src.core.context_gatherer import ContextGatherer
from src.core.models import Context, SystemState, ToolResult
from src.core.response_pipeline import ResponsePipeline
from src.rag.evidence_renderer import CITATION_INSTRUCTION, render_for_prompt
from tests.unit.rag.evidence_pipeline_fixtures import (
    CURRENT_METRICS,
    QUERY,
    make_retriever,
)


class _Recorder:
    """litellm.acompletion 대역: 메시지를 기록하고 고정 답을 돌려준다."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs["messages"])
        message = SimpleNamespace(content="lip_care HHI는 0.0681입니다.")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


async def test_gather_passes_cards_and_summary_through(tmp_path):
    retriever = make_retriever(tmp_path)
    unified = await retriever.retrieve_unified(QUERY, current_metrics=CURRENT_METRICS)
    gatherer = ContextGatherer(hybrid_retriever=retriever)

    context = await gatherer.gather(QUERY, current_metrics=CURRENT_METRICS)

    assert [c.id for c in context.prompt_evidence] == [c.id for c in unified.prompt_evidence]
    assert [c.id for c in context.evidence] == [c.id for c in unified.evidence]
    assert context.summary.startswith("[시스템 상태]")
    assert context.summary.endswith(render_for_prompt(context.prompt_evidence))
    assert "0.0681" in context.summary
    assert "42.4" not in context.summary


async def test_gather_without_cards_does_not_render_kg_facts(tmp_path):
    retriever = make_retriever(tmp_path)

    class NoCardRetriever:
        kg = retriever.kg

        async def initialize(self):
            return None

        async def retrieve(self, query, current_metrics=None, include_explanations=True):
            ctx = await retriever.retrieve(query, current_metrics=current_metrics)
            ctx.evidence, ctx.prompt_evidence, ctx.combined_context = [], [], ""
            return ctx

    context = await ContextGatherer(hybrid_retriever=NoCardRetriever()).gather(QUERY)

    assert context.kg_facts  # 원자료는 남지만
    assert "42.4" not in context.summary  # KG 메타데이터 수치를 카드 밖에서 렌더링하지 않는다
    assert "[관련 정보]" not in context.summary


async def test_response_messages_carry_citation_instruction_exactly_once(tmp_path):
    retriever = make_retriever(tmp_path)
    context = await ContextGatherer(hybrid_retriever=retriever).gather(
        QUERY, current_metrics=CURRENT_METRICS
    )

    messages = ResponsePipeline()._build_messages(QUERY, context)
    joined = "\n".join(m["content"] for m in messages)

    assert joined.count(CITATION_INSTRUCTION) == 1
    assert context.summary in joined


def test_no_citation_instruction_without_cards():
    context = Context(query="안녕", summary="[Retrieval skipped: greeting_or_command]")

    messages = ResponsePipeline()._build_messages("안녕", context)

    assert all(CITATION_INSTRUCTION not in m["content"] for m in messages)


async def test_generate_and_fast_path_prompts_cite_once_and_sources_are_strings(
    tmp_path, monkeypatch
):
    import litellm

    recorder = _Recorder()
    monkeypatch.setattr(litellm, "acompletion", recorder)
    retriever = make_retriever(tmp_path)
    context = await ContextGatherer(hybrid_retriever=retriever).gather(
        QUERY, current_metrics=CURRENT_METRICS
    )
    pipeline = ResponsePipeline(openai_client=object())

    response = await pipeline.generate(QUERY, context)
    fast = await pipeline._call_llm_fast(QUERY, context)

    assert fast and response.text
    assert len(recorder.calls) == 2
    for messages in recorder.calls:
        assert "\n".join(m["content"] for m in messages).count(CITATION_INSTRUCTION) == 1
    assert isinstance(response.sources, list)
    assert all(isinstance(source, str) for source in response.sources)
    assert "sqlite:market_metrics (2026-08-31)" in response.sources
    assert "KG" in response.sources
    assert "HHI 해석 가이드" in response.sources
    assert any(source.startswith("rule:") for source in response.sources)
    assert len(response.sources) == len(set(response.sources))


def test_sources_are_empty_without_cards():
    context = Context(
        query="q",
        rag_docs=[{"metadata": {"title": "문서"}}],
        kg_inferences=[{"insight": "x"}],
        system_state=SystemState(),
    )

    assert ResponsePipeline()._extract_sources(context) == []


async def test_tool_result_context_keeps_cards(tmp_path, monkeypatch):
    retriever = make_retriever(tmp_path)
    context = await ContextGatherer(hybrid_retriever=retriever).gather(
        QUERY, current_metrics=CURRENT_METRICS
    )
    pipeline = ResponsePipeline()
    seen = {}

    async def capture(query, ctx, decision=None, tool_result=None):
        seen["context"] = ctx
        return None

    monkeypatch.setattr(pipeline, "generate", capture)
    await pipeline.generate_with_tool_result(
        QUERY, context, ToolResult(tool_name="query_data", success=True, data={})
    )

    assert [c.id for c in seen["context"].prompt_evidence] == [
        c.id for c in context.prompt_evidence
    ]
    assert seen["context"].evidence == context.evidence


# ----------------------------------------------------------------------
# v1 챗봇 프롬프트 조립부
# ----------------------------------------------------------------------


async def test_v1_chatbot_prompt_cites_same_cards_once(tmp_path, monkeypatch):
    from src.agents import hybrid_chatbot_agent as module
    from src.agents.hybrid_chatbot_agent import HybridChatbotAgent
    from src.rag.context_builder import ContextBuilder
    from src.rag.router import QueryType

    recorder = _Recorder()
    monkeypatch.setattr(module, "acompletion", recorder)
    retriever = make_retriever(tmp_path)
    agent = HybridChatbotAgent(knowledge_graph=retriever.kg, reasoner=retriever.reasoner)
    agent.hybrid_retriever = retriever

    hybrid_context = await retriever.retrieve(QUERY, current_metrics=CURRENT_METRICS)
    agent._last_hybrid_context = hybrid_context
    context_text = ContextBuilder(max_tokens=3000).build(hybrid_context, None, QUERY, retriever.kg)

    await agent._generate_response(
        user_message=QUERY,
        query_type=QueryType.DATA_QUERY,
        context=context_text,
        inferences=hybrid_context.inferences,
        prompt_evidence=hybrid_context.prompt_evidence,
    )

    user_prompt = recorder.calls[0][1]["content"]
    assert user_prompt.count(CITATION_INSTRUCTION) == 1
    for card in hybrid_context.prompt_evidence:
        assert f"[{card.id}]" in user_prompt
    # 추론·계층 정보를 카드 밖에서 다시 렌더링하지 않는다
    assert "## 온톨로지 추론 결과" not in user_prompt
    assert "## 카테고리 계층 정보" not in user_prompt
    for inference in hybrid_context.inferences:
        assert user_prompt.count(inference.insight) == 1

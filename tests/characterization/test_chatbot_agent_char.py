"""
Characterization: src.agents.hybrid_chatbot_agent.HybridChatbotAgent.chat

Collaborators:
- knowledge_graph / reasoner / context_manager: injected via constructor
- hybrid_retriever: the constructor builds its own (over a real
  DocumentRetriever pointed at an empty tmp docs dir). It is replaced on the
  instance with a HybridRetriever wired to the FakeDocRetriever from
  conftest so no ChromaDB/embedding work happens.
- external signal manager: replaced through the DI container's public
  ``Container.override`` hook (otherwise it would open an aiohttp session).
- verification pipeline: singleton factory pointed at a tmp SQLite path.
- LLM boundary: ``litellm.acompletion`` is imported by name into the agent
  module, so the bound name in that module is replaced with an echo fake that
  records the messages it receives.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import src.agents.hybrid_chatbot_agent as chatbot_module
from src.agents.hybrid_chatbot_agent import HybridChatbotAgent
from src.core.verification_pipeline import VerificationPipelineFactory
from src.infrastructure.container import Container
from src.memory.context import ContextManager
from src.rag.hybrid_retriever import HybridContext, HybridRetriever
from tests.characterization.conftest import FakeDocRetriever

QUERY = "LANEIGE Lip Care SoS는?"


class FakeSignalManager:
    def __init__(self):
        self.calls: list[tuple[str, dict | None]] = []

    async def collect(self, query: str, entities=None, signal_types=None) -> list[Any]:
        self.calls.append((query, entities))
        return []

    def get_failed_collectors(self) -> list[str]:
        return []


class EchoLLM:
    """Replacement for litellm.acompletion: echoes the last message back."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, model: str, messages: list[dict], **kwargs: Any):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="ECHO:" + messages[-1]["content"]))
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> EchoLLM:
    echo = EchoLLM()
    monkeypatch.setattr(chatbot_module, "acompletion", echo)
    return echo


@pytest.fixture
def signals() -> FakeSignalManager:
    fake = FakeSignalManager()
    Container.override("external_signal_manager", fake)  # reset by root conftest
    return fake


@pytest.fixture
def agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kg, reasoner, llm, signals
) -> HybridChatbotAgent:
    monkeypatch.chdir(tmp_path)  # ./logs audit files
    monkeypatch.delenv("LLM_CHATBOT_TEMPERATURE", raising=False)
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    VerificationPipelineFactory.reset()
    VerificationPipelineFactory.get_instance(db_path=str(tmp_path / "verify.db"))

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    bot = HybridChatbotAgent(
        model="gpt-4.1-mini",
        docs_dir=str(docs_dir),
        knowledge_graph=kg,
        reasoner=reasoner,
        context_manager=ContextManager(),
    )
    bot.hybrid_retriever = HybridRetriever(
        knowledge_graph=kg,
        reasoner=reasoner,
        doc_retriever=FakeDocRetriever(),
        auto_init_rules=False,
    )
    yield bot
    VerificationPipelineFactory.reset()


def test_constructor_config_defaults(agent: HybridChatbotAgent) -> None:
    # PINS CURRENT BEHAVIOR: config/thresholds.json sets temperature 0.7, which
    # overrides the "chatbot uses low 0.4 temperature" documented in __init__.
    assert agent.temperature == 0.7
    assert agent.model == "gpt-4.1-mini"
    assert agent._enable_verification is True
    assert agent.max_context_tokens == 8000


async def test_chat_response_shape_and_prompt(
    agent: HybridChatbotAgent, llm: EchoLLM, signals: FakeSignalManager
) -> None:
    result = await agent.chat(QUERY)

    assert sorted(result) == [
        "entities",
        "hybrid_context",
        "inferences",
        "is_fallback",
        "query_info",
        "query_type",
        "response",
        "sources",
        "stats",
        "suggestions",
        "verification",
    ]
    # PINS CURRENT BEHAVIOR: RAGRouter puts this metric question in DEFINITION
    # (only the "sos" indicator scores, exactly at the 1.5 threshold).
    assert result["query_type"] == "definition"
    assert result["is_fallback"] is False
    assert result["entities"] == {
        "brands": ["laneige"],
        "categories": ["lip_care"],
        "indicators": ["sos"],
        "time_range": [],
        "products": [],
        "sentiments": [],
        "sentiment_clusters": [],
        "concepts": ["sos"],
    }
    assert result["query_info"] == {"original": QUERY, "rewritten": None, "was_rewritten": False}
    assert isinstance(result["hybrid_context"], HybridContext)
    assert result["hybrid_context"].metadata["query_intent"] == "metric"
    assert set(result["stats"]) == {
        "inferences_count",
        "rag_chunks_count",
        "kg_facts_count",
        "response_time_ms",
    }
    assert result["stats"]["inferences_count"] == 1
    assert result["stats"]["rag_chunks_count"] == 2
    assert result["stats"]["kg_facts_count"] == 3

    # Sources: KG summary, ontology rule, the two RAG chunks, AI disclaimer
    assert len(result["sources"]) == 5
    assert result["sources"][0]["type"] == "knowledge_graph"
    assert result["sources"][0]["fact_count"] == 3
    assert result["sources"][1]["type"] == "ontology_inference"
    assert result["sources"][1]["rule_name"] == "category_entry_opportunity"
    assert result["sources"][1]["confidence"] == 0.7

    # Exactly one LLM call with system + user messages, config temperature, 800 tokens
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["model"] == "gpt-4.1-mini"
    assert call["temperature"] == 0.7
    assert call["max_tokens"] == 800
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    user_prompt = call["messages"][1]["content"]
    assert "LANEIGE" in user_prompt
    assert f"## 사용자 질문\n{QUERY}" in user_prompt
    assert "### SoS 정의 [2]\nSoS(Share of Shelf)는 점유율 지표입니다." in user_prompt
    assert "**lip_care** Top 브랜드: LANEIGE" in user_prompt
    assert "## 이전 대화\n이전 대화 없음" in user_prompt
    assert "- [entry_opportunity] lip_care는 분산된 시장 구조(HHI: 0.000)" in user_prompt
    assert call["messages"][0]["content"].startswith(
        "당신은 Amazon 베스트셀러 순위 분석 전문가입니다."
    )

    # External signals were asked for with the extracted entities
    assert signals.calls == [(QUERY, result["entities"])]


async def test_chat_response_text_and_brand_normalization(agent: HybridChatbotAgent) -> None:
    result = await agent.chat(QUERY)
    text = result["response"]

    assert text.startswith("ECHO:\n## 사용자 질문\nLANEIGE Lip Care SoS는?")
    # Perplexity-style source appendix is appended to the LLM answer
    assert "**📚 출처 및 참고자료:**" in text
    assert "3. 📄 **SoS 정의**\n   - 관련도: 0.90" in text
    assert "5. 🤖 **AI 분석: gpt-4.1-mini**" in text
    # FIXED (D24): _normalize_response_brands no longer rewrites the category name
    # "Beauty & Personal Care" (the bare token "Beauty" is protected inside it).
    assert "Beauty & Personal Care > Skin Care > Lip Care" in text
    assert "Beauty of Joseon & Personal Care" not in text

    # Conversation memory got both turns
    history = agent.get_conversation_history()
    assert [h["role"] for h in history] == ["user", "assistant"]
    assert history[0]["content"] == QUERY
    assert history[1]["content"] == text


async def test_chat_inference_without_data_context(agent: HybridChatbotAgent) -> None:
    result = await agent.chat(QUERY)

    # PINS CURRENT BEHAVIOR: no set_data_context() -> empty metrics -> the rule
    # engine sees HHI/SoS as 0 and emits a spurious "entry opportunity" insight.
    assert len(result["inferences"]) == 1
    inf = result["inferences"][0]
    assert inf["rule_name"] == "category_entry_opportunity"
    assert inf["insight_type"] == "entry_opportunity"
    assert inf["confidence"] == 0.7
    assert inf["evidence"]["satisfied_conditions"] == [
        "hhi_below_0.15",
        "low_target_presence",
        "is_target",
    ]
    assert inf["metadata"]["hhi"] is None and inf["metadata"]["current_sos"] is None
    assert result["suggestions"] == [
        "트렌드 상세 분석",
        "점유율 개선 전략은?",
        "laneige 제품별 성과 분석",
    ]

    verification = result["verification"]
    assert verification["grade"] == "red"
    assert verification["verified_claims"] == 0


@pytest.mark.parametrize("query", ["안녕", "hello"])
async def test_chat_unknown_intent_returns_router_fallback(
    agent: HybridChatbotAgent, llm: EchoLLM, query: str
) -> None:
    result = await agent.chat(query)

    assert result == {
        "response": (
            "질문의 의도를 정확히 파악하지 못했습니다. 다음 중 어떤 정보가 필요하신가요?\n\n"
            "1. 지표 정의 (예: 'SoS가 뭐야?')\n"
            "2. 지표 해석 (예: 'HHI가 높으면 어떤 의미야?')\n"
            "3. 지표 조합 해석 (예: 'CPI 높고 평점 낮으면?')\n"
            "4. 순위/데이터 조회 (예: '라네즈 현재 순위?')\n"
            "5. 분석 요청 (예: '라네즈 3개월 성과 분석해줘')"
        ),
        "query_type": "unknown",
        "is_fallback": True,
        "inferences": [],
        "sources": [],
        "suggestions": [
            "SoS(점유율)에 대해 알려주세요",
            "오늘의 주요 인사이트는?",
            "LANEIGE 현재 순위는?",
        ],
    }
    assert llm.calls == []
    assert agent.get_conversation_history() == []  # fallback turns are not remembered

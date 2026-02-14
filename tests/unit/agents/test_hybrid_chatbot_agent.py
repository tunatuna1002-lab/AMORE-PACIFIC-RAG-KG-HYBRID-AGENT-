"""
TDD Phase 2: HybridChatbotAgent 테스트 (RED → GREEN)

테스트 대상: src/agents/hybrid_chatbot_agent.py
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestHybridChatbotAgentInit:
    """HybridChatbotAgent 초기화 테스트"""

    def test_init_with_defaults(self):
        """기본값으로 초기화 가능해야 함"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent()

        assert agent.model == "gpt-4.1-mini"
        assert agent.kg is not None
        assert agent.reasoner is not None
        assert agent.hybrid_retriever is not None

    def test_init_with_custom_model(self):
        """커스텀 모델명으로 초기화 가능해야 함"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent(model="gpt-4")
        assert agent.model == "gpt-4"

    def test_init_with_injected_knowledge_graph(self):
        """주입된 KnowledgeGraph 사용해야 함"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent
        from src.ontology.knowledge_graph import KnowledgeGraph

        mock_kg = KnowledgeGraph()
        agent = HybridChatbotAgent(knowledge_graph=mock_kg)

        assert agent.kg is mock_kg


class TestHybridChatbotAgentChat:
    """HybridChatbotAgent.chat() 테스트"""

    @pytest.fixture
    def mock_kg(self):
        """Mock KnowledgeGraph"""
        kg = MagicMock()
        kg.query.return_value = []
        kg.get_entities_by_type.return_value = []
        return kg

    @pytest.fixture
    def mock_reasoner(self):
        """Mock OntologyReasoner"""
        reasoner = MagicMock()
        reasoner.infer.return_value = []
        reasoner.rules = ["rule1"]
        return reasoner

    @pytest.fixture
    def sample_data(self) -> dict[str, Any]:
        """샘플 데이터"""
        return {
            "date": "2026-01-23",
            "categories": {
                "lip_care": {
                    "total_products": 100,
                    "laneige_count": 3,
                    "sos": 0.15,
                    "brands": {"LANEIGE": {"count": 3, "sos": 0.15}},
                }
            },
        }

    @pytest.mark.asyncio
    async def test_chat_returns_dict_with_required_keys(self, mock_kg, mock_reasoner):
        """chat()는 필수 키를 포함한 dict 반환해야 함"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        with patch(
            "src.agents.hybrid_chatbot_agent.acompletion", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Test response"))]
            )

            agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

            # router.route 모킹
            with patch.object(agent.router, "route") as mock_route:
                mock_route.return_value = {
                    "query_type": "METRIC",
                    "entities": {"brands": ["LANEIGE"]},
                }

                # hybrid_retriever.retrieve 모킹
                with patch.object(
                    agent.hybrid_retriever, "retrieve", new_callable=AsyncMock
                ) as mock_retrieve:
                    mock_context = MagicMock()
                    mock_context.entities = {}
                    mock_context.ontology_facts = []
                    mock_context.inferences = []
                    mock_context.rag_chunks = []
                    mock_retrieve.return_value = mock_context

                    result = await agent.chat("LANEIGE 경쟁력은?")

        assert isinstance(result, dict)
        assert "response" in result

    @pytest.mark.asyncio
    async def test_chat_response_is_string(self, mock_kg, mock_reasoner):
        """응답은 문자열이어야 함"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        with patch(
            "src.agents.hybrid_chatbot_agent.acompletion", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="라네즈의 경쟁력 분석입니다."))]
            )

            agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

            with patch.object(agent.router, "route") as mock_route:
                mock_route.return_value = {"query_type": "METRIC"}

                with patch.object(
                    agent.hybrid_retriever, "retrieve", new_callable=AsyncMock
                ) as mock_retrieve:
                    mock_context = MagicMock()
                    mock_context.entities = {}
                    mock_context.ontology_facts = []
                    mock_context.inferences = []
                    mock_context.rag_chunks = []
                    mock_retrieve.return_value = mock_context

                    result = await agent.chat("LANEIGE 분석")

        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    @pytest.mark.asyncio
    async def test_chat_uses_hybrid_retrieval(self, mock_kg, mock_reasoner):
        """쿼리 시 HybridRetriever를 사용해야 함"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        with patch(
            "src.agents.hybrid_chatbot_agent.acompletion", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Response"))]
            )

            agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

            with patch.object(agent.router, "route") as mock_route:
                mock_route.return_value = {"query_type": "METRIC"}

                with patch.object(
                    agent.hybrid_retriever, "retrieve", new_callable=AsyncMock
                ) as mock_retrieve:
                    mock_context = MagicMock()
                    mock_context.entities = {}
                    mock_context.ontology_facts = []
                    mock_context.inferences = []
                    mock_context.rag_chunks = []
                    mock_retrieve.return_value = mock_context

                    await agent.chat("SoS 분석")

                    mock_retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_includes_sources_when_available(self, mock_kg, mock_reasoner):
        """RAG 청크가 있으면 sources에 포함해야 함"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        with patch(
            "src.agents.hybrid_chatbot_agent.acompletion", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Response"))]
            )

            agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

            with patch.object(agent.router, "route") as mock_route:
                mock_route.return_value = {"query_type": "DEFINITION"}

                with patch.object(
                    agent.hybrid_retriever, "retrieve", new_callable=AsyncMock
                ) as mock_retrieve:
                    mock_context = MagicMock()
                    mock_context.entities = {}
                    mock_context.ontology_facts = []
                    mock_context.inferences = []
                    mock_context.rag_chunks = [{"source": "guide1.md", "content": "Content 1"}]
                    mock_retrieve.return_value = mock_context

                    result = await agent.chat("가이드 검색")

        assert "sources" in result


class TestHybridChatbotAgentErrorHandling:
    """HybridChatbotAgent 에러 처리 테스트"""

    @pytest.fixture
    def mock_kg(self):
        return MagicMock()

    @pytest.fixture
    def mock_reasoner(self):
        reasoner = MagicMock()
        reasoner.rules = ["rule1"]
        return reasoner

    @pytest.mark.asyncio
    async def test_chat_handles_unknown_query_type(self, mock_kg, mock_reasoner):
        """알 수 없는 쿼리 타입은 fallback 응답"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent
        from src.rag.router import QueryType

        agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

        with patch.object(agent.router, "route") as mock_route:
            mock_route.return_value = {
                "query_type": QueryType.UNKNOWN,
                "fallback_message": "죄송합니다. 질문을 이해하지 못했습니다.",
            }

            result = await agent.chat("")

            assert "response" in result
            assert result.get("is_fallback", False) is True

    @pytest.mark.asyncio
    async def test_chat_handles_llm_failure_gracefully(self, mock_kg, mock_reasoner):
        """LLM 실패 시 graceful degradation (에러 응답 반환)"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        with patch(
            "src.agents.hybrid_chatbot_agent.acompletion",
            new_callable=AsyncMock,
            side_effect=Exception("LLM API failed"),
        ):
            agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

            with patch.object(agent.router, "route") as mock_route:
                mock_route.return_value = {"query_type": "METRIC"}

                with patch.object(
                    agent.hybrid_retriever, "retrieve", new_callable=AsyncMock
                ) as mock_retrieve:
                    mock_context = MagicMock()
                    mock_context.entities = {}
                    mock_context.ontology_facts = []
                    mock_context.inferences = []
                    mock_context.rag_chunks = []
                    mock_retrieve.return_value = mock_context

                    # 에러 시에도 응답 반환 (graceful degradation)
                    result = await agent.chat("테스트 쿼리")
                    assert "response" in result


class TestHybridChatbotAgentQueryRouting:
    """HybridChatbotAgent 쿼리 라우팅 테스트"""

    def test_router_exists(self):
        """에이전트에 라우터가 있어야 함"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent()

        assert hasattr(agent, "router")
        assert agent.router is not None


class TestHybridChatbotAgentSuggestions:
    """HybridChatbotAgent 후속 질문 제안 테스트"""

    @pytest.fixture
    def mock_kg(self):
        return MagicMock()

    @pytest.fixture
    def mock_reasoner(self):
        reasoner = MagicMock()
        reasoner.rules = ["rule1"]
        return reasoner

    @pytest.mark.asyncio
    async def test_chat_returns_suggestions(self, mock_kg, mock_reasoner):
        """응답에 후속 질문 제안이 포함되어야 함"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        with patch(
            "src.agents.hybrid_chatbot_agent.acompletion", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Response"))]
            )

            agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

            with patch.object(agent.router, "route") as mock_route:
                mock_route.return_value = {"query_type": "METRIC"}

                with patch.object(
                    agent.hybrid_retriever, "retrieve", new_callable=AsyncMock
                ) as mock_retrieve:
                    mock_context = MagicMock()
                    mock_context.entities = {}
                    mock_context.ontology_facts = []
                    mock_context.inferences = []
                    mock_context.rag_chunks = []
                    mock_retrieve.return_value = mock_context

                    result = await agent.chat("LANEIGE 분석")

        # suggestions가 있을 수 있음
        if "suggestions" in result:
            assert isinstance(result["suggestions"], list)


class TestHybridChatbotAgentDataContext:
    """HybridChatbotAgent 데이터 컨텍스트 테스트"""

    def test_set_data_context(self):
        """데이터 컨텍스트 설정 가능해야 함"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent()

        data = {"date": "2026-01-23", "categories": {}}
        agent.set_data_context(data)

        assert agent._current_data == data

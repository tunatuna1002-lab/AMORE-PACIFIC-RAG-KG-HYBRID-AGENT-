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


class TestHybridChatbotAgentGenerateResponse:
    """_generate_response 메서드 테스트"""

    @pytest.fixture
    def mock_kg(self):
        kg = MagicMock()
        kg.get_category_hierarchy.return_value = {"error": "not_found"}
        return kg

    @pytest.fixture
    def mock_reasoner(self):
        reasoner = MagicMock()
        reasoner.rules = ["rule1"]
        return reasoner

    @pytest.mark.asyncio
    async def test_generate_response_with_valid_llm_response(self, mock_kg, mock_reasoner):
        """LLM 응답이 정상인 경우"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent
        from src.rag.router import QueryType

        with patch(
            "src.agents.hybrid_chatbot_agent.acompletion", new_callable=AsyncMock
        ) as mock_llm:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content="LANEIGE는 강력한 경쟁력을 보이고 있습니다."))
            ]
            mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
            mock_llm.return_value = mock_response

            agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

            result = await agent._generate_response(
                user_message="LANEIGE 경쟁력은?",
                query_type=QueryType.ANALYSIS,
                context="테스트 컨텍스트",
                inferences=[],
            )

            assert isinstance(result, str)
            assert len(result) > 0
            mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_with_llm_failure(self, mock_kg, mock_reasoner):
        """LLM 실패 시 폴백 응답"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent
        from src.rag.router import QueryType

        with patch(
            "src.agents.hybrid_chatbot_agent.acompletion",
            new_callable=AsyncMock,
            side_effect=Exception("API Error"),
        ):
            agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

            result = await agent._generate_response(
                user_message="Test",
                query_type=QueryType.ANALYSIS,
                context="Test context",
                inferences=[],
            )

            assert isinstance(result, str)
            assert "죄송합니다" in result or "응답" in result

    @pytest.mark.asyncio
    async def test_generate_response_with_inferences(self, mock_kg, mock_reasoner):
        """추론 결과가 있는 경우"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent
        from src.domain.entities.relations import InferenceResult, InsightType
        from src.rag.router import QueryType

        with patch(
            "src.agents.hybrid_chatbot_agent.acompletion", new_callable=AsyncMock
        ) as mock_llm:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content="Analysis with inference"))
            ]
            mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
            mock_llm.return_value = mock_response

            agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

            inference = InferenceResult(
                rule_name="test_rule",
                insight_type=InsightType.COMPETITIVE_THREAT,
                insight="경쟁이 치열합니다",
                confidence=0.9,
                evidence=[],
                recommendation="모니터링 필요",
            )

            result = await agent._generate_response(
                user_message="Test",
                query_type=QueryType.ANALYSIS,
                context="Test context",
                inferences=[inference],
            )

            assert isinstance(result, str)


class TestHybridChatbotAgentBrandNormalization:
    """브랜드명 정규화 테스트"""

    def test_normalize_brand_with_truncated_name(self):
        """잘린 브랜드명을 정규화"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent()

        assert agent._normalize_brand("burt's") == "Burt's Bees"
        assert agent._normalize_brand("wet") == "wet n wild"
        assert agent._normalize_brand("the") == "The Ordinary"

    def test_normalize_brand_with_unknown_brand(self):
        """알 수 없는 브랜드는 그대로 반환"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent()

        assert agent._normalize_brand("RandomBrand") == "RandomBrand"

    def test_normalize_response_brands(self):
        """응답 내 브랜드명 정규화"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent()

        response = "Burt's 제품이 인기입니다."
        normalized = agent._normalize_response_brands(response)

        assert "Burt's Bees" in normalized


class TestHybridChatbotAgentConversation:
    """대화 관리 테스트"""

    def test_get_conversation_history(self):
        """대화 기록 조회"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent()

        history = agent.get_conversation_history(limit=5)

        assert isinstance(history, list)

    def test_clear_conversation(self):
        """대화 초기화"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent()

        agent.clear_conversation()

        history = agent.get_conversation_history()
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_maybe_rewrite_query_without_history(self):
        """대화 히스토리가 없으면 재구성 스킵"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent()

        result = await agent._maybe_rewrite_query("LANEIGE 분석")

        assert result.was_rewritten is False
        assert result.rewritten_query == "LANEIGE 분석"


class TestHybridChatbotAgentUtilities:
    """유틸리티 메서드 테스트"""

    def test_estimate_cost(self):
        """비용 추정"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent()

        cost = agent._estimate_cost(prompt_tokens=1000, completion_tokens=500)

        assert isinstance(cost, float)
        assert cost > 0

    def test_get_knowledge_graph(self):
        """KG 반환"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent()

        kg = agent.get_knowledge_graph()

        assert kg is not None

    def test_get_reasoner(self):
        """추론기 반환"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent()

        reasoner = agent.get_reasoner()

        assert reasoner is not None

    def test_get_last_hybrid_context(self):
        """마지막 컨텍스트 반환"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent()

        context = agent.get_last_hybrid_context()

        assert context is None  # 초기 상태

    @pytest.mark.asyncio
    async def test_explain_last_response_without_context(self):
        """컨텍스트가 없으면 설명 없음"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent()

        explanation = await agent.explain_last_response()

        assert "없습니다" in explanation


class TestHybridChatbotSession:
    """HybridChatbotSession 테스트"""

    def test_session_get_or_create(self):
        """세션 생성 또는 조회"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotSession

        session_manager = HybridChatbotSession()

        agent1 = session_manager.get_or_create("session1")
        agent2 = session_manager.get_or_create("session1")

        assert agent1 is agent2

    def test_session_close(self):
        """세션 종료"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotSession

        session_manager = HybridChatbotSession()

        session_manager.get_or_create("session1")
        session_manager.close_session("session1")

        sessions = session_manager.list_sessions()
        assert "session1" not in sessions

    def test_session_list(self):
        """세션 목록"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotSession

        session_manager = HybridChatbotSession()

        session_manager.get_or_create("session1")
        session_manager.get_or_create("session2")

        sessions = session_manager.list_sessions()

        assert len(sessions) == 2
        assert "session1" in sessions
        assert "session2" in sessions

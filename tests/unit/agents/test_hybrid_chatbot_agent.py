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


class TestHybridChatbotAgentSuggestionGeneration:
    """후속 질문 제안 생성 테스트"""

    @pytest.fixture
    def mock_kg(self):
        kg = MagicMock()
        kg.get_related_brands.return_value = ["CeraVe", "La Roche-Posay"]
        return kg

    @pytest.fixture
    def mock_reasoner(self):
        reasoner = MagicMock()
        reasoner.rules = ["rule1"]
        return reasoner

    def test_generate_suggestions_with_entities(self, mock_kg, mock_reasoner):
        """엔티티 기반 제안 생성"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent
        from src.rag.router import QueryType

        agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

        entities = {"brands": ["LANEIGE"], "categories": ["Lip Care"], "indicators": ["SoS"]}

        suggestions = agent._generate_suggestions(
            query_type=QueryType.ANALYSIS,
            entities=entities,
            inferences=[],
            response="LANEIGE의 시장 점유율이 증가하고 있습니다.",
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3
        assert all(isinstance(s, str) for s in suggestions)

    def test_extract_response_keywords(self, mock_kg, mock_reasoner):
        """응답에서 키워드 추출"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

        response = "순위가 하락하고 있습니다. 경쟁사 분석이 필요합니다."
        keywords = agent._extract_response_keywords(response)

        assert isinstance(keywords, list)
        assert len(keywords) <= 2

    def test_generate_entity_suggestions(self, mock_kg, mock_reasoner):
        """엔티티 기반 제안"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

        entities = {"brands": ["LANEIGE"], "categories": ["Lip Care"]}
        suggestions = agent._generate_entity_suggestions(entities)

        assert isinstance(suggestions, list)

    def test_get_fallback_suggestions(self, mock_kg, mock_reasoner):
        """폴백 제안"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

        suggestions = agent._get_fallback_suggestions()

        assert isinstance(suggestions, list)
        assert len(suggestions) == 3


class TestHybridChatbotAgentSourceExtraction:
    """출처 추출 테스트"""

    @pytest.fixture
    def mock_kg(self):
        kg = MagicMock()
        kg.get_category_hierarchy.return_value = {
            "name": "Lip Care",
            "level": 2,
            "ancestors": [{"name": "Skin Care"}, {"name": "Beauty"}],
            "descendants": [],
        }
        return kg

    @pytest.fixture
    def mock_reasoner(self):
        reasoner = MagicMock()
        reasoner.rules = ["rule1"]
        return reasoner

    @pytest.fixture
    def sample_hybrid_context(self):
        """샘플 하이브리드 컨텍스트"""
        from src.domain.entities.relations import InferenceResult, InsightType

        context = MagicMock()
        context.entities = {"brands": ["LANEIGE"], "categories": ["Lip Care"]}
        context.ontology_facts = [
            {"subject": "LANEIGE", "predicate": "competes_with", "object": "CeraVe"}
        ]
        context.inferences = [
            InferenceResult(
                rule_name="competitive_analysis",
                insight_type=InsightType.COMPETITIVE_THREAT,
                insight="경쟁 심화",
                confidence=0.85,
                evidence=["fact1"],
                recommendation="모니터링",
            )
        ]
        context.rag_chunks = [
            {
                "content": "Test content",
                "metadata": {
                    "doc_id": "doc1",
                    "title": "가이드라인",
                    "file_path": "/path/to/guide.md",
                    "section": "Section 1",
                },
                "score": 0.92,
            }
        ]
        return context

    def test_extract_sources_with_crawled_data(self, mock_kg, mock_reasoner, sample_hybrid_context):
        """크롤링 데이터 출처 추출"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)
        agent._current_data = {
            "metadata": {"data_date": "2026-01-23"},
            "categories": {
                "lip_care": {
                    "rank_records": [
                        {
                            "asin": "B001TEST",
                            "brand": "LANEIGE",
                            "product_name": "Lip Sleeping Mask",
                            "rank": 4,
                        }
                    ]
                }
            },
        }

        sources = agent._extract_sources(sample_hybrid_context, external_signals=[])

        assert isinstance(sources, list)
        assert len(sources) > 0
        assert any(s["type"] == "crawled_data" for s in sources)
        assert any(s["type"] == "ai_model" for s in sources)

    def test_extract_sources_with_kg_facts(self, mock_kg, mock_reasoner, sample_hybrid_context):
        """KG 팩트 출처 추출"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

        sources = agent._extract_sources(sample_hybrid_context, external_signals=[])

        kg_sources = [s for s in sources if s["type"] == "knowledge_graph"]
        assert len(kg_sources) > 0
        assert kg_sources[0]["fact_count"] > 0

    def test_extract_sources_with_inferences(self, mock_kg, mock_reasoner, sample_hybrid_context):
        """추론 결과 출처 추출"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

        sources = agent._extract_sources(sample_hybrid_context, external_signals=[])

        inference_sources = [s for s in sources if s["type"] == "ontology_inference"]
        assert len(inference_sources) > 0
        assert inference_sources[0]["confidence"] == 0.85

    def test_extract_sources_with_rag_chunks(self, mock_kg, mock_reasoner, sample_hybrid_context):
        """RAG 청크 출처 추출"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

        sources = agent._extract_sources(sample_hybrid_context, external_signals=[])

        rag_sources = [s for s in sources if s["type"] == "rag_document"]
        assert len(rag_sources) > 0
        assert rag_sources[0]["relevance_score"] == 0.92

    def test_format_sources_for_response(self, mock_kg, mock_reasoner):
        """출처 포맷팅"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

        sources = [
            {
                "type": "crawled_data",
                "icon": "📊",
                "description": "Amazon 데이터",
                "collected_at": "2026-01-23",
                "url": "https://amazon.com",
                "details": {"total_products": 100},
            },
            {
                "type": "ai_model",
                "icon": "🤖",
                "description": "AI 분석",
                "model": "gpt-4.1-mini",
                "disclaimer": "AI 생성",
            },
        ]

        formatted = agent._format_sources_for_response(sources)

        assert isinstance(formatted, str)
        assert "📊" in formatted
        assert "🤖" in formatted
        assert "출처" in formatted

    def test_extract_entity_names(self, mock_kg, mock_reasoner):
        """엔티티 이름 추출"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

        facts = [
            {"subject": "LANEIGE", "predicate": "competes_with", "object": "CeraVe"},
            {"subject": "CeraVe", "predicate": "has_product", "object": "Moisturizer"},
        ]

        entities = agent._extract_entity_names(facts)

        assert isinstance(entities, list)
        assert "LANEIGE" in entities
        assert "CeraVe" in entities
        assert len(entities) <= 5

    def test_extract_relation_types(self, mock_kg, mock_reasoner):
        """관계 타입 추출"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

        facts = [
            {"subject": "LANEIGE", "predicate": "competes_with", "object": "CeraVe"},
            {"subject": "CeraVe", "predicate": "has_product", "object": "Moisturizer"},
        ]

        relations = agent._extract_relation_types(facts)

        assert isinstance(relations, list)
        assert "competes_with" in relations
        assert "has_product" in relations


class TestHybridChatbotAgentExternalSignals:
    """외부 신호 수집 테스트"""

    @pytest.fixture
    def mock_kg(self):
        return MagicMock()

    @pytest.fixture
    def mock_reasoner(self):
        reasoner = MagicMock()
        reasoner.rules = ["rule1"]
        return reasoner

    @pytest.mark.asyncio
    async def test_collect_external_signals_with_entities(self, mock_kg, mock_reasoner):
        """엔티티가 있는 경우 외부 신호 수집"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

        # Mock external signal collector
        mock_collector = MagicMock()
        mock_collector.fetch_tavily_news = AsyncMock(return_value=[])
        mock_collector.fetch_all_rss_feeds = AsyncMock(return_value=[])
        agent._external_signal_collector = mock_collector

        entities = {"brands": ["LANEIGE"], "categories": ["Lip Care"]}

        signals = await agent._collect_external_signals("LANEIGE 뉴스", entities)

        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_collect_external_signals_without_collector(self, mock_kg, mock_reasoner):
        """수집기가 없는 경우 빈 리스트 반환"""
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        agent = HybridChatbotAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)
        agent._external_signal_collector = None

        with patch(
            "src.tools.collectors.external_signal_collector.ExternalSignalCollector",
            side_effect=ImportError("mocked"),
        ):
            signals = await agent._collect_external_signals("test", None)

            assert isinstance(signals, list)
            assert len(signals) == 0


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

"""
TDD Phase 2: HybridInsightAgent 테스트 (RED → GREEN)

테스트 대상: src/agents/hybrid_insight_agent.py
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestHybridInsightAgentInit:
    """HybridInsightAgent 초기화 테스트"""

    def test_init_with_defaults(self):
        """기본값으로 초기화 가능해야 함"""
        from src.agents.hybrid_insight_agent import HybridInsightAgent

        agent = HybridInsightAgent()

        assert agent.model == "gpt-4.1-mini"
        assert agent.kg is not None
        assert agent.reasoner is not None
        assert agent.hybrid_retriever is not None

    def test_init_with_custom_model(self):
        """커스텀 모델명으로 초기화 가능해야 함"""
        from src.agents.hybrid_insight_agent import HybridInsightAgent

        agent = HybridInsightAgent(model="gpt-4")
        assert agent.model == "gpt-4"

    def test_init_with_injected_knowledge_graph(self):
        """주입된 KnowledgeGraph 사용해야 함"""
        from src.agents.hybrid_insight_agent import HybridInsightAgent
        from src.ontology.knowledge_graph import KnowledgeGraph

        mock_kg = KnowledgeGraph()
        agent = HybridInsightAgent(knowledge_graph=mock_kg)

        assert agent.kg is mock_kg

    def test_init_with_injected_reasoner(self):
        """주입된 Reasoner 사용해야 함"""
        from src.agents.hybrid_insight_agent import HybridInsightAgent
        from src.ontology.knowledge_graph import KnowledgeGraph
        from src.ontology.reasoner import OntologyReasoner

        kg = KnowledgeGraph()
        mock_reasoner = OntologyReasoner(kg)
        agent = HybridInsightAgent(knowledge_graph=kg, reasoner=mock_reasoner)

        assert agent.reasoner is mock_reasoner


class TestHybridInsightAgentExecute:
    """HybridInsightAgent.execute() 테스트"""

    @pytest.fixture
    def mock_kg(self):
        """Mock KnowledgeGraph"""
        kg = MagicMock()
        kg.load_from_crawl_data.return_value = 10
        kg.load_from_metrics_data.return_value = 5
        kg.query.return_value = []
        kg.get_entities_by_type.return_value = []
        return kg

    @pytest.fixture
    def mock_reasoner(self):
        """Mock OntologyReasoner"""
        reasoner = MagicMock()
        reasoner.infer.return_value = []
        reasoner.rules = ["rule1"]  # 규칙이 이미 등록됨
        return reasoner

    @pytest.fixture
    def sample_metrics_data(self) -> dict[str, Any]:
        """샘플 메트릭 데이터"""
        return {
            "date": "2026-01-23",
            "categories": {
                "lip_care": {
                    "total_products": 100,
                    "laneige_count": 3,
                    "sos": 0.15,
                    "hhi": 0.05,
                    "brands": {"LANEIGE": {"count": 3, "sos": 0.15, "avg_rank": 25}},
                }
            },
            "alerts": [],
        }

    @pytest.fixture
    def sample_crawl_data(self) -> dict[str, Any]:
        """샘플 크롤 데이터"""
        return {
            "categories": {
                "lip_care": {
                    "products": [
                        {
                            "asin": "B0BSHRYY1S",
                            "title": "LANEIGE Lip Sleeping Mask",
                            "brand": "LANEIGE",
                            "rank": 1,
                            "price": 24.00,
                        }
                    ]
                }
            }
        }

    @pytest.mark.asyncio
    async def test_execute_returns_dict_with_required_keys(
        self, mock_kg, mock_reasoner, sample_metrics_data
    ):
        """execute()는 필수 키를 포함한 dict 반환해야 함"""
        from src.agents.hybrid_insight_agent import HybridInsightAgent

        with patch.object(
            HybridInsightAgent,
            "_generate_daily_insight",
            new_callable=AsyncMock,
            return_value="Test insight",
        ):
            with patch.object(
                HybridInsightAgent, "_run_hybrid_retrieval", new_callable=AsyncMock
            ) as mock_retrieval:
                # HybridContext mock
                mock_context = MagicMock()
                mock_context.inferences = []
                mock_context.rag_chunks = []
                mock_context.ontology_facts = []
                mock_retrieval.return_value = mock_context

                with patch.object(
                    HybridInsightAgent,
                    "_collect_external_signals",
                    new_callable=AsyncMock,
                    return_value={"signals": []},
                ):
                    agent = HybridInsightAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)
                    result = await agent.execute(sample_metrics_data)

        assert isinstance(result, dict)
        assert "status" in result
        assert "daily_insight" in result
        assert "action_items" in result
        assert "highlights" in result
        assert "inferences" in result

    @pytest.mark.asyncio
    async def test_execute_status_completed_on_success(
        self, mock_kg, mock_reasoner, sample_metrics_data
    ):
        """성공 시 status는 'completed'여야 함"""
        from src.agents.hybrid_insight_agent import HybridInsightAgent

        with patch.object(
            HybridInsightAgent,
            "_generate_daily_insight",
            new_callable=AsyncMock,
            return_value="Test insight",
        ):
            with patch.object(
                HybridInsightAgent, "_run_hybrid_retrieval", new_callable=AsyncMock
            ) as mock_retrieval:
                mock_context = MagicMock()
                mock_context.inferences = []
                mock_context.rag_chunks = []
                mock_context.ontology_facts = []
                mock_retrieval.return_value = mock_context

                with patch.object(
                    HybridInsightAgent,
                    "_collect_external_signals",
                    new_callable=AsyncMock,
                    return_value={"signals": []},
                ):
                    agent = HybridInsightAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)
                    result = await agent.execute(sample_metrics_data)

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_execute_updates_knowledge_graph(
        self, mock_kg, mock_reasoner, sample_metrics_data, sample_crawl_data
    ):
        """execute()는 KnowledgeGraph를 업데이트해야 함"""
        from src.agents.hybrid_insight_agent import HybridInsightAgent

        with patch.object(
            HybridInsightAgent,
            "_generate_daily_insight",
            new_callable=AsyncMock,
            return_value="Test insight",
        ):
            with patch.object(
                HybridInsightAgent, "_run_hybrid_retrieval", new_callable=AsyncMock
            ) as mock_retrieval:
                mock_context = MagicMock()
                mock_context.inferences = []
                mock_context.rag_chunks = []
                mock_context.ontology_facts = []
                mock_retrieval.return_value = mock_context

                with patch.object(
                    HybridInsightAgent,
                    "_collect_external_signals",
                    new_callable=AsyncMock,
                    return_value={"signals": []},
                ):
                    agent = HybridInsightAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)
                    await agent.execute(sample_metrics_data, crawl_data=sample_crawl_data)

        mock_kg.load_from_crawl_data.assert_called_once()
        mock_kg.load_from_metrics_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_includes_hybrid_stats(self, mock_kg, mock_reasoner, sample_metrics_data):
        """결과에 hybrid_stats가 포함되어야 함"""
        from src.agents.hybrid_insight_agent import HybridInsightAgent

        with patch.object(
            HybridInsightAgent,
            "_generate_daily_insight",
            new_callable=AsyncMock,
            return_value="Test insight",
        ):
            with patch.object(
                HybridInsightAgent, "_run_hybrid_retrieval", new_callable=AsyncMock
            ) as mock_retrieval:
                mock_context = MagicMock()
                mock_context.inferences = []
                mock_context.rag_chunks = []
                mock_context.ontology_facts = []
                mock_retrieval.return_value = mock_context

                with patch.object(
                    HybridInsightAgent,
                    "_collect_external_signals",
                    new_callable=AsyncMock,
                    return_value={"signals": []},
                ):
                    agent = HybridInsightAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)
                    result = await agent.execute(sample_metrics_data)

        assert "hybrid_stats" in result
        assert "inferences_count" in result["hybrid_stats"]
        assert "rag_chunks_count" in result["hybrid_stats"]


class TestHybridInsightAgentErrorHandling:
    """HybridInsightAgent 에러 처리 테스트"""

    @pytest.fixture
    def mock_kg(self):
        kg = MagicMock()
        kg.load_from_crawl_data.return_value = 0
        kg.load_from_metrics_data.return_value = 0
        return kg

    @pytest.fixture
    def mock_reasoner(self):
        reasoner = MagicMock()
        reasoner.rules = ["rule1"]
        return reasoner

    @pytest.mark.asyncio
    async def test_execute_raises_on_llm_failure(self, mock_kg, mock_reasoner):
        """LLM 실패 시 예외 발생해야 함"""
        from src.agents.hybrid_insight_agent import HybridInsightAgent

        with patch.object(
            HybridInsightAgent, "_run_hybrid_retrieval", new_callable=AsyncMock
        ) as mock_retrieval:
            mock_context = MagicMock()
            mock_context.inferences = []
            mock_context.rag_chunks = []
            mock_context.ontology_facts = []
            mock_retrieval.return_value = mock_context

            with patch.object(
                HybridInsightAgent,
                "_collect_external_signals",
                new_callable=AsyncMock,
                return_value={"signals": []},
            ):
                with patch.object(
                    HybridInsightAgent,
                    "_generate_daily_insight",
                    new_callable=AsyncMock,
                    side_effect=Exception("LLM API failed"),
                ):
                    agent = HybridInsightAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)

                    with pytest.raises(Exception) as exc_info:
                        await agent.execute({"categories": {}})

                    assert "LLM API failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_handles_empty_metrics_data(self, mock_kg, mock_reasoner):
        """빈 메트릭 데이터 처리 가능해야 함"""
        from src.agents.hybrid_insight_agent import HybridInsightAgent

        with patch.object(
            HybridInsightAgent,
            "_generate_daily_insight",
            new_callable=AsyncMock,
            return_value="No data insight",
        ):
            with patch.object(
                HybridInsightAgent, "_run_hybrid_retrieval", new_callable=AsyncMock
            ) as mock_retrieval:
                mock_context = MagicMock()
                mock_context.inferences = []
                mock_context.rag_chunks = []
                mock_context.ontology_facts = []
                mock_retrieval.return_value = mock_context

                with patch.object(
                    HybridInsightAgent,
                    "_collect_external_signals",
                    new_callable=AsyncMock,
                    return_value={"signals": []},
                ):
                    agent = HybridInsightAgent(knowledge_graph=mock_kg, reasoner=mock_reasoner)
                    result = await agent.execute({})

        assert result["status"] == "completed"


class TestHybridInsightAgentHelperMethods:
    """HybridInsightAgent 헬퍼 메소드 테스트"""

    def test_extract_action_items_from_inferences(self):
        """추론 결과에서 액션 아이템 추출해야 함"""
        from src.agents.hybrid_insight_agent import HybridInsightAgent
        from src.domain.entities.relations import InferenceResult, InsightType

        agent = HybridInsightAgent()

        inferences = [
            InferenceResult(
                rule_name="sos_opportunity_rule",
                insight_type=InsightType.GROWTH_OPPORTUNITY,
                insight="LANEIGE SoS 상승 기회",
                confidence=0.8,
                evidence={"sos": 0.15},
            )
        ]

        metrics_data = {"categories": {"lip_care": {"sos": 0.15}}}

        action_items = agent._extract_action_items(inferences, metrics_data)

        assert isinstance(action_items, list)

    def test_extract_highlights_from_inferences(self):
        """추론 결과에서 하이라이트 추출해야 함"""
        from src.agents.hybrid_insight_agent import HybridInsightAgent
        from src.domain.entities.relations import InferenceResult, InsightType

        agent = HybridInsightAgent()

        inferences = [
            InferenceResult(
                rule_name="market_position_rule",
                insight_type=InsightType.MARKET_POSITION,
                insight="Lip Sleeping Mask #1 유지",
                confidence=0.95,
                evidence={"rank": 1},
            )
        ]

        metrics_data = {"categories": {}}

        highlights = agent._extract_highlights(inferences, metrics_data)

        assert isinstance(highlights, list)

    def test_generate_explanations_for_inferences(self):
        """추론에 대한 설명 생성해야 함"""
        from src.agents.hybrid_insight_agent import HybridInsightAgent
        from src.domain.entities.relations import InferenceResult, InsightType

        agent = HybridInsightAgent()

        inferences = [
            InferenceResult(
                rule_name="growth_momentum_rule",
                insight_type=InsightType.GROWTH_MOMENTUM,
                insight="K-Beauty 트렌드 상승",
                confidence=0.7,
                evidence={"fact1": "data1", "fact2": "data2"},
            )
        ]

        explanations = agent._generate_explanations(inferences)

        assert isinstance(explanations, list)


class TestHybridInsightAgentIntegration:
    """HybridInsightAgent 통합 테스트 (실제 컴포넌트 사용)"""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="통합 테스트는 별도 실행")
    async def test_full_execution_with_real_components(self):
        """실제 컴포넌트로 전체 실행 테스트"""
        from src.agents.hybrid_insight_agent import HybridInsightAgent

        agent = HybridInsightAgent()

        metrics_data = {
            "date": "2026-01-23",
            "categories": {"lip_care": {"total_products": 100, "laneige_count": 3, "sos": 0.15}},
        }

        result = await agent.execute(metrics_data)

        assert result["status"] == "completed"
        assert result["daily_insight"]

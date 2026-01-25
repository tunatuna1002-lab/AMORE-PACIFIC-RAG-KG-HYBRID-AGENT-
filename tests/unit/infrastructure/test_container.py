"""
TDD Phase 4: DI 컨테이너 테스트 (RED → GREEN)

테스트 대상: src/infrastructure/container.py
"""
import pytest
from unittest.mock import MagicMock


class TestContainerSingleton:
    """Container 싱글톤 동작 테스트"""

    def test_get_knowledge_graph_singleton(self):
        """KnowledgeGraph는 싱글톤이어야 함"""
        from src.infrastructure.container import Container

        Container.reset()  # 초기화

        kg1 = Container.get_knowledge_graph()
        kg2 = Container.get_knowledge_graph()

        assert kg1 is kg2

    def test_get_reasoner_singleton(self):
        """Reasoner는 싱글톤이어야 함"""
        from src.infrastructure.container import Container

        Container.reset()

        reasoner1 = Container.get_reasoner()
        reasoner2 = Container.get_reasoner()

        assert reasoner1 is reasoner2

    def test_get_hybrid_retriever_singleton(self):
        """HybridRetriever는 싱글톤이어야 함"""
        from src.infrastructure.container import Container

        Container.reset()

        retriever1 = Container.get_hybrid_retriever()
        retriever2 = Container.get_hybrid_retriever()

        assert retriever1 is retriever2


class TestContainerDependencyInjection:
    """Container 의존성 주입 테스트"""

    def test_insight_agent_uses_container_kg(self):
        """InsightAgent는 Container의 KG를 사용해야 함"""
        from src.infrastructure.container import Container

        Container.reset()

        agent = Container.get_insight_agent()
        kg = Container.get_knowledge_graph()

        assert agent.kg is kg

    def test_insight_agent_uses_container_reasoner(self):
        """InsightAgent는 Container의 Reasoner를 사용해야 함"""
        from src.infrastructure.container import Container

        Container.reset()

        agent = Container.get_insight_agent()
        reasoner = Container.get_reasoner()

        assert agent.reasoner is reasoner

    def test_chatbot_agent_uses_container_kg(self):
        """ChatbotAgent는 Container의 KG를 사용해야 함"""
        from src.infrastructure.container import Container

        Container.reset()

        agent = Container.get_chatbot_agent()
        kg = Container.get_knowledge_graph()

        assert agent.kg is kg

    def test_crawler_agent_creation(self):
        """CrawlerAgent 생성 테스트"""
        from src.infrastructure.container import Container

        Container.reset()

        agent = Container.get_crawler_agent()

        assert agent is not None


class TestContainerReset:
    """Container reset 기능 테스트"""

    def test_reset_clears_all_instances(self):
        """reset()은 모든 인스턴스 초기화"""
        from src.infrastructure.container import Container

        # 인스턴스 생성
        Container.get_knowledge_graph()
        Container.get_reasoner()

        # 리셋
        Container.reset()

        # 내부 상태 확인
        assert len(Container._instances) == 0
        assert len(Container._overrides) == 0

    def test_reset_creates_new_instances(self):
        """reset() 후 새 인스턴스 생성"""
        from src.infrastructure.container import Container

        # 첫 번째 인스턴스
        kg1 = Container.get_knowledge_graph()

        # 리셋
        Container.reset()

        # 새 인스턴스
        kg2 = Container.get_knowledge_graph()

        # 다른 인스턴스여야 함
        assert kg1 is not kg2


class TestContainerOverride:
    """Container override 기능 테스트 (테스트용 Mock 주입)"""

    def test_override_knowledge_graph(self):
        """KnowledgeGraph Mock 주입 가능"""
        from src.infrastructure.container import Container

        Container.reset()

        mock_kg = MagicMock()
        Container.override('knowledge_graph', mock_kg)

        result = Container.get_knowledge_graph()

        assert result is mock_kg

    def test_override_reasoner(self):
        """Reasoner Mock 주입 가능"""
        from src.infrastructure.container import Container

        Container.reset()

        mock_reasoner = MagicMock()
        Container.override('reasoner', mock_reasoner)

        result = Container.get_reasoner()

        assert result is mock_reasoner

    def test_override_affects_agents(self):
        """Override가 에이전트에 반영되어야 함"""
        from src.infrastructure.container import Container

        Container.reset()

        mock_kg = MagicMock()
        Container.override('knowledge_graph', mock_kg)

        agent = Container.get_insight_agent()

        # 에이전트가 Mock을 사용해야 함
        assert agent.kg is mock_kg

    def test_override_cleared_on_reset(self):
        """reset()이 override도 초기화"""
        from src.infrastructure.container import Container

        Container.reset()

        mock_kg = MagicMock()
        Container.override('knowledge_graph', mock_kg)

        Container.reset()

        # 이제 실제 KG를 반환해야 함
        result = Container.get_knowledge_graph()
        assert result is not mock_kg


class TestContainerAgentCreation:
    """Container 에이전트 생성 테스트"""

    def test_get_insight_agent_returns_hybrid_insight_agent(self):
        """get_insight_agent()는 HybridInsightAgent 반환"""
        from src.infrastructure.container import Container
        from src.agents.hybrid_insight_agent import HybridInsightAgent

        Container.reset()

        agent = Container.get_insight_agent()

        assert isinstance(agent, HybridInsightAgent)

    def test_get_chatbot_agent_returns_hybrid_chatbot_agent(self):
        """get_chatbot_agent()는 HybridChatbotAgent 반환"""
        from src.infrastructure.container import Container
        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        Container.reset()

        agent = Container.get_chatbot_agent()

        assert isinstance(agent, HybridChatbotAgent)

    def test_get_crawler_agent_returns_crawler_agent(self):
        """get_crawler_agent()는 CrawlerAgent 반환"""
        from src.infrastructure.container import Container
        from src.agents.crawler_agent import CrawlerAgent

        Container.reset()

        agent = Container.get_crawler_agent()

        assert isinstance(agent, CrawlerAgent)

    def test_agents_are_not_cached(self):
        """에이전트는 매번 새로 생성되어야 함 (싱글톤 아님)"""
        from src.infrastructure.container import Container

        Container.reset()

        agent1 = Container.get_insight_agent()
        agent2 = Container.get_insight_agent()

        # 에이전트는 다른 인스턴스
        assert agent1 is not agent2

        # 하지만 같은 KG를 공유
        assert agent1.kg is agent2.kg


class TestContainerContextManager:
    """Container context manager 테스트"""

    def test_container_as_context_manager(self):
        """Container를 context manager로 사용 가능"""
        from src.infrastructure.container import Container

        Container.reset()

        # override를 컨텍스트 매니저로 사용
        mock_kg = MagicMock()

        with Container.test_override('knowledge_graph', mock_kg):
            result = Container.get_knowledge_graph()
            assert result is mock_kg

        # 컨텍스트 종료 후 원래 값 복원
        result = Container.get_knowledge_graph()
        assert result is not mock_kg

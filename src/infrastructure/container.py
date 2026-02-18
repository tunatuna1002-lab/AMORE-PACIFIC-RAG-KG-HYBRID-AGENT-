"""
DI (Dependency Injection) 컨테이너

이 모듈은 애플리케이션의 의존성을 중앙에서 관리합니다.

사용 예:
    from src.infrastructure.container import Container

    # 싱글톤 컴포넌트 획득
    kg = Container.get_knowledge_graph()
    reasoner = Container.get_reasoner()

    # 에이전트 생성 (의존성 자동 주입)
    agent = Container.get_insight_agent()

    # 테스트용 Mock 주입
    Container.override('knowledge_graph', mock_kg)

    # 초기화
    Container.reset()
"""

from contextlib import contextmanager
from typing import Any

from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.retriever import DocumentRetriever


class Container:
    """
    Dependency Injection Container for AMORE Agent

    싱글톤 컴포넌트:
    - KnowledgeGraph
    - OntologyReasoner
    - HybridRetriever
    - DocumentRetriever
    - SuggestionEngine
    - SourceProvider
    - ExternalSignalManager
    - MarketIntelligenceEngine

    팩토리 컴포넌트 (매번 새로 생성):
    - HybridInsightAgent
    - HybridChatbotAgent
    - CrawlerAgent
    - AlertAgent
    - MetricsAgent
    - StorageAgent
    - BatchWorkflow
    """

    _instances: dict[str, Any] = {}
    _overrides: dict[str, Any] = {}

    # ========================================
    # 싱글톤 컴포넌트 (캐시됨)
    # ========================================

    @classmethod
    def get_knowledge_graph(cls) -> KnowledgeGraph:
        """
        KnowledgeGraph 싱글톤 반환

        Returns:
            KnowledgeGraph 인스턴스
        """
        if "knowledge_graph" in cls._overrides:
            return cls._overrides["knowledge_graph"]

        if "knowledge_graph" not in cls._instances:
            cls._instances["knowledge_graph"] = KnowledgeGraph()

        return cls._instances["knowledge_graph"]

    @classmethod
    def get_reasoner(cls) -> OntologyReasoner:
        """
        OntologyReasoner 싱글톤 반환

        Returns:
            OntologyReasoner 인스턴스
        """
        if "reasoner" in cls._overrides:
            return cls._overrides["reasoner"]

        if "reasoner" not in cls._instances:
            kg = cls.get_knowledge_graph()
            cls._instances["reasoner"] = OntologyReasoner(kg)

        return cls._instances["reasoner"]

    @classmethod
    def get_document_retriever(cls, docs_dir: str = ".") -> DocumentRetriever:
        """
        DocumentRetriever 싱글톤 반환

        Args:
            docs_dir: 문서 디렉토리 경로

        Returns:
            DocumentRetriever 인스턴스
        """
        if "document_retriever" in cls._overrides:
            return cls._overrides["document_retriever"]

        if "document_retriever" not in cls._instances:
            cls._instances["document_retriever"] = DocumentRetriever(docs_dir)

        return cls._instances["document_retriever"]

    @classmethod
    def get_hybrid_retriever(cls) -> HybridRetriever:
        """
        HybridRetriever 싱글톤 반환

        Returns:
            HybridRetriever 인스턴스
        """
        if "hybrid_retriever" in cls._overrides:
            return cls._overrides["hybrid_retriever"]

        if "hybrid_retriever" not in cls._instances:
            cls._instances["hybrid_retriever"] = HybridRetriever(
                knowledge_graph=cls.get_knowledge_graph(),
                reasoner=cls.get_reasoner(),
                doc_retriever=cls.get_document_retriever(),
                auto_init_rules=True,
            )

        return cls._instances["hybrid_retriever"]

    @classmethod
    def get_unified_retriever(cls, docs_path: str = "./docs"):
        """
        HybridRetriever (OWL strategy 포함) 싱글톤 반환.

        Feature flag에 따라 OWL strategy를 주입합니다.
        이전 UnifiedRetriever facade를 대체합니다.

        Args:
            docs_path: 문서 디렉토리 경로

        Returns:
            HybridRetriever 인스턴스 (retrieve_unified() 지원)
        """
        if "unified_retriever" in cls._overrides:
            return cls._overrides["unified_retriever"]

        if "unified_retriever" not in cls._instances:
            from src.infrastructure.feature_flags import FeatureFlags
            from src.rag.hybrid_retriever import HybridRetriever

            kg = cls.get_knowledge_graph()
            owl_strategy = None

            flags = FeatureFlags.get_instance()
            if flags.use_owl_strategy():
                try:
                    from src.ontology.owl_reasoner import OWLREADY2_AVAILABLE, OWLReasoner

                    if OWLREADY2_AVAILABLE:
                        from src.rag.retrieval_strategy import OWLRetrievalStrategy

                        owl_reasoner = OWLReasoner()
                        owl_strategy = OWLRetrievalStrategy(
                            knowledge_graph=kg,
                            owl_reasoner=owl_reasoner,
                            docs_path=docs_path,
                        )
                except Exception:
                    pass  # fall back to legacy

            cls._instances["unified_retriever"] = HybridRetriever(
                knowledge_graph=kg,
                reasoner=cls.get_reasoner(),
                doc_retriever=cls.get_document_retriever(),
                auto_init_rules=True,
                owl_strategy=owl_strategy,
            )

        return cls._instances["unified_retriever"]

    @classmethod
    def get_category_service(cls):
        """
        CategoryService 싱글톤 반환

        Returns:
            CategoryService 인스턴스
        """
        if "category_service" in cls._overrides:
            return cls._overrides["category_service"]

        if "category_service" not in cls._instances:
            from src.ontology.category_service import CategoryService

            cls._instances["category_service"] = CategoryService(cls.get_knowledge_graph())

        return cls._instances["category_service"]

    @classmethod
    def get_sentiment_service(cls):
        """
        SentimentService 싱글톤 반환

        Returns:
            SentimentService 인스턴스
        """
        if "sentiment_service" in cls._overrides:
            return cls._overrides["sentiment_service"]

        if "sentiment_service" not in cls._instances:
            from src.ontology.sentiment_service import SentimentService

            cls._instances["sentiment_service"] = SentimentService(cls.get_knowledge_graph())

        return cls._instances["sentiment_service"]

    # ========================================
    # 팩토리 메서드 (매번 새로 생성)
    # ========================================

    @classmethod
    def get_insight_agent(cls):
        """
        HybridInsightAgent 생성 (매번 새 인스턴스)

        의존성은 Container에서 주입됩니다.

        Returns:
            HybridInsightAgent 인스턴스
        """
        if "insight_agent" in cls._overrides:
            return cls._overrides["insight_agent"]

        from src.agents.hybrid_insight_agent import HybridInsightAgent

        return HybridInsightAgent(
            knowledge_graph=cls.get_knowledge_graph(),
            reasoner=cls.get_reasoner(),
            signal_collector=cls.get_external_signal_collector(),
        )

    @classmethod
    def get_chatbot_agent(cls):
        """
        HybridChatbotAgent 생성 (매번 새 인스턴스)

        의존성은 Container에서 주입됩니다.

        Returns:
            HybridChatbotAgent 인스턴스
        """
        if "chatbot_agent" in cls._overrides:
            return cls._overrides["chatbot_agent"]

        from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

        return HybridChatbotAgent(
            knowledge_graph=cls.get_knowledge_graph(), reasoner=cls.get_reasoner()
        )

    @classmethod
    def get_crawler_agent(cls):
        """
        CrawlerAgent 생성 (매번 새 인스턴스)

        Returns:
            CrawlerAgent 인스턴스
        """
        if "crawler_agent" in cls._overrides:
            return cls._overrides["crawler_agent"]

        from src.agents.crawler_agent import CrawlerAgent

        return CrawlerAgent()

    @classmethod
    def get_batch_workflow(cls, **kwargs):
        """
        BatchWorkflow 생성 (매번 새 인스턴스)

        Args:
            **kwargs: BatchWorkflow 생성자 파라미터

        Returns:
            BatchWorkflow 인스턴스
        """
        if "batch_workflow" in cls._overrides:
            return cls._overrides["batch_workflow"]

        from src.application.workflows.batch_workflow import BatchWorkflow

        return BatchWorkflow(**kwargs)

    @classmethod
    def get_alert_agent(cls, **kwargs):
        """
        AlertAgent 생성 (매번 새 인스턴스)

        Args:
            **kwargs: AlertAgent 생성자 파라미터 (state_manager 등)

        Returns:
            AlertAgent 인스턴스
        """
        if "alert_agent" in cls._overrides:
            return cls._overrides["alert_agent"]

        from src.agents.alert_agent import AlertAgent

        return AlertAgent(**kwargs)

    @classmethod
    def get_period_insight_agent(cls, model: str = None):
        """
        PeriodInsightAgent 생성 (매번 새 인스턴스)

        Args:
            model: LLM 모델명 (기본값: gpt-4.1-mini)

        Returns:
            PeriodInsightAgent 인스턴스
        """
        if "period_insight_agent" in cls._overrides:
            return cls._overrides["period_insight_agent"]

        from src.agents.period_insight_agent import PeriodInsightAgent

        if model is not None:
            return PeriodInsightAgent(model=model)
        return PeriodInsightAgent()

    @classmethod
    def get_metrics_agent(cls, **kwargs):
        """
        MetricsAgent 생성 (매번 새 인스턴스)

        Args:
            **kwargs: MetricsAgent 생성자 파라미터 (config_path 등)

        Returns:
            MetricsAgent 인스턴스
        """
        if "metrics_agent" in cls._overrides:
            return cls._overrides["metrics_agent"]

        from src.agents.metrics_agent import MetricsAgent

        return MetricsAgent(**kwargs)

    @classmethod
    def get_storage_agent(cls, **kwargs):
        """
        StorageAgent 생성 (매번 새 인스턴스)

        Args:
            **kwargs: StorageAgent 생성자 파라미터 (spreadsheet_id 등)

        Returns:
            StorageAgent 인스턴스
        """
        if "storage_agent" in cls._overrides:
            return cls._overrides["storage_agent"]

        from src.agents.storage_agent import StorageAgent

        return StorageAgent(**kwargs)

    # ========================================
    # 싱글톤 서비스 컴포넌트
    # ========================================

    @classmethod
    def get_suggestion_engine(cls):
        """
        SuggestionEngine 싱글톤 반환

        Returns:
            SuggestionEngine 인스턴스
        """
        if "suggestion_engine" in cls._overrides:
            return cls._overrides["suggestion_engine"]

        if "suggestion_engine" not in cls._instances:
            from src.agents.suggestion_engine import SuggestionEngine

            cls._instances["suggestion_engine"] = SuggestionEngine(
                knowledge_graph=cls.get_knowledge_graph()
            )

        return cls._instances["suggestion_engine"]

    @classmethod
    def get_source_provider(cls):
        """
        SourceProvider 싱글톤 반환

        Returns:
            SourceProvider 인스턴스
        """
        if "source_provider" in cls._overrides:
            return cls._overrides["source_provider"]

        if "source_provider" not in cls._instances:
            from src.agents.source_provider import SourceProvider

            cls._instances["source_provider"] = SourceProvider(
                knowledge_graph=cls.get_knowledge_graph()
            )

        return cls._instances["source_provider"]

    @classmethod
    def get_external_signal_manager(cls):
        """
        ExternalSignalManager 싱글톤 반환

        Returns:
            ExternalSignalManager 인스턴스
        """
        if "external_signal_manager" in cls._overrides:
            return cls._overrides["external_signal_manager"]

        if "external_signal_manager" not in cls._instances:
            from src.agents.external_signal_manager import ExternalSignalManager

            cls._instances["external_signal_manager"] = ExternalSignalManager()

        return cls._instances["external_signal_manager"]

    @classmethod
    def get_market_intelligence_engine(cls):
        """
        MarketIntelligenceEngine 싱글톤 반환

        Returns:
            MarketIntelligenceEngine 인스턴스
        """
        if "market_intelligence_engine" in cls._overrides:
            return cls._overrides["market_intelligence_engine"]

        if "market_intelligence_engine" not in cls._instances:
            from src.tools.intelligence.market_intelligence import MarketIntelligenceEngine

            cls._instances["market_intelligence_engine"] = MarketIntelligenceEngine()

        return cls._instances["market_intelligence_engine"]

    # ========================================
    # Application Layer 워크플로우
    # ========================================

    @classmethod
    def get_chat_workflow(cls):
        """
        ChatWorkflow 생성 (매번 새 인스턴스)

        의존성은 Container에서 주입됩니다.

        Returns:
            ChatWorkflow 인스턴스
        """
        if "chat_workflow" in cls._overrides:
            return cls._overrides["chat_workflow"]

        from src.application.workflows.chat_workflow import ChatWorkflow

        return ChatWorkflow(
            chatbot=cls.get_chatbot_agent(),
            retriever=cls.get_hybrid_retriever(),
        )

    @classmethod
    def get_crawl_workflow(cls, scraper=None, storage=None, metric_calculator=None):
        """
        CrawlWorkflow 생성 (매번 새 인스턴스)

        Args:
            scraper: ScraperProtocol 구현 (None이면 기본 CrawlerAgent)
            storage: StorageProtocol 구현 (None이면 기본 StorageAgent)
            metric_calculator: MetricCalculatorProtocol 구현

        Returns:
            CrawlWorkflow 인스턴스
        """
        if "crawl_workflow" in cls._overrides:
            return cls._overrides["crawl_workflow"]

        from src.application.workflows.crawl_workflow import CrawlWorkflow

        if scraper is None:
            scraper = cls.get_crawler_agent()
        if storage is None:
            storage = cls.get_storage_agent()
        if metric_calculator is None:
            metric_calculator = cls.get_metrics_agent()

        return CrawlWorkflow(
            scraper=scraper,
            storage=storage,
            metric_calculator=metric_calculator,
        )

    @classmethod
    def get_insight_workflow(cls, storage=None):
        """
        InsightWorkflow 생성 (매번 새 인스턴스)

        Args:
            storage: StorageProtocol 구현 (None이면 기본 StorageAgent)

        Returns:
            InsightWorkflow 인스턴스
        """
        if "insight_workflow" in cls._overrides:
            return cls._overrides["insight_workflow"]

        from src.application.workflows.insight_workflow import InsightWorkflow

        if storage is None:
            storage = cls.get_storage_agent()

        return InsightWorkflow(
            insight_agent=cls.get_insight_agent(),
            storage=storage,
        )

    @classmethod
    def get_sqlite_storage(cls):
        """Get SQLite storage singleton."""
        if "sqlite_storage" in cls._overrides:
            return cls._overrides["sqlite_storage"]
        if "sqlite_storage" not in cls._instances:
            from src.tools.storage.sqlite_storage import get_sqlite_storage

            cls._instances["sqlite_storage"] = get_sqlite_storage()
        return cls._instances["sqlite_storage"]

    @classmethod
    def get_state_manager(cls):
        """Get state manager singleton."""
        if "state_manager" in cls._overrides:
            return cls._overrides["state_manager"]
        if "state_manager" not in cls._instances:
            from src.core.state_manager import get_state_manager

            cls._instances["state_manager"] = get_state_manager()
        return cls._instances["state_manager"]

    @classmethod
    def get_alert_service(cls):
        """Get AlertService singleton."""
        if "alert_service" in cls._overrides:
            return cls._overrides["alert_service"]
        if "alert_service" not in cls._instances:
            from src.tools.notifications.alert_service import get_alert_service

            cls._instances["alert_service"] = get_alert_service()
        return cls._instances["alert_service"]

    @classmethod
    def get_period_analyzer(cls):
        """PeriodAnalyzer 싱글톤 반환"""
        if "period_analyzer" in cls._overrides:
            return cls._overrides["period_analyzer"]
        if "period_analyzer" not in cls._instances:
            from src.tools.calculators.period_analyzer import PeriodAnalyzer

            cls._instances["period_analyzer"] = PeriodAnalyzer()
        return cls._instances["period_analyzer"]

    @classmethod
    def get_rag_router(cls):
        """Get RAGRouter singleton."""
        if "rag_router" in cls._overrides:
            return cls._overrides["rag_router"]
        if "rag_router" not in cls._instances:
            from src.rag.router import RAGRouter

            cls._instances["rag_router"] = RAGRouter()
        return cls._instances["rag_router"]

    @classmethod
    def get_external_signal_collector(cls):
        """Get ExternalSignalCollector singleton."""
        if "external_signal_collector" in cls._overrides:
            return cls._overrides["external_signal_collector"]
        if "external_signal_collector" not in cls._instances:
            from src.tools.collectors.external_signal_collector import ExternalSignalCollector

            cls._instances["external_signal_collector"] = ExternalSignalCollector()
        return cls._instances["external_signal_collector"]

    # ========================================
    # 인텐트 기반 전략 선택
    # ========================================

    @classmethod
    def get_retrieval_config_for_intent(cls, intent):
        """
        인텐트에 맞는 RetrievalConfig 반환

        Phase 1A의 UnifiedIntent와 Phase 2A의 IntentRetrievalConfig를 연결합니다.

        Args:
            intent: UnifiedIntent enum 값

        Returns:
            IntentRetrievalConfig (weights, top_k, doc_type_filter)
        """
        from src.rag.retrieval_strategy import get_intent_retrieval_config

        return get_intent_retrieval_config(intent)

    # ========================================
    # 유틸리티 메서드
    # ========================================

    @classmethod
    def override(cls, name: str, instance: Any) -> None:
        """
        테스트용 Mock 주입

        Args:
            name: 컴포넌트 이름
            instance: 주입할 인스턴스

        Example:
            Container.override('knowledge_graph', mock_kg)
        """
        cls._overrides[name] = instance

    @classmethod
    def reset(cls) -> None:
        """
        모든 인스턴스 및 오버라이드 초기화

        테스트 간 격리 또는 애플리케이션 재시작 시 사용
        """
        cls._instances.clear()
        cls._overrides.clear()

    @classmethod
    @contextmanager
    def test_override(cls, name: str, instance: Any):
        """
        테스트용 임시 오버라이드 (context manager)

        Args:
            name: 컴포넌트 이름
            instance: 주입할 인스턴스

        Example:
            with Container.test_override('knowledge_graph', mock_kg):
                agent = Container.get_insight_agent()
                # mock_kg 사용
            # 원래 값 복원

        Yields:
            None
        """
        # 현재 상태 저장
        had_override = name in cls._overrides
        old_override = cls._overrides.get(name)

        had_instance = name in cls._instances
        old_instance = cls._instances.get(name)

        # 오버라이드 설정
        cls._overrides[name] = instance

        # 인스턴스 캐시 제거 (오버라이드가 적용되도록)
        if name in cls._instances:
            del cls._instances[name]

        try:
            yield
        finally:
            # 오버라이드 복원
            if had_override:
                cls._overrides[name] = old_override
            elif name in cls._overrides:
                del cls._overrides[name]

            # 인스턴스 복원
            if had_instance:
                cls._instances[name] = old_instance
            elif name in cls._instances:
                del cls._instances[name]


# 편의를 위한 별칭
container = Container

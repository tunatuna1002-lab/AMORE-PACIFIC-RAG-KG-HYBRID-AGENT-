"""
ApplicationContainer (bootstrap) 단위 테스트

테스트 대상: src/infrastructure/bootstrap.py
Coverage target: 40%+
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.bootstrap import (
    ApplicationContainer,
    get_container,
    initialize_app,
)

# =============================================================================
# Fixture
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """각 테스트 전후로 싱글톤 리셋"""
    ApplicationContainer.reset()
    yield
    ApplicationContainer.reset()


# =============================================================================
# 싱글톤 패턴 테스트
# =============================================================================


class TestSingletonPattern:
    """ApplicationContainer 싱글톤 테스트"""

    def test_not_initialized_by_default(self):
        """기본적으로 초기화되지 않은 상태"""
        assert ApplicationContainer.is_initialized() is False

    def test_get_raises_when_not_initialized(self):
        """초기화 전 get() 호출 시 RuntimeError"""
        with pytest.raises(RuntimeError, match="not initialized"):
            ApplicationContainer.get()

    def test_reset(self):
        """reset() 호출 시 싱글톤 초기화"""
        ApplicationContainer._instance = MagicMock()
        assert ApplicationContainer.is_initialized() is True

        ApplicationContainer.reset()
        assert ApplicationContainer.is_initialized() is False

    def test_is_initialized_true(self):
        """인스턴스 있으면 True"""
        ApplicationContainer._instance = MagicMock()
        assert ApplicationContainer.is_initialized() is True

    def test_get_container_alias(self):
        """get_container는 ApplicationContainer.get()의 별칭"""
        with pytest.raises(RuntimeError, match="not initialized"):
            get_container()


# =============================================================================
# create 테스트
# =============================================================================


class TestApplicationContainerCreate:
    """ApplicationContainer.create 테스트"""

    @pytest.mark.asyncio
    async def test_create_with_env_config(self):
        """환경변수 기반 설정으로 생성"""
        with (
            patch("src.infrastructure.bootstrap.AppConfig.from_env") as mock_from_env,
            patch.object(
                ApplicationContainer,
                "_initialize_components",
                new_callable=AsyncMock,
            ),
        ):
            mock_config = MagicMock()
            mock_from_env.return_value = mock_config

            container = await ApplicationContainer.create()

            assert container is not None
            assert container.config is mock_config
            assert ApplicationContainer.is_initialized() is True
            mock_from_env.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_with_config_path(self, tmp_path):
        """설정 파일 경로로 생성"""
        with (
            patch("src.infrastructure.bootstrap.AppConfig.from_files") as mock_from_files,
            patch.object(
                ApplicationContainer,
                "_initialize_components",
                new_callable=AsyncMock,
            ),
        ):
            mock_config = MagicMock()
            mock_from_files.return_value = mock_config

            container = await ApplicationContainer.create(config_path=tmp_path)

            assert container.config is mock_config
            mock_from_files.assert_called_once_with(tmp_path)

    @pytest.mark.asyncio
    async def test_create_sets_singleton(self):
        """create 후 싱글톤 접근 가능"""
        with (
            patch("src.infrastructure.bootstrap.AppConfig.from_env") as mock_from_env,
            patch.object(
                ApplicationContainer,
                "_initialize_components",
                new_callable=AsyncMock,
            ),
        ):
            mock_from_env.return_value = MagicMock()

            container = await ApplicationContainer.create()
            same = ApplicationContainer.get()

            assert container is same


# =============================================================================
# _initialize_components 테스트
# =============================================================================


class TestInitializeComponents:
    """ApplicationContainer._initialize_components 테스트

    Note: _initialize_components uses local imports (from src.ontology... import ...),
    so we must patch at the source module paths, not at src.infrastructure.bootstrap.
    """

    @pytest.mark.asyncio
    async def test_initializes_core_components(self):
        """KnowledgeGraph + OntologyReasoner 초기화"""
        mock_config = MagicMock()
        mock_kg = MagicMock()
        mock_reasoner = MagicMock()

        with (
            patch(
                "src.ontology.knowledge_graph.KnowledgeGraph", return_value=mock_kg
            ) as mock_kg_cls,
            patch(
                "src.ontology.reasoner.OntologyReasoner", return_value=mock_reasoner
            ) as mock_reasoner_cls,
            patch(
                "src.rag.hybrid_retriever.HybridRetriever",
                side_effect=ImportError("not available"),
            ),
            patch(
                "src.core.brain.get_brain",
                side_effect=ImportError("not available"),
            ),
            patch(
                "src.core.crawl_manager.get_crawl_manager",
                side_effect=ImportError("not available"),
            ),
        ):
            container = ApplicationContainer(config=mock_config)
            await container._initialize_components()

            assert container._knowledge_graph is mock_kg
            assert container._reasoner is mock_reasoner
            mock_reasoner_cls.assert_called_once_with(mock_kg)

    @pytest.mark.asyncio
    async def test_initializes_hybrid_retriever(self):
        """HybridRetriever 초기화"""
        mock_config = MagicMock()
        mock_hr = MagicMock()
        mock_hr.initialize = AsyncMock()

        with (
            patch("src.ontology.knowledge_graph.KnowledgeGraph", return_value=MagicMock()),
            patch("src.ontology.reasoner.OntologyReasoner", return_value=MagicMock()),
            patch("src.rag.hybrid_retriever.HybridRetriever", return_value=mock_hr),
            patch(
                "src.core.brain.get_brain",
                side_effect=ImportError("not available"),
            ),
            patch(
                "src.core.crawl_manager.get_crawl_manager",
                side_effect=ImportError("not available"),
            ),
        ):
            container = ApplicationContainer(config=mock_config)
            await container._initialize_components()

            assert container._hybrid_retriever is mock_hr
            mock_hr.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_retriever_import_error(self):
        """HybridRetriever import 실패 시 None"""
        mock_config = MagicMock()

        with (
            patch("src.ontology.knowledge_graph.KnowledgeGraph", return_value=MagicMock()),
            patch("src.ontology.reasoner.OntologyReasoner", return_value=MagicMock()),
            patch(
                "src.rag.hybrid_retriever.HybridRetriever",
                side_effect=ImportError("not available"),
            ),
            patch(
                "src.core.brain.get_brain",
                side_effect=ImportError("not available"),
            ),
            patch(
                "src.core.crawl_manager.get_crawl_manager",
                side_effect=ImportError("not available"),
            ),
        ):
            container = ApplicationContainer(config=mock_config)
            await container._initialize_components()
            assert container._hybrid_retriever is None

    @pytest.mark.asyncio
    async def test_brain_import_error(self):
        """UnifiedBrain import 실패 시 None"""
        mock_config = MagicMock()

        with (
            patch("src.ontology.knowledge_graph.KnowledgeGraph", return_value=MagicMock()),
            patch("src.ontology.reasoner.OntologyReasoner", return_value=MagicMock()),
            patch(
                "src.rag.hybrid_retriever.HybridRetriever",
                side_effect=ImportError("not available"),
            ),
            patch(
                "src.core.brain.get_brain",
                side_effect=ImportError("not available"),
            ),
            patch(
                "src.core.crawl_manager.get_crawl_manager",
                side_effect=ImportError("not available"),
            ),
        ):
            container = ApplicationContainer(config=mock_config)
            await container._initialize_components()
            assert container._brain is None

    @pytest.mark.asyncio
    async def test_crawl_manager_import_error(self):
        """CrawlManager import 실패 시 None"""
        mock_config = MagicMock()

        with (
            patch("src.ontology.knowledge_graph.KnowledgeGraph", return_value=MagicMock()),
            patch("src.ontology.reasoner.OntologyReasoner", return_value=MagicMock()),
            patch(
                "src.rag.hybrid_retriever.HybridRetriever",
                side_effect=ImportError("not available"),
            ),
            patch(
                "src.core.brain.get_brain",
                side_effect=ImportError("not available"),
            ),
            patch(
                "src.core.crawl_manager.get_crawl_manager",
                side_effect=ImportError("not available"),
            ),
        ):
            container = ApplicationContainer(config=mock_config)
            await container._initialize_components()
            assert container._crawl_manager is None

    @pytest.mark.asyncio
    async def test_core_component_error_raises(self):
        """핵심 컴포넌트 오류 시 예외 전파"""
        mock_config = MagicMock()

        with patch(
            "src.ontology.knowledge_graph.KnowledgeGraph",
            side_effect=Exception("KG init failed"),
        ):
            container = ApplicationContainer(config=mock_config)
            with pytest.raises(Exception, match="KG init failed"):
                await container._initialize_components()


# =============================================================================
# Property 테스트
# =============================================================================


class TestProperties:
    """Property accessor 테스트"""

    def test_brain_property(self):
        """brain 프로퍼티"""
        mock_config = MagicMock()
        container = ApplicationContainer(config=mock_config)
        container._brain = MagicMock()
        assert container.brain is container._brain

    def test_crawl_manager_property(self):
        """crawl_manager 프로퍼티"""
        mock_config = MagicMock()
        container = ApplicationContainer(config=mock_config)
        container._crawl_manager = MagicMock()
        assert container.crawl_manager is container._crawl_manager

    def test_knowledge_graph_property(self):
        """knowledge_graph 프로퍼티"""
        mock_config = MagicMock()
        container = ApplicationContainer(config=mock_config)
        container._knowledge_graph = MagicMock()
        assert container.knowledge_graph is container._knowledge_graph

    def test_reasoner_property(self):
        """reasoner 프로퍼티"""
        mock_config = MagicMock()
        container = ApplicationContainer(config=mock_config)
        container._reasoner = MagicMock()
        assert container.reasoner is container._reasoner

    def test_hybrid_retriever_property(self):
        """hybrid_retriever 프로퍼티"""
        mock_config = MagicMock()
        container = ApplicationContainer(config=mock_config)
        container._hybrid_retriever = MagicMock()
        assert container.hybrid_retriever is container._hybrid_retriever

    def test_properties_default_none(self):
        """프로퍼티 기본값 None"""
        mock_config = MagicMock()
        container = ApplicationContainer(config=mock_config)
        assert container.brain is None
        assert container.crawl_manager is None
        assert container.knowledge_graph is None
        assert container.reasoner is None
        assert container.hybrid_retriever is None


# =============================================================================
# initialize_app 테스트
# =============================================================================


class TestInitializeApp:
    """initialize_app 함수 테스트"""

    @pytest.mark.asyncio
    async def test_initialize_app(self):
        """initialize_app은 ApplicationContainer.create() 호출"""
        with (
            patch("src.infrastructure.bootstrap.AppConfig.from_env") as mock_from_env,
            patch.object(
                ApplicationContainer,
                "_initialize_components",
                new_callable=AsyncMock,
            ),
        ):
            mock_from_env.return_value = MagicMock()

            container = await initialize_app()

            assert isinstance(container, ApplicationContainer)
            assert ApplicationContainer.is_initialized() is True

    @pytest.mark.asyncio
    async def test_initialize_app_with_path(self, tmp_path):
        """config_path 전달"""
        with (
            patch("src.infrastructure.bootstrap.AppConfig.from_files") as mock_from_files,
            patch.object(
                ApplicationContainer,
                "_initialize_components",
                new_callable=AsyncMock,
            ),
        ):
            mock_from_files.return_value = MagicMock()

            container = await initialize_app(config_path=tmp_path)

            assert isinstance(container, ApplicationContainer)
            mock_from_files.assert_called_once_with(tmp_path)

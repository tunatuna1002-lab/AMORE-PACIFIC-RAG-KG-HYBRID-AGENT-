"""
Application Bootstrap & Dependency Injection Container
======================================================
애플리케이션 부트스트랩 및 의존성 주입 컨테이너

모든 의존성을 중앙에서 생성하고 연결합니다.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

from src.infrastructure.config.config_manager import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class ApplicationContainer:
    """
    Application Dependency Container

    모든 애플리케이션 의존성을 관리합니다.
    Singleton 패턴으로 전역 접근을 제공합니다.

    Usage:
        # Initialize (once at startup)
        container = await ApplicationContainer.create()

        # Access anywhere
        container = ApplicationContainer.get()
        brain = container.brain
    """

    config: AppConfig

    # Lazy-loaded components (Optional to avoid import issues during init)
    _brain: object | None = None
    _crawl_manager: object | None = None
    _knowledge_graph: object | None = None
    _reasoner: object | None = None
    _hybrid_retriever: object | None = None

    # Singleton instance
    _instance: ClassVar[Optional["ApplicationContainer"]] = None

    @classmethod
    async def create(cls, config_path: Path | None = None) -> "ApplicationContainer":
        """
        Create and initialize the application container.

        Args:
            config_path: Optional path to config directory

        Returns:
            Initialized ApplicationContainer
        """
        logger.info("Creating ApplicationContainer...")

        # Load configuration
        if config_path:
            config = AppConfig.from_files(config_path)
        else:
            config = AppConfig.from_env()

        container = cls(config=config)

        # Initialize components lazily (avoid circular imports)
        await container._initialize_components()

        cls._instance = container
        logger.info("ApplicationContainer created successfully")
        return container

    async def _initialize_components(self) -> None:
        """Initialize all application components with proper dependency wiring."""
        try:
            # Import here to avoid circular dependencies
            from src.ontology.knowledge_graph import KnowledgeGraph
            from src.ontology.reasoner import OntologyReasoner

            # Create core domain components
            self._knowledge_graph = KnowledgeGraph()
            self._reasoner = OntologyReasoner(self._knowledge_graph)

            logger.info("Core components initialized: KnowledgeGraph, OntologyReasoner")

            # Try to initialize optional components
            try:
                from src.rag.hybrid_retriever import HybridRetriever

                self._hybrid_retriever = HybridRetriever(
                    knowledge_graph=self._knowledge_graph, reasoner=self._reasoner
                )
                await self._hybrid_retriever.initialize()
                logger.info("HybridRetriever initialized")
            except ImportError as e:
                logger.warning(f"HybridRetriever not available: {e}")

            # Try to initialize Brain (optional, may have dependencies)
            try:
                from src.core.brain import UnifiedBrain, get_brain

                self._brain = await get_brain()  # Use existing singleton (async)
                logger.info("UnifiedBrain connected")
            except ImportError as e:
                logger.warning(f"UnifiedBrain not available: {e}")

            # Try to initialize CrawlManager (optional)
            try:
                from src.core.crawl_manager import CrawlManager, get_crawl_manager

                self._crawl_manager = await get_crawl_manager()
                logger.info("CrawlManager connected")
            except ImportError as e:
                logger.warning(f"CrawlManager not available: {e}")

        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise

    @classmethod
    def get(cls) -> "ApplicationContainer":
        """
        Get the singleton instance.

        Raises:
            RuntimeError: If container not initialized

        Returns:
            The application container
        """
        if cls._instance is None:
            raise RuntimeError(
                "ApplicationContainer not initialized. "
                "Call 'await ApplicationContainer.create()' first."
            )
        return cls._instance

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if container is initialized."""
        return cls._instance is not None

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._instance = None

    # Property accessors for lazy-loaded components
    @property
    def brain(self):
        """Get UnifiedBrain instance."""
        return self._brain

    @property
    def crawl_manager(self):
        """Get CrawlManager instance."""
        return self._crawl_manager

    @property
    def knowledge_graph(self):
        """Get KnowledgeGraph instance."""
        return self._knowledge_graph

    @property
    def reasoner(self):
        """Get OntologyReasoner instance."""
        return self._reasoner

    @property
    def hybrid_retriever(self):
        """Get HybridRetriever instance."""
        return self._hybrid_retriever


# Convenience function for quick access
def get_container() -> ApplicationContainer:
    """Get the application container (alias for ApplicationContainer.get())."""
    return ApplicationContainer.get()


async def initialize_app(config_path: Path | None = None) -> ApplicationContainer:
    """
    Initialize the application.

    This is the main entry point for application setup.

    Args:
        config_path: Optional path to config directory

    Returns:
        Initialized ApplicationContainer
    """
    return await ApplicationContainer.create(config_path)

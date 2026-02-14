"""
Unified Retriever Facade
=========================
Facade that wraps both HybridRetriever and TrueHybridRetriever behind a single interface.

Architecture:
    User Code
        ↓
    UnifiedRetriever (Facade)
        ↓
    ┌─────────────┴─────────────┐
    │                           │
    HybridRetriever    TrueHybridRetriever
    (legacy)           (OWL-based)

Selection logic:
1. Feature flag `use_unified_retriever` must be True
2. If `use_true_hybrid_retriever` flag is True AND owlready2 available → TrueHybridRetriever
3. Otherwise → HybridRetriever (legacy)

Returns:
- UnifiedRetrievalResult regardless of backend used
- Converts HybridContext → UnifiedRetrievalResult
- Converts HybridResult → UnifiedRetrievalResult
"""

import logging
from typing import Any

from src.domain.interfaces.retriever import UnifiedRetrievalResult
from src.infrastructure.feature_flags import FeatureFlags

logger = logging.getLogger(__name__)


class UnifiedRetriever:
    """Facade that provides a unified interface to both retriever backends.

    This class automatically selects the appropriate backend based on:
    - Feature flags (use_true_hybrid_retriever)
    - owlready2 availability
    - Fallback to legacy HybridRetriever if TrueHybridRetriever unavailable

    All results are normalized to UnifiedRetrievalResult for consistent downstream handling.

    Attributes:
        backend: The actual retriever instance (HybridRetriever or TrueHybridRetriever)
        backend_type: String identifier of active backend ("hybrid" | "true_hybrid")
        knowledge_graph: Shared knowledge graph instance
        config: Optional configuration dict
    """

    def __init__(self, knowledge_graph=None, config: dict[str, Any] | None = None):
        """Initialize UnifiedRetriever with backend selection.

        Args:
            knowledge_graph: KnowledgeGraph instance (required for both backends)
            config: Optional configuration dict
        """
        self.knowledge_graph = knowledge_graph
        self.config = config or {}
        self.backend = None
        self.backend_type = "unknown"

        # Get feature flags
        flags = FeatureFlags.get_instance()

        # Determine which backend to use
        use_true_hybrid = flags.use_true_hybrid_retriever()

        if use_true_hybrid:
            # Try to use TrueHybridRetriever with OWL
            try:
                from src.ontology.owl_reasoner import OWLREADY2_AVAILABLE

                if OWLREADY2_AVAILABLE:
                    from src.rag.true_hybrid_retriever import get_true_hybrid_retriever

                    logger.info("UnifiedRetriever: Selecting TrueHybridRetriever (OWL available)")
                    self.backend = get_true_hybrid_retriever(
                        knowledge_graph=knowledge_graph,
                        docs_path=self.config.get("docs_path", "./docs"),
                    )
                    self.backend_type = "true_hybrid"
                else:
                    raise ImportError("owlready2 not available")
            except Exception as e:
                logger.warning(
                    f"UnifiedRetriever: TrueHybridRetriever unavailable ({e}), "
                    f"falling back to HybridRetriever"
                )
                use_true_hybrid = False

        if not use_true_hybrid:
            # Use legacy HybridRetriever
            from src.ontology.reasoner import OntologyReasoner
            from src.rag.hybrid_retriever import HybridRetriever
            from src.rag.retriever import DocumentRetriever

            logger.info("UnifiedRetriever: Selecting HybridRetriever (legacy)")

            reasoner = OntologyReasoner(knowledge_graph)
            doc_retriever = DocumentRetriever(docs_dir=self.config.get("docs_path", "./docs"))

            self.backend = HybridRetriever(
                knowledge_graph=knowledge_graph,
                reasoner=reasoner,
                document_retriever=doc_retriever,
            )
            self.backend_type = "hybrid"

        logger.info(f"UnifiedRetriever initialized with backend: {self.backend_type}")

    async def initialize(self) -> None:
        """Initialize the underlying retriever backend."""
        if self.backend and hasattr(self.backend, "initialize"):
            await self.backend.initialize()

    async def retrieve(
        self,
        query: str,
        current_metrics: dict[str, Any] | None = None,
        top_k: int = 5,
        **kwargs,
    ) -> UnifiedRetrievalResult:
        """Retrieve context using the selected backend.

        Args:
            query: User query
            current_metrics: Current dashboard metrics (optional)
            top_k: Maximum number of results to return
            **kwargs: Additional backend-specific arguments

        Returns:
            UnifiedRetrievalResult with normalized fields

        Raises:
            ValueError: If backend not initialized
        """
        if self.backend is None:
            raise ValueError("UnifiedRetriever backend not initialized")

        # Call the appropriate backend
        if self.backend_type == "true_hybrid":
            result = await self.backend.retrieve(
                query=query, current_metrics=current_metrics, top_k=top_k, **kwargs
            )
            return self._convert_hybrid_result(result, query)
        elif self.backend_type == "hybrid":
            # HybridRetriever uses different parameter names
            context = await self.backend.retrieve(
                query=query,
                current_metrics=current_metrics,
                include_explanations=kwargs.get("include_explanations", True),
            )
            return self._convert_hybrid_context(context, query)
        else:
            raise ValueError(f"Unknown backend type: {self.backend_type}")

    async def search(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search documents using the backend's document retriever.

        Args:
            query: Search query
            top_k: Maximum number of documents
            doc_filter: Optional document filter

        Returns:
            List of document dicts with content, metadata, score
        """
        if self.backend is None:
            raise ValueError("UnifiedRetriever backend not initialized")

        if hasattr(self.backend, "search"):
            return await self.backend.search(query=query, top_k=top_k, doc_filter=doc_filter)
        elif hasattr(self.backend, "document_retriever"):
            # HybridRetriever delegates to document_retriever
            return await self.backend.document_retriever.search(
                query=query, top_k=top_k, doc_filter=doc_filter
            )
        else:
            logger.warning("Backend does not support search(), returning empty list")
            return []

    def _convert_hybrid_context(self, ctx, query: str) -> UnifiedRetrievalResult:
        """Convert HybridContext (from HybridRetriever) to UnifiedRetrievalResult.

        HybridContext fields:
        - query: str
        - entities: dict[str, list[str]]
        - ontology_facts: list[dict]
        - inferences: list[InferenceResult]
        - rag_chunks: list[dict]
        - combined_context: str
        - metadata: dict

        Args:
            ctx: HybridContext instance
            query: Original query string

        Returns:
            UnifiedRetrievalResult
        """
        # Convert InferenceResult objects to dicts
        inferences_dicts = []
        if hasattr(ctx, "inferences"):
            for inf in ctx.inferences:
                if hasattr(inf, "to_dict"):
                    inferences_dicts.append(inf.to_dict())
                elif isinstance(inf, dict):
                    inferences_dicts.append(inf)

        return UnifiedRetrievalResult(
            query=query,
            entities=ctx.entities if hasattr(ctx, "entities") else {},
            ontology_facts=ctx.ontology_facts if hasattr(ctx, "ontology_facts") else [],
            inferences=inferences_dicts,
            rag_chunks=ctx.rag_chunks if hasattr(ctx, "rag_chunks") else [],
            combined_context=ctx.combined_context if hasattr(ctx, "combined_context") else "",
            confidence=0.0,  # HybridRetriever doesn't provide confidence
            entity_links=[],  # HybridRetriever doesn't have entity links
            metadata=ctx.metadata if hasattr(ctx, "metadata") else {},
            retriever_type="hybrid",
        )

    def _convert_hybrid_result(self, result, query: str) -> UnifiedRetrievalResult:
        """Convert HybridResult (from TrueHybridRetriever) to UnifiedRetrievalResult.

        HybridResult fields:
        - query: str
        - documents: list[dict]  (RAG results)
        - ontology_context: dict (with 'facts' and 'inferences' keys)
        - entity_links: list[LinkedEntity]
        - confidence: float
        - combined_context: str
        - metadata: dict

        Args:
            result: HybridResult instance
            query: Original query string

        Returns:
            UnifiedRetrievalResult
        """
        # Extract entities from entity_links
        entities_dict = {"brands": [], "categories": [], "indicators": [], "products": []}

        if hasattr(result, "entity_links"):
            for entity in result.entity_links:
                entity_type = (
                    entity.entity_type.value
                    if hasattr(entity.entity_type, "value")
                    else entity.entity_type
                )
                entity_id = (
                    entity.ontology_id
                    if hasattr(entity, "ontology_id")
                    else getattr(entity, "concept_label", entity.text)
                )

                if entity_type in entities_dict:
                    entities_dict[entity_type].append(entity_id)
                elif entity_type == "brand":
                    entities_dict["brands"].append(entity_id)
                elif entity_type == "category":
                    entities_dict["categories"].append(entity_id)
                elif entity_type == "indicator":
                    entities_dict["indicators"].append(entity_id)
                elif entity_type == "product":
                    entities_dict["products"].append(entity_id)

        # Extract ontology facts and inferences
        ontology_context = result.ontology_context if hasattr(result, "ontology_context") else {}
        ontology_facts = ontology_context.get("facts", [])
        inferences = ontology_context.get("inferences", [])

        # documents → rag_chunks
        rag_chunks = result.documents if hasattr(result, "documents") else []

        return UnifiedRetrievalResult(
            query=query,
            entities=entities_dict,
            ontology_facts=ontology_facts,
            inferences=inferences,
            rag_chunks=rag_chunks,
            combined_context=result.combined_context if hasattr(result, "combined_context") else "",
            confidence=result.confidence if hasattr(result, "confidence") else 0.0,
            entity_links=result.entity_links if hasattr(result, "entity_links") else [],
            metadata=result.metadata if hasattr(result, "metadata") else {},
            retriever_type="true_hybrid",
        )


# Factory function for convenience
def get_unified_retriever(knowledge_graph=None, config: dict[str, Any] | None = None):
    """Factory function to create a UnifiedRetriever instance.

    Args:
        knowledge_graph: KnowledgeGraph instance
        config: Optional configuration dict (can include 'docs_path')

    Returns:
        UnifiedRetriever instance
    """
    return UnifiedRetriever(knowledge_graph=knowledge_graph, config=config)

"""
Base Hybrid Agent
=================
Shared foundation for ontology-RAG hybrid agents.
Template Method pattern: subclasses override specialized steps.

Shared components:
- KnowledgeGraph + OntologyReasoner (ontology layer)
- HybridRetriever (RAG + KG unified search)
- ContextBuilder + ResponseTemplates (context / formatting)
- AgentLogger + ExecutionTracer + QualityMetrics (monitoring)
"""

from __future__ import annotations

import logging
from typing import Any

from src.monitoring.logger import AgentLogger
from src.monitoring.metrics import QualityMetrics
from src.monitoring.tracer import ExecutionTracer
from src.ontology.business_rules import register_all_rules
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.rag.context_builder import ContextBuilder
from src.rag.hybrid_retriever import HybridContext, HybridRetriever
from src.rag.retriever import DocumentRetriever
from src.rag.templates import ResponseTemplates

logger = logging.getLogger(__name__)


class BaseHybridAgent:
    """
    Base class for ontology-RAG hybrid agents.

    Provides shared initialisation and retrieval pipeline.
    Subclasses implement their specific processing logic.

    Attributes set by this base:
        kg, reasoner, doc_retriever, hybrid_retriever,
        context_builder, templates, logger, tracer, metrics,
        model, _last_hybrid_context
    """

    # Subclasses override this for default logger name
    AGENT_NAME: str = "base_hybrid"

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        docs_dir: str = ".",
        knowledge_graph: KnowledgeGraph | None = None,
        reasoner: OntologyReasoner | None = None,
        agent_logger: AgentLogger | None = None,
        tracer: ExecutionTracer | None = None,
        metrics: QualityMetrics | None = None,
        context_builder_max_tokens: int = 4000,
        auto_init_rules: bool = True,
    ):
        self.model = model

        # Ontology components
        self.kg = knowledge_graph or KnowledgeGraph()
        self.reasoner = reasoner or OntologyReasoner(self.kg)

        # Business rules
        if auto_init_rules and not self.reasoner.rules:
            register_all_rules(self.reasoner)

        # RAG components
        self.doc_retriever = DocumentRetriever(docs_dir)
        self.hybrid_retriever = HybridRetriever(
            knowledge_graph=self.kg,
            reasoner=self.reasoner,
            doc_retriever=self.doc_retriever,
            auto_init_rules=False,
        )

        # Context building
        self.context_builder = ContextBuilder(max_tokens=context_builder_max_tokens)
        self.templates = ResponseTemplates()

        # Monitoring
        self.logger = agent_logger or AgentLogger(self.AGENT_NAME)
        self.tracer = tracer
        self.metrics = metrics

        # State
        self._last_hybrid_context: HybridContext | None = None

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    async def _run_hybrid_retrieval(
        self,
        query: str,
        current_metrics: dict[str, Any] | None = None,
        include_explanations: bool = True,
    ) -> HybridContext:
        """
        Run the hybrid retrieval pipeline (ontology + RAG).

        Args:
            query: Search query
            current_metrics: Current dashboard metrics
            include_explanations: Include reasoning explanations

        Returns:
            HybridContext with ontology facts, inferences, RAG chunks
        """
        if not self.hybrid_retriever._initialized:
            await self.hybrid_retriever.initialize()

        context = await self.hybrid_retriever.retrieve(
            query=query,
            current_metrics=current_metrics,
            include_explanations=include_explanations,
        )
        self._last_hybrid_context = context
        return context

    def _trace_span(self, name: str, action: str = "start") -> None:
        """Helper for tracing spans."""
        if self.tracer:
            if action == "start":
                self.tracer.start_span(name)
            else:
                self.tracer.end_span(action)

    # ------------------------------------------------------------------
    # Common accessors
    # ------------------------------------------------------------------

    def get_last_hybrid_context(self) -> HybridContext | None:
        """Return the last hybrid context."""
        return self._last_hybrid_context

    def get_knowledge_graph(self) -> KnowledgeGraph:
        """Return the knowledge graph instance."""
        return self.kg

    def get_reasoner(self) -> OntologyReasoner:
        """Return the ontology reasoner instance."""
        return self.reasoner

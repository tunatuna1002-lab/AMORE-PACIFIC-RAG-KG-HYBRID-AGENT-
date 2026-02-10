"""
Evaluation Runner
=================
Runs evaluation items through the agent and captures traces.

Features:
- Agent invocation with trace capture
- L1-L5 metric computation
- Result aggregation
- Concurrent evaluation support
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any

from eval.judge.interface import JudgeInterface
from eval.judge.stub import StubJudge
from eval.metrics.aggregator import MetricAggregator
from eval.metrics.l1_query import L1QueryMetrics
from eval.metrics.l2_retrieval import L2RetrievalMetrics
from eval.metrics.l3_kg import L3KGMetrics
from eval.metrics.l4_ontology import L4OntologyMetrics
from eval.metrics.l5_answer import L5AnswerMetrics
from eval.schemas import (
    AnswerTrace,
    DocRetrievalTrace,
    EntityLinkingTrace,
    EvalConfig,
    EvalItem,
    EvalTrace,
    ItemResult,
    KGQueryTrace,
    OntologyReasoningTrace,
)
from eval.validators.ontology_validator import OntologyValidator

logger = logging.getLogger(__name__)


class EvalRunner:
    """
    Runs evaluation items through the HybridChatbotAgent.

    Captures traces at each layer and computes metrics.
    """

    def __init__(
        self,
        agent: Any,  # HybridChatbotAgent
        config: EvalConfig | None = None,
        judge: JudgeInterface | None = None,
        validator: OntologyValidator | None = None,
        use_semantic_similarity: bool = False,
    ):
        """
        Initialize evaluation runner.

        Args:
            agent: HybridChatbotAgent instance
            config: Evaluation configuration
            judge: Judge interface (uses StubJudge if not provided)
            validator: Ontology validator (creates new if not provided)
            use_semantic_similarity: Whether to compute semantic similarity
        """
        self.agent = agent
        self.config = config or EvalConfig()
        self.judge = judge or StubJudge()
        self.validator = validator or OntologyValidator()
        self.use_semantic_similarity = use_semantic_similarity

        # Initialize metric calculators
        self.l1_metrics = L1QueryMetrics()
        self.l2_metrics = L2RetrievalMetrics(default_k=self.config.top_k)
        self.l3_metrics = L3KGMetrics(default_k=self.config.top_k)
        self.l4_metrics = L4OntologyMetrics(validator=self.validator)
        self.l5_metrics = L5AnswerMetrics(
            judge=self.judge,
            use_semantic_similarity=use_semantic_similarity,
        )
        self.aggregator = MetricAggregator()

    async def run_item(self, item: EvalItem) -> ItemResult:
        """
        Run a single evaluation item through the agent.

        Args:
            item: Evaluation item

        Returns:
            ItemResult with metrics and trace
        """
        start_time = time.time()
        error: str | None = None

        try:
            # Call the agent
            result = await self._invoke_agent(item.question)

            # Capture traces
            trace = await self._capture_trace(item.id, result, start_time)

            # Compute metrics
            l1 = self.l1_metrics.compute(trace.l1_entity_linking, trace.l4_ontology, item.gold)
            l2 = self.l2_metrics.compute(trace.l2_doc_retrieval, item.gold)
            l3 = self.l3_metrics.compute(trace.l3_kg_query, item.gold)
            l4 = self.l4_metrics.compute(trace.l4_ontology, trace.l3_kg_query, item.gold)

            # L5 metrics (with optional judge)
            context = self._build_context_string(trace)
            l5 = await self.l5_metrics.compute(
                trace.l5_answer,
                item.gold,
                item.question,
                context,
                use_judge=self.config.use_judge,
            )

        except Exception as e:
            logger.error(f"Error evaluating item {item.id}: {e}")
            error = str(e)

            # Create empty trace on error
            trace = self._create_empty_trace(item.id, start_time, error)

            # Create zeroed metrics on error
            l1, l2, l3, l4, l5 = self._create_zeroed_metrics()

        # Aggregate results
        return self.aggregator.aggregate(
            item_id=item.id,
            l1=l1,
            l2=l2,
            l3=l3,
            l4=l4,
            l5=l5,
            trace=trace,
            metadata=item.metadata,
        )

    async def run_dataset(
        self,
        items: list[EvalItem],
        concurrency: int = 1,
    ) -> list[ItemResult]:
        """
        Run evaluation on a full dataset.

        Args:
            items: List of evaluation items
            concurrency: Maximum concurrent evaluations

        Returns:
            List of ItemResult for each item
        """
        if concurrency <= 1:
            # Sequential execution
            results = []
            for i, item in enumerate(items):
                logger.info(f"Evaluating item {i + 1}/{len(items)}: {item.id}")
                result = await self.run_item(item)
                results.append(result)
            return results

        # Concurrent execution with semaphore
        semaphore = asyncio.Semaphore(concurrency)

        async def run_with_semaphore(item: EvalItem) -> ItemResult:
            async with semaphore:
                return await self.run_item(item)

        tasks = [run_with_semaphore(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in item {items[i].id}: {result}")
                # Create failed result
                trace = self._create_empty_trace(items[i].id, time.time(), str(result))
                l1, l2, l3, l4, l5 = self._create_zeroed_metrics()
                result = self.aggregator.aggregate(
                    items[i].id, l1, l2, l3, l4, l5, trace, items[i].metadata
                )
            processed_results.append(result)

        return processed_results

    async def _invoke_agent(self, question: str) -> dict[str, Any]:
        """
        Invoke the agent with a question.

        Args:
            question: User question

        Returns:
            Agent response dict
        """
        # Call the agent's chat method
        if hasattr(self.agent, "chat"):
            result = await self.agent.chat(question)
        elif hasattr(self.agent, "process_query"):
            result = await self.agent.process_query(question)
        else:
            raise AttributeError("Agent must have 'chat' or 'process_query' method")

        return result if isinstance(result, dict) else {"response": str(result)}

    async def _capture_trace(
        self,
        item_id: str,
        result: dict[str, Any],
        start_time: float,
    ) -> EvalTrace:
        """
        Capture evaluation trace from agent result.

        Args:
            item_id: Evaluation item ID
            result: Agent response
            start_time: Evaluation start time

        Returns:
            EvalTrace with all layer traces
        """
        latency_ms = (time.time() - start_time) * 1000

        # Extract hybrid context if available
        hybrid_ctx = None
        if hasattr(self.agent, "get_last_hybrid_context"):
            hybrid_ctx = self.agent.get_last_hybrid_context()

        # L1: Entity linking trace
        l1_trace = self._extract_l1_trace(result, hybrid_ctx)

        # L2: Document retrieval trace
        l2_trace = self._extract_l2_trace(result, hybrid_ctx)

        # L3: KG query trace
        l3_trace = self._extract_l3_trace(result, hybrid_ctx)

        # L4: Ontology reasoning trace
        l4_trace = self._extract_l4_trace(result, hybrid_ctx)

        # L5: Answer trace
        l5_trace = self._extract_l5_trace(result)

        return EvalTrace(
            item_id=item_id,
            timestamp=datetime.now(),
            l1_entity_linking=l1_trace,
            l2_doc_retrieval=l2_trace,
            l3_kg_query=l3_trace,
            l4_ontology=l4_trace,
            l5_answer=l5_trace,
            latency_ms=latency_ms,
            error=None,
        )

    def _extract_l1_trace(self, result: dict[str, Any], hybrid_ctx: Any) -> EntityLinkingTrace:
        """Extract L1 entity linking trace."""
        entities = result.get("entities", {})

        # Try to get from hybrid context first
        if hybrid_ctx:
            brands = getattr(hybrid_ctx, "brands", []) or []
            categories = getattr(hybrid_ctx, "categories", []) or []
            indicators = getattr(hybrid_ctx, "indicators", []) or []
            products = getattr(hybrid_ctx, "products", []) or []
        else:
            brands = entities.get("brands", [])
            categories = entities.get("categories", [])
            indicators = entities.get("indicators", [])
            products = entities.get("products", [])

        return EntityLinkingTrace(
            extracted_brands=brands,
            extracted_categories=categories,
            extracted_indicators=indicators,
            extracted_products=products,
        )

    def _extract_l2_trace(self, result: dict[str, Any], hybrid_ctx: Any) -> DocRetrievalTrace:
        """Extract L2 document retrieval trace."""
        # Try to get from hybrid context
        if hybrid_ctx and hasattr(hybrid_ctx, "doc_chunks"):
            chunks = hybrid_ctx.doc_chunks or []
            return DocRetrievalTrace(
                chunk_ids=[c.get("id", f"chunk_{i}") for i, c in enumerate(chunks)],
                snippets=[c.get("text", "")[:200] for c in chunks],
                scores=[c.get("score", 0.0) for c in chunks],
            )

        # Fallback to result sources
        sources = result.get("sources", [])
        return DocRetrievalTrace(
            chunk_ids=[s.get("id", f"src_{i}") for i, s in enumerate(sources)],
            snippets=[s.get("snippet", s.get("text", ""))[:200] for s in sources],
            scores=[s.get("score", 0.0) for s in sources],
        )

    def _extract_l3_trace(self, result: dict[str, Any], hybrid_ctx: Any) -> KGQueryTrace:
        """Extract L3 KG query trace."""
        # Try to get from hybrid context
        if hybrid_ctx:
            kg_entities = getattr(hybrid_ctx, "kg_entities", []) or []
            kg_edges = getattr(hybrid_ctx, "kg_edges", []) or []
            ontology_facts = getattr(hybrid_ctx, "ontology_facts", []) or []
            competitor_network = getattr(hybrid_ctx, "competitor_network", []) or []
        else:
            kg_entities = result.get("kg_entities", [])
            kg_edges = result.get("kg_edges", [])
            ontology_facts = result.get("ontology_facts", [])
            competitor_network = result.get("competitor_network", [])

        return KGQueryTrace(
            kg_entities_found=kg_entities,
            kg_edges_found=kg_edges,
            ontology_facts=ontology_facts,
            competitor_network=competitor_network,
        )

    def _extract_l4_trace(self, result: dict[str, Any], hybrid_ctx: Any) -> OntologyReasoningTrace:
        """Extract L4 ontology reasoning trace."""
        inferences = result.get("inferences", [])
        applied_rules = []
        constraint_violations = []

        # Extract rule names from inferences
        for inf in inferences:
            if isinstance(inf, dict):
                rule_name = inf.get("rule_name", "")
                if rule_name:
                    applied_rules.append(rule_name)

        # Try to get from hybrid context
        if hybrid_ctx and hasattr(hybrid_ctx, "reasoning_result"):
            reasoning = hybrid_ctx.reasoning_result
            if reasoning:
                inferences = getattr(reasoning, "inferences", inferences)
                constraint_violations = getattr(reasoning, "violations", [])

        return OntologyReasoningTrace(
            inferences=inferences,
            applied_rules=list(set(applied_rules)),
            constraint_violations=constraint_violations,
        )

    def _extract_l5_trace(self, result: dict[str, Any]) -> AnswerTrace:
        """Extract L5 answer trace."""
        response = result.get("response", "")
        citations = result.get("citations", [])
        confidence = result.get("confidence")

        return AnswerTrace(
            final_answer=response,
            citations=citations,
            confidence=confidence,
        )

    def _build_context_string(self, trace: EvalTrace) -> str:
        """Build context string for groundedness checking."""
        parts = []

        # Add document snippets
        for snippet in trace.l2_doc_retrieval.snippets:
            if snippet:
                parts.append(snippet)

        # Add KG facts
        for fact in trace.l3_kg_query.ontology_facts:
            if isinstance(fact, dict):
                parts.append(str(fact))

        return "\n\n".join(parts)

    def _create_empty_trace(self, item_id: str, start_time: float, error: str) -> EvalTrace:
        """Create an empty trace for error cases."""
        return EvalTrace(
            item_id=item_id,
            timestamp=datetime.now(),
            l1_entity_linking=EntityLinkingTrace(
                extracted_brands=[],
                extracted_categories=[],
                extracted_indicators=[],
                extracted_products=[],
            ),
            l2_doc_retrieval=DocRetrievalTrace(chunk_ids=[], snippets=[], scores=[]),
            l3_kg_query=KGQueryTrace(
                kg_entities_found=[],
                kg_edges_found=[],
                ontology_facts=[],
                competitor_network=[],
            ),
            l4_ontology=OntologyReasoningTrace(
                inferences=[], applied_rules=[], constraint_violations=[]
            ),
            l5_answer=AnswerTrace(final_answer="", citations=[], confidence=None),
            latency_ms=(time.time() - start_time) * 1000,
            error=error,
        )

    def _create_zeroed_metrics(self):
        """Create zeroed metrics for error cases."""
        from eval.schemas import L1Metrics, L2Metrics, L3Metrics, L4Metrics, L5Metrics

        l1 = L1Metrics(entity_link_f1=0.0, concept_map_f1=0.0, constraint_extraction_f1=0.0)
        l2 = L2Metrics(context_recall_at_k=0.0, context_precision_at_k=0.0, mrr=0.0)
        l3 = L3Metrics(hits_at_k=0.0, kg_edge_f1=0.0)
        l4 = L4Metrics(constraint_violation_rate=1.0, type_consistency_rate=0.0)
        l5 = L5Metrics(
            answer_exact_match=0.0,
            answer_f1=0.0,
            groundedness_score=None,
            answer_relevance_score=None,
        )
        return l1, l2, l3, l4, l5

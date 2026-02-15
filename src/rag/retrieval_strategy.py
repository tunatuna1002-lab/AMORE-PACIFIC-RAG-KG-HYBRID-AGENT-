"""
Retrieval Strategy Protocol
============================
Strategy pattern for retriever backends.

HybridRetriever delegates to a RetrievalStrategy implementation:
- LegacyRetrievalStrategy: KG + rule-based reasoning + RAG (default)
- OWLRetrievalStrategy: OWL ontology + entity linking + confidence fusion + RAG

Architecture:
    HybridRetriever (single entry point)
        ├─ mode="legacy" → LegacyRetrievalStrategy
        └─ mode="owl"    → OWLRetrievalStrategy

Both strategies return UnifiedRetrievalResult.
"""

import logging
from typing import Any, Protocol, runtime_checkable

from src.domain.value_objects.retrieval_result import UnifiedRetrievalResult

logger = logging.getLogger(__name__)


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Protocol for retrieval strategy implementations.

    Each strategy encapsulates a different retrieval pipeline
    (legacy rule-based vs OWL ontology-based).
    """

    async def initialize(self) -> None:
        """Initialize strategy-specific resources."""
        ...

    async def retrieve(
        self,
        query: str,
        current_metrics: dict[str, Any] | None = None,
        top_k: int = 5,
        **kwargs: Any,
    ) -> UnifiedRetrievalResult:
        """Execute retrieval pipeline.

        Args:
            query: User query
            current_metrics: Current dashboard metrics
            top_k: Maximum number of results
            **kwargs: Strategy-specific arguments

        Returns:
            UnifiedRetrievalResult
        """
        ...

    async def search(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search documents.

        Args:
            query: Search query
            top_k: Maximum results
            doc_filter: Optional document filter

        Returns:
            List of document dicts
        """
        ...


class OWLRetrievalStrategy:
    """OWL-based retrieval strategy.

    OWL-based retrieval pipeline (ported from former TrueHybridRetriever):
    - Entity linking (query → ontology concepts)
    - OWL reasoning (owlready2-based formal inference)
    - Ontology-guided vector search
    - Cross-encoder reranking
    - Confidence fusion (multi-source)

    This strategy requires owlready2 to be available.
    """

    def __init__(
        self,
        knowledge_graph: Any | None = None,
        owl_reasoner: Any | None = None,
        doc_retriever: Any | None = None,
        ontology_kg: Any | None = None,
        unified_reasoner: Any | None = None,
        use_reranking: bool = True,
        use_query_expansion: bool = True,
    ):
        from .confidence_fusion import ConfidenceFusion
        from .entity_linker import EntityLinker
        from .reranker import get_reranker
        from .retriever import DocumentRetriever

        self.kg = knowledge_graph
        self.owl_reasoner = owl_reasoner
        self.ontology_kg = ontology_kg
        self.unified_reasoner = unified_reasoner

        self.doc_retriever = doc_retriever or DocumentRetriever(
            use_semantic_chunking=True,
            use_reranker=use_reranking,
            use_query_expansion=use_query_expansion,
        )

        self.entity_linker = EntityLinker(knowledge_graph=self.kg)
        self.confidence_fusion = ConfidenceFusion()
        self._reranker = None
        self.use_reranking = use_reranking
        self._initialized = False
        self._get_reranker = get_reranker

    async def initialize(self) -> None:
        """Initialize OWL strategy resources."""
        if self._initialized:
            return

        await self.doc_retriever.initialize()

        if self.ontology_kg:
            try:
                await self.ontology_kg.initialize()
                logger.info("OntologyKnowledgeGraph initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OntologyKnowledgeGraph: {e}")

        if self.unified_reasoner:
            logger.info("UnifiedReasoner ready")

        if self.owl_reasoner:
            await self.owl_reasoner.initialize()

            if self.kg:
                try:
                    count = self.owl_reasoner.import_from_knowledge_graph(self.kg)
                    if count > 0:
                        logger.info(f"Imported {count} entities to OWL ontology")
                        self.owl_reasoner.run_reasoner()
                        self.owl_reasoner.infer_market_positions()
                except Exception as e:
                    logger.warning(f"Failed to migrate KG to OWL: {e}")

        self._initialized = True
        logger.info("OWLRetrievalStrategy initialized")

    async def retrieve(
        self,
        query: str,
        current_metrics: dict[str, Any] | None = None,
        top_k: int = 5,
        **kwargs: Any,
    ) -> UnifiedRetrievalResult:
        """Execute OWL retrieval pipeline."""
        from datetime import datetime

        from .entity_linker import LinkedEntity

        if not self._initialized:
            await self.initialize()

        start_time = datetime.now()

        # Initialize result fields
        entity_links: list[LinkedEntity] = []
        ontology_context: dict[str, Any] = {"inferences": [], "facts": [], "related_docs": []}
        documents: list[dict[str, Any]] = []
        confidence = 0.0
        combined_context = ""
        metadata: dict[str, Any] = {}

        try:
            # 1. Entity Linking
            entity_links = self.entity_linker.link(query)
            entity_confidence = (
                sum(e.confidence for e in entity_links) / len(entity_links) if entity_links else 0.5
            )

            # 2. Ontology-Guided Filters
            ontology_filters = self.entity_linker.get_ontology_filters(entity_links)

            # 3. Query Expansion
            expanded_queries = await self.doc_retriever.expand_query(query)

            # 4. Ontology-Guided Vector Search
            vector_results = await self._ontology_guided_search(
                expanded_queries, ontology_filters, top_k * 3
            )

            # 5. OWL Ontology Reasoning
            ontology_context = await self._infer_with_ontology(entity_links, current_metrics)

            # 6. Cross-Encoder Reranking
            if self.use_reranking and vector_results:
                reranked_results = await self._rerank(query, vector_results, top_k * 2)
            else:
                reranked_results = vector_results[: top_k * 2]

            # 7. Confidence Fusion
            fused_docs = self._fuse_results(
                vector_results=vector_results[: top_k * 2],
                ontology_results=ontology_context.get("related_docs", []),
                reranked_results=reranked_results,
                entity_confidence=entity_confidence,
            )

            # Extract final documents
            fused_documents = fused_docs.documents if hasattr(fused_docs, "documents") else []
            documents = [
                {
                    "id": doc.get("id", str(i)),
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": doc.get("score", 0.5),
                    "rank": i + 1,
                }
                for i, doc in enumerate(fused_documents[:top_k])
            ]

            # 8. Combined Context
            combined_context = self._build_combined_context(
                entity_links, ontology_context, documents
            )

            # 9. Overall Confidence
            avg_doc_score = fused_docs.confidence if hasattr(fused_docs, "confidence") else 0.5
            ontology_coverage = len(ontology_context.get("inferences", [])) / 5
            confidence = min(
                max(
                    0.4 * entity_confidence
                    + 0.4 * avg_doc_score
                    + 0.2 * min(ontology_coverage, 1.0),
                    0.0,
                ),
                1.0,
            )

            metadata = {
                "retrieval_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "entity_count": len(entity_links),
                "entity_confidence": entity_confidence,
                "vector_results_count": len(vector_results),
                "reranked_results_count": len(reranked_results) if self.use_reranking else 0,
                "final_results_count": len(documents),
                "ontology_inferences_count": len(ontology_context.get("inferences", [])),
                "query_expanded": len(expanded_queries) > 1,
            }

        except Exception as e:
            logger.error(f"OWL retrieval failed: {e}")
            metadata["error"] = str(e)
            confidence = 0.0

        # Build entities dict from entity_links
        entities_dict: dict[str, list[str]] = {
            "brands": [],
            "categories": [],
            "indicators": [],
            "products": [],
        }
        for entity in entity_links:
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
            type_key = entity_type if entity_type in entities_dict else f"{entity_type}s"
            if type_key in entities_dict:
                entities_dict[type_key].append(entity_id)

        return UnifiedRetrievalResult(
            query=query,
            entities=entities_dict,
            ontology_facts=ontology_context.get("facts", []),
            inferences=ontology_context.get("inferences", []),
            rag_chunks=documents,
            combined_context=combined_context,
            confidence=confidence,
            entity_links=entity_links,
            metadata=metadata,
            retriever_type="owl",
        )

    async def search(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search documents via doc_retriever."""
        if hasattr(self.doc_retriever, "search"):
            return await self.doc_retriever.search(query=query, top_k=top_k, doc_filter=doc_filter)
        return []

    # ── Private methods ──

    async def _ontology_guided_search(
        self, queries: list[str], filters: dict[str, Any], top_k: int
    ) -> list[dict[str, Any]]:
        """Ontology-guided vector search with deduplication."""
        all_results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for q in queries:
            results = await self.doc_retriever.search(
                query=q,
                top_k=top_k // len(queries),
                use_query_expansion=False,
                use_reranking=False,
            )
            for result in results:
                if result["id"] not in seen_ids:
                    if self._matches_filters(result.get("metadata", {}), filters):
                        all_results.append(result)
                        seen_ids.add(result["id"])

        return all_results[:top_k]

    @staticmethod
    def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if metadata matches filter conditions."""
        if not filters:
            return True
        for key, condition in filters.items():
            meta_value = metadata.get(key)
            if isinstance(condition, dict):
                if "$in" in condition and meta_value not in condition["$in"]:
                    return False
            elif meta_value != condition:
                return False
        return True

    async def _infer_with_ontology(
        self, entities: list, current_metrics: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Execute OWL ontology reasoning."""
        context: dict[str, Any] = {"inferences": [], "facts": [], "related_docs": []}

        try:
            if self.unified_reasoner:
                for entity in entities:
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
                    if entity_type == "brand":
                        result = self.unified_reasoner.infer(
                            context={"brand": entity_id, **(current_metrics or {})},
                            query=f"{entity_id} market analysis",
                        )
                        if result and hasattr(result, "to_dict"):
                            context["inferences"].append(result.to_dict())
                        elif isinstance(result, list):
                            for r in result:
                                if hasattr(r, "to_dict"):
                                    context["inferences"].append(r.to_dict())

                if self.ontology_kg and hasattr(self.ontology_kg, "owl") and self.ontology_kg.owl:
                    try:
                        context["facts"] = self.ontology_kg.owl.get_inferred_facts()
                    except Exception as e:
                        logger.warning(f"Failed to get inferred facts from OntologyKG: {e}")
                return context

            # Fallback to OWL-only reasoning
            if self.owl_reasoner:
                inferred_facts = self.owl_reasoner.get_inferred_facts()
                context["facts"] = inferred_facts

                for entity in entities:
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
                    if entity_type == "brand":
                        brand_info = self.owl_reasoner.get_brand_info(entity_id)
                        if brand_info:
                            position = brand_info.get("market_position")
                            if position:
                                context["inferences"].append(
                                    {
                                        "type": "market_position",
                                        "brand": entity_id,
                                        "position": position,
                                        "sos": brand_info.get("sos", 0.0),
                                    }
                                )
                            competitors = brand_info.get("competitors", [])
                            if competitors:
                                context["inferences"].append(
                                    {
                                        "type": "competition",
                                        "brand": entity_id,
                                        "competitors": competitors[:5],
                                    }
                                )

        except Exception as e:
            logger.warning(f"Ontology inference failed: {e}")

        return context

    async def _rerank(
        self, query: str, documents: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """Cross-encoder reranking."""
        if self._reranker is None:
            self._reranker = self._get_reranker()

        try:
            ranked_docs = self._reranker.rerank(query, documents, top_k=top_k)
            return [
                {
                    "id": doc.metadata.get("chunk_id", ""),
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": doc.score,
                }
                for doc in ranked_docs
            ]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return documents[:top_k]

    def _fuse_results(
        self,
        vector_results: list[dict[str, Any]],
        ontology_results: list[dict[str, Any]],
        reranked_results: list[dict[str, Any]],
        entity_confidence: float,
    ) -> Any:
        """Confidence fusion to merge results from multiple sources."""
        from src.rag.confidence_fusion import InferenceResult
        from src.rag.confidence_fusion import LinkedEntity as FusionLinkedEntity
        from src.rag.confidence_fusion import SearchResult as FusionSearchResult

        fusion_vector = (
            [
                FusionSearchResult(
                    content=r.get("content", ""),
                    score=r.get("score", 0.5),
                    metadata=r.get("metadata", {}),
                    source="vector",
                )
                for r in vector_results
            ]
            if vector_results
            else None
        )

        fusion_ontology = (
            [
                InferenceResult(
                    insight=r.get("insight", ""),
                    confidence=r.get("confidence", 0.5),
                    evidence=r.get("evidence", {}),
                    rule_name=r.get("rule_name"),
                )
                for r in ontology_results
            ]
            if ontology_results
            else None
        )

        fusion_entities = (
            [
                FusionLinkedEntity(
                    entity_id=str(i),
                    entity_name=r.get("text", ""),
                    entity_type=r.get("type", "unknown"),
                    link_confidence=entity_confidence,
                    context=r.get("context", ""),
                    metadata=r.get("metadata", {}),
                )
                for i, r in enumerate(reranked_results)
            ]
            if reranked_results
            else None
        )

        return self.confidence_fusion.fuse(
            vector_results=fusion_vector,
            ontology_results=fusion_ontology,
            entity_links=fusion_entities,
            query=None,
        )

    @staticmethod
    def _build_combined_context(
        entity_links: list,
        ontology_context: dict[str, Any],
        documents: list[dict[str, Any]],
    ) -> str:
        """Build combined context string for LLM prompts."""
        parts: list[str] = []

        if entity_links:
            parts.append("## 추출된 엔티티\n")
            for entity in entity_links[:5]:
                entity_type = (
                    entity.entity_type.value
                    if hasattr(entity.entity_type, "value")
                    else entity.entity_type
                )
                parts.append(f"- {entity_type}: {entity.text} (신뢰도: {entity.confidence:.2f})")
            parts.append("")

        if ontology_context.get("inferences"):
            parts.append("## 온톨로지 추론 결과\n")
            for inference in ontology_context["inferences"][:3]:
                inf_type = inference.get("type", "")
                if inf_type == "market_position":
                    brand = inference.get("brand", "")
                    position = inference.get("position", "")
                    sos = inference.get("sos", 0.0)
                    parts.append(f"- {brand}의 시장 포지션: {position} (SoS: {sos:.2%})")
                elif inf_type == "competition":
                    brand = inference.get("brand", "")
                    competitors = inference.get("competitors", [])
                    parts.append(f"- {brand}의 경쟁사: {', '.join(competitors[:3])}")
            parts.append("")

        if documents:
            parts.append("## 관련 문서\n")
            for i, doc in enumerate(documents[:3], 1):
                title = doc.get("metadata", {}).get("title", "")
                content = doc.get("content", "")
                if title:
                    parts.append(f"### {i}. {title}")
                if len(content) > 500:
                    content = content[:500] + "..."
                parts.append(content)
                parts.append("")

        return "\n".join(parts)

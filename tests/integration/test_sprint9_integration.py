"""
Sprint 9 Integration Tests (D-5)

Tests for multi-hop, AIS citation, SPARQL, IRI roundtrip,
OWL consistency, and Self-RAG + hybrid retrieval integration.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.entities.relations import IRI, Relation, RelationType

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def knowledge_graph():
    """Create a KnowledgeGraph with test data only (no real data)."""
    from src.ontology.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(auto_load=False)
    kg.auto_save = False

    # Add brand-product relations
    kg.add_relation(
        Relation(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            object="B08R35S2QH",
            confidence=0.95,
        )
    )
    kg.add_relation(
        Relation(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            object="B0BQ2LY2JK",
            confidence=0.90,
        )
    )
    kg.add_relation(
        Relation(
            subject="COSRX",
            predicate=RelationType.HAS_PRODUCT,
            object="B00PBX3L7K",
            confidence=0.92,
        )
    )
    kg.add_relation(
        Relation(
            subject="COSRX",
            predicate=RelationType.HAS_PRODUCT,
            object="B07B5G7KFZ",
            confidence=0.88,
        )
    )
    kg.add_relation(
        Relation(
            subject="COSRX",
            predicate=RelationType.HAS_PRODUCT,
            object="B09QMJ2PLZ",
            confidence=0.85,
        )
    )

    # Brand-category relations
    kg.add_relation(
        Relation(
            subject="LANEIGE",
            predicate=RelationType.BELONGS_TO_CATEGORY,
            object="lip_care",
            confidence=0.95,
        )
    )
    kg.add_relation(
        Relation(
            subject="COSRX",
            predicate=RelationType.BELONGS_TO_CATEGORY,
            object="skin_care",
            confidence=0.90,
        )
    )

    # Competition relations
    kg.add_relation(
        Relation(
            subject="LANEIGE",
            predicate=RelationType.COMPETES_WITH,
            object="COSRX",
            confidence=0.80,
        )
    )
    kg.add_relation(
        Relation(
            subject="LANEIGE",
            predicate=RelationType.COMPETES_WITH,
            object="ANUA",
            confidence=0.75,
        )
    )

    return kg


# =========================================================================
# 1. Multi-hop tests (5 questions)
# =========================================================================


class TestMultiHopIntegration:
    """Integration tests for multi-hop retrieval via ReActAgent."""

    @pytest.fixture
    def react_agent(self):
        """Create ReActAgent with mocked tool_executor."""
        from src.core.react_agent import ReActAgent

        agent = ReActAgent(
            max_iterations=6,
            max_hops=2,
            ircot_enabled=True,
        )

        # Mock tool executor
        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(
            return_value=MagicMock(
                success=True,
                data={"results": [{"brand": "LANEIGE", "rank": 1}]},
                error=None,
            )
        )
        agent.tool_executor = mock_executor
        return agent

    @pytest.mark.asyncio
    async def test_multihop_competitor_avg_price(self, react_agent):
        """LANEIGE Lip Sleeping Mask의 경쟁 제품들의 평균 가격은?"""
        from src.core.react_agent import ReActStep

        query = "LANEIGE Lip Sleeping Mask의 경쟁 제품들의 평균 가격은?"
        step_idx = 0

        async def mock_execute_step(q, ctx, steps):
            nonlocal step_idx
            sequence = [
                ReActStep(
                    thought="Need to find LANEIGE competitors first",
                    action="get_competitors",
                    action_input={"brand": "LANEIGE"},
                ),
                ReActStep(
                    thought="추가 검색 필요: competitor pricing",
                    action="refine_search",
                    action_input={
                        "refined_query": "competitor prices for LANEIGE",
                        "reason": "need pricing data",
                        "focus_entities": ["LANEIGE"],
                    },
                ),
                ReActStep(
                    thought="Got data. Average price is $24.99.",
                    action="final_answer",
                    observation="경쟁 제품 평균 가격은 $24.99입니다.",
                ),
            ]
            result = sequence[min(step_idx, len(sequence) - 1)]
            step_idx += 1
            return result

        with (
            patch.object(react_agent, "_execute_step", side_effect=mock_execute_step),
            patch.object(
                react_agent,
                "_reflect",
                return_value={"quality_score": 0.8, "needs_improvement": False},
            ),
        ):
            result = await react_agent.run(query, context="Test context")

        assert result is not None
        assert result.hop_count >= 1

    @pytest.mark.asyncio
    async def test_multihop_category_leader_performance(self, react_agent):
        """Lip Care 카테고리 1위 브랜드의 다른 카테고리 성과는?"""
        from src.core.react_agent import ReActStep

        step_idx = 0

        async def mock_execute_step(q, ctx, steps):
            nonlocal step_idx
            sequence = [
                ReActStep(
                    thought="Find Lip Care category leader first",
                    action="search_products",
                    action_input={"category": "lip_care"},
                ),
                ReActStep(
                    thought="추가 검색 필요: cross-category performance",
                    action="refine_search",
                    action_input={
                        "refined_query": "LANEIGE other categories",
                        "reason": "need cross-category data",
                        "focus_entities": ["LANEIGE"],
                    },
                ),
                ReActStep(
                    thought="Found cross-category data.",
                    action="final_answer",
                    observation="LANEIGE는 Lip Care 1위이며 Skin Care에서도 활약합니다.",
                ),
            ]
            result = sequence[min(step_idx, len(sequence) - 1)]
            step_idx += 1
            return result

        with (
            patch.object(react_agent, "_execute_step", side_effect=mock_execute_step),
            patch.object(
                react_agent,
                "_reflect",
                return_value={"quality_score": 0.8, "needs_improvement": False},
            ),
        ):
            result = await react_agent.run(
                "Lip Care 카테고리 1위 브랜드의 다른 카테고리 성과는?",
                context="Test context",
            )

        assert result is not None
        assert result.hop_count >= 1

    @pytest.mark.asyncio
    async def test_multihop_cosrx_top3_hhi(self, react_agent):
        """COSRX의 Top 3 제품이 속한 카테고리들의 시장 집중도는?"""
        from src.core.react_agent import ReActStep

        step_idx = 0

        async def mock_execute_step(q, ctx, steps):
            nonlocal step_idx
            sequence = [
                ReActStep(
                    thought="First find COSRX top products",
                    action="search_products",
                    action_input={"brand": "COSRX", "limit": 3},
                ),
                ReActStep(
                    thought="추가 검색 필요: HHI for categories",
                    action="refine_search",
                    action_input={
                        "refined_query": "HHI for COSRX product categories",
                        "reason": "need market concentration",
                        "focus_entities": ["COSRX"],
                    },
                ),
                ReActStep(
                    thought="HHI data retrieved.",
                    action="final_answer",
                    observation="COSRX 제품 카테고리의 평균 HHI는 0.08입니다.",
                ),
            ]
            result = sequence[min(step_idx, len(sequence) - 1)]
            step_idx += 1
            return result

        with (
            patch.object(react_agent, "_execute_step", side_effect=mock_execute_step),
            patch.object(
                react_agent,
                "_reflect",
                return_value={"quality_score": 0.8, "needs_improvement": False},
            ),
        ):
            result = await react_agent.run(
                "COSRX의 Top 3 제품이 속한 카테고리들의 시장 집중도는?",
                context="Test context",
            )

        assert result is not None

    @pytest.mark.asyncio
    async def test_multihop_fastest_growing_competitor(self, react_agent):
        """LANEIGE와 경쟁하는 브랜드 중 성장세가 가장 큰 브랜드는?"""
        from src.core.react_agent import ReActStep

        step_idx = 0

        async def mock_execute_step(q, ctx, steps):
            nonlocal step_idx
            sequence = [
                ReActStep(
                    thought="Get LANEIGE competitors first",
                    action="get_competitors",
                    action_input={"brand": "LANEIGE"},
                ),
                ReActStep(
                    thought="추가 검색 필요: growth rate details",
                    action="refine_search",
                    action_input={
                        "refined_query": "ANUA growth rate details",
                        "reason": "need growth data",
                        "focus_entities": ["ANUA"],
                    },
                ),
                ReActStep(
                    thought="ANUA is fastest growing.",
                    action="final_answer",
                    observation="ANUA가 25.3%로 가장 높은 성장세를 보입니다.",
                ),
            ]
            result = sequence[min(step_idx, len(sequence) - 1)]
            step_idx += 1
            return result

        with (
            patch.object(react_agent, "_execute_step", side_effect=mock_execute_step),
            patch.object(
                react_agent,
                "_reflect",
                return_value={"quality_score": 0.8, "needs_improvement": False},
            ),
        ):
            result = await react_agent.run(
                "LANEIGE와 경쟁하는 브랜드 중 성장세가 가장 큰 브랜드는?",
                context="Test context",
            )

        assert result is not None

    @pytest.mark.asyncio
    async def test_multihop_brand_sentiment_profile(self, react_agent):
        """Lip Care에서 SoS가 가장 높은 브랜드의 감성 프로필은?"""
        from src.core.react_agent import ReActStep

        step_idx = 0

        async def mock_execute_step(q, ctx, steps):
            nonlocal step_idx
            sequence = [
                ReActStep(
                    thought="Find top SoS brand in Lip Care",
                    action="search_products",
                    action_input={"category": "lip_care"},
                ),
                ReActStep(
                    thought="추가 검색 필요: sentiment data",
                    action="refine_search",
                    action_input={
                        "refined_query": "LANEIGE sentiment profile",
                        "reason": "need sentiment",
                        "focus_entities": ["LANEIGE"],
                    },
                ),
                ReActStep(
                    thought="Got sentiment data.",
                    action="final_answer",
                    observation="LANEIGE의 감성 프로필: 긍정 80%, 중립 15%, 부정 5%",
                ),
            ]
            result = sequence[min(step_idx, len(sequence) - 1)]
            step_idx += 1
            return result

        with (
            patch.object(react_agent, "_execute_step", side_effect=mock_execute_step),
            patch.object(
                react_agent,
                "_reflect",
                return_value={"quality_score": 0.8, "needs_improvement": False},
            ),
        ):
            result = await react_agent.run(
                "Lip Care에서 SoS가 가장 높은 브랜드의 감성 프로필은?",
                context="Test context",
            )

        assert result is not None


# =========================================================================
# 2. AIS Citation Test
# =========================================================================


class TestAISCitationIntegration:
    """Test AIS inline citation with ContextBuilder."""

    def test_build_ais_response_with_citations(self):
        """Build context with ContextBuilder and verify citation rate >= 0.80."""
        from src.rag.context_builder import ContextBuilder

        builder = ContextBuilder(enable_ais=True)

        # Register sources that overlap with the response text
        builder._register_source("KG", "LANEIGE 브랜드", "LANEIGE Lip Sleeping Mask SoS 점유율 15%")
        builder._register_source(
            "RAG", "시장 분석 가이드", "Lip Care 카테고리 경쟁 분석 시장 트렌드"
        )
        builder._register_source(
            "Inference", "경쟁력 인사이트", "LANEIGE 경쟁 브랜드 COSRX ANUA 점유율"
        )

        # Response text with high overlap with registered sources
        response = (
            "LANEIGE Lip Sleeping Mask는 현재 SoS 점유율 15%를 기록하고 있습니다. "
            "Lip Care 카테고리에서 경쟁 분석 결과 시장 트렌드가 긍정적입니다. "
            "주요 경쟁 브랜드인 COSRX와 ANUA의 점유율도 주목할 필요가 있습니다. "
            "LANEIGE의 경쟁력은 Lip Care 시장에서 강세를 보이고 있습니다. "
            "전반적으로 LANEIGE 브랜드의 시장 포지션은 안정적입니다."
        )

        annotated = builder.build_ais_response(response)
        stats = builder.get_citation_stats()

        assert stats["total_sentences"] > 0
        assert stats["citation_rate"] >= 0.80, f"Citation rate {stats['citation_rate']:.2f} < 0.80"
        # Verify citation tags present
        assert "[출처" in annotated


# =========================================================================
# 3. SPARQL Tests (5 queries)
# =========================================================================


class TestSPARQLIntegration:
    """Test query_sparql_rdflib() with real SPARQL queries."""

    def test_sparql_select_all_products(self, knowledge_graph):
        """SELECT all products for a brand using STR FILTER on URI."""
        results = knowledge_graph.query_sparql_rdflib("""
            PREFIX amore: <http://amore.ontology/>
            SELECT ?product
            WHERE {
                <http://amore.ontology/entity/LANEIGE> amore:hasProduct ?product .
            }
        """)

        assert len(results) >= 2
        product_ids = [r.get("product", "") for r in results]
        assert any("B08R35S2QH" in pid for pid in product_ids)

    def test_sparql_select_competitors(self, knowledge_graph):
        """SELECT competitors of a brand using STR FILTER on URI."""
        results = knowledge_graph.query_sparql_rdflib("""
            PREFIX amore: <http://amore.ontology/>
            SELECT ?competitor
            WHERE {
                <http://amore.ontology/entity/LANEIGE> amore:competesWith ?competitor .
            }
        """)

        assert len(results) >= 2
        competitor_ids = [r.get("competitor", "") for r in results]
        assert any("COSRX" in cid for cid in competitor_ids)
        assert any("ANUA" in cid for cid in competitor_ids)

    def test_sparql_select_brand_category(self, knowledge_graph):
        """SELECT brand-category relations."""
        results = knowledge_graph.query_sparql_rdflib("""
            PREFIX amore: <http://amore.ontology/>
            SELECT ?brand ?category
            WHERE {
                ?brand amore:belongsToCategory ?category .
            }
        """)

        assert len(results) >= 2

    def test_sparql_filter_by_literal(self, knowledge_graph):
        """SELECT with FILTER on object literal."""
        results = knowledge_graph.query_sparql_rdflib("""
            PREFIX amore: <http://amore.ontology/>
            SELECT ?brand
            WHERE {
                ?brand amore:belongsToCategory ?cat .
                FILTER(?cat = "lip_care")
            }
        """)

        assert len(results) >= 1
        brand_ids = [r.get("brand", "") for r in results]
        assert any("LANEIGE" in bid for bid in brand_ids)

    def test_sparql_count_products(self, knowledge_graph):
        """COUNT products for COSRX brand."""
        results = knowledge_graph.query_sparql_rdflib("""
            PREFIX amore: <http://amore.ontology/>
            SELECT (COUNT(?product) AS ?count)
            WHERE {
                <http://amore.ontology/entity/COSRX> amore:hasProduct ?product .
            }
        """)

        assert len(results) >= 1
        # COSRX has 3 products (may count 6 with URI+literal dual triples)
        count = int(results[0].get("count", 0))
        assert count >= 3


# =========================================================================
# 4. IRI Roundtrip Test
# =========================================================================


class TestIRIRoundtrip:
    """Test IRI create -> migrate -> query roundtrip."""

    def test_iri_bare_to_iri_to_bare_roundtrip(self):
        """Test bare ID -> IRI -> bare ID conversion."""
        iri = IRI.to_iri("brand", "LANEIGE")
        assert IRI.is_iri(iri)
        assert "brand" in iri
        assert "LANEIGE" in iri

        entity_type, entity_id = IRI.from_iri(iri)
        assert entity_type == "brand"
        assert entity_id == "LANEIGE"

    def test_iri_product_roundtrip(self):
        """Test product IRI roundtrip."""
        iri = IRI.to_iri("product", "B08R35S2QH")
        assert IRI.is_iri(iri)

        entity_type, entity_id = IRI.from_iri(iri)
        assert entity_type == "product"
        assert entity_id == "B08R35S2QH"

    def test_iri_category_roundtrip(self):
        """Test category IRI roundtrip."""
        iri = IRI.to_iri("category", "lip_care")
        assert IRI.is_iri(iri)

        entity_type, entity_id = IRI.from_iri(iri)
        assert entity_type == "category"
        assert entity_id == "lip_care"

    def test_kg_migrate_to_iri_and_query(self, knowledge_graph):
        """Migrate KG to IRI, verify queries still return results."""
        # Before migration: query with bare IDs
        before_results = knowledge_graph.query(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
        )
        assert len(before_results) >= 2

        # Migrate
        stats = knowledge_graph.migrate_to_iri()
        assert stats["converted"] > 0

        # After migration: subjects are now IRI form, query with IRI directly
        iri_subject = IRI.to_iri("brand", "LANEIGE")
        after_results = knowledge_graph.query(
            subject=iri_subject,
            predicate=RelationType.HAS_PRODUCT,
        )
        assert len(after_results) >= 2

    def test_kg_export_as_iri(self, knowledge_graph):
        """Export KG as IRI format (non-destructive)."""
        export = knowledge_graph.export_as_iri()

        assert "triples" in export
        assert len(export["triples"]) > 0

        iri_count = sum(1 for t in export["triples"] if t.get("subject", "").startswith("amore:"))
        assert iri_count > 0


# =========================================================================
# 5. OWL Consistency Test
# =========================================================================


class TestOWLConsistencyIntegration:
    """Test OWL consistency checking."""

    def test_clean_ontology_is_consistent(self):
        """A fresh OWL reasoner should report consistency."""
        from src.ontology.owl_reasoner import ConsistencyReport, OWLReasoner

        reasoner = OWLReasoner.__new__(OWLReasoner)
        reasoner.onto = None
        reasoner.reasoner_type = "hermit"

        report = reasoner.check_consistency()

        assert isinstance(report, ConsistencyReport)
        assert report.is_consistent is True
        assert report.checked_at != ""

    def test_consistency_report_fields(self):
        """ConsistencyReport has expected fields."""
        from src.ontology.owl_reasoner import ConsistencyReport

        report = ConsistencyReport(
            is_consistent=True,
            violations=[],
            warnings=["test warning"],
            check_method="test",
        )

        assert report.is_consistent is True
        assert report.violations == []
        assert len(report.warnings) == 1
        assert report.check_method == "test"
        assert report.checked_at != ""

    def test_consistency_report_with_violation(self):
        """ConsistencyReport correctly reports violations."""
        from src.ontology.owl_reasoner import ConsistencyReport

        report = ConsistencyReport(
            is_consistent=False,
            violations=[
                {
                    "type": "disjointness",
                    "entity": "TestBrand",
                    "description": "Brand in multiple disjoint classes",
                    "severity": "error",
                }
            ],
            warnings=[],
            check_method="rule_based_fallback",
        )

        assert report.is_consistent is False
        assert len(report.violations) == 1
        assert report.violations[0]["severity"] == "error"


# =========================================================================
# 6. Self-RAG + Hybrid Retrieval Test
# =========================================================================


class TestSelfRAGHybridIntegration:
    """Test Self-RAG gate + hybrid search integration."""

    @pytest.fixture
    def hybrid_retriever(self):
        """Create HybridRetriever with mocked doc_retriever."""
        from src.rag.hybrid_retriever import HybridRetriever

        mock_kg = MagicMock()
        mock_kg.get_entity_metadata.return_value = {}
        mock_kg.get_brand_products.return_value = []
        mock_kg.get_competitors.return_value = []
        mock_kg.get_neighbors.return_value = {"outgoing": [], "incoming": []}
        mock_kg.query.return_value = []
        mock_kg.get_category_brands.return_value = []
        mock_kg.get_category_hierarchy.return_value = {}
        mock_kg.get_product_sentiments.return_value = {}
        mock_kg.get_brand_sentiment_profile.return_value = {}
        mock_kg.find_products_by_sentiment.return_value = []
        mock_kg.load_category_hierarchy.return_value = 0
        mock_kg.get_stats.return_value = {}

        mock_reasoner = MagicMock()
        mock_reasoner.rules = ["rule1"]
        mock_reasoner.infer.return_value = []
        mock_reasoner.get_inference_stats.return_value = {}

        mock_doc_retriever = MagicMock()
        mock_doc_retriever.initialize = AsyncMock()
        mock_doc_retriever.search = AsyncMock(
            return_value=[
                {"id": "d1", "content": "LANEIGE analysis", "score": 0.9, "metadata": {}},
            ]
        )
        mock_doc_retriever.search_bm25 = MagicMock(
            return_value=[
                {"id": "d2", "content": "BM25 result", "score": 0.7, "metadata": {}},
            ]
        )
        mock_doc_retriever.reciprocal_rank_fusion = MagicMock(
            return_value=[
                {"id": "d1", "content": "LANEIGE analysis", "score": 0.9, "metadata": {}},
                {"id": "d2", "content": "BM25 result", "score": 0.7, "metadata": {}},
            ]
        )

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=mock_reasoner,
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )
        retriever._initialized = True
        return retriever

    def test_should_retrieve_various_queries(self, hybrid_retriever):
        """Test should_retrieve with various query types."""
        should, reason, conf = hybrid_retriever.should_retrieve("LANEIGE SoS 분석")
        assert should is True
        assert conf == 1.0

        should, reason, conf = hybrid_retriever.should_retrieve("안녕하세요")
        assert should is False
        assert conf == 0.0

        should, reason, conf = hybrid_retriever.should_retrieve("이 질문은 일반적입니다")
        assert should is True
        assert conf == 0.8

    @pytest.mark.asyncio
    async def test_hybrid_search_with_mocked_bm25(self, hybrid_retriever):
        """Test _hybrid_search returns hybrid_rrf when BM25 available."""
        results, method = await hybrid_retriever._hybrid_search("LANEIGE 분석", top_k=5)

        assert method == "hybrid_rrf"
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_hybrid_search_metadata_in_retrieve(self, hybrid_retriever):
        """retrieve() should include search_method in metadata."""
        context = await hybrid_retriever.retrieve("LANEIGE 경쟁력 분석")

        assert "search_method" in context.metadata
        assert context.metadata["search_method"] in ("hybrid_rrf", "dense_only")
        assert "selfrag_confidence" in context.metadata
        assert "bm25_available" in context.metadata

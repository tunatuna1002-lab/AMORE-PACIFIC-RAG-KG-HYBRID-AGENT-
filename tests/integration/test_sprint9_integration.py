"""
Sprint 9 Integration Tests (D-5)

Tests for multi-hop, AIS citation, IRI roundtrip,
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
    """Integration tests for multi-hop retrieval via ReActAgent.

    5-C: ReAct가 네이티브 function calling으로 바뀌면서 스텝 생성 이음매가
    ``_execute_step``(프롬프트 JSON 파싱) → ``_next_move``(tool_calls 읽기)로 옮겨졌다.
    여기서는 그 이음매를 대역으로 두고 **루프·홉 집계·최종 답 추출**을 실제 코드로 검증한다.
    LLM 호출은 일어나지 않는다 (``_next_move``와 ``_reflect``가 대역이다).
    """

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

    @staticmethod
    def _scripted_moves(moves):
        """(thought, action, action_input) 목록 → ``_next_move`` 대역.

        마지막 항목이 소진되면 그대로 반복한다 (루프가 스스로 멈춰야 한다).
        """
        index = {"n": 0}

        async def _next_move(messages, tools, run=None):
            thought, action, action_input = moves[min(index["n"], len(moves) - 1)]
            index["n"] += 1
            if action is None:
                return thought, None
            return thought, (action, action_input, f"call_{index['n']}")

        return _next_move

    async def _run_multihop(self, agent, query, moves):
        from unittest.mock import AsyncMock as _AsyncMock

        with (
            patch.object(agent, "_next_move", side_effect=self._scripted_moves(moves)),
            patch.object(
                agent,
                "_reflect",
                _AsyncMock(return_value={"quality_score": 0.8, "needs_improvement": False}),
            ),
        ):
            return await agent.run(query, context="Test context")

    @pytest.mark.asyncio
    async def test_multihop_competitor_avg_price(self, react_agent):
        """LANEIGE Lip Sleeping Mask의 경쟁 제품들의 평균 가격은?"""
        result = await self._run_multihop(
            react_agent,
            "LANEIGE Lip Sleeping Mask의 경쟁 제품들의 평균 가격은?",
            [
                ("경쟁 관계부터 확인", "kg_neighbors", {"entity": "LANEIGE"}),
                (
                    "추가 검색 필요: competitor pricing",
                    "refine_search",
                    {
                        "refined_query": "competitor prices for LANEIGE",
                        "reason": "need pricing data",
                        "focus_entities": ["LANEIGE"],
                    },
                ),
                (
                    "정리",
                    "final_answer",
                    {"answer": "경쟁 제품 평균 가격은 $24.99입니다.", "confidence": 0.8},
                ),
            ],
        )

        assert result.hop_count >= 1
        assert result.final_answer == "경쟁 제품 평균 가격은 $24.99입니다."
        assert [step.action for step in result.steps] == [
            "kg_neighbors",
            "refine_search",
            "final_answer",
        ]

    @pytest.mark.asyncio
    async def test_multihop_category_leader_performance(self, react_agent):
        """Lip Care 카테고리 1위 브랜드의 다른 카테고리 성과는?"""
        result = await self._run_multihop(
            react_agent,
            "Lip Care 카테고리 1위 브랜드의 다른 카테고리 성과는?",
            [
                ("카테고리 1위부터", "get_metrics", {"category": "Lip Care"}),
                (
                    "추가 검색 필요: cross-category performance",
                    "refine_search",
                    {
                        "refined_query": "LANEIGE other categories",
                        "reason": "need cross-category data",
                        "focus_entities": ["LANEIGE"],
                    },
                ),
                (
                    "정리",
                    "final_answer",
                    {"answer": "LANEIGE는 Lip Care 1위이며 Skin Care에서도 활약합니다."},
                ),
            ],
        )

        assert result.hop_count >= 1
        assert "Lip Care 1위" in result.final_answer

    @pytest.mark.asyncio
    async def test_multihop_cosrx_top3_hhi(self, react_agent):
        """COSRX의 Top 3 제품이 속한 카테고리들의 시장 집중도는?"""
        result = await self._run_multihop(
            react_agent,
            "COSRX의 Top 3 제품이 속한 카테고리들의 시장 집중도는?",
            [
                ("제품부터", "get_metrics", {"brand": "COSRX"}),
                (
                    "추가 검색 필요: HHI for categories",
                    "refine_search",
                    {
                        "refined_query": "HHI for COSRX product categories",
                        "reason": "need market concentration",
                        "focus_entities": ["COSRX"],
                    },
                ),
                (
                    "정리",
                    "final_answer",
                    {"answer": "COSRX 제품 카테고리의 평균 HHI는 0.08입니다."},
                ),
            ],
        )

        assert result.hop_count >= 1
        assert "HHI" in result.final_answer

    @pytest.mark.asyncio
    async def test_multihop_fastest_growing_competitor(self, react_agent):
        """LANEIGE와 경쟁하는 브랜드 중 성장세가 가장 큰 브랜드는?"""
        result = await self._run_multihop(
            react_agent,
            "LANEIGE와 경쟁하는 브랜드 중 성장세가 가장 큰 브랜드는?",
            [
                ("경쟁사부터", "kg_neighbors", {"entity": "LANEIGE"}),
                (
                    "추가 검색 필요: growth rate details",
                    "refine_search",
                    {
                        "refined_query": "ANUA growth rate details",
                        "reason": "need growth data",
                        "focus_entities": ["ANUA"],
                    },
                ),
                ("정리", "final_answer", {"answer": "ANUA가 25.3%로 가장 높은 성장세를 보입니다."}),
            ],
        )

        assert result.hop_count >= 1
        assert "ANUA" in result.final_answer

    @pytest.mark.asyncio
    async def test_multihop_brand_sentiment_profile(self, react_agent):
        """Lip Care에서 SoS가 가장 높은 브랜드의 감성 프로필은?"""
        result = await self._run_multihop(
            react_agent,
            "Lip Care에서 SoS가 가장 높은 브랜드의 감성 프로필은?",
            [
                ("SoS 1위부터", "get_metrics", {"category": "Lip Care"}),
                (
                    "추가 검색 필요: sentiment data",
                    "refine_search",
                    {
                        "refined_query": "LANEIGE sentiment profile",
                        "reason": "need sentiment",
                        "focus_entities": ["LANEIGE"],
                    },
                ),
                (
                    "정리",
                    "final_answer",
                    {"answer": "LANEIGE의 감성 프로필: 긍정 80%, 중립 15%, 부정 5%"},
                ),
            ],
        )

        assert result.hop_count >= 1
        assert "긍정 80%" in result.final_answer

    @pytest.mark.asyncio
    async def test_multihop_loop_stops_without_final_answer(self, react_agent):
        """final_answer 없이 같은 도구만 반복하면 반복 한도에서 멈춘다 (무한 루프 방지)."""
        from unittest.mock import AsyncMock as _AsyncMock

        moves = [("계속 조회", "kg_neighbors", {"entity": "LANEIGE"})]
        with (
            patch.object(react_agent, "_next_move", side_effect=self._scripted_moves(moves)),
            patch.object(react_agent, "_force_final_answer", _AsyncMock(return_value="요약 답변")),
            patch.object(
                react_agent,
                "_reflect",
                _AsyncMock(return_value={"quality_score": 0.5, "needs_improvement": True}),
            ),
        ):
            result = await react_agent.run("질문", context="Test context")

        assert result.iterations == react_agent.max_iterations
        assert result.final_answer == "요약 답변"


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

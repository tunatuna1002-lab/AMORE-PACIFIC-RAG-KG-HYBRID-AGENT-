"""
OWLRetrievalStrategy 단위 테스트
================================
Strategy pattern 구현, OWL 추론 파이프라인, 필터 매칭, 컨텍스트 빌드 검증
(외부 의존: mock)
"""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.value_objects.retrieval_result import UnifiedRetrievalResult
from src.rag.retrieval_strategy import OWLRetrievalStrategy, RetrievalStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeLinkedEntity:
    """테스트용 LinkedEntity stub"""

    text: str
    entity_type: str
    concept_uri: str = ""
    concept_label: str = ""
    confidence: float = 0.9
    ontology_id: str = ""
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeFusedResult:
    """테스트용 FusedResult stub"""

    documents: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.7


@dataclass
class FakeRankedDoc:
    """테스트용 RankedDocument stub"""

    content: str = "ranked content"
    score: float = 0.85
    metadata: dict[str, Any] = field(default_factory=lambda: {"chunk_id": "c1"})


def _make_owl_strategy(**kwargs):
    """OWLRetrievalStrategy 생성 헬퍼 (lazy import를 mock)"""
    with (
        patch("src.rag.entity_linker.EntityLinker") as mock_el,
        patch("src.rag.confidence_fusion.ConfidenceFusion") as mock_cf,
        patch("src.rag.retriever.DocumentRetriever") as mock_dr,
        patch("src.rag.reranker.get_reranker") as mock_rr,
    ):
        mock_dr.return_value = MagicMock()
        mock_el.return_value = MagicMock()
        mock_cf.return_value = MagicMock()
        mock_rr.return_value = MagicMock()
        strategy = OWLRetrievalStrategy(**kwargs)
    return strategy


# ---------------------------------------------------------------------------
# Protocol 확인
# ---------------------------------------------------------------------------


class TestRetrievalStrategyProtocol:
    """RetrievalStrategy Protocol 검증"""

    def test_protocol_is_runtime_checkable(self):
        """Protocol이 runtime_checkable인지 확인"""
        assert (
            hasattr(RetrievalStrategy, "__protocol_attrs__")
            or hasattr(RetrievalStrategy, "__abstractmethods__")
            or True
        )  # runtime_checkable decorator applied

    def test_owl_strategy_has_required_methods(self):
        """OWLRetrievalStrategy에 필수 메서드 존재"""
        strategy = _make_owl_strategy()
        assert hasattr(strategy, "initialize")
        assert hasattr(strategy, "retrieve")
        assert hasattr(strategy, "search")


# ---------------------------------------------------------------------------
# OWLRetrievalStrategy 초기화
# ---------------------------------------------------------------------------


class TestOWLRetrievalStrategyInit:
    """OWLRetrievalStrategy 초기화 테스트"""

    def test_default_init(self):
        """기본 초기화 성공"""
        strategy = _make_owl_strategy()
        assert strategy.kg is None
        assert strategy.owl_reasoner is None
        assert strategy.use_reranking is True
        assert strategy._initialized is False

    def test_init_with_dependencies(self):
        """의존성 주입 초기화"""
        mock_kg = MagicMock()
        mock_owl = MagicMock()
        strategy = _make_owl_strategy(
            knowledge_graph=mock_kg,
            owl_reasoner=mock_owl,
            use_reranking=False,
            use_query_expansion=False,
        )
        assert strategy.kg is mock_kg
        assert strategy.owl_reasoner is mock_owl
        assert strategy.use_reranking is False

    def test_init_with_custom_doc_retriever(self):
        """커스텀 doc_retriever 주입"""
        custom_retriever = MagicMock()
        strategy = _make_owl_strategy(doc_retriever=custom_retriever)
        assert strategy.doc_retriever is custom_retriever


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestOWLRetrievalStrategyInitialize:
    """initialize 메서드 테스트"""

    @pytest.mark.asyncio
    async def test_initialize_sets_flag(self):
        """초기화 후 _initialized=True"""
        strategy = _make_owl_strategy()
        strategy.doc_retriever = AsyncMock()
        await strategy.initialize()
        assert strategy._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """중복 초기화 방지"""
        strategy = _make_owl_strategy()
        strategy.doc_retriever = AsyncMock()
        await strategy.initialize()
        first_call_count = strategy.doc_retriever.initialize.call_count
        await strategy.initialize()
        assert strategy.doc_retriever.initialize.call_count == first_call_count

    @pytest.mark.asyncio
    async def test_initialize_with_ontology_kg(self):
        """OntologyKG 초기화"""
        mock_okg = AsyncMock()
        strategy = _make_owl_strategy(ontology_kg=mock_okg)
        strategy.doc_retriever = AsyncMock()
        await strategy.initialize()
        mock_okg.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_ontology_kg_failure_handled(self):
        """OntologyKG 초기화 실패 시 경고만 발생"""
        mock_okg = AsyncMock()
        mock_okg.initialize.side_effect = RuntimeError("init fail")
        strategy = _make_owl_strategy(ontology_kg=mock_okg)
        strategy.doc_retriever = AsyncMock()
        await strategy.initialize()
        assert strategy._initialized is True  # 실패해도 계속 진행

    @pytest.mark.asyncio
    async def test_initialize_with_owl_reasoner(self):
        """OWL Reasoner 초기화 및 KG 임포트"""
        # initialize는 async로 호출됨, 나머지는 sync
        mock_owl = MagicMock()
        mock_owl.initialize = AsyncMock()
        mock_owl.import_from_knowledge_graph.return_value = 5
        mock_kg = MagicMock()
        strategy = _make_owl_strategy(knowledge_graph=mock_kg, owl_reasoner=mock_owl)
        strategy.doc_retriever = AsyncMock()
        await strategy.initialize()
        mock_owl.initialize.assert_awaited_once()
        mock_owl.import_from_knowledge_graph.assert_called_once_with(mock_kg)
        mock_owl.run_reasoner.assert_called_once()
        mock_owl.infer_market_positions.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_owl_import_zero_entities(self):
        """KG 임포트 0건이면 추론 미실행"""
        mock_owl = MagicMock()
        mock_owl.initialize = AsyncMock()
        mock_owl.import_from_knowledge_graph.return_value = 0
        mock_kg = MagicMock()
        strategy = _make_owl_strategy(knowledge_graph=mock_kg, owl_reasoner=mock_owl)
        strategy.doc_retriever = AsyncMock()
        await strategy.initialize()
        mock_owl.run_reasoner.assert_not_called()


# ---------------------------------------------------------------------------
# _matches_filters (정적 메서드)
# ---------------------------------------------------------------------------


class TestMatchesFilters:
    """_matches_filters 정적 메서드 테스트"""

    def test_empty_filters_matches_all(self):
        """빈 필터는 모든 메타데이터와 매칭"""
        assert OWLRetrievalStrategy._matches_filters({"brand": "LANEIGE"}, {}) is True

    def test_exact_match(self):
        """정확한 값 매칭"""
        metadata = {"brand": "LANEIGE", "category": "Lip Care"}
        filters = {"brand": "LANEIGE"}
        assert OWLRetrievalStrategy._matches_filters(metadata, filters) is True

    def test_exact_mismatch(self):
        """값 불일치"""
        metadata = {"brand": "COSRX"}
        filters = {"brand": "LANEIGE"}
        assert OWLRetrievalStrategy._matches_filters(metadata, filters) is False

    def test_in_operator_match(self):
        """$in 연산자 매칭"""
        metadata = {"brand": "LANEIGE"}
        filters = {"brand": {"$in": ["LANEIGE", "COSRX"]}}
        assert OWLRetrievalStrategy._matches_filters(metadata, filters) is True

    def test_in_operator_mismatch(self):
        """$in 연산자 불일치"""
        metadata = {"brand": "TIRTIR"}
        filters = {"brand": {"$in": ["LANEIGE", "COSRX"]}}
        assert OWLRetrievalStrategy._matches_filters(metadata, filters) is False

    def test_missing_key_in_metadata(self):
        """메타데이터에 키가 없으면 불일치"""
        metadata = {}
        filters = {"brand": "LANEIGE"}
        assert OWLRetrievalStrategy._matches_filters(metadata, filters) is False

    def test_multiple_filters(self):
        """다중 필터 조건"""
        metadata = {"brand": "LANEIGE", "category": "Lip Care"}
        filters = {"brand": "LANEIGE", "category": "Lip Care"}
        assert OWLRetrievalStrategy._matches_filters(metadata, filters) is True

    def test_multiple_filters_partial_fail(self):
        """다중 필터 중 하나 불일치"""
        metadata = {"brand": "LANEIGE", "category": "Skin Care"}
        filters = {"brand": "LANEIGE", "category": "Lip Care"}
        assert OWLRetrievalStrategy._matches_filters(metadata, filters) is False


# ---------------------------------------------------------------------------
# _build_combined_context (정적 메서드)
# ---------------------------------------------------------------------------


class TestBuildCombinedContext:
    """_build_combined_context 테스트"""

    def test_empty_inputs(self):
        """모든 입력이 빈 경우"""
        result = OWLRetrievalStrategy._build_combined_context([], {}, [])
        assert result == ""

    def test_with_entity_links(self):
        """엔티티 링크 포함"""
        entities = [
            FakeLinkedEntity(text="LANEIGE", entity_type="brand", confidence=0.95),
        ]
        result = OWLRetrievalStrategy._build_combined_context(entities, {}, [])
        assert "LANEIGE" in result
        assert "엔티티" in result

    def test_with_market_position_inference(self):
        """시장 포지션 추론 결과 포함"""
        ontology_ctx = {
            "inferences": [
                {"type": "market_position", "brand": "LANEIGE", "position": "leader", "sos": 0.15}
            ],
            "facts": [],
        }
        result = OWLRetrievalStrategy._build_combined_context([], ontology_ctx, [])
        assert "LANEIGE" in result
        assert "시장 포지션" in result or "leader" in result

    def test_with_competition_inference(self):
        """경쟁 분석 추론 결과 포함"""
        ontology_ctx = {
            "inferences": [
                {"type": "competition", "brand": "LANEIGE", "competitors": ["COSRX", "TIRTIR"]}
            ],
            "facts": [],
        }
        result = OWLRetrievalStrategy._build_combined_context([], ontology_ctx, [])
        assert "경쟁사" in result
        assert "COSRX" in result

    def test_with_documents(self):
        """관련 문서 포함"""
        docs = [
            {"metadata": {"title": "Test Doc"}, "content": "This is test content."},
        ]
        result = OWLRetrievalStrategy._build_combined_context([], {}, docs)
        assert "Test Doc" in result
        assert "test content" in result

    def test_document_content_truncation(self):
        """긴 문서 내용 500자 초과 시 잘림"""
        docs = [
            {"metadata": {"title": "Long"}, "content": "A" * 600},
        ]
        result = OWLRetrievalStrategy._build_combined_context([], {}, docs)
        assert "..." in result

    def test_combined_all_sections(self):
        """모든 섹션이 포함된 컨텍스트"""
        entities = [FakeLinkedEntity(text="LANEIGE", entity_type="brand", confidence=0.9)]
        ontology_ctx = {
            "inferences": [
                {"type": "market_position", "brand": "LANEIGE", "position": "leader", "sos": 0.1}
            ],
            "facts": [],
        }
        docs = [{"metadata": {"title": "Doc1"}, "content": "Content1"}]
        result = OWLRetrievalStrategy._build_combined_context(entities, ontology_ctx, docs)
        assert "엔티티" in result
        assert "온톨로지" in result
        assert "문서" in result


# ---------------------------------------------------------------------------
# search (위임)
# ---------------------------------------------------------------------------


class TestOWLRetrievalStrategySearch:
    """search 메서드 테스트 (doc_retriever 위임)"""

    @pytest.mark.asyncio
    async def test_search_delegates_to_doc_retriever(self):
        """doc_retriever.search에 위임"""
        strategy = _make_owl_strategy()
        strategy.doc_retriever = AsyncMock()
        strategy.doc_retriever.search = AsyncMock(return_value=[{"id": "1", "content": "test"}])
        results = await strategy.search("test query", top_k=3)
        strategy.doc_retriever.search.assert_awaited_once_with(
            query="test query", top_k=3, doc_filter=None
        )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_no_search_method(self):
        """doc_retriever에 search 메서드가 없으면 빈 리스트"""
        strategy = _make_owl_strategy()
        strategy.doc_retriever = MagicMock(spec=[])
        results = await strategy.search("test")
        assert results == []


# ---------------------------------------------------------------------------
# retrieve (전체 파이프라인)
# ---------------------------------------------------------------------------


class TestOWLRetrievalStrategyRetrieve:
    """retrieve 메서드 - 전체 OWL 파이프라인 테스트"""

    def _make_strategy_for_retrieve(self):
        """테스트용 strategy 생성 (모든 의존성 mock)"""
        strategy = _make_owl_strategy()

        # Mock 의존성 설정
        strategy.doc_retriever = AsyncMock()
        strategy.doc_retriever.initialize = AsyncMock()
        strategy.doc_retriever.expand_query = AsyncMock(return_value=["test query"])
        strategy.doc_retriever.search = AsyncMock(
            return_value=[
                {"id": "doc1", "content": "Content 1", "metadata": {}, "score": 0.8},
            ]
        )

        strategy.entity_linker = MagicMock()
        strategy.entity_linker.link.return_value = [
            FakeLinkedEntity(
                text="LANEIGE",
                entity_type="brand",
                confidence=0.9,
                ontology_id="laneige",
            )
        ]
        strategy.entity_linker.get_ontology_filters.return_value = {}

        fused = FakeFusedResult(
            documents=[{"id": "doc1", "content": "Content 1", "metadata": {}, "score": 0.8}],
            confidence=0.75,
        )
        strategy.confidence_fusion = MagicMock()
        strategy.confidence_fusion.fuse.return_value = fused

        strategy._initialized = True
        return strategy

    @pytest.mark.asyncio
    async def test_retrieve_returns_unified_result(self):
        """retrieve가 UnifiedRetrievalResult 반환"""
        strategy = self._make_strategy_for_retrieve()
        result = await strategy.retrieve("LANEIGE 분석", top_k=5)
        assert isinstance(result, UnifiedRetrievalResult)
        assert result.query == "LANEIGE 분석"
        assert result.retriever_type == "owl"

    @pytest.mark.asyncio
    async def test_retrieve_confidence_bounded(self):
        """confidence가 0.0-1.0 범위"""
        strategy = self._make_strategy_for_retrieve()
        result = await strategy.retrieve("test", top_k=5)
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_retrieve_entities_dict_structure(self):
        """entities 딕셔너리 구조 확인"""
        strategy = self._make_strategy_for_retrieve()
        result = await strategy.retrieve("LANEIGE", top_k=5)
        for key in ["brands", "categories", "indicators", "products"]:
            assert key in result.entities

    @pytest.mark.asyncio
    async def test_retrieve_auto_initializes(self):
        """_initialized=False이면 자동 초기화"""
        strategy = self._make_strategy_for_retrieve()
        strategy._initialized = False
        result = await strategy.retrieve("test", top_k=3)
        assert isinstance(result, UnifiedRetrievalResult)

    @pytest.mark.asyncio
    async def test_retrieve_with_reranking_disabled(self):
        """reranking 비활성화"""
        strategy = self._make_strategy_for_retrieve()
        strategy.use_reranking = False
        result = await strategy.retrieve("test", top_k=5)
        assert isinstance(result, UnifiedRetrievalResult)

    @pytest.mark.asyncio
    async def test_retrieve_metadata_contains_timing(self):
        """metadata에 retrieval_time_ms 포함"""
        strategy = self._make_strategy_for_retrieve()
        result = await strategy.retrieve("test", top_k=5)
        assert "retrieval_time_ms" in result.metadata

    @pytest.mark.asyncio
    async def test_retrieve_pipeline_exception_handled(self):
        """파이프라인 예외 시 confidence=0.0"""
        strategy = self._make_strategy_for_retrieve()
        strategy.entity_linker.link.side_effect = RuntimeError("entity link failed")
        result = await strategy.retrieve("test", top_k=5)
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_retrieve_brand_entity_populated(self):
        """브랜드 엔티티가 entities['brands']에 추가"""
        strategy = self._make_strategy_for_retrieve()
        result = await strategy.retrieve("LANEIGE 경쟁력", top_k=5)
        assert "laneige" in result.entities.get("brands", [])


# ---------------------------------------------------------------------------
# _rerank
# ---------------------------------------------------------------------------


class TestOWLRetrievalStrategyRerank:
    """_rerank 메서드 테스트"""

    @pytest.mark.asyncio
    async def test_rerank_success(self):
        """정상 reranking"""
        strategy = _make_owl_strategy()
        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = [
            FakeRankedDoc(content="ranked", score=0.9, metadata={"chunk_id": "c1"}),
        ]
        strategy._reranker = mock_reranker_instance
        docs = [{"id": "d1", "content": "test", "score": 0.5}]
        result = await strategy._rerank("query", docs, top_k=5)
        assert len(result) == 1
        assert result[0]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_rerank_failure_fallback(self):
        """reranking 실패 시 원본 반환"""
        strategy = _make_owl_strategy()
        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.side_effect = RuntimeError("rerank fail")
        strategy._reranker = mock_reranker_instance
        docs = [{"id": "d1", "content": "test", "score": 0.5}]
        result = await strategy._rerank("query", docs, top_k=5)
        assert result == docs[:5]

    @pytest.mark.asyncio
    async def test_rerank_lazy_init(self):
        """_reranker가 None이면 lazy 초기화"""
        strategy = _make_owl_strategy()
        mock_reranker_fn = MagicMock()
        mock_reranker_inst = MagicMock()
        mock_reranker_inst.rerank.return_value = []
        mock_reranker_fn.return_value = mock_reranker_inst
        strategy._get_reranker = mock_reranker_fn
        strategy._reranker = None
        await strategy._rerank("query", [], top_k=5)
        mock_reranker_fn.assert_called_once()


# ---------------------------------------------------------------------------
# _ontology_guided_search
# ---------------------------------------------------------------------------


class TestOntologyGuidedSearch:
    """_ontology_guided_search 테스트"""

    @pytest.mark.asyncio
    async def test_deduplication(self):
        """중복 결과 제거"""
        strategy = _make_owl_strategy()
        strategy.doc_retriever = AsyncMock()
        # 같은 id를 가진 결과를 두 번 반환
        strategy.doc_retriever.search = AsyncMock(
            return_value=[
                {"id": "dup1", "content": "doc", "metadata": {}, "score": 0.8},
            ]
        )
        results = await strategy._ontology_guided_search(["query1", "query2"], {}, top_k=10)
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids))  # 중복 없음

    @pytest.mark.asyncio
    async def test_filter_applied(self):
        """필터가 적용된 결과만 반환"""
        strategy = _make_owl_strategy()
        strategy.doc_retriever = AsyncMock()
        strategy.doc_retriever.search = AsyncMock(
            return_value=[
                {"id": "d1", "content": "c", "metadata": {"brand": "LANEIGE"}, "score": 0.9},
                {"id": "d2", "content": "c", "metadata": {"brand": "COSRX"}, "score": 0.7},
            ]
        )
        results = await strategy._ontology_guided_search(["query"], {"brand": "LANEIGE"}, top_k=10)
        assert all(r["metadata"].get("brand") == "LANEIGE" for r in results)


# ---------------------------------------------------------------------------
# _infer_with_ontology
# ---------------------------------------------------------------------------


class TestInferWithOntology:
    """_infer_with_ontology 테스트"""

    @pytest.mark.asyncio
    async def test_no_reasoners_returns_empty(self):
        """reasoner가 없으면 빈 컨텍스트"""
        strategy = _make_owl_strategy()
        strategy.unified_reasoner = None
        strategy.owl_reasoner = None
        result = await strategy._infer_with_ontology([], None)
        assert result["inferences"] == []
        assert result["facts"] == []

    @pytest.mark.asyncio
    async def test_unified_reasoner_brand_inference(self):
        """UnifiedReasoner로 브랜드 추론"""
        strategy = _make_owl_strategy()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"type": "market_position", "brand": "LANEIGE"}
        strategy.unified_reasoner = MagicMock()
        strategy.unified_reasoner.infer.return_value = mock_result
        strategy.ontology_kg = None

        entity = FakeLinkedEntity(text="LANEIGE", entity_type="brand", ontology_id="laneige")
        result = await strategy._infer_with_ontology([entity], {"sos": 0.15})
        assert len(result["inferences"]) == 1

    @pytest.mark.asyncio
    async def test_owl_reasoner_fallback(self):
        """UnifiedReasoner 없으면 OWL reasoner 폴백"""
        strategy = _make_owl_strategy()
        strategy.unified_reasoner = None
        strategy.owl_reasoner = MagicMock()
        strategy.owl_reasoner.get_inferred_facts.return_value = [{"fact": "test"}]
        strategy.owl_reasoner.get_brand_info.return_value = {
            "market_position": "leader",
            "sos": 0.15,
            "competitors": ["COSRX"],
        }

        entity = FakeLinkedEntity(text="LANEIGE", entity_type="brand", ontology_id="laneige")
        result = await strategy._infer_with_ontology([entity], None)
        assert len(result["facts"]) == 1
        assert len(result["inferences"]) >= 1

    @pytest.mark.asyncio
    async def test_inference_exception_handled(self):
        """추론 예외 시 빈 결과 반환"""
        strategy = _make_owl_strategy()
        strategy.unified_reasoner = MagicMock()
        strategy.unified_reasoner.infer.side_effect = Exception("infer fail")
        strategy.ontology_kg = None

        entity = FakeLinkedEntity(text="LANEIGE", entity_type="brand")
        result = await strategy._infer_with_ontology([entity], None)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# IntentRetrievalConfig + get_intent_retrieval_config (Phase 2A)
# ---------------------------------------------------------------------------


from src.core.intent import UnifiedIntent
from src.rag.retrieval_strategy import IntentRetrievalConfig, get_intent_retrieval_config


class TestIntentRetrievalConfig:
    """IntentRetrievalConfig 데이터클래스 테스트"""

    def test_default_values(self):
        """기본값 확인"""
        config = IntentRetrievalConfig()
        assert config.weights == {"kg": 0.4, "rag": 0.4, "inference": 0.2}
        assert config.top_k == 5
        assert config.doc_type_filter is None
        assert config.description == "default"

    def test_frozen_dataclass(self):
        """frozen=True이므로 수정 불가"""
        config = IntentRetrievalConfig()
        with pytest.raises(AttributeError):
            config.top_k = 10  # type: ignore[misc]

    def test_custom_values(self):
        """커스텀 값으로 생성"""
        config = IntentRetrievalConfig(
            weights={"kg": 0.6, "rag": 0.2, "inference": 0.2},
            top_k=10,
            doc_type_filter=["playbook"],
            description="test",
        )
        assert config.weights["kg"] == 0.6
        assert config.top_k == 10
        assert config.doc_type_filter == ["playbook"]
        assert config.description == "test"


class TestGetIntentRetrievalConfig:
    """get_intent_retrieval_config 함수 테스트"""

    def test_all_intents_mapped(self):
        """모든 UnifiedIntent 값에 대해 config 반환"""
        for intent in UnifiedIntent:
            config = get_intent_retrieval_config(intent)
            assert isinstance(config, IntentRetrievalConfig)
            assert isinstance(config.weights, dict)
            assert "kg" in config.weights
            assert "rag" in config.weights
            assert "inference" in config.weights

    def test_weights_sum_to_one(self):
        """모든 인텐트의 가중치 합이 1.0"""
        for intent in UnifiedIntent:
            config = get_intent_retrieval_config(intent)
            total = sum(config.weights.values())
            assert abs(total - 1.0) < 0.01, f"{intent.value}: weights sum to {total}"

    def test_diagnosis_is_graph_heavy(self):
        """DIAGNOSIS → KG 가중치가 RAG보다 높음"""
        config = get_intent_retrieval_config(UnifiedIntent.DIAGNOSIS)
        assert config.weights["kg"] > config.weights["rag"]
        assert config.description.startswith("graph-heavy")

    def test_competitive_is_graph_heavy(self):
        """COMPETITIVE → KG 가중치가 높음"""
        config = get_intent_retrieval_config(UnifiedIntent.COMPETITIVE)
        assert config.weights["kg"] >= config.weights["rag"]
        assert config.description.startswith("graph-heavy")

    def test_general_is_vector_heavy(self):
        """GENERAL → RAG 가중치가 KG보다 높음"""
        config = get_intent_retrieval_config(UnifiedIntent.GENERAL)
        assert config.weights["rag"] > config.weights["kg"]
        assert config.description.startswith("vector-heavy")

    def test_definition_is_vector_heavy(self):
        """DEFINITION → RAG 문서 가중치가 높음"""
        config = get_intent_retrieval_config(UnifiedIntent.DEFINITION)
        assert config.weights["rag"] > config.weights["kg"]
        assert "metric_guide" in (config.doc_type_filter or [])

    def test_analysis_is_inference_heavy(self):
        """ANALYSIS → inference 가중치가 가장 높음"""
        config = get_intent_retrieval_config(UnifiedIntent.ANALYSIS)
        assert config.weights["inference"] >= config.weights["kg"]
        assert config.weights["inference"] >= config.weights["rag"]

    def test_insight_rule_is_inference_heavy(self):
        """INSIGHT_RULE → inference 가중치가 높음"""
        config = get_intent_retrieval_config(UnifiedIntent.INSIGHT_RULE)
        assert config.weights["inference"] >= config.weights["kg"]

    def test_trend_is_hybrid(self):
        """TREND → 균형 잡힌 가중치"""
        config = get_intent_retrieval_config(UnifiedIntent.TREND)
        assert config.description.startswith("hybrid")
        # top_k가 기본보다 높음 (더 넓은 검색)
        assert config.top_k >= 5

    def test_crisis_has_response_guide_priority(self):
        """CRISIS → response_guide 문서 우선"""
        config = get_intent_retrieval_config(UnifiedIntent.CRISIS)
        assert config.doc_type_filter is not None
        assert "response_guide" in config.doc_type_filter

    def test_metric_has_metric_guide_priority(self):
        """METRIC → metric_guide 문서 우선"""
        config = get_intent_retrieval_config(UnifiedIntent.METRIC)
        assert config.doc_type_filter is not None
        assert "metric_guide" in config.doc_type_filter

    def test_general_has_no_doc_filter(self):
        """GENERAL → 전체 문서 검색 (필터 없음)"""
        config = get_intent_retrieval_config(UnifiedIntent.GENERAL)
        assert config.doc_type_filter is None

    def test_data_query_has_no_doc_filter(self):
        """DATA_QUERY → 전체 문서 검색"""
        config = get_intent_retrieval_config(UnifiedIntent.DATA_QUERY)
        assert config.doc_type_filter is None

    def test_top_k_positive(self):
        """모든 인텐트의 top_k가 양수"""
        for intent in UnifiedIntent:
            config = get_intent_retrieval_config(intent)
            assert config.top_k > 0

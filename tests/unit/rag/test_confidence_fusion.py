"""
ConfidenceFusion 단위 테스트
===========================
다중 소스 신뢰도 융합 엔진 검증
"""

from src.rag.confidence_fusion import (
    ConfidenceFusion,
    FusedResult,
    FusionStrategy,
    InferenceResult,
    LinkedEntity,
    ScoreNormalizationMethod,
    SearchResult,
    SourceScore,
)

# ---------------------------------------------------------------------------
# Enum 테스트
# ---------------------------------------------------------------------------


class TestEnums:
    """Enum 값 검증"""

    def test_normalization_methods(self):
        assert ScoreNormalizationMethod.MIN_MAX == "min_max"
        assert ScoreNormalizationMethod.SOFTMAX == "softmax"
        assert ScoreNormalizationMethod.Z_SCORE == "z_score"
        assert ScoreNormalizationMethod.NONE == "none"

    def test_fusion_strategies(self):
        assert FusionStrategy.WEIGHTED_SUM == "weighted_sum"
        assert FusionStrategy.HARMONIC_MEAN == "harmonic_mean"
        assert FusionStrategy.GEOMETRIC_MEAN == "geometric_mean"
        assert FusionStrategy.MAX_SCORE == "max_score"
        assert FusionStrategy.MIN_SCORE == "min_score"


# ---------------------------------------------------------------------------
# Dataclass 테스트
# ---------------------------------------------------------------------------


class TestSearchResult:
    """SearchResult 데이터클래스"""

    def test_create(self):
        r = SearchResult(content="test", score=0.8, metadata={}, source="vector")
        assert r.content == "test"
        assert r.score == 0.8
        assert r.source == "vector"

    def test_default_source(self):
        r = SearchResult(content="test", score=0.5, metadata={})
        assert r.source == "vector"


class TestInferenceResult:
    """InferenceResult 데이터클래스"""

    def test_create(self):
        r = InferenceResult(insight="test", confidence=0.7, evidence={})
        assert r.insight == "test"
        assert r.confidence == 0.7

    def test_optional_rule_name(self):
        r = InferenceResult(insight="test", confidence=0.7, evidence={}, rule_name="rule1")
        assert r.rule_name == "rule1"

    def test_default_rule_name_none(self):
        r = InferenceResult(insight="test", confidence=0.7, evidence={})
        assert r.rule_name is None


class TestLinkedEntity:
    """LinkedEntity 데이터클래스"""

    def test_create(self):
        e = LinkedEntity(
            entity_id="e1",
            entity_name="LANEIGE",
            entity_type="brand",
            link_confidence=0.9,
        )
        assert e.entity_name == "LANEIGE"
        assert e.link_confidence == 0.9

    def test_optional_fields(self):
        e = LinkedEntity(
            entity_id="e1",
            entity_name="LANEIGE",
            entity_type="brand",
            link_confidence=0.9,
            context="Lip Care context",
            metadata={"category": "lip_care"},
        )
        assert e.context == "Lip Care context"
        assert e.metadata["category"] == "lip_care"


class TestSourceScore:
    """SourceScore 데이터클래스"""

    def test_create(self):
        s = SourceScore(
            source_name="vector",
            raw_score=0.8,
            normalized_score=0.9,
            weight=0.4,
            contribution=0.36,
            confidence_level="high",
            explanation="벡터 검색 결과",
        )
        assert s.source_name == "vector"
        assert s.contribution == 0.36


class TestFusedResult:
    """FusedResult 데이터클래스"""

    def test_create(self):
        r = FusedResult(
            documents=[{"content": "test"}],
            confidence=0.85,
            explanation="test",
            source_scores=[],
            fusion_strategy="weighted_sum",
        )
        assert r.confidence == 0.85
        assert len(r.documents) == 1

    def test_to_dict(self):
        r = FusedResult(
            documents=[{"content": "test"}],
            confidence=0.85,
            explanation="설명",
            source_scores=[],
            fusion_strategy="weighted_sum",
        )
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "confidence" in d
        assert "documents" in d

    def test_default_warnings(self):
        r = FusedResult(
            documents=[],
            confidence=0.5,
            explanation="",
            source_scores=[],
            fusion_strategy="weighted_sum",
        )
        assert r.warnings == []


# ---------------------------------------------------------------------------
# ConfidenceFusion 초기화
# ---------------------------------------------------------------------------


class TestConfidenceFusionInit:
    """융합 엔진 초기화"""

    def test_default_weights(self):
        fusion = ConfidenceFusion()
        assert hasattr(fusion, "weights")
        weights = fusion.weights
        assert "vector" in weights
        assert "ontology" in weights
        assert "entity" in weights

    def test_weights_sum_to_one(self):
        fusion = ConfidenceFusion()
        total = sum(fusion.weights.values())
        assert abs(total - 1.0) < 0.01

    def test_default_normalization(self):
        fusion = ConfidenceFusion()
        assert hasattr(fusion, "normalization")

    def test_default_strategy(self):
        fusion = ConfidenceFusion()
        assert hasattr(fusion, "strategy")


# ---------------------------------------------------------------------------
# _normalize_scores (Dict 기반)
# ---------------------------------------------------------------------------


class TestNormalizeScores:
    """점수 정규화 (Dict[str, float] 입력)"""

    def test_normalize_scores_returns_dict(self):
        fusion = ConfidenceFusion()
        scores = {"vector": 0.8, "ontology": 0.6, "entity": 0.7}
        normalized = fusion._normalize_scores(scores)
        assert isinstance(normalized, dict)
        assert len(normalized) == 3

    def test_empty_scores(self):
        fusion = ConfidenceFusion()
        normalized = fusion._normalize_scores({})
        assert normalized == {}

    def test_single_score(self):
        fusion = ConfidenceFusion()
        normalized = fusion._normalize_scores({"vector": 0.5})
        assert isinstance(normalized, dict)
        assert len(normalized) == 1

    def test_no_normalization(self):
        fusion = ConfidenceFusion(normalization=ScoreNormalizationMethod.NONE)
        scores = {"vector": 0.8, "ontology": 0.6}
        normalized = fusion._normalize_scores(scores)
        assert normalized == scores


# ---------------------------------------------------------------------------
# fuse
# ---------------------------------------------------------------------------


class TestFuse:
    """융합 실행"""

    def test_fuse_with_vector_only(self):
        fusion = ConfidenceFusion()
        vector_results = [
            SearchResult(content="doc1", score=0.8, metadata={}, source="vector"),
            SearchResult(content="doc2", score=0.6, metadata={}, source="vector"),
        ]
        result = fusion.fuse(vector_results=vector_results)
        assert isinstance(result, FusedResult)
        assert result.confidence > 0

    def test_fuse_with_all_sources(self):
        fusion = ConfidenceFusion()
        vector_results = [
            SearchResult(content="doc1", score=0.8, metadata={}, source="vector"),
        ]
        ontology_results = [
            InferenceResult(insight="insight1", confidence=0.7, evidence={}),
        ]
        entity_links = [
            LinkedEntity(
                entity_id="e1", entity_name="LANEIGE", entity_type="brand", link_confidence=0.9
            ),
        ]
        result = fusion.fuse(
            vector_results=vector_results,
            ontology_results=ontology_results,
            entity_links=entity_links,
        )
        assert isinstance(result, FusedResult)
        assert result.confidence > 0

    def test_fuse_empty_returns_result(self):
        fusion = ConfidenceFusion()
        result = fusion.fuse()
        assert isinstance(result, FusedResult)

    def test_fused_result_has_documents(self):
        fusion = ConfidenceFusion()
        vector_results = [
            SearchResult(content="doc1", score=0.8, metadata={"title": "t1"}, source="vector"),
        ]
        result = fusion.fuse(vector_results=vector_results)
        assert hasattr(result, "documents")

    def test_fuse_has_explanation(self):
        fusion = ConfidenceFusion()
        vector_results = [
            SearchResult(content="doc1", score=0.8, metadata={}, source="vector"),
        ]
        result = fusion.fuse(vector_results=vector_results)
        assert isinstance(result.explanation, str)

    def test_fuse_has_source_scores(self):
        fusion = ConfidenceFusion()
        vector_results = [
            SearchResult(content="doc1", score=0.8, metadata={}, source="vector"),
        ]
        result = fusion.fuse(vector_results=vector_results)
        assert isinstance(result.source_scores, list)


# ---------------------------------------------------------------------------
# _detect_conflicts (Dict 기반)
# ---------------------------------------------------------------------------


class TestDetectConflicts:
    """충돌 감지"""

    def test_no_conflict_with_single_source(self):
        fusion = ConfidenceFusion()
        warnings = fusion._detect_conflicts({"vector": 0.8})
        assert isinstance(warnings, list)

    def test_no_conflict_with_similar_scores(self):
        fusion = ConfidenceFusion()
        warnings = fusion._detect_conflicts({"vector": 0.8, "ontology": 0.7})
        assert isinstance(warnings, list)

    def test_empty_scores_no_conflict(self):
        fusion = ConfidenceFusion()
        warnings = fusion._detect_conflicts({})
        assert warnings == []


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class TestFactoryFunctions:
    """팩토리 함수"""

    def test_create_default_fusion(self):
        from src.rag.confidence_fusion import create_default_fusion

        fusion = create_default_fusion()
        assert isinstance(fusion, ConfidenceFusion)

    def test_create_conservative_fusion(self):
        from src.rag.confidence_fusion import create_conservative_fusion

        fusion = create_conservative_fusion()
        assert isinstance(fusion, ConfidenceFusion)

    def test_create_optimistic_fusion(self):
        from src.rag.confidence_fusion import create_optimistic_fusion

        fusion = create_optimistic_fusion()
        assert isinstance(fusion, ConfidenceFusion)

    def test_create_rrf_fusion(self):
        from src.rag.confidence_fusion import create_rrf_fusion

        fusion = create_rrf_fusion()
        assert isinstance(fusion, ConfidenceFusion)
        assert fusion.strategy == FusionStrategy.RRF

    def test_create_rrf_fusion_with_k_parameter(self):
        from src.rag.confidence_fusion import create_rrf_fusion

        fusion = create_rrf_fusion(k=30)
        assert isinstance(fusion, ConfidenceFusion)
        assert fusion.strategy == FusionStrategy.RRF


# ---------------------------------------------------------------------------
# RRF Document Fusion 테스트
# ---------------------------------------------------------------------------


class TestRRFDocumentFusion:
    """Document-level RRF 융합 테스트"""

    def test_rrf_document_fusion_basic(self):
        """기본 RRF document fusion 테스트"""
        fusion = ConfidenceFusion(strategy=FusionStrategy.RRF)
        ranked_lists = {
            "vector": [
                {"content": "LANEIGE SoS analysis", "score": 0.9},
                {"content": "COSRX market share", "score": 0.7},
            ],
            "bm25": [
                {"content": "COSRX market share", "score": 0.85},
                {"content": "LANEIGE SoS analysis", "score": 0.6},
            ],
        }
        results = fusion.fuse_documents_rrf(ranked_lists, k=60, top_n=5)
        assert len(results) > 0
        assert all("rrf_score" in doc for doc in results)
        assert all("rrf_sources" in doc for doc in results)

    def test_rrf_empty_lists(self):
        """빈 ranked_lists 처리 테스트"""
        fusion = ConfidenceFusion(strategy=FusionStrategy.RRF)
        results = fusion.fuse_documents_rrf({}, k=60, top_n=5)
        assert results == []

    def test_rrf_single_list(self):
        """단일 소스 리스트 테스트"""
        fusion = ConfidenceFusion(strategy=FusionStrategy.RRF)
        ranked_lists = {"vector": [{"content": "doc A", "score": 0.9}]}
        results = fusion.fuse_documents_rrf(ranked_lists)
        assert len(results) == 1
        assert results[0]["rrf_score"] > 0
        assert "vector" in results[0]["rrf_sources"]

    def test_rrf_document_dedup(self):
        """동일 문서가 여러 소스에서 나타날 때 중복 제거 + 점수 합산"""
        fusion = ConfidenceFusion(strategy=FusionStrategy.RRF)
        ranked_lists = {
            "vector": [{"content": "shared document", "score": 0.9}],
            "bm25": [{"content": "shared document", "score": 0.8}],
            "ontology": [{"content": "shared document", "score": 0.7}],
        }
        results = fusion.fuse_documents_rrf(ranked_lists, k=60, top_n=5)
        # 동일 content이므로 하나의 문서로 합쳐져야 함
        assert len(results) == 1
        # 3개 소스에서 모두 1위이므로 점수가 합산됨
        assert results[0]["rrf_score"] > 1.0 / (60 + 1)
        assert len(results[0]["rrf_sources"]) == 3

    def test_rrf_top_n_limit(self):
        """top_n 제한 테스트"""
        fusion = ConfidenceFusion(strategy=FusionStrategy.RRF)
        docs = [{"content": f"doc_{i}", "score": 0.9 - i * 0.1} for i in range(10)]
        ranked_lists = {"vector": docs}
        results = fusion.fuse_documents_rrf(ranked_lists, k=60, top_n=3)
        assert len(results) == 3

    def test_rrf_k_parameter_effect(self):
        """k 파라미터 효과 테스트: k가 작으면 상위 순위 가중치가 더 높음"""
        fusion = ConfidenceFusion(strategy=FusionStrategy.RRF)
        ranked_lists = {
            "vector": [
                {"content": "doc1", "score": 0.9},
                {"content": "doc2", "score": 0.5},
            ],
        }
        # Small k: bigger gap between rank 1 and rank 2
        results_small_k = fusion.fuse_documents_rrf(ranked_lists, k=1, top_n=5)
        # Large k: smaller gap
        results_large_k = fusion.fuse_documents_rrf(ranked_lists, k=1000, top_n=5)

        if len(results_small_k) >= 2 and len(results_large_k) >= 2:
            gap_small = results_small_k[0]["rrf_score"] - results_small_k[1]["rrf_score"]
            gap_large = results_large_k[0]["rrf_score"] - results_large_k[1]["rrf_score"]
            assert gap_small > gap_large

    def test_rrf_with_insight_key(self):
        """insight 키를 사용하는 문서 테스트 (ontology 결과)"""
        fusion = ConfidenceFusion(strategy=FusionStrategy.RRF)
        ranked_lists = {
            "ontology": [
                {"insight": "market insight 1", "confidence": 0.9},
                {"insight": "market insight 2", "confidence": 0.7},
            ],
        }
        results = fusion.fuse_documents_rrf(ranked_lists, k=60, top_n=5)
        assert len(results) == 2

    def test_rrf_scores_descending(self):
        """결과가 RRF 점수 내림차순으로 정렬되는지 테스트"""
        fusion = ConfidenceFusion(strategy=FusionStrategy.RRF)
        ranked_lists = {
            "vector": [
                {"content": "doc1", "score": 0.9},
                {"content": "doc2", "score": 0.7},
                {"content": "doc3", "score": 0.5},
            ],
        }
        results = fusion.fuse_documents_rrf(ranked_lists, k=60, top_n=5)
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

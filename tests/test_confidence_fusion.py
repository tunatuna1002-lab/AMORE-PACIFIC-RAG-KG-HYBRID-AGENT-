"""
Confidence Fusion 모듈 테스트

Usage:
    python -m pytest tests/test_confidence_fusion.py -v
"""

import pytest

from src.rag.confidence_fusion import (
    ConfidenceFusion,
    FusedResult,
    FusionStrategy,
    InferenceResult,
    LinkedEntity,
    ScoreNormalizationMethod,
    SearchResult,
    create_conservative_fusion,
    create_default_fusion,
    create_optimistic_fusion,
)

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def sample_vector_results():
    """샘플 벡터 검색 결과"""
    return [
        SearchResult(
            content="LANEIGE Lip Sleeping Mask is a best-selling lip care product.",
            score=0.92,
            metadata={"doc_id": "doc_1", "source": "vector"},
            source="vector",
        ),
        SearchResult(
            content="Lip care products include lip balms, masks, and treatments.",
            score=0.85,
            metadata={"doc_id": "doc_2", "source": "vector"},
            source="vector",
        ),
        SearchResult(
            content="Amazon bestseller rankings update hourly.",
            score=0.45,
            metadata={"doc_id": "doc_3", "source": "vector"},
            source="vector",
        ),
    ]


@pytest.fixture
def sample_ontology_results():
    """샘플 온톨로지 추론 결과"""
    return [
        InferenceResult(
            insight="LANEIGE는 Lip Care 카테고리에서 지배적 포지션을 보유",
            confidence=0.88,
            evidence={"sos": 0.35, "rank": 1, "category": "Lip Care"},
            rule_name="market_dominance_rule",
        ),
        InferenceResult(
            insight="LANEIGE는 Beauty & Personal Care에서 강력한 브랜드 파워 보유",
            confidence=0.75,
            evidence={"brand_product_count": 5, "avg_rank": 12.4},
            rule_name="brand_strength_rule",
        ),
    ]


@pytest.fixture
def sample_entity_links():
    """샘플 엔티티 연결 결과"""
    return [
        LinkedEntity(
            entity_id="brand_laneige",
            entity_name="LANEIGE",
            entity_type="Brand",
            link_confidence=0.95,
            context="Query mentioned LANEIGE explicitly",
            metadata={"linked_by": "exact_match"},
        ),
        LinkedEntity(
            entity_id="cat_lip_care",
            entity_name="Lip Care",
            entity_type="Category",
            link_confidence=0.80,
            context="Lip Sleeping Mask belongs to Lip Care",
            metadata={"linked_by": "semantic_match"},
        ),
    ]


# =========================================================================
# 기본 테스트
# =========================================================================


def test_fusion_initialization():
    """Fusion 객체 초기화 테스트"""
    fusion = ConfidenceFusion()
    assert fusion.weights == ConfidenceFusion.DEFAULT_WEIGHTS
    assert fusion.strategy == FusionStrategy.WEIGHTED_SUM
    assert fusion.normalization == ScoreNormalizationMethod.MIN_MAX


def test_invalid_weights():
    """잘못된 가중치 검증 테스트"""
    with pytest.raises(ValueError):
        ConfidenceFusion(weights={"vector": 0.5, "ontology": 0.3})  # 합이 1.0 아님


def test_fuse_all_sources(sample_vector_results, sample_ontology_results, sample_entity_links):
    """모든 소스를 융합하는 테스트"""
    fusion = create_default_fusion()

    result = fusion.fuse(
        vector_results=sample_vector_results,
        ontology_results=sample_ontology_results,
        entity_links=sample_entity_links,
        query="LANEIGE Lip Sleeping Mask 분석",
    )

    # 기본 검증
    assert isinstance(result, FusedResult)
    assert 0 <= result.confidence <= 1
    assert len(result.source_scores) == 3  # vector, ontology, entity
    assert len(result.documents) > 0
    assert result.explanation != ""

    # 소스별 점수 검증
    source_names = [s.source_name for s in result.source_scores]
    assert "vector" in source_names
    assert "ontology" in source_names
    assert "entity" in source_names

    # 기여도 검증
    total_contribution = sum(s.contribution for s in result.source_scores)
    assert abs(total_contribution - result.confidence) < 0.01  # 부동소수점 오차 허용

    print("\n=== All Sources Fusion Result ===")
    print(f"Final Confidence: {result.confidence:.3f}")
    print(f"Explanation: {result.explanation}")
    for source in result.source_scores:
        print(
            f"  {source.source_name}: raw={source.raw_score:.3f}, "
            f"norm={source.normalized_score:.3f}, contrib={source.contribution:.3f}"
        )


def test_fuse_two_sources(sample_vector_results, sample_ontology_results):
    """2개 소스만 융합 (Entity 없음)"""
    fusion = create_default_fusion()

    result = fusion.fuse(
        vector_results=sample_vector_results,
        ontology_results=sample_ontology_results,
        entity_links=None,
    )

    assert len(result.source_scores) == 2  # vector, ontology만
    assert "entity" not in [s.source_name for s in result.source_scores]

    # 가중치 재조정 확인
    adjusted_weights = result.metadata["adjusted_weights"]
    assert "entity" not in adjusted_weights
    assert abs(sum(adjusted_weights.values()) - 1.0) < 0.01

    print("\n=== Two Sources Fusion Result ===")
    print(f"Final Confidence: {result.confidence:.3f}")
    print(f"Adjusted Weights: {adjusted_weights}")


def test_fuse_single_source(sample_vector_results):
    """1개 소스만 융합"""
    fusion = create_default_fusion()

    result = fusion.fuse(
        vector_results=sample_vector_results, ontology_results=None, entity_links=None
    )

    assert len(result.source_scores) == 1
    assert result.source_scores[0].source_name == "vector"
    assert result.confidence > 0

    print("\n=== Single Source Fusion Result ===")
    print(f"Final Confidence: {result.confidence:.3f}")


def test_fuse_empty_sources():
    """빈 소스로 융합 시도"""
    fusion = create_default_fusion()

    result = fusion.fuse(vector_results=None, ontology_results=None, entity_links=None)

    assert result.confidence == 0.0
    assert len(result.documents) == 0
    assert len(result.warnings) > 0
    assert "Insufficient sources" in result.warnings[0]


# =========================================================================
# 정규화 방법 테스트
# =========================================================================


def test_min_max_normalization(sample_vector_results, sample_ontology_results):
    """Min-Max 정규화 테스트"""
    fusion = ConfidenceFusion(normalization=ScoreNormalizationMethod.MIN_MAX)

    result = fusion.fuse(
        vector_results=sample_vector_results, ontology_results=sample_ontology_results
    )

    # 정규화 점수는 0~1 범위
    for source in result.source_scores:
        assert 0 <= source.normalized_score <= 1

    print("\n=== Min-Max Normalization ===")
    for source in result.source_scores:
        print(f"{source.source_name}: {source.raw_score:.3f} → {source.normalized_score:.3f}")


def test_softmax_normalization(sample_vector_results, sample_ontology_results):
    """Softmax 정규화 테스트"""
    fusion = ConfidenceFusion(normalization=ScoreNormalizationMethod.SOFTMAX)

    result = fusion.fuse(
        vector_results=sample_vector_results, ontology_results=sample_ontology_results
    )

    # Softmax는 확률분포 (합이 1.0)
    normalized_sum = sum(s.normalized_score for s in result.source_scores)
    assert abs(normalized_sum - 1.0) < 0.01

    print("\n=== Softmax Normalization ===")
    for source in result.source_scores:
        print(f"{source.source_name}: {source.normalized_score:.3f} (sum={normalized_sum:.3f})")


def test_no_normalization(sample_vector_results, sample_ontology_results):
    """정규화 없음 테스트"""
    fusion = ConfidenceFusion(normalization=ScoreNormalizationMethod.NONE)

    result = fusion.fuse(
        vector_results=sample_vector_results, ontology_results=sample_ontology_results
    )

    # 정규화 점수 == 원본 점수
    for source in result.source_scores:
        assert source.normalized_score == source.raw_score


# =========================================================================
# 융합 전략 테스트
# =========================================================================


def test_weighted_sum_strategy(sample_vector_results, sample_ontology_results, sample_entity_links):
    """가중합 전략 테스트"""
    fusion = ConfidenceFusion(strategy=FusionStrategy.WEIGHTED_SUM)

    result = fusion.fuse(
        vector_results=sample_vector_results,
        ontology_results=sample_ontology_results,
        entity_links=sample_entity_links,
    )

    # 가중합 검증
    expected = sum(s.contribution for s in result.source_scores)
    assert abs(result.confidence - expected) < 0.01

    print("\n=== Weighted Sum Strategy ===")
    print(f"Confidence: {result.confidence:.3f}")


def test_harmonic_mean_strategy(sample_vector_results, sample_ontology_results):
    """조화평균 전략 테스트 (보수적)"""
    fusion = ConfidenceFusion(strategy=FusionStrategy.HARMONIC_MEAN)

    result = fusion.fuse(
        vector_results=sample_vector_results, ontology_results=sample_ontology_results
    )

    # 조화평균은 가중합보다 낮거나 같음 (보수적)
    weighted_fusion = ConfidenceFusion(strategy=FusionStrategy.WEIGHTED_SUM)
    weighted_result = weighted_fusion.fuse(
        vector_results=sample_vector_results, ontology_results=sample_ontology_results
    )

    assert result.confidence <= weighted_result.confidence + 0.01

    print("\n=== Harmonic Mean Strategy (Conservative) ===")
    print(f"Confidence: {result.confidence:.3f}")


def test_max_score_strategy(sample_vector_results, sample_ontology_results):
    """최대값 전략 테스트 (낙관적)"""
    fusion = ConfidenceFusion(strategy=FusionStrategy.MAX_SCORE)

    result = fusion.fuse(
        vector_results=sample_vector_results, ontology_results=sample_ontology_results
    )

    # 최대값은 모든 전략 중 가장 높음
    max_raw = max(s.raw_score for s in result.source_scores)
    assert result.confidence >= max_raw - 0.1  # 정규화 고려

    print("\n=== Max Score Strategy (Optimistic) ===")
    print(f"Confidence: {result.confidence:.3f}")


# =========================================================================
# 상충 감지 테스트
# =========================================================================


def test_conflict_detection():
    """상충되는 점수 감지 테스트"""
    # 매우 높은 벡터 점수
    vector_results = [SearchResult(content="High score document", score=0.95, metadata={})]

    # 매우 낮은 온톨로지 점수
    ontology_results = [
        InferenceResult(insight="Low confidence inference", confidence=0.20, evidence={})
    ]

    fusion = ConfidenceFusion(conflict_threshold=0.3)

    result = fusion.fuse(vector_results=vector_results, ontology_results=ontology_results)

    # 경고 발생 확인
    assert len(result.warnings) > 0
    assert "점수 불일치" in result.warnings[0]

    print("\n=== Conflict Detection ===")
    print(f"Warnings: {result.warnings}")


# =========================================================================
# 편의 함수 테스트
# =========================================================================


def test_create_default_fusion():
    """기본 Fusion 생성 테스트"""
    fusion = create_default_fusion()
    assert fusion.strategy == FusionStrategy.WEIGHTED_SUM
    assert fusion.min_sources == 1


def test_create_conservative_fusion():
    """보수적 Fusion 생성 테스트"""
    fusion = create_conservative_fusion()
    assert fusion.strategy == FusionStrategy.HARMONIC_MEAN
    assert fusion.min_sources == 2  # 더 엄격


def test_create_optimistic_fusion():
    """낙관적 Fusion 생성 테스트"""
    fusion = create_optimistic_fusion()
    assert fusion.strategy == FusionStrategy.MAX_SCORE


# =========================================================================
# 실전 시나리오 테스트
# =========================================================================


def test_high_confidence_scenario():
    """높은 신뢰도 시나리오 (정규화 없음)"""
    vector_results = [SearchResult(content="Perfect match", score=0.95, metadata={})]
    ontology_results = [
        InferenceResult(insight="High confidence inference", confidence=0.90, evidence={})
    ]
    entity_links = [
        LinkedEntity(
            entity_id="e1", entity_name="Entity1", entity_type="Brand", link_confidence=0.92
        )
    ]

    # 정규화 없이 원본 점수 사용
    fusion = ConfidenceFusion(normalization=ScoreNormalizationMethod.NONE)
    result = fusion.fuse(
        vector_results=vector_results, ontology_results=ontology_results, entity_links=entity_links
    )

    # 높은 신뢰도 예상 (모든 점수가 0.9 이상)
    assert result.confidence > 0.85
    assert "높은 신뢰도" in result.explanation

    print("\n=== High Confidence Scenario ===")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Explanation: {result.explanation}")


def test_low_confidence_scenario():
    """낮은 신뢰도 시나리오 (정규화 없음)"""
    vector_results = [SearchResult(content="Weak match", score=0.35, metadata={})]
    ontology_results = [
        InferenceResult(insight="Low confidence inference", confidence=0.30, evidence={})
    ]

    # 정규화 없이 원본 점수 사용
    fusion = ConfidenceFusion(normalization=ScoreNormalizationMethod.NONE)
    result = fusion.fuse(vector_results=vector_results, ontology_results=ontology_results)

    # 낮은 신뢰도 예상 (모든 점수가 0.35 이하)
    assert result.confidence < 0.40
    assert "낮은 신뢰도" in result.explanation or "중간 수준" in result.explanation

    print("\n=== Low Confidence Scenario ===")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Explanation: {result.explanation}")


def test_to_dict_serialization(sample_vector_results, sample_ontology_results):
    """to_dict() 직렬화 테스트"""
    fusion = create_default_fusion()
    result = fusion.fuse(
        vector_results=sample_vector_results, ontology_results=sample_ontology_results
    )

    result_dict = result.to_dict()

    # 필수 필드 확인
    assert "confidence" in result_dict
    assert "explanation" in result_dict
    assert "source_scores" in result_dict
    assert "documents" in result_dict
    assert "fusion_strategy" in result_dict

    # JSON 직렬화 가능 확인
    import json

    json_str = json.dumps(result_dict, ensure_ascii=False, indent=2)
    assert len(json_str) > 0

    print("\n=== Serialized Result ===")
    print(json_str[:500])  # 처음 500자만 출력


# =========================================================================
# 실행 예제
# =========================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Confidence Fusion 모듈 데모")
    print("=" * 80)

    # 샘플 데이터 생성
    vector_results = [
        SearchResult(
            content="LANEIGE Lip Sleeping Mask는 립 케어 베스트셀러입니다.",
            score=0.92,
            metadata={"doc": "strategic_doc"},
        )
    ]

    ontology_results = [
        InferenceResult(
            insight="LANEIGE는 Lip Care 카테고리에서 지배적 포지션 보유",
            confidence=0.88,
            evidence={"sos": 0.35, "rank": 1},
        )
    ]

    entity_links = [
        LinkedEntity(
            entity_id="brand_laneige",
            entity_name="LANEIGE",
            entity_type="Brand",
            link_confidence=0.95,
        )
    ]

    # Fusion 실행
    fusion = create_default_fusion()
    result = fusion.fuse(
        vector_results=vector_results,
        ontology_results=ontology_results,
        entity_links=entity_links,
        query="LANEIGE Lip Sleeping Mask 분석",
    )

    # 결과 출력
    print(f"\n최종 신뢰도: {result.confidence:.3f}")
    print(f"\n설명: {result.explanation}")
    print("\n소스별 점수:")
    for source in result.source_scores:
        print(f"  • {source.source_name}:")
        print(f"    - Raw: {source.raw_score:.3f}")
        print(f"    - Normalized: {source.normalized_score:.3f}")
        print(f"    - Weight: {source.weight:.2f}")
        print(f"    - Contribution: {source.contribution:.3f}")
        print(f"    - Level: {source.confidence_level}")
        print(f"    - Explanation: {source.explanation}")

    if result.warnings:
        print("\n경고:")
        for warning in result.warnings:
            print(f"  ⚠️  {warning}")

    print("\n" + "=" * 80)

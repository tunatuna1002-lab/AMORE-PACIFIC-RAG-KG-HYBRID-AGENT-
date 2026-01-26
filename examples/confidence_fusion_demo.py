"""
Confidence Fusion 실전 사용 예제

실제 RAG 시스템에서 다중 소스 통합 시나리오 시연
"""

from src.rag.confidence_fusion import (
    ConfidenceFusion,
    SearchResult,
    InferenceResult,
    LinkedEntity,
    ScoreNormalizationMethod,
    FusionStrategy,
    create_default_fusion,
    create_conservative_fusion,
    create_optimistic_fusion
)


# =========================================================================
# 시나리오 1: LANEIGE Lip Sleeping Mask 분석
# =========================================================================

def scenario_laneige_analysis():
    """LANEIGE 제품 분석 쿼리에 대한 다중 소스 융합"""

    print("=" * 80)
    print("시나리오 1: LANEIGE Lip Sleeping Mask 분석")
    print("=" * 80)

    # 1. 벡터 검색 결과 (문서 기반)
    vector_results = [
        SearchResult(
            content="LANEIGE Lip Sleeping Mask는 Amazon Lip Care 카테고리의 대표 베스트셀러입니다.",
            score=0.92,
            metadata={
                "doc_id": "strategic_doc_001",
                "doc_type": "market_analysis",
                "source": "docs/market/K-뷰티 초격차의 서막.md"
            },
            source="vector"
        ),
        SearchResult(
            content="립 케어 제품은 Lip Balm, Lip Mask, Lip Treatment로 분류됩니다.",
            score=0.78,
            metadata={
                "doc_id": "category_guide_002",
                "doc_type": "category_definition"
            },
            source="keyword"
        )
    ]

    # 2. 온톨로지 추론 결과 (규칙 기반)
    ontology_results = [
        InferenceResult(
            insight="LANEIGE는 Lip Care 카테고리에서 지배적 포지션 보유 (SoS 35%, Rank #1)",
            confidence=0.88,
            evidence={
                "rule": "market_dominance_rule",
                "sos": 0.35,
                "rank": 1,
                "category": "Lip Care",
                "threshold_sos": 0.30
            },
            rule_name="market_dominance_rule"
        ),
        InferenceResult(
            insight="LANEIGE Lip Sleeping Mask는 안정적 순위 유지 (30일 변동률 < 5%)",
            confidence=0.82,
            evidence={
                "rule": "stability_rule",
                "rank_volatility": 0.03,
                "days_tracked": 30
            },
            rule_name="stability_rule"
        )
    ]

    # 3. 엔티티 연결 결과 (Knowledge Graph)
    entity_links = [
        LinkedEntity(
            entity_id="brand_laneige",
            entity_name="LANEIGE",
            entity_type="Brand",
            link_confidence=0.95,
            context="Query explicitly mentioned 'LANEIGE'",
            metadata={
                "linked_by": "exact_match",
                "product_count": 5,
                "avg_rank": 12.4
            }
        ),
        LinkedEntity(
            entity_id="product_B074PXJGSB",
            entity_name="Lip Sleeping Mask",
            entity_type="Product",
            link_confidence=0.90,
            context="Top product of LANEIGE in Lip Care",
            metadata={
                "asin": "B074PXJGSB",
                "rank": 1,
                "category": "Lip Care"
            }
        ),
        LinkedEntity(
            entity_id="cat_lip_care",
            entity_name="Lip Care",
            entity_type="Category",
            link_confidence=0.85,
            context="Product belongs to Lip Care category"
        )
    ]

    # Fusion 실행
    fusion = create_default_fusion()
    result = fusion.fuse(
        vector_results=vector_results,
        ontology_results=ontology_results,
        entity_links=entity_links,
        query="LANEIGE Lip Sleeping Mask 시장 포지션 분석"
    )

    # 결과 출력
    print(f"\n📊 최종 신뢰도: {result.confidence:.3f}")
    print(f"\n📝 종합 설명:\n{result.explanation}")

    print(f"\n🔍 소스별 상세 점수:")
    for source in result.source_scores:
        print(f"\n  [{source.source_name.upper()}]")
        print(f"    • Raw Score: {source.raw_score:.3f}")
        print(f"    • Normalized: {source.normalized_score:.3f}")
        print(f"    • Weight: {source.weight:.2f}")
        print(f"    • Contribution: {source.contribution:.3f}")
        print(f"    • Level: {source.confidence_level.upper()}")
        print(f"    • {source.explanation}")

    if result.warnings:
        print(f"\n⚠️  경고:")
        for warning in result.warnings:
            print(f"    • {warning}")

    print(f"\n📄 통합 문서 수: {len(result.documents)}")
    print(f"💡 전략: {result.fusion_strategy}")

    return result


# =========================================================================
# 시나리오 2: 모호한 쿼리 (낮은 신뢰도)
# =========================================================================

def scenario_ambiguous_query():
    """모호한 쿼리에 대한 낮은 신뢰도 결과"""

    print("\n" + "=" * 80)
    print("시나리오 2: 모호한 쿼리 - '립 제품 시장 변화'")
    print("=" * 80)

    # 약한 벡터 유사도
    vector_results = [
        SearchResult(
            content="뷰티 시장은 지속적으로 변화하고 있습니다.",
            score=0.45,
            metadata={"doc": "general_trends"}
        )
    ]

    # 낮은 추론 신뢰도
    ontology_results = [
        InferenceResult(
            insight="일부 립 제품 카테고리에서 변동 감지",
            confidence=0.38,
            evidence={"volatility": 0.15, "confidence": "low"}
        )
    ]

    # 약한 엔티티 연결
    entity_links = [
        LinkedEntity(
            entity_id="cat_lip_care",
            entity_name="Lip Care",
            entity_type="Category",
            link_confidence=0.50,
            context="Generic category match"
        )
    ]

    fusion = create_default_fusion()
    result = fusion.fuse(
        vector_results=vector_results,
        ontology_results=ontology_results,
        entity_links=entity_links,
        query="립 제품 시장 변화"
    )

    print(f"\n📊 최종 신뢰도: {result.confidence:.3f} (낮음)")
    print(f"\n📝 설명: {result.explanation}")

    return result


# =========================================================================
# 시나리오 3: 상충되는 정보 (Conflict Detection)
# =========================================================================

def scenario_conflicting_sources():
    """소스 간 점수 불일치 감지"""

    print("\n" + "=" * 80)
    print("시나리오 3: 상충되는 정보 감지")
    print("=" * 80)

    # 매우 높은 벡터 점수
    vector_results = [
        SearchResult(
            content="CeraVe는 매우 강력한 브랜드입니다.",
            score=0.95,
            metadata={"doc": "brand_analysis"}
        )
    ]

    # 매우 낮은 온톨로지 점수
    ontology_results = [
        InferenceResult(
            insight="CeraVe는 현재 하락세를 보이고 있음",
            confidence=0.25,
            evidence={"rank_drop": -15, "sos_decline": -0.08}
        )
    ]

    fusion = ConfidenceFusion(conflict_threshold=0.3)
    result = fusion.fuse(
        vector_results=vector_results,
        ontology_results=ontology_results,
        query="CeraVe 브랜드 분석"
    )

    print(f"\n📊 최종 신뢰도: {result.confidence:.3f}")
    print(f"\n⚠️  감지된 상충:")
    for warning in result.warnings:
        print(f"    • {warning}")

    print(f"\n📝 설명: {result.explanation}")

    return result


# =========================================================================
# 시나리오 4: 전략 비교 (Conservative vs Optimistic)
# =========================================================================

def scenario_strategy_comparison():
    """다양한 융합 전략 비교"""

    print("\n" + "=" * 80)
    print("시나리오 4: 융합 전략 비교")
    print("=" * 80)

    vector_results = [
        SearchResult(content="Document A", score=0.85, metadata={})
    ]
    ontology_results = [
        InferenceResult(insight="Inference B", confidence=0.75, evidence={})
    ]
    entity_links = [
        LinkedEntity(
            entity_id="e1",
            entity_name="Entity",
            entity_type="Brand",
            link_confidence=0.80
        )
    ]

    strategies = [
        ("Default (Weighted Sum)", create_default_fusion()),
        ("Conservative (Harmonic Mean)", create_conservative_fusion()),
        ("Optimistic (Max Score)", create_optimistic_fusion())
    ]

    print("\n동일한 입력에 대해 3가지 전략 비교:\n")

    results = []
    for name, fusion in strategies:
        result = fusion.fuse(
            vector_results=vector_results,
            ontology_results=ontology_results,
            entity_links=entity_links
        )
        results.append((name, result))
        print(f"  {name:30s} → Confidence: {result.confidence:.3f}")

    print("\n💡 전략 선택 가이드:")
    print("  • Weighted Sum: 균형잡힌 기본 전략 (일반적 상황)")
    print("  • Harmonic Mean: 보수적 (모든 소스가 높아야 높음)")
    print("  • Max Score: 낙관적 (하나라도 높으면 높음)")

    return results


# =========================================================================
# 시나리오 5: 실시간 챗봇 응답 신뢰도
# =========================================================================

def scenario_chatbot_response():
    """챗봇 응답의 신뢰도 평가"""

    print("\n" + "=" * 80)
    print("시나리오 5: 챗봇 응답 신뢰도 평가")
    print("=" * 80)

    user_query = "LANEIGE와 CeraVe 중 어느 브랜드가 더 강한가요?"

    # RAG 문서 검색
    vector_results = [
        SearchResult(
            content="LANEIGE는 K-Beauty의 대표 브랜드로 립 케어에서 독보적 위치",
            score=0.87,
            metadata={"relevance": "high"}
        ),
        SearchResult(
            content="CeraVe는 스킨케어 카테고리에서 강세",
            score=0.82,
            metadata={"relevance": "high"}
        )
    ]

    # 온톨로지 비교 추론
    ontology_results = [
        InferenceResult(
            insight="LANEIGE는 Lip Care에서, CeraVe는 Skin Care에서 각각 강점 보유",
            confidence=0.85,
            evidence={
                "laneige_sos_lip_care": 0.35,
                "cerave_sos_skin_care": 0.28
            },
            rule_name="brand_comparison_rule"
        )
    ]

    # 양쪽 브랜드 엔티티 연결
    entity_links = [
        LinkedEntity(
            entity_id="brand_laneige",
            entity_name="LANEIGE",
            entity_type="Brand",
            link_confidence=0.92
        ),
        LinkedEntity(
            entity_id="brand_cerave",
            entity_name="CeraVe",
            entity_type="Brand",
            link_confidence=0.90
        )
    ]

    fusion = create_default_fusion()
    result = fusion.fuse(
        vector_results=vector_results,
        ontology_results=ontology_results,
        entity_links=entity_links,
        query=user_query
    )

    print(f"\n💬 사용자 질문: {user_query}")
    print(f"\n📊 응답 신뢰도: {result.confidence:.3f}")

    # 신뢰도 기반 응답 생성
    if result.confidence > 0.75:
        tone = "확신 있는 답변"
        response_prefix = "데이터에 따르면,"
    elif result.confidence > 0.50:
        tone = "중립적 답변"
        response_prefix = "분석 결과,"
    else:
        tone = "조심스러운 답변"
        response_prefix = "현재 데이터로는 명확히 말하기 어렵지만,"

    print(f"💡 답변 톤: {tone}")
    print(f"📝 답변 예시:\n  {response_prefix} {ontology_results[0].insight}")

    return result


# =========================================================================
# 메인 실행
# =========================================================================

if __name__ == "__main__":
    print("\n🚀 Confidence Fusion 실전 데모 시작\n")

    # 시나리오 실행
    result1 = scenario_laneige_analysis()
    result2 = scenario_ambiguous_query()
    result3 = scenario_conflicting_sources()
    result4 = scenario_strategy_comparison()
    result5 = scenario_chatbot_response()

    # 요약
    print("\n" + "=" * 80)
    print("📊 전체 시나리오 요약")
    print("=" * 80)

    scenarios = [
        ("LANEIGE 분석", result1.confidence),
        ("모호한 쿼리", result2.confidence),
        ("상충 감지", result3.confidence),
        ("챗봇 응답", result5.confidence)
    ]

    print("\n신뢰도 비교:")
    for name, confidence in scenarios:
        level = "🟢 HIGH" if confidence > 0.75 else "🟡 MEDIUM" if confidence > 0.50 else "🔴 LOW"
        print(f"  {name:20s} {confidence:.3f}  {level}")

    print("\n" + "=" * 80)
    print("✅ 데모 완료")
    print("=" * 80)

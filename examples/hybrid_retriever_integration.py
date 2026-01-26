"""
Confidence Fusion을 Hybrid Retriever에 통합하는 예제

기존 HybridRetriever에서 Confidence Fusion 모듈을 활용하여
다중 소스 신뢰도를 통합하는 방법을 보여줍니다.
"""

from typing import List, Dict, Any
from src.rag.confidence_fusion import (
    ConfidenceFusion,
    SearchResult,
    InferenceResult,
    LinkedEntity,
    FusedResult,
    create_default_fusion
)


class EnhancedHybridRetriever:
    """
    Confidence Fusion이 통합된 Hybrid Retriever

    기존 HybridRetriever의 기능에 신뢰도 융합을 추가
    """

    def __init__(self):
        """초기화"""
        self.fusion = create_default_fusion()

    def retrieve_with_confidence(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        쿼리에 대한 결과를 검색하고 신뢰도를 계산

        Args:
            query: 사용자 쿼리
            top_k: 반환할 문서 수

        Returns:
            검색 결과와 신뢰도 정보
        """
        # 1. 벡터 검색 (DocumentRetriever)
        vector_results = self._vector_search(query, top_k)

        # 2. 온톨로지 추론 (KnowledgeGraph + Reasoner)
        ontology_results = self._ontology_inference(query)

        # 3. 엔티티 연결 (EntityLinker)
        entity_links = self._entity_linking(query)

        # 4. Confidence Fusion 적용
        fused_result = self.fusion.fuse(
            vector_results=vector_results,
            ontology_results=ontology_results,
            entity_links=entity_links,
            query=query
        )

        # 5. 결과 구성
        return {
            "query": query,
            "documents": fused_result.documents[:top_k],
            "confidence": fused_result.confidence,
            "explanation": fused_result.explanation,
            "source_breakdown": {
                source.source_name: {
                    "score": source.raw_score,
                    "contribution": source.contribution,
                    "level": source.confidence_level
                }
                for source in fused_result.source_scores
            },
            "warnings": fused_result.warnings
        }

    def _vector_search(self, query: str, top_k: int) -> List[SearchResult]:
        """
        벡터 검색 수행

        실제 구현에서는 DocumentRetriever.search() 호출
        """
        # 모의 벡터 검색 결과
        if "LANEIGE" in query:
            return [
                SearchResult(
                    content="LANEIGE Lip Sleeping Mask는 립 케어 베스트셀러",
                    score=0.92,
                    metadata={"doc": "market_analysis.md", "chunk_id": 1},
                    source="vector"
                ),
                SearchResult(
                    content="LANEIGE는 K-Beauty 대표 브랜드",
                    score=0.85,
                    metadata={"doc": "brand_guide.md", "chunk_id": 3},
                    source="vector"
                )
            ]
        elif "순위" in query or "ranking" in query.lower():
            return [
                SearchResult(
                    content="Amazon BSR은 실시간으로 업데이트됩니다",
                    score=0.78,
                    metadata={"doc": "ranking_guide.md"},
                    source="vector"
                )
            ]
        else:
            return [
                SearchResult(
                    content="일반 뷰티 시장 정보",
                    score=0.50,
                    metadata={"doc": "general.md"},
                    source="vector"
                )
            ]

    def _ontology_inference(self, query: str) -> List[InferenceResult]:
        """
        온톨로지 추론 수행

        실제 구현에서는 OntologyReasoner.infer() 호출
        """
        # 모의 추론 결과
        if "LANEIGE" in query:
            return [
                InferenceResult(
                    insight="LANEIGE는 Lip Care에서 지배적 포지션 보유",
                    confidence=0.88,
                    evidence={
                        "rule": "market_dominance",
                        "sos": 0.35,
                        "rank": 1
                    },
                    rule_name="market_dominance_rule"
                ),
                InferenceResult(
                    insight="LANEIGE는 안정적인 순위 유지 중",
                    confidence=0.82,
                    evidence={
                        "rule": "stability",
                        "volatility": 0.03
                    },
                    rule_name="stability_rule"
                )
            ]
        else:
            return [
                InferenceResult(
                    insight="일반적인 시장 트렌드 관찰됨",
                    confidence=0.45,
                    evidence={"rule": "general_trend"},
                    rule_name="trend_analysis"
                )
            ]

    def _entity_linking(self, query: str) -> List[LinkedEntity]:
        """
        엔티티 연결 수행

        실제 구현에서는 EntityLinker.link() 호출
        """
        # 모의 엔티티 연결 결과
        entities = []

        if "LANEIGE" in query:
            entities.append(
                LinkedEntity(
                    entity_id="brand_laneige",
                    entity_name="LANEIGE",
                    entity_type="Brand",
                    link_confidence=0.95,
                    context="Exact brand name match",
                    metadata={"match_type": "exact"}
                )
            )

        if "Lip" in query or "립" in query:
            entities.append(
                LinkedEntity(
                    entity_id="cat_lip_care",
                    entity_name="Lip Care",
                    entity_type="Category",
                    link_confidence=0.85,
                    context="Lip-related category",
                    metadata={"match_type": "keyword"}
                )
            )

        if "Sleeping Mask" in query:
            entities.append(
                LinkedEntity(
                    entity_id="product_B074PXJGSB",
                    entity_name="Lip Sleeping Mask",
                    entity_type="Product",
                    link_confidence=0.90,
                    context="Product name match",
                    metadata={"asin": "B074PXJGSB"}
                )
            )

        return entities


# =========================================================================
# 실전 사용 예제
# =========================================================================

def example_chatbot_query():
    """챗봇 쿼리 처리 예제"""

    retriever = EnhancedHybridRetriever()

    # 사용자 쿼리
    queries = [
        "LANEIGE Lip Sleeping Mask의 시장 포지션은?",
        "립 케어 시장 트렌드는?",
        "순위가 급변한 이유는?"
    ]

    print("=" * 80)
    print("Enhanced Hybrid Retriever with Confidence Fusion")
    print("=" * 80)

    for query in queries:
        print(f"\n📝 Query: {query}")
        print("-" * 80)

        result = retriever.retrieve_with_confidence(query, top_k=3)

        # 신뢰도 기반 응답 톤 결정
        conf = result["confidence"]
        if conf > 0.75:
            tone = "🟢 HIGH"
        elif conf > 0.50:
            tone = "🟡 MEDIUM"
        else:
            tone = "🔴 LOW"

        print(f"\n신뢰도: {conf:.3f} {tone}")
        print(f"\n설명: {result['explanation']}")

        print(f"\n소스별 기여도:")
        for source_name, scores in result["source_breakdown"].items():
            print(f"  • {source_name:10s}: "
                  f"score={scores['score']:.3f}, "
                  f"contrib={scores['contribution']:.3f}, "
                  f"level={scores['level']}")

        if result["warnings"]:
            print(f"\n⚠️  경고:")
            for warning in result["warnings"]:
                print(f"  • {warning}")

        print(f"\n검색된 문서 ({len(result['documents'])}개):")
        for i, doc in enumerate(result["documents"][:3], 1):
            if doc["source"] == "vector":
                print(f"  {i}. [VECTOR] {doc['content'][:60]}...")
            elif doc["source"] == "ontology":
                print(f"  {i}. [ONTOLOGY] {doc['content'][:60]}...")
            elif doc["source"] == "entity":
                print(f"  {i}. [ENTITY] {doc['entity_name']} ({doc['entity_type']})")


def example_adaptive_strategy():
    """신뢰도에 따른 적응적 전략 예제"""

    retriever = EnhancedHybridRetriever()

    query = "LANEIGE Lip Sleeping Mask 분석"
    result = retriever.retrieve_with_confidence(query)

    print("\n" + "=" * 80)
    print("Adaptive Strategy based on Confidence")
    print("=" * 80)

    conf = result["confidence"]

    if conf > 0.75:
        print("\n✅ HIGH CONFIDENCE - 직접 답변 제공")
        print("   → 확신 있는 톤으로 답변")
        print("   → 근거 자료 간략히 언급")
        print(f"   예: '데이터에 따르면, {result['explanation']}'")

    elif conf > 0.50:
        print("\n⚠️  MEDIUM CONFIDENCE - 중립적 답변")
        print("   → 조건부 답변 제공")
        print("   → 추가 컨텍스트 제시")
        print(f"   예: '분석 결과, {result['explanation']}'")

    elif conf > 0.25:
        print("\n❌ LOW CONFIDENCE - 조심스러운 답변")
        print("   → 불확실성 명시")
        print("   → 추가 정보 요청")
        print(f"   예: '현재 데이터로는 명확하지 않지만, {result['explanation']}'")

    else:
        print("\n🚫 VERY LOW CONFIDENCE - 답변 보류")
        print("   → 정보 부족 명시")
        print("   → 다른 방법 제안")
        print("   예: '죄송합니다. 충분한 정보가 없어 답변이 어렵습니다.'")


def example_source_contribution_analysis():
    """소스별 기여도 분석 예제"""

    retriever = EnhancedHybridRetriever()

    query = "LANEIGE Lip Sleeping Mask의 경쟁력은?"
    result = retriever.retrieve_with_confidence(query)

    print("\n" + "=" * 80)
    print("Source Contribution Analysis")
    print("=" * 80)

    print(f"\n쿼리: {query}")
    print(f"최종 신뢰도: {result['confidence']:.3f}")

    # 소스별 기여도 시각화
    breakdown = result["source_breakdown"]

    print("\n📊 소스별 기여도 (contribution):")
    max_contrib = max(s["contribution"] for s in breakdown.values())

    for source_name, scores in sorted(
        breakdown.items(),
        key=lambda x: x[1]["contribution"],
        reverse=True
    ):
        contrib = scores["contribution"]
        percentage = (contrib / result["confidence"]) * 100 if result["confidence"] > 0 else 0
        bar_length = int((contrib / max_contrib) * 30) if max_contrib > 0 else 0
        bar = "█" * bar_length

        print(f"  {source_name:10s} {bar:30s} "
              f"{contrib:.3f} ({percentage:.1f}%)")

    # 주요 근거 소스 식별
    major_sources = [
        name for name, scores in breakdown.items()
        if scores["contribution"] > 0.15
    ]

    if major_sources:
        print(f"\n💡 주요 근거: {', '.join(major_sources)}")


# =========================================================================
# 메인 실행
# =========================================================================

if __name__ == "__main__":
    print("\n🚀 Hybrid Retriever with Confidence Fusion\n")

    # 예제 실행
    example_chatbot_query()
    example_adaptive_strategy()
    example_source_contribution_analysis()

    print("\n" + "=" * 80)
    print("✅ 통합 예제 완료")
    print("=" * 80)

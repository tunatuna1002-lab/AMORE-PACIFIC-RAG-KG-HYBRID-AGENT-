"""
Entity Linker Integration Example
==================================
EntityLinker를 HybridRetriever 및 OWLReasoner와 통합하는 예제

## 통합 시나리오
1. EntityLinker로 쿼리에서 엔티티 추출 및 온톨로지 링크
2. 링크된 엔티티를 HybridRetriever에 전달
3. OWLReasoner에서 형식 추론 실행
4. 통합 컨텍스트 생성
"""

import sys
from pathlib import Path
import asyncio

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.entity_linker import EntityLinker, LinkedEntity
from src.rag.hybrid_retriever import HybridRetriever
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner


async def example_basic_integration():
    """기본 통합 예제"""
    print("=" * 80)
    print("Example 1: Basic Integration")
    print("=" * 80)

    # 1. 컴포넌트 초기화
    kg = KnowledgeGraph()
    reasoner = OntologyReasoner(kg)
    retriever = HybridRetriever(kg, reasoner)
    linker = EntityLinker(knowledge_graph=kg)

    await retriever.initialize()

    # 2. 쿼리 입력
    query = "LANEIGE Lip Care SoS 분석해줘"
    print(f"\n🔍 Query: {query}\n")

    # 3. Entity Linking
    print("Step 1: Entity Linking")
    print("-" * 80)
    entities = linker.link(query)

    for ent in entities:
        print(f"  ✅ [{ent.entity_type.upper()}] {ent.text}")
        print(f"     → {ent.concept_label} (confidence: {ent.confidence:.2f})")
        print(f"     → {ent.concept_uri}")

    # 4. Hybrid Retrieval
    print("\nStep 2: Hybrid Retrieval")
    print("-" * 80)

    # EntityLinker 결과를 HybridRetriever 형식으로 변환
    entity_dict = {
        "brands": [e.concept_label for e in entities if e.entity_type == "brand"],
        "categories": [e.context.get("matched_key", e.text.lower()) for e in entities if e.entity_type == "category"],
        "indicators": [e.context.get("matched_key", e.text.lower()) for e in entities if e.entity_type == "metric"]
    }

    print(f"Entities for retriever: {entity_dict}")

    context = await retriever.retrieve(
        query=query,
        current_metrics={"summary": {"laneige_sos_by_category": {"lip_care": 0.12}}}
    )

    print(f"\nOntology facts: {len(context.ontology_facts)}")
    print(f"Inferences: {len(context.inferences)}")
    print(f"RAG chunks: {len(context.rag_chunks)}")


async def example_entity_uri_usage():
    """엔티티 URI를 사용한 고급 예제"""
    print("\n" + "=" * 80)
    print("Example 2: Using Entity URIs")
    print("=" * 80)

    linker = EntityLinker()

    query = "LANEIGE vs COSRX 경쟁력 비교"
    print(f"\n🔍 Query: {query}\n")

    entities = linker.link(query)

    # URI 기반 그룹핑
    print("Entities grouped by type:")
    print("-" * 80)

    entity_groups = {}
    for ent in entities:
        if ent.entity_type not in entity_groups:
            entity_groups[ent.entity_type] = []
        entity_groups[ent.entity_type].append(ent)

    for ent_type, ent_list in entity_groups.items():
        print(f"\n{ent_type.upper()}:")
        for ent in ent_list:
            print(f"  • {ent.concept_label}")
            print(f"    URI: {ent.concept_uri}")

    # URI를 사용한 SPARQL 쿼리 구성 (예시)
    print("\nGenerated SPARQL-like query pattern:")
    print("-" * 80)
    if "brand" in entity_groups and len(entity_groups["brand"]) >= 2:
        brand1 = entity_groups["brand"][0]
        brand2 = entity_groups["brand"][1]
        print(f"SELECT ?relation")
        print(f"WHERE {{")
        print(f"  <{brand1.concept_uri}> ?relation <{brand2.concept_uri}> .")
        print(f"}}")


async def example_confidence_filtering():
    """신뢰도 기반 필터링 예제"""
    print("\n" + "=" * 80)
    print("Example 3: Confidence-based Filtering")
    print("=" * 80)

    linker = EntityLinker()

    query = "LANEIGE Lip Care 제품 분석"
    print(f"\n🔍 Query: {query}\n")

    # 높은 신뢰도만
    high_conf = linker.link(query, min_confidence=0.9)
    print(f"High confidence entities (>= 0.9): {len(high_conf)}")
    for ent in high_conf:
        print(f"  • {ent.concept_label} ({ent.confidence:.2f})")

    # 중간 신뢰도
    mid_conf = linker.link(query, min_confidence=0.7)
    print(f"\nMedium confidence entities (>= 0.7): {len(mid_conf)}")
    for ent in mid_conf:
        print(f"  • {ent.concept_label} ({ent.confidence:.2f})")

    # 모든 엔티티
    all_ents = linker.link(query, min_confidence=0.5)
    print(f"\nAll entities (>= 0.5): {len(all_ents)}")
    for ent in all_ents:
        print(f"  • {ent.concept_label} ({ent.confidence:.2f})")


async def example_type_specific_extraction():
    """타입별 추출 예제"""
    print("\n" + "=" * 80)
    print("Example 4: Type-specific Extraction")
    print("=" * 80)

    linker = EntityLinker()

    query = "LANEIGE Peptide 성분 Lip Care SoS 15% 달성"
    print(f"\n🔍 Query: {query}\n")

    # 타입별 추출
    entity_types = ["brand", "category", "metric", "ingredient"]

    for ent_type in entity_types:
        entities = linker.link(query, entity_types=[ent_type])
        print(f"{ent_type.upper()}: {len(entities)} found")
        for ent in entities:
            print(f"  • {ent.text} → {ent.concept_label}")


async def example_multi_language():
    """다국어 엔티티 인식 예제"""
    print("\n" + "=" * 80)
    print("Example 5: Multi-language Entity Recognition")
    print("=" * 80)

    linker = EntityLinker()

    # 한/영 혼합 쿼리
    queries = [
        "라네즈 립케어 제품 분석",
        "LANEIGE Lip Care analysis",
        "COSRX 스킨케어 vs 라네즈",
        "Beauty of Joseon 조선미녀 제품"
    ]

    for query in queries:
        print(f"\n🔍 Query: {query}")
        entities = linker.link(query)
        for ent in entities:
            print(f"  [{ent.entity_type}] {ent.text} → {ent.concept_label}")


def example_serialization():
    """직렬화 예제"""
    print("\n" + "=" * 80)
    print("Example 6: Entity Serialization")
    print("=" * 80)

    linker = EntityLinker()

    query = "LANEIGE SoS 분석"
    print(f"\n🔍 Query: {query}\n")

    entities = linker.link(query)

    # JSON 직렬화
    import json

    serialized = [ent.to_dict() for ent in entities]
    json_str = json.dumps(serialized, indent=2, ensure_ascii=False)

    print("Serialized entities (JSON):")
    print("-" * 80)
    print(json_str)

    # 역직렬화
    print("\nDeserialized:")
    print("-" * 80)
    for ent_dict in serialized:
        print(f"  {ent_dict['text']} → {ent_dict['concept_label']} ({ent_dict['confidence']})")


async def main():
    """전체 예제 실행"""
    try:
        await example_basic_integration()
        await example_entity_uri_usage()
        await example_confidence_filtering()
        await example_type_specific_extraction()
        await example_multi_language()
        example_serialization()

        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

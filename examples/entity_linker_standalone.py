"""
Entity Linker Standalone Example
=================================
전체 시스템 없이 EntityLinker만 사용하는 독립 예제
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.entity_linker import EntityLinker


def example_1_basic_usage():
    """기본 사용법"""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)

    linker = EntityLinker(use_spacy=False)

    query = "LANEIGE Lip Care 경쟁력 분석해줘"
    print(f"\n🔍 Query: {query}\n")

    entities = linker.link(query)

    for ent in entities:
        print(f"[{ent.entity_type.upper()}] {ent.text}")
        print(f"  → Label: {ent.concept_label}")
        print(f"  → URI: {ent.concept_uri}")
        print(f"  → Confidence: {ent.confidence:.2f}\n")


def example_2_type_filtering():
    """엔티티 유형별 필터링"""
    print("=" * 80)
    print("Example 2: Type-based Filtering")
    print("=" * 80)

    linker = EntityLinker(use_spacy=False)

    query = "LANEIGE Peptide Lip Care SoS 15% 달성"
    print(f"\n🔍 Query: {query}\n")

    entity_types = ["brand", "category", "metric", "ingredient"]

    for ent_type in entity_types:
        entities = linker.link(query, entity_types=[ent_type])
        print(f"\n{ent_type.upper()}: {len(entities)} found")
        for ent in entities:
            print(f"  • {ent.text} → {ent.concept_label} (conf: {ent.confidence:.2f})")


def example_3_confidence_levels():
    """신뢰도 레벨별 필터링"""
    print("\n" + "=" * 80)
    print("Example 3: Confidence Levels")
    print("=" * 80)

    linker = EntityLinker(use_spacy=False)

    query = "LANEIGE Lip Care 제품 분석"
    print(f"\n🔍 Query: {query}\n")

    thresholds = [0.5, 0.7, 0.9]

    for threshold in thresholds:
        entities = linker.link(query, min_confidence=threshold)
        print(f"\nMin confidence >= {threshold}: {len(entities)} entities")
        for ent in entities:
            print(f"  [{ent.entity_type}] {ent.text} ({ent.confidence:.2f})")


def example_4_multilingual():
    """한/영 다국어 인식"""
    print("\n" + "=" * 80)
    print("Example 4: Multilingual Recognition")
    print("=" * 80)

    linker = EntityLinker(use_spacy=False)

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


def example_5_json_serialization():
    """JSON 직렬화"""
    print("\n" + "=" * 80)
    print("Example 5: JSON Serialization")
    print("=" * 80)

    linker = EntityLinker(use_spacy=False)

    query = "LANEIGE SoS 분석"
    print(f"\n🔍 Query: {query}\n")

    entities = linker.link(query)

    import json

    # 전체 직렬화
    serialized = [ent.to_dict() for ent in entities]
    json_str = json.dumps(serialized, indent=2, ensure_ascii=False)

    print("JSON Output:")
    print("-" * 80)
    print(json_str)


def example_6_uri_generation():
    """URI 생성 패턴"""
    print("\n" + "=" * 80)
    print("Example 6: Ontology URI Generation")
    print("=" * 80)

    linker = EntityLinker(use_spacy=False)

    query = "LANEIGE vs COSRX 경쟁 분석"
    print(f"\n🔍 Query: {query}\n")

    entities = linker.link(query)

    print("Generated URIs:")
    print("-" * 80)
    for ent in entities:
        print(f"\nText: '{ent.text}'")
        print(f"Type: {ent.entity_type}")
        print(f"URI:  {ent.concept_uri}")
        print(f"Label: {ent.concept_label}")


def example_7_context_information():
    """컨텍스트 정보 활용"""
    print("\n" + "=" * 80)
    print("Example 7: Context Information")
    print("=" * 80)

    linker = EntityLinker(use_spacy=False)

    query = "LANEIGE Lip Care SoS 분석"
    print(f"\n🔍 Query: {query}\n")

    entities = linker.link(query)

    for ent in entities:
        print(f"\n[{ent.entity_type}] {ent.text}")
        print(f"  Concept: {ent.concept_label}")
        print(f"  Confidence: {ent.confidence:.2f}")
        print(f"  Context:")
        for key, value in ent.context.items():
            print(f"    - {key}: {value}")


def example_8_entity_statistics():
    """엔티티 통계"""
    print("\n" + "=" * 80)
    print("Example 8: Entity Linker Statistics")
    print("=" * 80)

    linker = EntityLinker(use_spacy=False)

    queries = [
        "LANEIGE Lip Care 분석",
        "COSRX vs 라네즈 비교",
        "Peptide 성분 트렌드",
        "SoS와 HHI 지표 해석",
        "Beauty of Joseon 제품"
    ]

    for query in queries:
        entities = linker.link(query)

    print("\nLinker Statistics:")
    print("-" * 80)
    stats = linker.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


def example_9_comprehensive():
    """종합 예제"""
    print("\n" + "=" * 80)
    print("Example 9: Comprehensive Query Analysis")
    print("=" * 80)

    linker = EntityLinker(use_spacy=False)

    query = "LANEIGE vs COSRX Peptide Lip Care 제품 SoS 15% 달성 분석"
    print(f"\n🔍 Query: {query}\n")

    entities = linker.link(query)

    # 유형별 그룹핑
    entity_groups = {}
    for ent in entities:
        if ent.entity_type not in entity_groups:
            entity_groups[ent.entity_type] = []
        entity_groups[ent.entity_type].append(ent)

    print("Entities by Type:")
    print("-" * 80)
    for ent_type, ent_list in sorted(entity_groups.items()):
        print(f"\n{ent_type.upper()} ({len(ent_list)}):")
        for ent in ent_list:
            print(f"  • {ent.text} → {ent.concept_label} (conf: {ent.confidence:.2f})")

    print("\n\nOntology URIs:")
    print("-" * 80)
    for ent_type, ent_list in sorted(entity_groups.items()):
        for ent in ent_list:
            print(f"{ent.concept_label}: {ent.concept_uri}")


def main():
    """전체 예제 실행"""
    try:
        example_1_basic_usage()
        example_2_type_filtering()
        example_3_confidence_levels()
        example_4_multilingual()
        example_5_json_serialization()
        example_6_uri_generation()
        example_7_context_information()
        example_8_entity_statistics()
        example_9_comprehensive()

        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

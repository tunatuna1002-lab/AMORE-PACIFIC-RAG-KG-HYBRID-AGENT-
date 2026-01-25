"""
Entity Linker 테스트 스크립트
============================
NER 기반 엔티티 추출 및 온톨로지 링킹 검증
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag.entity_linker import EntityLinker, get_entity_linker


def test_basic_linking():
    """기본 엔티티 링킹 테스트"""
    print("=" * 80)
    print("Entity Linker - Basic Test")
    print("=" * 80)

    linker = EntityLinker(use_spacy=False)  # 규칙 기반 사용 (spaCy 없어도 동작)

    # 테스트 쿼리
    test_queries = [
        "LANEIGE Lip Care 경쟁력 분석해줘",
        "COSRX vs 라네즈 비교",
        "Peptide 성분 트렌드는?",
        "SoS와 HHI 지표 해석",
        "Beauty of Joseon 스킨케어 제품",
        "B0BSHRYY1S ASIN 제품 정보",
        "글래스스킨 트렌드 분석"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 80)

        entities = linker.link(query)

        if not entities:
            print("  ❌ No entities found")
            continue

        for ent in entities:
            print(f"  ✅ [{ent.entity_type.upper()}] {ent.text}")
            print(f"     → Concept: {ent.concept_label}")
            print(f"     → URI: {ent.concept_uri}")
            print(f"     → Confidence: {ent.confidence:.2f}")

    # 통계 출력
    print("\n" + "=" * 80)
    print("Statistics")
    print("=" * 80)
    stats = linker.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def test_entity_type_filtering():
    """엔티티 유형 필터링 테스트"""
    print("\n" + "=" * 80)
    print("Entity Type Filtering Test")
    print("=" * 80)

    linker = EntityLinker(use_spacy=False)

    query = "LANEIGE Lip Care SoS 분석 with Peptide"

    # 브랜드만
    print(f"\n🔍 Query: {query}")
    print("-" * 80)
    print("Filter: brands only")
    entities = linker.link(query, entity_types=["brand"])
    for ent in entities:
        print(f"  [{ent.entity_type}] {ent.text} → {ent.concept_label} (conf: {ent.confidence:.2f})")

    # 지표만
    print("\nFilter: metrics only")
    entities = linker.link(query, entity_types=["metric"])
    for ent in entities:
        print(f"  [{ent.entity_type}] {ent.text} → {ent.concept_label} (conf: {ent.confidence:.2f})")

    # 카테고리만
    print("\nFilter: categories only")
    entities = linker.link(query, entity_types=["category"])
    for ent in entities:
        print(f"  [{ent.entity_type}] {ent.text} → {ent.concept_label} (conf: {ent.confidence:.2f})")


def test_confidence_threshold():
    """신뢰도 임계값 테스트"""
    print("\n" + "=" * 80)
    print("Confidence Threshold Test")
    print("=" * 80)

    linker = EntityLinker(use_spacy=False)

    query = "LANEIGE Lip Care 분석"

    for threshold in [0.5, 0.7, 0.9]:
        print(f"\n🔍 Min confidence: {threshold}")
        print("-" * 80)
        entities = linker.link(query, min_confidence=threshold)
        print(f"Found {len(entities)} entities:")
        for ent in entities:
            print(f"  [{ent.entity_type}] {ent.text} (conf: {ent.confidence:.2f})")


def test_fuzzy_matching():
    """퍼지 매칭 테스트"""
    print("\n" + "=" * 80)
    print("Fuzzy Matching Test")
    print("=" * 80)

    linker = EntityLinker(use_spacy=False)

    # 오타/변형 테스트
    test_queries = [
        "Lanege 제품",      # 오타
        "라네즈 립케어",     # 한글
        "스킨 케어 제품",    # 띄어쓰기
        "peptid 성분",      # 오타
    ]

    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        entities = linker.link(query, min_confidence=0.5)
        for ent in entities:
            print(f"  [{ent.entity_type}] {ent.text} → {ent.concept_label} (conf: {ent.confidence:.2f})")


def test_to_dict():
    """딕셔너리 변환 테스트"""
    print("\n" + "=" * 80)
    print("Dictionary Serialization Test")
    print("=" * 80)

    linker = EntityLinker(use_spacy=False)

    query = "LANEIGE SoS 분석"
    entities = linker.link(query)

    import json
    for ent in entities:
        print(f"\n{ent.text}:")
        print(json.dumps(ent.to_dict(), indent=2, ensure_ascii=False))


def main():
    """전체 테스트 실행"""
    try:
        test_basic_linking()
        test_entity_type_filtering()
        test_confidence_threshold()
        test_fuzzy_matching()
        test_to_dict()

        print("\n" + "=" * 80)
        print("✅ All tests completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

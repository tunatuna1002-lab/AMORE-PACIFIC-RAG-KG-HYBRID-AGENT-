"""
OWL Reasoner 테스트 스크립트
"""
import asyncio
import logging
from src.ontology.owl_reasoner import OWLReasoner, OWLREADY2_AVAILABLE

logging.basicConfig(level=logging.INFO)


async def main():
    print("=" * 60)
    print("OWL Reasoner 테스트")
    print("=" * 60)
    print(f"owlready2 available: {OWLREADY2_AVAILABLE}\n")

    # OWLReasoner 초기화
    reasoner = OWLReasoner(owl_file="data/ontology/test_amore.owl", reasoner_type="pellet")
    await reasoner.initialize()

    if not OWLREADY2_AVAILABLE:
        print("⚠️  owlready2가 설치되지 않았습니다. 테스트를 건너뜁니다.")
        print("설치 방법: pip install owlready2")
        return

    print("\n1. 브랜드 추가")
    print("-" * 60)
    reasoner.add_brand("LANEIGE", sos=0.25, avg_rank=15.5, product_count=5)
    reasoner.add_brand("COSRX", sos=0.18, avg_rank=12.3, product_count=8)
    reasoner.add_brand("TIRTIR", sos=0.08, avg_rank=22.1, product_count=3)
    reasoner.add_brand("e.l.f.", sos=0.35, avg_rank=8.2, product_count=12)
    print("✓ 4개 브랜드 추가 완료")

    print("\n2. 제품 추가")
    print("-" * 60)
    reasoner.add_product(
        asin="B08XYZ123",
        brand="LANEIGE",
        category="lip_care",
        rank=10,
        price=22.0,
        rating=4.5
    )
    reasoner.add_product(
        asin="B09ABC456",
        brand="COSRX",
        category="skin_care",
        rank=5,
        price=18.0,
        rating=4.7
    )
    reasoner.add_product(
        asin="B0ADEF789",
        brand="e.l.f.",
        category="lip_care",
        rank=3,
        price=8.0,
        rating=4.3
    )
    print("✓ 3개 제품 추가 완료")

    print("\n3. 경쟁 관계 추가")
    print("-" * 60)
    reasoner.add_competitor_relation("LANEIGE", "COSRX")
    reasoner.add_competitor_relation("LANEIGE", "TIRTIR")
    reasoner.add_competitor_relation("e.l.f.", "LANEIGE")
    print("✓ 경쟁 관계 추가 완료 (대칭 관계)")

    print("\n4. 트렌드 추가")
    print("-" * 60)
    reasoner.add_trend("K-Beauty", ["LANEIGE", "COSRX", "TIRTIR"])
    reasoner.add_trend("Glass Skin", ["LANEIGE", "COSRX"])
    print("✓ 트렌드 키워드 추가 완료")

    print("\n5. 시장 포지션 추론 (SoS 기반)")
    print("-" * 60)
    positions = reasoner.infer_market_positions()
    for brand, position in positions.items():
        print(f"  - {brand}: {position}")

    print("\n6. 추론기 실행 (Pellet)")
    print("-" * 60)
    success = reasoner.run_reasoner()
    if success:
        print("✓ 추론기 실행 성공")
    else:
        print("✗ 추론기 실행 실패")

    print("\n7. 추론된 사실 조회")
    print("-" * 60)
    facts = reasoner.get_inferred_facts()
    print(f"총 {len(facts)}개 추론 결과:")
    for fact in facts[:10]:  # 상위 10개만 출력
        if fact["type"] == "market_position":
            print(f"  - {fact['subject']}: {fact['position']} (SoS: {fact['sos']:.2%})")
        elif fact["type"] == "competition":
            print(f"  - {fact['subject']} ↔ {fact['object']}")

    print("\n8. 브랜드 정보 조회")
    print("-" * 60)
    laneige = reasoner.get_brand_info("LANEIGE")
    if laneige:
        print(f"브랜드: {laneige['name']}")
        print(f"  SoS: {laneige['sos']:.2%}")
        print(f"  평균 순위: {laneige['avg_rank']}")
        print(f"  제품 수: {laneige['product_count']}")
        print(f"  시장 포지션: {laneige['market_position']}")
        print(f"  제품: {laneige['products']}")
        print(f"  경쟁사: {laneige['competitors']}")

    print("\n9. 경쟁사 조회 (대칭 관계)")
    print("-" * 60)
    competitors = reasoner.get_competitors("LANEIGE")
    print(f"LANEIGE 경쟁사: {competitors}")

    print("\n10. 카테고리별 브랜드 조회")
    print("-" * 60)
    lip_brands = reasoner.get_category_brands("lip_care")
    print(f"Lip Care 카테고리 브랜드: {len(lip_brands)}개")
    for brand_info in lip_brands:
        print(f"  - {brand_info['brand']}: SoS {brand_info['sos']:.2%}, 제품 {brand_info['product_count']}개")

    print("\n11. 온톨로지 통계")
    print("-" * 60)
    stats = reasoner.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n12. OWL 파일 저장")
    print("-" * 60)
    save_success = reasoner.save()
    if save_success:
        print("✓ OWL 파일 저장 완료: data/ontology/test_amore.owl")
    else:
        print("✗ OWL 파일 저장 실패")

    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

"""
실제 데이터로 Knowledge Graph 구축 및 추론 테스트
dashboard_data.json의 실제 크롤링/메트릭 데이터를 사용
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ontology.knowledge_graph import KnowledgeGraph
from ontology.reasoner import OntologyReasoner
from ontology.business_rules import register_all_rules, ALL_BUSINESS_RULES
from ontology.relations import RelationType, Relation, InsightType


def load_dashboard_data() -> dict:
    """대시보드 데이터 로드"""
    data_path = PROJECT_ROOT / "data" / "dashboard_data.json"
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_knowledge_graph_from_dashboard(data: dict) -> KnowledgeGraph:
    """
    대시보드 데이터에서 Knowledge Graph 구축

    구축되는 관계:
    1. Brand → Product (hasProduct)
    2. Product → Category (belongsToCategory)
    3. Brand ↔ Brand (competesWith)
    """
    kg = KnowledgeGraph()

    print("\n" + "=" * 60)
    print("📊 Knowledge Graph 구축 시작")
    print("=" * 60)

    # 1. 제품 정보에서 Brand-Product, Product-Category 관계 추출
    products = data.get("products", {})
    print(f"\n📦 제품 데이터: {len(products)}개")

    for asin, product in products.items():
        name = product.get("name", "")
        category = product.get("category", "unknown")
        rank = product.get("rank", 0)
        rating = product.get("rating", 0)
        price = product.get("price", "0")

        # 브랜드 추출 (이름에서)
        brand = "LANEIGE"  # 이 데이터셋에서는 LANEIGE만 있음

        # Brand → Product 관계
        rel1 = Relation(
            subject=brand,
            predicate=RelationType.HAS_PRODUCT,
            object=asin,
            properties={
                "product_name": name[:50],  # 이름 축약
                "rank": rank,
                "rating": rating,
                "price": price,
                "category": category
            },
            source="dashboard"
        )
        kg.add_relation(rel1)

        # Product → Category 관계
        rel2 = Relation(
            subject=asin,
            predicate=RelationType.BELONGS_TO_CATEGORY,
            object=category,
            properties={
                "rank": rank
            },
            source="dashboard"
        )
        kg.add_relation(rel2)

        print(f"  ✅ {brand} --hasProduct--> {asin[:10]}... ({category})")

    # 2. 경쟁사 정보에서 Brand 관계 추출
    competitors = data.get("brand", {}).get("competitors", [])
    print(f"\n🏢 경쟁사 데이터: {len(competitors)}개")

    for comp in competitors:
        brand_name = comp.get("brand", "")
        sos = comp.get("sos", 0)
        avg_rank = comp.get("avg_rank", 0)
        product_count = comp.get("product_count", 0)
        is_laneige = comp.get("is_laneige", brand_name.upper() == "LANEIGE")

        # 브랜드 메타데이터 설정
        kg.set_entity_metadata(brand_name, {
            "type": "brand",
            "sos": sos / 100,  # 퍼센트 → 비율
            "avg_rank": avg_rank,
            "product_count": product_count,
            "is_target": is_laneige
        })

        # LANEIGE와 다른 브랜드 간 경쟁 관계
        if not is_laneige and brand_name != "LANEIGE":
            # LANEIGE → Competitor
            rel = Relation(
                subject="LANEIGE",
                predicate=RelationType.COMPETES_WITH,
                object=brand_name,
                properties={
                    "competitor_sos": sos / 100,
                    "competitor_avg_rank": avg_rank,
                    "category": "beauty"  # 전체 카테고리
                },
                source="dashboard"
            )
            kg.add_relation(rel)
            print(f"  ✅ LANEIGE --competesWith--> {brand_name} (SoS: {sos}%)")

    # 3. 카테고리 정보
    categories = data.get("categories", {})
    print(f"\n📁 카테고리 데이터: {len(categories)}개")

    for cat_id, cat_data in categories.items():
        kg.set_entity_metadata(cat_id, {
            "type": "category",
            "name": cat_data.get("name"),
            "sos": cat_data.get("sos", 0) / 100,
            "best_rank": cat_data.get("best_rank"),
            "cpi": cat_data.get("cpi"),
            "product_count": cat_data.get("product_count"),
            "laneige_count": cat_data.get("laneige_count")
        })
        print(f"  ✅ Category: {cat_id} ({cat_data.get('name')})")

    # 통계 출력
    stats = kg.get_stats()
    print(f"\n📈 Knowledge Graph 통계:")
    print(f"  - 총 트리플: {stats['total_triples']}")
    print(f"  - 고유 주체: {stats['unique_subjects']}")
    print(f"  - 고유 객체: {stats['unique_objects']}")
    print(f"  - 관계 유형별:")
    for rel_type, count in stats.get('relations_by_type', {}).items():
        print(f"      {rel_type}: {count}")

    return kg


def run_inferences(kg: KnowledgeGraph, data: dict) -> list:
    """
    실제 데이터로 온톨로지 추론 실행
    """
    print("\n" + "=" * 60)
    print("🧠 온톨로지 추론 실행")
    print("=" * 60)

    # Reasoner 초기화 및 규칙 등록
    reasoner = OntologyReasoner(kg)
    register_all_rules(reasoner)

    print(f"\n📋 등록된 규칙: {len(reasoner.rules)}개")
    for rule in reasoner.list_rules():
        print(f"  - {rule['name']}: {rule['description'][:40]}...")

    # 대시보드에서 추론 컨텍스트 구성
    brand_kpis = data.get("brand", {}).get("kpis", {})

    # LANEIGE 브랜드 메타데이터
    laneige_meta = kg.get_entity_metadata("LANEIGE")

    # 시장 지표
    inference_context = {
        # 브랜드 지표
        "brand": "LANEIGE",
        "is_target": True,
        "sos": brand_kpis.get("sos", 0) / 100,  # 2.3% → 0.023
        "avg_rank": brand_kpis.get("avg_rank", 0),
        "product_count": laneige_meta.get("product_count", 0) if laneige_meta else 0,

        # 시장 지표
        "hhi": brand_kpis.get("hhi", 0),  # 0.02 (매우 분산된 시장)
        "top1_sos": 0.07,  # e.l.f.가 7%로 1위

        # 카테고리 지표 (lip_care 기준)
        "category": "lip_care",
        "cpi": data.get("categories", {}).get("lip_care", {}).get("cpi", 100),
        "best_rank": data.get("categories", {}).get("lip_care", {}).get("best_rank", 0),

        # 경쟁 지표
        "competitor_count": len(data.get("brand", {}).get("competitors", [])) - 1,  # LANEIGE 제외

        # 제품 지표 (대표 제품 기준)
        "current_rank": 3,  # Lip Glowy Balm 3위
        "rank_change_7d": 0,  # 유지
        "streak_days": 7,  # 가정
        "rating_gap": 0.1  # 평균 대비 우위 (4.7 vs 4.5 추정)
    }

    print(f"\n📊 추론 컨텍스트:")
    for key, value in inference_context.items():
        print(f"  - {key}: {value}")

    # 추론 실행
    print("\n🔍 추론 결과:")
    inferences = reasoner.infer(inference_context)

    if not inferences:
        print("  ⚠️ 추론된 인사이트가 없습니다.")

        # 개별 규칙 디버깅
        print("\n🔧 규칙별 조건 검사:")
        for rule_name, rule in reasoner.rules.items():
            all_satisfied, satisfied = rule.evaluate_conditions(inference_context)
            status = "✅" if all_satisfied else "❌"
            print(f"  {status} {rule_name}: {satisfied}")
    else:
        for i, inf in enumerate(inferences, 1):
            print(f"\n  📌 인사이트 {i}: [{inf.insight_type.value}]")
            print(f"     결론: {inf.insight}")
            if inf.recommendation:
                print(f"     권장: {inf.recommendation}")
            print(f"     신뢰도: {inf.confidence:.0%}")
            print(f"     규칙: {inf.rule_name}")

    return inferences


def test_graph_queries(kg: KnowledgeGraph):
    """
    Knowledge Graph 쿼리 테스트
    """
    print("\n" + "=" * 60)
    print("🔎 Knowledge Graph 쿼리 테스트")
    print("=" * 60)

    # 1. LANEIGE의 모든 제품 조회
    print("\n1️⃣ LANEIGE의 제품 목록:")
    products = kg.get_brand_products("LANEIGE")
    for p in products:
        print(f"   - {p.get('asin')}: {p.get('name', '')[:30]}... (순위: {p.get('rank')})")

    # 2. LANEIGE의 경쟁사 조회
    print("\n2️⃣ LANEIGE의 경쟁사:")
    competitors = kg.get_competitors("LANEIGE")
    for c in competitors:
        print(f"   - {c.get('brand')} (SoS: {c.get('competitor_sos', 0)*100:.1f}%)")

    # 3. lip_care 카테고리의 브랜드
    print("\n3️⃣ lip_care 카테고리 제품:")
    lip_products = kg.query(predicate=RelationType.BELONGS_TO_CATEGORY, object_="lip_care")
    for rel in lip_products:
        print(f"   - {rel.subject} (순위: {rel.properties.get('rank')})")

    # 4. 그래프 탐색 (LANEIGE에서 2홉)
    print("\n4️⃣ LANEIGE 중심 그래프 탐색 (depth=2):")
    traversal = kg.bfs_traverse("LANEIGE", max_depth=2)
    for depth, entities in traversal.items():
        print(f"   Depth {depth}: {entities[:5]}{'...' if len(entities) > 5 else ''}")

    # 5. 엔티티 컨텍스트
    print("\n5️⃣ LANEIGE 엔티티 컨텍스트:")
    context = kg.get_entity_context("LANEIGE", depth=1)
    print(f"   엔티티: {context.get('entity')}")
    print(f"   메타데이터: {context.get('metadata')}")
    outgoing = context.get('relations', {}).get('outgoing', {})
    for rel_type, targets in outgoing.items():
        print(f"   {rel_type}: {len(targets)}개 연결")


def generate_inference_explanations(reasoner: OntologyReasoner, inferences: list):
    """
    추론 과정 설명 생성
    """
    print("\n" + "=" * 60)
    print("📝 추론 과정 설명 (Explainability)")
    print("=" * 60)

    if not inferences:
        print("  설명할 추론 결과가 없습니다.")
        return

    for inf in inferences:
        explanation = reasoner.explain_inference(inf)
        print(f"\n{explanation}")


def main():
    """메인 실행"""
    print("=" * 60)
    print("🧪 실제 데이터 기반 Knowledge Graph 테스트")
    print(f"   실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 데이터 로드
    try:
        data = load_dashboard_data()
        print(f"\n✅ 대시보드 데이터 로드 완료")
        print(f"   - 생성일: {data.get('metadata', {}).get('generated_at')}")
        print(f"   - 총 제품: {data.get('metadata', {}).get('total_products')}")
        print(f"   - LANEIGE 제품: {data.get('metadata', {}).get('laneige_products')}")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    # 2. Knowledge Graph 구축
    kg = build_knowledge_graph_from_dashboard(data)

    # 3. 그래프 쿼리 테스트
    test_graph_queries(kg)

    # 4. 추론 실행
    reasoner = OntologyReasoner(kg)
    register_all_rules(reasoner)
    inferences = run_inferences(kg, data)

    # 5. 추론 설명 생성
    generate_inference_explanations(reasoner, inferences)

    # 6. Knowledge Graph 저장 (선택적)
    kg_path = PROJECT_ROOT / "data" / "knowledge_graph.json"
    kg.save(str(kg_path))
    print(f"\n💾 Knowledge Graph 저장: {kg_path}")

    print("\n" + "=" * 60)
    print("✅ 테스트 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()

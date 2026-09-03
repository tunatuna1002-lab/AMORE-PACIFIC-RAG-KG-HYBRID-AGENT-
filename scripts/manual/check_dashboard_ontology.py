"""
대시보드 온톨로지 인사이트 통합 테스트
기존 dashboard_data.json을 로드하여 온톨로지 인사이트 추가
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.domain.entities.relations import Relation, RelationType
from src.ontology.business_rules import register_all_rules
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner


def load_dashboard_data() -> dict:
    """대시보드 데이터 로드"""
    data_path = PROJECT_ROOT / "data" / "dashboard_data.json"
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


def generate_ontology_insights(dashboard_data: dict) -> dict:
    """온톨로지 인사이트 생성"""
    # 1. Knowledge Graph 구축
    kg = KnowledgeGraph()
    reasoner = OntologyReasoner(kg)
    register_all_rules(reasoner)

    stats = {"brand_product": 0, "product_category": 0, "competition": 0}

    # 제품 데이터에서 관계 추출
    products = dashboard_data.get("products", {})
    for asin, product in products.items():
        # Brand → Product
        rel1 = Relation(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            object=asin,
            properties={
                "product_name": product.get("name", "")[:50],
                "rank": product.get("rank"),
                "category": product.get("category"),
            },
        )
        kg.add_relation(rel1)
        stats["brand_product"] += 1

        # Product → Category
        category = product.get("category")
        if category:
            rel2 = Relation(
                subject=asin,
                predicate=RelationType.BELONGS_TO_CATEGORY,
                object=category,
                properties={"rank": product.get("rank")},
            )
            kg.add_relation(rel2)
            stats["product_category"] += 1

    # 경쟁사 관계 추출
    competitors = dashboard_data.get("brand", {}).get("competitors", [])
    for comp in competitors:
        brand_name = comp.get("brand", "")
        is_laneige = brand_name.upper() == "LANEIGE"

        # 메타데이터 설정
        kg.set_entity_metadata(
            brand_name,
            {
                "type": "brand",
                "sos": comp.get("sos", 0) / 100,
                "avg_rank": comp.get("avg_rank"),
                "product_count": comp.get("product_count"),
                "is_target": is_laneige,
            },
        )

        if not is_laneige:
            rel = Relation(
                subject="LANEIGE",
                predicate=RelationType.COMPETES_WITH,
                object=brand_name,
                properties={"competitor_sos": comp.get("sos", 0) / 100},
            )
            kg.add_relation(rel)
            stats["competition"] += 1

    # 2. 추론 컨텍스트 구성
    brand_kpis = dashboard_data.get("brand", {}).get("kpis", {})
    categories = dashboard_data.get("categories", {})

    # 최고 순위 제품 찾기
    best_rank = 999
    for _asin, product in products.items():
        rank = product.get("rank", 999)
        if rank < best_rank:
            best_rank = rank

    first_cat = next(iter(categories.values()), {}) if categories else {}

    inference_context = {
        "brand": "LANEIGE",
        "is_target": True,
        "sos": brand_kpis.get("sos", 0) / 100,
        "avg_rank": brand_kpis.get("avg_rank", 0),
        "product_count": len(products),
        "hhi": brand_kpis.get("hhi", 0),
        "top1_sos": competitors[0].get("sos", 0) / 100 if competitors else 0,
        "category": next(iter(categories.keys()), "unknown"),
        "cpi": first_cat.get("cpi", 100),
        "best_rank": first_cat.get("best_rank", best_rank),
        "competitor_count": len(
            [c for c in competitors if c.get("brand", "").upper() != "LANEIGE"]
        ),
        "current_rank": best_rank,
        "rank_change_7d": 0,
        "streak_days": 7,
        "rating_gap": 0.1,
    }

    # 3. 추론 실행
    inferences = reasoner.infer(inference_context)

    # 4. 결과 포맷팅
    def get_priority(insight_type):
        high = {"risk_alert", "competitive_threat", "rank_shock"}
        medium = {"price_quality_gap", "competitive_advantage", "growth_opportunity"}
        if insight_type in high:
            return "high"
        elif insight_type in medium:
            return "medium"
        return "low"

    def get_icon(insight_type):
        icons = {
            "market_dominance": "crown",
            "market_position": "chart-bar",
            "competitive_advantage": "star",
            "competitive_threat": "exclamation-triangle",
            "growth_momentum": "arrow-up",
            "stability": "shield-check",
            "risk_alert": "bell",
            "entry_opportunity": "door-open",
            "price_position": "tag",
            "price_quality_gap": "balance-scale",
        }
        return icons.get(insight_type, "lightbulb")

    def get_color(insight_type):
        colors = {
            "market_dominance": "#10b981",
            "market_position": "#3b82f6",
            "competitive_advantage": "#8b5cf6",
            "competitive_threat": "#ef4444",
            "growth_momentum": "#22c55e",
            "stability": "#06b6d4",
            "risk_alert": "#f59e0b",
            "entry_opportunity": "#14b8a6",
            "price_position": "#6366f1",
            "price_quality_gap": "#f97316",
        }
        return colors.get(insight_type, "#6b7280")

    formatted_inferences = []
    for inf in inferences:
        insight_type = inf.insight_type.value
        formatted_inferences.append(
            {
                "rule_name": inf.rule_name,
                "insight_type": insight_type,
                "insight": inf.insight,
                "recommendation": inf.recommendation,
                "confidence": inf.confidence,
                "priority": get_priority(insight_type),
                "icon": get_icon(insight_type),
                "color": get_color(insight_type),
            }
        )

    # 우선순위로 정렬
    formatted_inferences.sort(
        key=lambda x: 0 if x["priority"] == "high" else 1 if x["priority"] == "medium" else 2
    )

    # 요약 생성
    positive_types = {"market_dominance", "competitive_advantage", "growth_momentum", "stability"}
    warning_types = {"risk_alert", "competitive_threat", "price_quality_gap"}
    opportunity_types = {"entry_opportunity", "growth_opportunity"}

    positive = sum(1 for inf in inferences if inf.insight_type.value in positive_types)
    warnings = sum(1 for inf in inferences if inf.insight_type.value in warning_types)
    opportunities = sum(1 for inf in inferences if inf.insight_type.value in opportunity_types)

    top_inference = max(inferences, key=lambda x: x.confidence) if inferences else None

    kg_stats = kg.get_stats()

    return {
        "enabled": True,
        "total_rules": len(reasoner.rules),
        "triggered_rules": len(inferences),
        "kg_stats": {
            "total_triples": kg_stats.get("total_triples", 0),
            "brand_product": stats["brand_product"],
            "product_category": stats["product_category"],
            "competition": stats["competition"],
        },
        "inferences": formatted_inferences,
        "summary": {
            "total_insights": len(inferences),
            "positive_count": positive,
            "warning_count": warnings,
            "opportunity_count": opportunities,
            "headline": top_inference.insight if top_inference else "추론된 인사이트가 없습니다.",
            "top_rule": top_inference.rule_name if top_inference else None,
        },
        "context": inference_context,
    }


def main():
    """메인 실행"""
    print("=" * 60)
    print("🎯 대시보드 온톨로지 인사이트 통합 테스트")
    print(f"   실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 대시보드 데이터 로드
    print("\n📊 대시보드 데이터 로드 중...")
    dashboard_data = load_dashboard_data()
    print(f"   - 생성일: {dashboard_data.get('metadata', {}).get('generated_at')}")
    print(f"   - 제품 수: {len(dashboard_data.get('products', {}))}")

    # 2. 온톨로지 인사이트 생성
    print("\n🧠 온톨로지 인사이트 생성 중...")
    ontology_insights = generate_ontology_insights(dashboard_data)

    print("\n📈 Knowledge Graph 통계:")
    kg_stats = ontology_insights.get("kg_stats", {})
    print(f"   - 총 트리플: {kg_stats.get('total_triples', 0)}")
    print(f"   - 브랜드-제품: {kg_stats.get('brand_product', 0)}")
    print(f"   - 제품-카테고리: {kg_stats.get('product_category', 0)}")
    print(f"   - 경쟁 관계: {kg_stats.get('competition', 0)}")

    print("\n🔍 추론 결과:")
    print(f"   - 총 규칙: {ontology_insights.get('total_rules', 0)}개")
    print(f"   - 트리거된 규칙: {ontology_insights.get('triggered_rules', 0)}개")

    summary = ontology_insights.get("summary", {})
    print("\n📋 요약:")
    print(f"   - 긍정적 인사이트: {summary.get('positive_count', 0)}개")
    print(f"   - 경고: {summary.get('warning_count', 0)}개")
    print(f"   - 기회: {summary.get('opportunity_count', 0)}개")
    print(f"   - 대표 인사이트: {summary.get('headline', '')[:50]}...")

    # 3. 인사이트 상세 출력
    print("\n" + "=" * 60)
    print("📝 상세 인사이트")
    print("=" * 60)

    for i, inf in enumerate(ontology_insights.get("inferences", []), 1):
        priority_emoji = (
            "🔴" if inf["priority"] == "high" else "🟡" if inf["priority"] == "medium" else "🟢"
        )
        print(f"\n{i}. {priority_emoji} [{inf['insight_type']}] (신뢰도: {inf['confidence']:.0%})")
        print(f"   💡 {inf['insight']}")
        if inf.get("recommendation"):
            print(f"   📌 권장: {inf['recommendation']}")
        print(f"   🎨 색상: {inf['color']} | 아이콘: {inf['icon']}")

    # 4. 대시보드 데이터에 추가하여 저장
    dashboard_data["ontology_insights"] = ontology_insights
    dashboard_data["metadata"]["ontology_enabled"] = True

    output_path = PROJECT_ROOT / "data" / "dashboard_data_with_ontology.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 저장 완료: {output_path}")

    print("\n" + "=" * 60)
    print("✅ 테스트 완료")
    print("=" * 60)

    # JSON 구조 미리보기
    print("\n📄 ontology_insights 구조 미리보기:")
    print(
        json.dumps(
            {
                "enabled": ontology_insights["enabled"],
                "total_rules": ontology_insights["total_rules"],
                "triggered_rules": ontology_insights["triggered_rules"],
                "kg_stats": ontology_insights["kg_stats"],
                "summary": ontology_insights["summary"],
                "inferences_count": len(ontology_insights["inferences"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

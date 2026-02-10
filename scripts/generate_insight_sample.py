#!/usr/bin/env python3
"""
인사이트 샘플 생성 스크립트
HybridInsightAgent를 사용하여 실제 인사이트 샘플을 생성합니다.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 환경변수 로드
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.agents.hybrid_insight_agent import HybridInsightAgent
from src.domain.entities.relations import Relation, RelationType
from src.ontology.business_rules import register_all_rules
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner


def load_dashboard_data() -> dict:
    """대시보드 데이터 로드"""
    data_path = PROJECT_ROOT / "data" / "dashboard_data.json"
    if not data_path.exists():
        print(f"❌ 데이터 파일이 없습니다: {data_path}")
        print("기본 샘플 데이터를 생성합니다...")
        return create_sample_data()

    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


def create_sample_data() -> dict:
    """샘플 데이터 생성"""
    return {
        "brand": {
            "kpis": {"hhi": 0.12},
            "competitors": [
                {"brand": "LANEIGE", "sos": 6.2, "avg_rank": 12, "product_count": 6},
                {"brand": "COSRX", "sos": 8.5, "avg_rank": 9, "product_count": 7},
                {"brand": "TIRTIR", "sos": 5.1, "avg_rank": 18, "product_count": 4},
                {"brand": "Beauty of Joseon", "sos": 7.3, "avg_rank": 11, "product_count": 5},
            ],
        },
        "categories": {"lip_care": {"sos": 6.2, "cpi": 135}, "skin_care": {"sos": 4.4, "cpi": 120}},
        "products": {
            "B000TEST01": {
                "name": "LANEIGE Lip Sleeping Mask",
                "category": "lip_care",
                "rank": 7,
                "rating": 4.6,
                "volatility": 2.1,
            },
            "B000TEST02": {
                "name": "LANEIGE Water Bank Cream",
                "category": "skin_care",
                "rank": 18,
                "rating": 4.5,
                "volatility": 3.0,
            },
            "B000TEST03": {
                "name": "LANEIGE Lip Glowy Balm",
                "category": "lip_care",
                "rank": 15,
                "rating": 4.4,
                "volatility": 1.8,
            },
        },
    }


def build_metrics_data_from_dashboard(data: dict) -> dict:
    """대시보드 데이터에서 메트릭 데이터 구성"""
    brand_kpis = data.get("brand", {}).get("kpis", {})
    categories = data.get("categories", {})
    products = data.get("products", {})
    competitors = data.get("brand", {}).get("competitors", [])

    # 브랜드 메트릭
    brand_metrics = []
    for comp in competitors:
        brand_metrics.append(
            {
                "brand_name": comp.get("brand"),
                "share_of_shelf": comp.get("sos", 0) / 100,
                "avg_rank": comp.get("avg_rank"),
                "product_count": comp.get("product_count"),
                "is_laneige": comp.get("brand", "").upper() == "LANEIGE",
            }
        )

    # 제품 메트릭
    product_metrics = []
    for asin, product in products.items():
        product_metrics.append(
            {
                "asin": asin,
                "product_title": product.get("name", ""),
                "category_id": product.get("category"),
                "current_rank": product.get("rank"),
                "rank_change_1d": 0,
                "rank_change_7d": -2,  # 샘플: 7일 전 대비 2위 상승
                "rating": product.get("rating"),
                "rank_volatility": product.get("volatility", 0),
            }
        )

    # 마켓 메트릭
    market_metrics = []
    for cat_id, cat_data in categories.items():
        market_metrics.append(
            {
                "category_id": cat_id,
                "hhi": brand_kpis.get("hhi", 0.02),
                "cpi": cat_data.get("cpi", 100),
                "avg_rating_gap": 0.1,
            }
        )

    # 서머리
    summary = {
        "laneige_products_tracked": len(product_metrics),
        "laneige_sos_by_category": {
            cat_id: cat_data.get("sos", 0) / 100 for cat_id, cat_data in categories.items()
        },
        "alert_count": 0,
    }

    return {
        "summary": summary,
        "brand_metrics": brand_metrics,
        "product_metrics": product_metrics,
        "market_metrics": market_metrics,
        "alerts": [],
    }


def build_knowledge_graph_from_dashboard(data: dict) -> KnowledgeGraph:
    """대시보드 데이터에서 KG 구축"""
    kg = KnowledgeGraph()

    # 제품 정보
    products = data.get("products", {})
    for asin, product in products.items():
        brand = "LANEIGE"
        category = product.get("category", "unknown")

        # Brand → Product
        kg.add_relation(
            Relation(
                subject=brand,
                predicate=RelationType.HAS_PRODUCT,
                object=asin,
                properties={
                    "product_name": product.get("name", "")[:50],
                    "rank": product.get("rank"),
                    "category": category,
                },
            )
        )

        # Product → Category
        kg.add_relation(
            Relation(
                subject=asin,
                predicate=RelationType.BELONGS_TO_CATEGORY,
                object=category,
                properties={"rank": product.get("rank")},
            )
        )

    # 경쟁사 정보
    competitors = data.get("brand", {}).get("competitors", [])
    for comp in competitors:
        brand_name = comp.get("brand", "")
        is_laneige = brand_name.upper() == "LANEIGE"

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
            kg.add_relation(
                Relation(
                    subject="LANEIGE",
                    predicate=RelationType.COMPETES_WITH,
                    object=brand_name,
                    properties={"competitor_sos": comp.get("sos", 0) / 100},
                )
            )

    return kg


async def generate_insight_sample():
    """인사이트 샘플 생성"""
    print("=" * 80)
    print("📊 인사이트 샘플 생성")
    print(f"   실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-your"):
        print("\n⚠️  OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   LLM 기반 인사이트 생성은 스킵되고, 추론 결과만 표시됩니다.")
        use_llm = False
    else:
        print(f"\n✅ OPENAI_API_KEY 확인됨 (마지막 4자리: ...{api_key[-4:]})")
        use_llm = True

    # 1. 데이터 로드
    print("\n📊 데이터 로드 중...")
    data = load_dashboard_data()
    metrics_data = build_metrics_data_from_dashboard(data)
    print(f"   - 제품 메트릭: {len(metrics_data['product_metrics'])}개")
    print(f"   - 브랜드 메트릭: {len(metrics_data['brand_metrics'])}개")
    print(f"   - 카테고리 메트릭: {len(metrics_data['market_metrics'])}개")

    # 2. KG 구축
    print("\n📈 Knowledge Graph 구축 중...")
    kg = build_knowledge_graph_from_dashboard(data)
    stats = kg.get_stats()
    print(f"   - 트리플: {stats.get('total_triples', len(kg.triples))}개")
    unique_subjects = stats.get("unique_subjects", 0)
    unique_objects = stats.get("unique_objects", 0)
    print(f"   - 주체 엔티티: {unique_subjects}개, 객체 엔티티: {unique_objects}개")

    # 3. Reasoner 초기화
    print("\n🧠 Reasoner 초기화 중...")
    reasoner = OntologyReasoner(kg)
    register_all_rules(reasoner)
    print(f"   - 규칙: {len(reasoner.rules)}개")

    # 4. HybridInsightAgent 생성
    print("\n🔧 HybridInsightAgent 초기화 중...")
    model = "gpt-4o-mini" if use_llm else None

    agent = HybridInsightAgent(
        model=model, knowledge_graph=kg, reasoner=reasoner, docs_dir=str(PROJECT_ROOT)
    )
    print(f"   - 모델: {model or 'N/A (추론만)'}")

    # 5. 인사이트 생성 실행
    print("\n" + "=" * 80)
    print("🚀 하이브리드 인사이트 생성 실행")
    print("=" * 80)

    try:
        result = await agent.execute(metrics_data=metrics_data, crawl_data=None, crawl_summary=None)

        print("\n✅ 인사이트 생성 완료!")
        print(f"   - 상태: {result.get('status')}")
        print(f"   - 추론 결과: {len(result.get('inferences', []))}개")
        print(f"   - 액션 아이템: {len(result.get('action_items', []))}개")
        print(f"   - 하이라이트: {len(result.get('highlights', []))}개")

        # 일일 인사이트 출력
        print("\n" + "=" * 80)
        print("📝 일일 인사이트")
        print("=" * 80)
        daily_insight = result.get("daily_insight", "")
        if daily_insight:
            print(daily_insight)
        else:
            print("(LLM 기반 인사이트가 생성되지 않았습니다. API 키를 확인해주세요.)")

        # 추론 결과 출력
        print("\n" + "=" * 80)
        print("🔍 온톨로지 추론 결과")
        print("=" * 80)
        inferences = result.get("inferences", [])
        if inferences:
            for i, inf in enumerate(inferences, 1):
                print(f"\n{i}. [{inf.get('insight_type', 'UNKNOWN')}]")
                print(f"   결론: {inf.get('insight', 'N/A')}")
                if inf.get("recommendation"):
                    print(f"   권장: {inf.get('recommendation')}")
                print(f"   신뢰도: {inf.get('confidence', 0):.0%}")
        else:
            print("(추론 결과가 없습니다.)")

        # 액션 아이템 출력
        print("\n" + "=" * 80)
        print("📋 액션 아이템")
        print("=" * 80)
        action_items = result.get("action_items", [])
        if action_items:
            for i, action in enumerate(action_items, 1):
                priority = action.get("priority", "low").upper()
                print(f"{i}. [{priority}] {action.get('action', 'N/A')}")
                if action.get("source"):
                    print(f"   소스: {action.get('source')}")
                if action.get("type"):
                    print(f"   유형: {action.get('type')}")
        else:
            print("(액션 아이템이 없습니다.)")

        # 하이브리드 통계
        print("\n" + "=" * 80)
        print("📊 하이브리드 시스템 통계")
        print("=" * 80)
        hybrid_stats = result.get("hybrid_stats", {})
        print(f"   - KG 업데이트: {hybrid_stats.get('kg_update', {})}")
        print(f"   - 추론 결과: {hybrid_stats.get('inferences_count', 0)}개")
        print(f"   - RAG 청크: {hybrid_stats.get('rag_chunks_count', 0)}개")
        print(f"   - 온톨로지 사실: {hybrid_stats.get('ontology_facts_count', 0)}개")

        # 결과 저장
        output_path = PROJECT_ROOT / "data" / "insight_sample.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 결과 저장: {output_path}")

        # 마크다운 샘플 생성
        markdown_path = PROJECT_ROOT / "data" / "insight_sample.md"
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write("# 인사이트 샘플\n\n")
            f.write(f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## 일일 인사이트\n\n")
            f.write(daily_insight or "(인사이트 없음)\n\n")
            f.write("\n## 추론 결과\n\n")
            for i, inf in enumerate(inferences, 1):
                f.write(f"### {i}. {inf.get('insight_type', 'UNKNOWN')}\n\n")
                f.write(f"**결론:** {inf.get('insight', 'N/A')}\n\n")
                if inf.get("recommendation"):
                    f.write(f"**권장:** {inf.get('recommendation')}\n\n")
                f.write(f"**신뢰도:** {inf.get('confidence', 0):.0%}\n\n")
            f.write("\n## 액션 아이템\n\n")
            for i, action in enumerate(action_items, 1):
                priority = action.get("priority", "low").upper()
                f.write(f"{i}. **[{priority}]** {action.get('action', 'N/A')}\n")
                if action.get("source"):
                    f.write(f"   - 소스: {action.get('source')}\n")
        print(f"📄 마크다운 샘플 저장: {markdown_path}")

        return True

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(generate_insight_sample())
    print("\n" + "=" * 80)
    if success:
        print("✅ 인사이트 샘플 생성 완료")
    else:
        print("⚠️  인사이트 샘플 생성 중 오류 발생")
    print("=" * 80)

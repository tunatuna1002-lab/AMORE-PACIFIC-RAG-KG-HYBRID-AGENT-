"""
LLM API 연동 테스트
HybridInsightAgent를 사용하여 실제 LLM 인사이트 생성 테스트
"""

import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
import os

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.ontology.business_rules import register_all_rules
from src.domain.entities.relations import Relation, RelationType

from src.rag.hybrid_retriever import HybridRetriever, HybridContext
from src.rag.context_builder import ContextBuilder

from src.agents.hybrid_insight_agent import HybridInsightAgent


def load_dashboard_data() -> dict:
    """대시보드 데이터 로드"""
    data_path = PROJECT_ROOT / "data" / "dashboard_data.json"
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_metrics_data_from_dashboard(data: dict) -> dict:
    """대시보드 데이터에서 메트릭 데이터 구성"""
    brand_kpis = data.get("brand", {}).get("kpis", {})
    categories = data.get("categories", {})
    products = data.get("products", {})
    competitors = data.get("brand", {}).get("competitors", [])

    # 브랜드 메트릭
    brand_metrics = []
    for comp in competitors:
        brand_metrics.append({
            "brand_name": comp.get("brand"),
            "share_of_shelf": comp.get("sos", 0) / 100,
            "avg_rank": comp.get("avg_rank"),
            "product_count": comp.get("product_count"),
            "is_laneige": comp.get("brand", "").upper() == "LANEIGE"
        })

    # 제품 메트릭
    product_metrics = []
    for asin, product in products.items():
        product_metrics.append({
            "asin": asin,
            "product_title": product.get("name", ""),
            "category_id": product.get("category"),
            "current_rank": product.get("rank"),
            "rank_change_1d": 0,  # 데이터에 없음
            "rank_change_7d": 0,
            "rating": product.get("rating"),
            "rank_volatility": product.get("volatility", 0)
        })

    # 마켓 메트릭
    market_metrics = []
    for cat_id, cat_data in categories.items():
        market_metrics.append({
            "category_id": cat_id,
            "hhi": brand_kpis.get("hhi", 0.02),
            "cpi": cat_data.get("cpi", 100),
            "avg_rating_gap": 0.1
        })

    # 서머리
    summary = {
        "laneige_products_tracked": len(product_metrics),
        "laneige_sos_by_category": {
            cat_id: cat_data.get("sos", 0) / 100
            for cat_id, cat_data in categories.items()
        },
        "alert_count": 0
    }

    return {
        "summary": summary,
        "brand_metrics": brand_metrics,
        "product_metrics": product_metrics,
        "market_metrics": market_metrics,
        "alerts": []
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
        kg.add_relation(Relation(
            subject=brand,
            predicate=RelationType.HAS_PRODUCT,
            object=asin,
            properties={
                "product_name": product.get("name", "")[:50],
                "rank": product.get("rank"),
                "category": category
            }
        ))

        # Product → Category
        kg.add_relation(Relation(
            subject=asin,
            predicate=RelationType.BELONGS_TO_CATEGORY,
            object=category,
            properties={"rank": product.get("rank")}
        ))

    # 경쟁사 정보
    competitors = data.get("brand", {}).get("competitors", [])
    for comp in competitors:
        brand_name = comp.get("brand", "")
        is_laneige = brand_name.upper() == "LANEIGE"

        kg.set_entity_metadata(brand_name, {
            "type": "brand",
            "sos": comp.get("sos", 0) / 100,
            "avg_rank": comp.get("avg_rank"),
            "product_count": comp.get("product_count"),
            "is_target": is_laneige
        })

        if not is_laneige:
            kg.add_relation(Relation(
                subject="LANEIGE",
                predicate=RelationType.COMPETES_WITH,
                object=brand_name,
                properties={"competitor_sos": comp.get("sos", 0) / 100}
            ))

    return kg


async def test_hybrid_insight_agent_with_llm():
    """HybridInsightAgent LLM 연동 테스트"""
    print("=" * 60)
    print("🤖 LLM API 연동 테스트")
    print(f"   실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-your"):
        print("\n⚠️  OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 유효한 API 키를 설정해주세요.")
        print("\n   테스트를 폴백 모드로 진행합니다...")
        use_fallback = True
    else:
        print(f"\n✅ OPENAI_API_KEY 확인됨 (마지막 4자리: ...{api_key[-4:]})")
        use_fallback = False

    # 1. 데이터 로드
    print("\n📊 데이터 로드 중...")
    data = load_dashboard_data()
    metrics_data = build_metrics_data_from_dashboard(data)
    print(f"   - 제품 메트릭: {len(metrics_data['product_metrics'])}개")
    print(f"   - 브랜드 메트릭: {len(metrics_data['brand_metrics'])}개")

    # 2. KG 구축
    print("\n📈 Knowledge Graph 구축 중...")
    kg = build_knowledge_graph_from_dashboard(data)
    stats = kg.get_stats()
    print(f"   - 트리플: {stats['total_triples']}개")

    # 3. Reasoner 초기화
    print("\n🧠 Reasoner 초기화 중...")
    reasoner = OntologyReasoner(kg)
    register_all_rules(reasoner)
    print(f"   - 규칙: {len(reasoner.rules)}개")

    # 4. HybridInsightAgent 생성
    print("\n🔧 HybridInsightAgent 초기화 중...")
    model = "gpt-4o-mini" if not use_fallback else "gpt-4o-mini"

    agent = HybridInsightAgent(
        model=model,
        knowledge_graph=kg,
        reasoner=reasoner,
        docs_dir=str(PROJECT_ROOT)
    )
    print(f"   - 모델: {model}")

    # 5. 인사이트 생성 실행
    print("\n" + "=" * 60)
    print("🚀 하이브리드 인사이트 생성 실행")
    print("=" * 60)

    try:
        result = await agent.execute(
            metrics_data=metrics_data,
            crawl_data=None,
            crawl_summary=None
        )

        print("\n✅ 인사이트 생성 완료!")
        print(f"   - 상태: {result.get('status')}")
        print(f"   - 추론 결과: {len(result.get('inferences', []))}개")
        print(f"   - 액션 아이템: {len(result.get('action_items', []))}개")
        print(f"   - 하이라이트: {len(result.get('highlights', []))}개")

        # 일일 인사이트 출력
        print("\n" + "=" * 60)
        print("📝 일일 인사이트")
        print("=" * 60)
        daily_insight = result.get("daily_insight", "")
        print(daily_insight)

        # 추론 결과 출력
        print("\n" + "=" * 60)
        print("🔍 온톨로지 추론 결과")
        print("=" * 60)
        for i, inf in enumerate(result.get("inferences", []), 1):
            print(f"\n{i}. [{inf.get('insight_type')}]")
            print(f"   결론: {inf.get('insight')}")
            if inf.get('recommendation'):
                print(f"   권장: {inf.get('recommendation')}")
            print(f"   신뢰도: {inf.get('confidence', 0):.0%}")

        # 액션 아이템 출력
        print("\n" + "=" * 60)
        print("📋 액션 아이템")
        print("=" * 60)
        for i, action in enumerate(result.get("action_items", []), 1):
            priority = action.get("priority", "low").upper()
            print(f"{i}. [{priority}] {action.get('action')}")
            print(f"   소스: {action.get('source')} / 유형: {action.get('type')}")

        # 하이브리드 통계
        print("\n" + "=" * 60)
        print("📊 하이브리드 시스템 통계")
        print("=" * 60)
        hybrid_stats = result.get("hybrid_stats", {})
        print(f"   - KG 업데이트: {hybrid_stats.get('kg_update', {})}")
        print(f"   - 추론 결과: {hybrid_stats.get('inferences_count', 0)}개")
        print(f"   - RAG 청크: {hybrid_stats.get('rag_chunks_count', 0)}개")
        print(f"   - 온톨로지 사실: {hybrid_stats.get('ontology_facts_count', 0)}개")

        # 결과 저장
        output_path = PROJECT_ROOT / "data" / "llm_insight_result.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 결과 저장: {output_path}")

        return True

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_context_builder_only():
    """컨텍스트 빌더만 테스트 (LLM 없이)"""
    print("\n" + "=" * 60)
    print("📋 컨텍스트 빌더 테스트 (LLM 없이)")
    print("=" * 60)

    # 데이터 준비
    data = load_dashboard_data()
    kg = build_knowledge_graph_from_dashboard(data)
    reasoner = OntologyReasoner(kg)
    register_all_rules(reasoner)

    # 추론 실행
    inference_context = {
        "brand": "LANEIGE",
        "is_target": True,
        "sos": 0.023,
        "hhi": 0.02,
        "category": "lip_care",
        "cpi": 212.0,
        "current_rank": 3
    }

    inferences = reasoner.infer(inference_context)
    print(f"\n추론 결과: {len(inferences)}개")

    # HybridContext 구성
    hybrid_context = HybridContext(
        query="LANEIGE 시장 분석",
        inferences=inferences,
        rag_chunks=[],
        ontology_facts=[]
    )

    # 컨텍스트 빌드
    builder = ContextBuilder()
    system_prompt = builder.build_system_prompt(hybrid_context)
    user_prompt = builder.build_user_prompt("시장 분석해줘", hybrid_context)

    print("\n" + "-" * 40)
    print("📝 시스템 프롬프트 (처음 500자)")
    print("-" * 40)
    print(system_prompt[:500] + "...")

    print("\n" + "-" * 40)
    print("📝 사용자 프롬프트 (처음 500자)")
    print("-" * 40)
    print(user_prompt[:500] + "...")

    return True


async def main():
    """메인 실행"""
    # 1. 컨텍스트 빌더 테스트 (LLM 없이)
    await test_context_builder_only()

    # 2. LLM 연동 테스트
    print("\n\n")
    success = await test_hybrid_insight_agent_with_llm()

    print("\n" + "=" * 60)
    if success:
        print("✅ LLM 연동 테스트 완료")
    else:
        print("⚠️  LLM 연동 테스트 실패 (API 키 확인 필요)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

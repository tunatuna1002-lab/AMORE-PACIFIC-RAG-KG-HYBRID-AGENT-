"""
Data dependencies
=================
Thin wrappers over ``DashboardDataService`` (the single dashboard-data access
path, F6-1) plus the v1 data-context renderer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.application.services.dashboard_data_service import (
    DashboardDataService,
    get_dashboard_data_service,
    resolve_data_dir,
)
from src.domain.brand import is_target_brand
from src.rag.router import QueryType

DOCS_PATH = "./"


def get_data_service() -> DashboardDataService:
    return get_dashboard_data_service()


def load_dashboard_data() -> dict[str, Any]:
    """대시보드 JSON 캐시 로드 (staleness 메타 포함, 없으면 {})"""
    return get_data_service().load_dashboard_json()


async def get_dashboard_data() -> dict[str, Any]:
    """JSON → SQLite 폴백 → 빈 스켈레톤"""
    return await get_data_service().get_dashboard_data()


def dashboard_data_path() -> Path:
    """현재 해석된 dashboard_data.json 경로"""
    return get_data_service().dashboard_json_path


def build_data_context(data: dict, query_type: QueryType, entities: dict) -> str:
    """
    데이터 컨텍스트 구성 (Ontology 기반)

    질문 유형과 추출된 엔티티에 따라 필요한 데이터만 선택
    """
    if not data:
        return "현재 데이터가 없습니다."

    context_parts = []

    # 메타데이터 (항상 포함)
    metadata = data.get("metadata", {})
    context_parts.append(f"""[데이터 현황]
- 기준일: {metadata.get("data_date", "N/A")}
- 총 제품 수: {metadata.get("total_products", 0)}개
- LANEIGE 제품 수: {metadata.get("laneige_products", 0)}개""")

    # 질문 유형별 데이터 선택
    brand_kpis = data.get("brand", {}).get("kpis", {})

    # 시장/브랜드 지표 (DEFINITION, INTERPRETATION, ANALYSIS)
    if query_type in [
        QueryType.DEFINITION,
        QueryType.INTERPRETATION,
        QueryType.ANALYSIS,
        QueryType.COMBINATION,
    ]:
        if brand_kpis:
            context_parts.append(f"""
[LANEIGE 브랜드 KPI] (Ontology: BrandMetrics)
- SoS (Share of Shelf): {brand_kpis.get("sos", 0)}% {brand_kpis.get("sos_delta", "")}
- Top 10 제품 수: {brand_kpis.get("top10_count", 0)}개
- 평균 순위: {brand_kpis.get("avg_rank", 0)}위
- HHI (시장 집중도): {brand_kpis.get("hhi", 0)}""")

    # 경쟁사 정보 (ANALYSIS, DATA_QUERY에서 경쟁사 언급 시)
    competitors = data.get("brand", {}).get("competitors", [])
    brands_mentioned = entities.get("brands", [])

    if query_type == QueryType.ANALYSIS or any(
        b for b in brands_mentioned if not is_target_brand(b)
    ):
        if competitors:
            top_comps = competitors[:5]
            comp_lines = [
                f"  - {c['brand']}: SoS {c['sos']}%, 평균 순위 {c['avg_rank']}위, 제품 {c['product_count']}개"
                for c in top_comps
            ]
            context_parts.append("[경쟁사 현황]\n" + "\n".join(comp_lines))

    # 제품 정보 (DATA_QUERY, 특정 제품 언급 시)
    products = data.get("products", {})
    products_mentioned = entities.get("products", [])

    if query_type == QueryType.DATA_QUERY or products_mentioned:
        if products:
            prod_lines = []
            for _asin, p in list(products.items())[:5]:
                prod_lines.append(f"""  - {p["name"][:40]}
    순위: #{p["rank"]} ({p["rank_delta"]}), 평점: {p["rating"]}, 변동성: {p.get("volatility_status", "N/A")}""")
            context_parts.append(
                "[LANEIGE 제품 현황] (Ontology: ProductMetrics)\n" + "\n".join(prod_lines)
            )

    # 카테고리 정보
    categories = data.get("categories", {})
    categories_mentioned = entities.get("categories", [])

    if categories_mentioned or query_type in [QueryType.ANALYSIS, QueryType.INTERPRETATION]:
        if categories:
            cat_lines = []
            for _cat_id, cat in categories.items():
                cat_lines.append(
                    f"  - {cat['name']}: SoS {cat['sos']}%, 최고 순위 #{cat['best_rank']}, CPI {cat.get('cpi', 100)}"
                )
            context_parts.append(
                "[카테고리 현황] (Ontology: MarketMetrics)\n" + "\n".join(cat_lines)
            )

    # 액션 아이템 (전략 질문)
    if query_type == QueryType.ANALYSIS:
        action_items = data.get("home", {}).get("action_items", [])
        if action_items:
            action_lines = [
                f"  - [{a['priority']}] {a['product_name']}: {a['signal']} → {a['action_tag']}"
                for a in action_items[:4]
            ]
            context_parts.append("[현재 액션 아이템]\n" + "\n".join(action_lines))

    return "\n\n".join(context_parts)


__all__ = [
    "DOCS_PATH",
    "build_data_context",
    "dashboard_data_path",
    "get_dashboard_data",
    "get_data_service",
    "load_dashboard_data",
    "resolve_data_dir",
]

"""
Dashboard Data Exporter
Google Sheets 데이터를 Dashboard용 JSON으로 변환

Usage:
    exporter = DashboardExporter()
    await exporter.initialize()
    await exporter.export_dashboard_data("./data/dashboard_data.json")

Ontology-RAG Hybrid Integration:
    - Knowledge Graph 기반 엔티티 관계
    - Ontology Reasoner 기반 추론 인사이트
    - 대시보드에 추론 결과 시각화
"""

import os
import json
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Any, List, Optional
from collections import defaultdict

# 한국 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))

from src.tools.sheets_writer import SheetsWriter

# Ontology components (optional import)
try:
    from ontology.knowledge_graph import KnowledgeGraph
    from ontology.reasoner import OntologyReasoner
    from ontology.business_rules import register_all_rules
    from ontology.relations import Relation, RelationType
    ONTOLOGY_AVAILABLE = True
except ImportError:
    ONTOLOGY_AVAILABLE = False


class DashboardExporter:
    """Dashboard용 데이터 내보내기 클래스"""

    # LANEIGE 브랜드 식별
    LANEIGE_BRAND = "laneige"

    # 카테고리 매핑
    CATEGORY_MAP = {
        "lip_care": "Lip Care",
        "face_moisturizer": "Face Moisturizer",
        "facial_skin_care": "Skin Care",
        "toner": "Toner",
        "beauty_skincare": "Beauty & Skin Care"
    }

    def __init__(
        self,
        spreadsheet_id: Optional[str] = None,
        enable_ontology: bool = True
    ):
        """
        Args:
            spreadsheet_id: Google Sheets ID
            enable_ontology: 온톨로지 추론 활성화 여부
        """
        self.sheets = SheetsWriter(spreadsheet_id=spreadsheet_id)
        self._initialized = False

        # Ontology components
        self.enable_ontology = enable_ontology and ONTOLOGY_AVAILABLE
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._reasoner: Optional[OntologyReasoner] = None

        if self.enable_ontology:
            self._init_ontology()

    def _init_ontology(self):
        """온톨로지 컴포넌트 초기화"""
        if not ONTOLOGY_AVAILABLE:
            return

        self._knowledge_graph = KnowledgeGraph()
        self._reasoner = OntologyReasoner(self._knowledge_graph)
        register_all_rules(self._reasoner)

    async def initialize(self) -> bool:
        """초기화"""
        result = await self.sheets.initialize()
        self._initialized = result
        return result

    async def export_dashboard_data(self, output_path: str = "./data/dashboard_data.json") -> Dict[str, Any]:
        """
        Dashboard용 JSON 데이터 생성

        Args:
            output_path: 출력 파일 경로

        Returns:
            생성된 데이터 딕셔너리
        """
        if not self._initialized:
            await self.initialize()

        # 전체 데이터 로드
        raw_data = await self._load_raw_data()

        if not raw_data:
            return {"error": "No data found"}

        # Dashboard 데이터 구조 생성
        dashboard_data = {
            "metadata": {
                "generated_at": datetime.now(KST).isoformat(),
                "data_date": self._get_latest_date(raw_data),
                "total_products": len(raw_data),
                "laneige_products": len([r for r in raw_data if self._is_laneige(r)]),
                "ontology_enabled": self.enable_ontology
            },
            "home": self._generate_home_data(raw_data),
            "brand": self._generate_brand_data(raw_data),
            "categories": self._generate_category_data(raw_data),
            "products": self._generate_product_data(raw_data),
            "charts": self._generate_chart_data(raw_data)
        }

        # 온톨로지 추론 결과 추가
        if self.enable_ontology:
            ontology_data = self._generate_ontology_insights(raw_data, dashboard_data)
            dashboard_data["ontology_insights"] = ontology_data

        # JSON 파일 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dashboard_data, f, ensure_ascii=False, indent=2)

        return dashboard_data

    async def _load_raw_data(self) -> List[Dict[str, Any]]:
        """RawData 시트에서 데이터 로드"""
        return await self.sheets.get_rank_history(days=30)

    def _get_latest_date(self, data: List[Dict]) -> str:
        """최신 데이터 날짜 반환"""
        dates = [r.get("snapshot_date", "") for r in data if r.get("snapshot_date")]
        return max(dates) if dates else datetime.now(KST).strftime("%Y-%m-%d")

    def _is_laneige(self, record: Dict) -> bool:
        """LANEIGE 제품 여부 확인"""
        brand = record.get("brand", "").lower()
        return self.LANEIGE_BRAND in brand

    def _generate_home_data(self, raw_data: List[Dict]) -> Dict[str, Any]:
        """Home 페이지 데이터 생성"""
        latest_date = self._get_latest_date(raw_data)
        latest_data = [r for r in raw_data if r.get("snapshot_date") == latest_date]

        # LANEIGE 제품 필터
        laneige_products = [r for r in latest_data if self._is_laneige(r)]

        # Top 10 내 LANEIGE 개수
        top10_count = len([p for p in laneige_products if self._safe_int(p.get("rank", 999)) <= 10])

        # 상태 판단
        if top10_count >= 3:
            status = "Stable Up"
            status_type = "success"
        elif top10_count >= 1:
            status = "Moderate"
            status_type = "info"
        else:
            status = "Needs Attention"
            status_type = "danger"

        # 액션 아이템 생성
        action_items = self._generate_action_items(laneige_products, raw_data)

        return {
            "insight_message": self._generate_daily_insight(laneige_products, latest_date),
            "status": {
                "exposure": status,
                "exposure_type": status_type,
                "position": f"Top {self._get_best_rank(laneige_products)}",
                "warning_count": len([a for a in action_items if a.get("priority") == "P1"])
            },
            "action_items": action_items[:4]  # 상위 4개만
        }

    def _generate_daily_insight(self, laneige_products: List[Dict], date_str: str) -> str:
        """일일 인사이트 메시지 생성"""
        if not laneige_products:
            return "오늘 LANEIGE 제품 데이터가 없습니다."

        best_product = min(laneige_products, key=lambda x: self._safe_int(x.get("rank", 999)))
        best_rank = self._safe_int(best_product.get("rank", 0))
        best_name = best_product.get("product_name", "Unknown")[:30]

        return (
            f"Amazon US 내 Laneige 브랜드는 "
            f"{'안정적인 상위권을 유지 중' if best_rank <= 10 else '중위권에 위치'}"
            f"입니다. <strong>{best_name}</strong>가 {best_rank}위를 기록했습니다."
        )

    def _generate_action_items(self, laneige_products: List[Dict], all_data: List[Dict]) -> List[Dict]:
        """액션 아이템 생성"""
        items = []

        for product in laneige_products:
            rank = self._safe_int(product.get("rank", 999))
            rating = self._safe_float(product.get("rating", 0))
            name = product.get("product_name", "Unknown")[:25]

            # 순위 상승/하락 체크
            rank_change = self._calculate_rank_change(product, all_data)

            priority = "P2"
            signal = ""
            action_tag = "MONITOR"

            if rank_change < -2:  # 순위 상승 (숫자가 작아짐)
                signal = f"순위 {rank+abs(rank_change)}→{rank} (상향)"
                priority = "P1"
                action_tag = "MONITOR"
            elif rank_change > 2:  # 순위 하락
                signal = f"순위 {rank-rank_change}→{rank} (하락)"
                priority = "P1"
                action_tag = "CHECK"
            elif rating < 4.0:  # 평점 낮음
                signal = f"평점 {rating:.2f} (하락)"
                priority = "P1"
                action_tag = "CHECK"
            elif rank <= 5:  # 상위권 유지
                signal = f"Top {rank} 유지 중"
                action_tag = "MONITOR"
            else:
                signal = f"순위 {rank}위"
                action_tag = "DEEP DIVE"

            items.append({
                "priority": priority,
                "product_name": name,
                "brand_variant": product.get("badge", ""),
                "signal": signal,
                "action_tag": action_tag,
                "asin": product.get("asin", "")
            })

        # 우선순위로 정렬
        items.sort(key=lambda x: (0 if x["priority"] == "P1" else 1, self._safe_int(x.get("rank", 999))))
        return items

    def _calculate_rank_change(self, product: Dict, all_data: List[Dict]) -> int:
        """순위 변동 계산 (어제 대비)"""
        asin = product.get("asin")
        current_date = product.get("snapshot_date")

        if not asin or not current_date:
            return 0

        # 이전 데이터 찾기
        prev_records = [
            r for r in all_data
            if r.get("asin") == asin
            and r.get("snapshot_date", "") < current_date
        ]

        if not prev_records:
            return 0

        # 가장 최근 이전 기록
        prev_record = max(prev_records, key=lambda x: x.get("snapshot_date", ""))
        prev_rank = self._safe_int(prev_record.get("rank", 0))
        curr_rank = self._safe_int(product.get("rank", 0))

        return curr_rank - prev_rank

    def _get_best_rank(self, products: List[Dict]) -> int:
        """최고 순위 반환"""
        ranks = [self._safe_int(p.get("rank", 999)) for p in products]
        return min(ranks) if ranks else 999

    def _generate_brand_data(self, raw_data: List[Dict]) -> Dict[str, Any]:
        """Brand View (L1) 데이터"""
        latest_date = self._get_latest_date(raw_data)
        latest_data = [r for r in raw_data if r.get("snapshot_date") == latest_date]

        # 브랜드별 집계
        brand_stats = defaultdict(lambda: {"products": [], "ranks": []})

        for record in latest_data:
            brand = record.get("brand", "Unknown")
            brand_stats[brand]["products"].append(record)
            brand_stats[brand]["ranks"].append(self._safe_int(record.get("rank", 999)))

        # LANEIGE 통계
        laneige_stats = brand_stats.get("LANEIGE", brand_stats.get("Laneige", {"products": [], "ranks": [999]}))

        laneige_ranks = laneige_stats["ranks"] if laneige_stats["ranks"] else [999]
        avg_rank = sum(laneige_ranks) / len(laneige_ranks) if laneige_ranks else 0

        # SoS 계산 (Share of Shelf = LANEIGE 제품 수 / 전체 Top 100)
        total_products = len(latest_data)
        laneige_count = len(laneige_stats["products"])
        sos = (laneige_count / total_products * 100) if total_products > 0 else 0

        # Top 10 내 개수
        top10_count = len([r for r in laneige_ranks if r <= 10])

        # HHI 계산 (간략화)
        hhi = self._calculate_hhi(brand_stats)

        return {
            "kpis": {
                "sos": round(sos, 1),
                "sos_delta": "+2.1%p",  # TODO: 실제 계산
                "top10_count": top10_count,
                "avg_rank": round(avg_rank, 1),
                "hhi": round(hhi, 2)
            },
            "competitors": self._generate_competitor_data(brand_stats)
        }

    def _calculate_hhi(self, brand_stats: Dict) -> float:
        """HHI (Herfindahl-Hirschman Index) 계산"""
        total = sum(len(stats["products"]) for stats in brand_stats.values())
        if total == 0:
            return 0

        hhi = sum(
            (len(stats["products"]) / total * 100) ** 2
            for stats in brand_stats.values()
        ) / 10000

        return hhi

    def _generate_competitor_data(self, brand_stats: Dict) -> List[Dict]:
        """경쟁사 데이터 생성"""
        competitors = []

        for brand, stats in brand_stats.items():
            if not stats["ranks"]:
                continue

            avg_rank = sum(stats["ranks"]) / len(stats["ranks"])
            product_count = len(stats["products"])

            competitors.append({
                "brand": brand,
                "sos": round(product_count / sum(len(s["products"]) for s in brand_stats.values()) * 100, 1),
                "avg_rank": round(avg_rank, 1),
                "product_count": product_count
            })

        # SoS 기준 정렬
        competitors.sort(key=lambda x: x["sos"], reverse=True)
        return competitors[:10]

    def _generate_category_data(self, raw_data: List[Dict]) -> Dict[str, Any]:
        """Category View (L2) 데이터"""
        latest_date = self._get_latest_date(raw_data)
        latest_data = [r for r in raw_data if r.get("snapshot_date") == latest_date]

        categories = {}

        for cat_id, cat_name in self.CATEGORY_MAP.items():
            cat_data = [r for r in latest_data if r.get("category_id") == cat_id]

            if not cat_data:
                continue

            laneige_in_cat = [r for r in cat_data if self._is_laneige(r)]

            # 카테고리 SoS
            sos = (len(laneige_in_cat) / len(cat_data) * 100) if cat_data else 0

            # LANEIGE 최고 순위
            laneige_ranks = [self._safe_int(r.get("rank", 999)) for r in laneige_in_cat]
            best_rank = min(laneige_ranks) if laneige_ranks else 999

            # 카테고리 평균 가격
            prices = [self._safe_float(r.get("price", 0)) for r in cat_data if r.get("price")]
            avg_price = sum(prices) / len(prices) if prices else 0

            # LANEIGE 평균 가격
            laneige_prices = [self._safe_float(r.get("price", 0)) for r in laneige_in_cat if r.get("price")]
            laneige_avg_price = sum(laneige_prices) / len(laneige_prices) if laneige_prices else 0

            # CPI (Competitive Price Index) = LANEIGE 가격 / 카테고리 평균 * 100
            cpi = (laneige_avg_price / avg_price * 100) if avg_price > 0 else 100

            categories[cat_id] = {
                "name": cat_name,
                "sos": round(sos, 1),
                "best_rank": best_rank,
                "cpi": round(cpi, 0),
                "product_count": len(cat_data),
                "laneige_count": len(laneige_in_cat)
            }

        return categories

    def _generate_product_data(self, raw_data: List[Dict]) -> Dict[str, Any]:
        """Product View (L3) 데이터"""
        latest_date = self._get_latest_date(raw_data)
        latest_data = [r for r in raw_data if r.get("snapshot_date") == latest_date]

        laneige_products = [r for r in latest_data if self._is_laneige(r)]

        products = {}

        for product in laneige_products:
            asin = product.get("asin", "")
            if not asin:
                continue

            rank = self._safe_int(product.get("rank", 999))
            rating = self._safe_float(product.get("rating", 0))

            # 순위 변동
            rank_change = self._calculate_rank_change(product, raw_data)

            # 변동성 계산
            volatility = self._calculate_volatility(asin, raw_data)

            products[asin] = {
                "asin": asin,
                "name": product.get("product_name", "Unknown"),
                "category": product.get("category_id", ""),
                "rank": rank,
                "rank_delta": self._format_rank_delta(rank_change),
                "rating": rating,
                "rating_delta": "상위 10%" if rating >= 4.3 else "양호" if rating >= 4.0 else "주의",
                "volatility": volatility,
                "volatility_status": "안정적" if volatility < 10 else "변동성 높음" if volatility > 20 else "보통",
                "price": product.get("price", ""),
                "reviews_count": product.get("reviews_count", "")
            }

        return products

    def _calculate_volatility(self, asin: str, all_data: List[Dict]) -> int:
        """순위 변동성 계산 (7일 표준편차 기반)"""
        product_history = [
            r for r in all_data
            if r.get("asin") == asin
        ][-7:]  # 최근 7일

        if len(product_history) < 2:
            return 0

        ranks = [self._safe_int(r.get("rank", 0)) for r in product_history]

        if not ranks:
            return 0

        # 간단한 변동성: max - min
        volatility = max(ranks) - min(ranks)
        return volatility

    def _format_rank_delta(self, change: int) -> str:
        """순위 변동 포맷팅"""
        if change < 0:
            return f"▲ {abs(change)}위 상승"
        elif change > 0:
            return f"▼ {change}위 하락"
        else:
            return "유지"

    def _generate_chart_data(self, raw_data: List[Dict]) -> Dict[str, Any]:
        """차트용 데이터 생성"""
        # 날짜별 데이터 그룹화
        date_groups = defaultdict(list)
        for record in raw_data:
            date_str = record.get("snapshot_date", "")
            if date_str:
                date_groups[date_str].append(record)

        # 최근 7일 정렬
        sorted_dates = sorted(date_groups.keys())[-7:]

        # SoS 추이 데이터
        sos_trend = []
        for date_str in sorted_dates:
            day_data = date_groups[date_str]
            laneige_count = len([r for r in day_data if self._is_laneige(r)])
            total = len(day_data)
            sos = (laneige_count / total * 100) if total > 0 else 0

            sos_trend.append({
                "date": date_str[-5:],  # MM-DD 형식
                "sos": round(sos, 1)
            })

        # 제품별 SoS (최신 데이터)
        latest_date = sorted_dates[-1] if sorted_dates else ""
        latest_data = date_groups.get(latest_date, [])

        # ASIN 기준 중복 제거 (순위가 가장 좋은 것 유지)
        laneige_raw = [r for r in latest_data if self._is_laneige(r)]
        seen_asins = {}
        for r in laneige_raw:
            asin = r.get("asin", "")
            if not asin:
                continue
            rank = self._safe_int(r.get("rank", 999))
            if asin not in seen_asins or rank < self._safe_int(seen_asins[asin].get("rank", 999)):
                seen_asins[asin] = r
        laneige_products = list(seen_asins.values())

        # 제품별 SoS 도넛 차트 데이터
        product_sos = []
        total_laneige = len(laneige_products)
        for product in sorted(laneige_products, key=lambda x: self._safe_int(x.get("rank", 999)))[:5]:
            rank = self._safe_int(product.get("rank", 999))
            # 순위 기반 가중치 (낮을수록 높은 비중)
            weight = max(1, 100 - rank)
            product_sos.append({
                "name": product.get("product_name", "Unknown")[:20],
                "weight": weight,
                "rank": rank
            })

        # 가중치 정규화
        total_weight = sum(p["weight"] for p in product_sos)
        for p in product_sos:
            p["sos"] = round(p["weight"] / total_weight * 100, 1) if total_weight > 0 else 0

        # Others 추가
        if len(laneige_products) > 5:
            others_sos = 100 - sum(p["sos"] for p in product_sos)
            product_sos.append({"name": "Others", "sos": round(others_sos, 1), "rank": 999})

        # 경쟁사 버블차트 데이터 (Brand Matrix)
        brand_stats = defaultdict(lambda: {"products": [], "ranks": []})
        for record in latest_data:
            brand = record.get("brand", "Unknown")
            brand_stats[brand]["products"].append(record)
            brand_stats[brand]["ranks"].append(self._safe_int(record.get("rank", 999)))

        brand_matrix = []
        for brand, stats in brand_stats.items():
            if not stats["ranks"] or len(stats["products"]) < 2:
                continue

            avg_rank = sum(stats["ranks"]) / len(stats["ranks"])
            product_count = len(stats["products"])
            sos = (product_count / len(latest_data) * 100) if latest_data else 0

            # 버블 크기 = 제품 수 기반
            bubble_size = min(20, max(8, product_count * 2))

            brand_matrix.append({
                "brand": brand,
                "sos": round(sos, 1),
                "avg_rank": round(avg_rank, 1),
                "bubble_size": bubble_size,
                "product_count": product_count,
                "is_laneige": self.LANEIGE_BRAND in brand.lower()
            })

        # SoS 기준 상위 10개만
        brand_matrix.sort(key=lambda x: x["sos"], reverse=True)
        brand_matrix = brand_matrix[:10]

        # 카테고리별 KPI 데이터
        category_kpis = {}
        for cat_id, cat_name in self.CATEGORY_MAP.items():
            cat_data = [r for r in latest_data if r.get("category_id") == cat_id]
            if not cat_data:
                continue

            laneige_in_cat = [r for r in cat_data if self._is_laneige(r)]
            sos = (len(laneige_in_cat) / len(cat_data) * 100) if cat_data else 0
            laneige_ranks = [self._safe_int(r.get("rank", 999)) for r in laneige_in_cat]
            best_rank = min(laneige_ranks) if laneige_ranks else 999

            # CPI 계산
            prices = [self._safe_float(r.get("price", 0)) for r in cat_data if r.get("price")]
            avg_price = sum(prices) / len(prices) if prices else 0
            laneige_prices = [self._safe_float(r.get("price", 0)) for r in laneige_in_cat if r.get("price")]
            laneige_avg_price = sum(laneige_prices) / len(laneige_prices) if laneige_prices else 0
            cpi = (laneige_avg_price / avg_price * 100) if avg_price > 0 else 100

            # 신규 경쟁자 수 (임시 계산 - 실제로는 시계열 비교 필요)
            unique_brands = len(set(r.get("brand", "") for r in cat_data))

            category_kpis[cat_id] = {
                "name": cat_name,
                "sos": round(sos, 1),
                "best_rank": best_rank,
                "cpi": round(cpi, 0),
                "new_competitors": unique_brands
            }

        # CPI 추이 차트 (최근 7일)
        cpi_trend = []
        for date_str in sorted_dates:
            day_data = date_groups[date_str]

            # Lip Care 카테고리 기준
            cat_data = [r for r in day_data if r.get("category_id") == "lip_care"]
            laneige_in_cat = [r for r in cat_data if self._is_laneige(r)]

            prices = [self._safe_float(r.get("price", 0)) for r in cat_data if r.get("price")]
            avg_price = sum(prices) / len(prices) if prices else 1
            laneige_prices = [self._safe_float(r.get("price", 0)) for r in laneige_in_cat if r.get("price")]
            laneige_avg_price = sum(laneige_prices) / len(laneige_prices) if laneige_prices else avg_price
            cpi = (laneige_avg_price / avg_price * 100) if avg_price > 0 else 100

            cpi_trend.append({
                "date": date_str[-5:],
                "cpi": round(cpi, 0)
            })

        # 제품 순위 추이 (L3 Product View용)
        product_rank_trend = {}
        for product in laneige_products[:3]:  # 상위 3개 제품
            asin = product.get("asin", "")
            name = product.get("product_name", "Unknown")[:25]

            ranks = []
            for date_str in sorted_dates:
                day_data = date_groups[date_str]
                product_day = next((r for r in day_data if r.get("asin") == asin), None)
                rank = self._safe_int(product_day.get("rank", 0)) if product_day else None
                ranks.append(rank)

            product_rank_trend[asin] = {
                "name": name,
                "ranks": ranks
            }

        # 제품 매트릭스 데이터 (순위 x 변동성)
        product_matrix = []
        for product in laneige_products:
            asin = product.get("asin", "")
            rank = self._safe_int(product.get("rank", 999))
            volatility = self._calculate_volatility(asin, raw_data)
            rating = self._safe_float(product.get("rating", 0))

            # 4분면 결정
            if rank <= 10 and volatility <= 15:
                quadrant = "king"  # 안정적 상위권
                color = "rgba(16,185,129,0.7)"  # 녹색
            elif rank <= 10 and volatility > 15:
                quadrant = "rising"  # 급부상
                color = "rgba(32,128,128,0.7)"  # 틸
            elif rank > 10 and volatility <= 15:
                quadrant = "lagging"  # 정체
                color = "rgba(245,158,11,0.7)"  # 주황
            else:
                quadrant = "risk"  # 위험
                color = "rgba(231,76,60,0.7)"  # 빨강

            product_matrix.append({
                "asin": asin,
                "name": product.get("product_name", "Unknown")[:20],
                "rank": rank,
                "volatility": volatility,
                "rating": rating,
                "quadrant": quadrant,
                "color": color,
                "bubble_size": max(8, min(18, 20 - rank))
            })

        return {
            "sos_trend": {
                "labels": [d["date"] for d in sos_trend],
                "data": [d["sos"] for d in sos_trend]
            },
            "product_sos": {
                "labels": [p["name"] for p in product_sos],
                "data": [p["sos"] for p in product_sos]
            },
            "brand_matrix": brand_matrix,
            "category_kpis": category_kpis,
            "cpi_trend": {
                "labels": [d["date"] for d in cpi_trend],
                "data": [d["cpi"] for d in cpi_trend]
            },
            "product_rank_trend": {
                "labels": [d[-5:] for d in sorted_dates],
                "products": product_rank_trend
            },
            "product_matrix": product_matrix
        }

    def _safe_int(self, value: Any) -> int:
        """안전한 int 변환"""
        try:
            if isinstance(value, str):
                value = value.replace(",", "").replace("#", "")
            return int(float(value))
        except (ValueError, TypeError):
            return 0

    def _safe_float(self, value: Any) -> float:
        """안전한 float 변환"""
        try:
            if isinstance(value, str):
                value = value.replace(",", "").replace("$", "")
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    # =========================================================================
    # Ontology-RAG Hybrid Integration
    # =========================================================================

    def _generate_ontology_insights(
        self,
        raw_data: List[Dict],
        dashboard_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        온톨로지 기반 추론 인사이트 생성

        Args:
            raw_data: 원본 데이터
            dashboard_data: 기본 대시보드 데이터

        Returns:
            온톨로지 인사이트 데이터
        """
        if not self.enable_ontology or not self._reasoner:
            return {"enabled": False}

        # 1. Knowledge Graph 구축
        kg_stats = self._build_knowledge_graph(raw_data, dashboard_data)

        # 2. 추론 컨텍스트 구성
        inference_context = self._build_inference_context(dashboard_data)

        # 3. 추론 실행
        inferences = self._reasoner.infer(inference_context)

        # 4. 결과 포맷팅
        formatted_inferences = []
        for inf in inferences:
            formatted_inferences.append({
                "rule_name": inf.rule_name,
                "insight_type": inf.insight_type.value,
                "insight": inf.insight,
                "recommendation": inf.recommendation,
                "confidence": inf.confidence,
                "priority": self._get_inference_priority(inf),
                "icon": self._get_insight_icon(inf.insight_type.value),
                "color": self._get_insight_color(inf.insight_type.value)
            })

        # 우선순위로 정렬
        formatted_inferences.sort(
            key=lambda x: (0 if x["priority"] == "high" else 1 if x["priority"] == "medium" else 2)
        )

        return {
            "enabled": True,
            "total_rules": len(self._reasoner.rules) if self._reasoner else 0,
            "triggered_rules": len(inferences),
            "kg_stats": kg_stats,
            "inferences": formatted_inferences,
            "summary": self._generate_inference_summary(inferences),
            "context": inference_context
        }

    def _build_knowledge_graph(
        self,
        raw_data: List[Dict],
        dashboard_data: Dict[str, Any]
    ) -> Dict[str, int]:
        """Knowledge Graph 구축"""
        if not self._knowledge_graph:
            return {"triples": 0}

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
                    "category": product.get("category")
                }
            )
            self._knowledge_graph.add_relation(rel1)
            stats["brand_product"] += 1

            # Product → Category
            category = product.get("category")
            if category:
                rel2 = Relation(
                    subject=asin,
                    predicate=RelationType.BELONGS_TO_CATEGORY,
                    object=category,
                    properties={"rank": product.get("rank")}
                )
                self._knowledge_graph.add_relation(rel2)
                stats["product_category"] += 1

        # 경쟁사 관계 추출
        competitors = dashboard_data.get("brand", {}).get("competitors", [])
        for comp in competitors:
            brand_name = comp.get("brand", "")
            is_laneige = brand_name.upper() == "LANEIGE"

            # 메타데이터 설정
            self._knowledge_graph.set_entity_metadata(brand_name, {
                "type": "brand",
                "sos": comp.get("sos", 0) / 100,
                "avg_rank": comp.get("avg_rank"),
                "product_count": comp.get("product_count"),
                "is_target": is_laneige
            })

            if not is_laneige:
                rel = Relation(
                    subject="LANEIGE",
                    predicate=RelationType.COMPETES_WITH,
                    object=brand_name,
                    properties={"competitor_sos": comp.get("sos", 0) / 100}
                )
                self._knowledge_graph.add_relation(rel)
                stats["competition"] += 1

        kg_full_stats = self._knowledge_graph.get_stats()
        return {
            "total_triples": kg_full_stats.get("total_triples", 0),
            "brand_product": stats["brand_product"],
            "product_category": stats["product_category"],
            "competition": stats["competition"]
        }

    def _build_inference_context(self, dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """추론 컨텍스트 구성"""
        brand_kpis = dashboard_data.get("brand", {}).get("kpis", {})
        categories = dashboard_data.get("categories", {})
        products = dashboard_data.get("products", {})
        competitors = dashboard_data.get("brand", {}).get("competitors", [])

        # 최고 순위 제품 찾기
        best_rank = 999
        for asin, product in products.items():
            rank = product.get("rank", 999)
            if rank < best_rank:
                best_rank = rank

        # 첫 번째 카테고리 기준
        first_cat = next(iter(categories.values()), {}) if categories else {}

        return {
            # 브랜드 지표
            "brand": "LANEIGE",
            "is_target": True,
            "sos": brand_kpis.get("sos", 0) / 100,
            "avg_rank": brand_kpis.get("avg_rank", 0),
            "product_count": len(products),

            # 시장 지표
            "hhi": brand_kpis.get("hhi", 0),
            "top1_sos": competitors[0].get("sos", 0) / 100 if competitors else 0,

            # 카테고리 지표
            "category": next(iter(categories.keys()), "unknown"),
            "cpi": first_cat.get("cpi", 100),
            "best_rank": first_cat.get("best_rank", best_rank),

            # 경쟁 지표
            "competitor_count": len([c for c in competitors if c.get("brand", "").upper() != "LANEIGE"]),

            # 제품 지표
            "current_rank": best_rank,
            "rank_change_7d": 0,  # 추후 계산 가능
            "streak_days": 7,  # 추후 계산 가능
            "rating_gap": 0.1  # 추후 계산 가능
        }

    def _get_inference_priority(self, inference) -> str:
        """추론 우선순위 결정"""
        high_priority_types = {"risk_alert", "competitive_threat", "rank_shock"}
        medium_priority_types = {"price_quality_gap", "competitive_advantage", "growth_opportunity"}

        insight_type = inference.insight_type.value
        if insight_type in high_priority_types:
            return "high"
        elif insight_type in medium_priority_types:
            return "medium"
        return "low"

    def _get_insight_icon(self, insight_type: str) -> str:
        """인사이트 유형별 아이콘"""
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
            "price_quality_gap": "balance-scale"
        }
        return icons.get(insight_type, "lightbulb")

    def _get_insight_color(self, insight_type: str) -> str:
        """인사이트 유형별 색상"""
        colors = {
            "market_dominance": "#10b981",  # green
            "market_position": "#3b82f6",  # blue
            "competitive_advantage": "#8b5cf6",  # purple
            "competitive_threat": "#ef4444",  # red
            "growth_momentum": "#22c55e",  # green
            "stability": "#06b6d4",  # cyan
            "risk_alert": "#f59e0b",  # amber
            "entry_opportunity": "#14b8a6",  # teal
            "price_position": "#6366f1",  # indigo
            "price_quality_gap": "#f97316"  # orange
        }
        return colors.get(insight_type, "#6b7280")  # gray

    def _generate_inference_summary(self, inferences: List) -> Dict[str, Any]:
        """추론 결과 요약"""
        if not inferences:
            return {
                "total_insights": 0,
                "positive_count": 0,
                "warning_count": 0,
                "opportunity_count": 0,
                "headline": "추론된 인사이트가 없습니다."
            }

        positive_types = {"market_dominance", "competitive_advantage", "growth_momentum", "stability"}
        warning_types = {"risk_alert", "competitive_threat", "price_quality_gap"}
        opportunity_types = {"entry_opportunity", "growth_opportunity"}

        positive = sum(1 for inf in inferences if inf.insight_type.value in positive_types)
        warnings = sum(1 for inf in inferences if inf.insight_type.value in warning_types)
        opportunities = sum(1 for inf in inferences if inf.insight_type.value in opportunity_types)

        # 대표 인사이트 선택 (가장 높은 신뢰도)
        top_inference = max(inferences, key=lambda x: x.confidence)
        headline = top_inference.insight

        return {
            "total_insights": len(inferences),
            "positive_count": positive,
            "warning_count": warnings,
            "opportunity_count": opportunities,
            "headline": headline,
            "top_rule": top_inference.rule_name
        }

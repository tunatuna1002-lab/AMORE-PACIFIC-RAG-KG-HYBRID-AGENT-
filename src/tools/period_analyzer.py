"""
기간별 데이터 분석기
====================
Amazon 크롤링 데이터를 기간별로 분석하여 인사이트 추출
"""

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PeriodAnalysis:
    """기간별 분석 결과"""

    start_date: str
    end_date: str
    total_days: int

    # LANEIGE 성과
    laneige_metrics: dict[str, Any] = field(default_factory=dict)

    # 시장 지표
    market_metrics: dict[str, Any] = field(default_factory=dict)

    # 브랜드별 성과
    brand_performance: list[dict[str, Any]] = field(default_factory=list)

    # 카테고리별 분석
    category_analysis: dict[str, Any] = field(default_factory=dict)

    # 제품별 분석 (급등/급락)
    top_movers: dict[str, list[dict]] = field(default_factory=dict)

    # 경쟁 구도 변화
    competitive_shifts: dict[str, Any] = field(default_factory=dict)

    # 일별 추이 데이터 (차트용)
    daily_trends: list[dict[str, Any]] = field(default_factory=list)


class PeriodAnalyzer:
    """기간별 데이터 분석기"""

    def __init__(self, sqlite_storage=None):
        """
        Args:
            sqlite_storage: SQLiteStorage 인스턴스 (None이면 자동 생성)
        """
        self.storage = sqlite_storage
        self._categories = [
            "beauty_personal_care",
            "skin_care",
            "lip_care",
            "lip_makeup",
            "face_powder",
        ]

    async def analyze(self, start_date: str, end_date: str) -> PeriodAnalysis:
        """
        지정 기간의 데이터를 분석

        Args:
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)

        Returns:
            PeriodAnalysis 객체
        """
        # 1. SQLite에서 기간 데이터 조회
        raw_data = await self._load_period_data(start_date, end_date)

        if not raw_data:
            logger.warning(f"No data found for period {start_date} ~ {end_date}")
            return PeriodAnalysis(start_date=start_date, end_date=end_date, total_days=0)

        # 2. 일별로 그룹핑
        daily_data = self._group_by_date(raw_data)

        # 3. 각 분석 수행
        analysis = PeriodAnalysis(
            start_date=start_date, end_date=end_date, total_days=len(daily_data)
        )

        # LANEIGE 분석
        analysis.laneige_metrics = self._analyze_laneige(daily_data)

        # 시장 지표 (HHI 등)
        analysis.market_metrics = self._analyze_market(daily_data)

        # 브랜드별 성과
        analysis.brand_performance = self._analyze_brands(daily_data)

        # 카테고리별 분석
        analysis.category_analysis = self._analyze_categories(daily_data)

        # 급등/급락 제품
        analysis.top_movers = self._identify_top_movers(daily_data)

        # 경쟁 구도 변화
        analysis.competitive_shifts = self._analyze_competitive_shifts(daily_data)

        # 일별 추이 (차트용)
        analysis.daily_trends = self._build_daily_trends(daily_data)

        logger.info(
            f"Period analysis completed: {start_date} ~ {end_date} ({len(daily_data)} days)"
        )

        return analysis

    async def _load_period_data(self, start_date: str, end_date: str) -> list[dict]:
        """SQLite에서 기간 데이터 로드"""
        if self.storage is None:
            from src.tools.sqlite_storage import SQLiteStorage

            self.storage = SQLiteStorage()
            await self.storage.initialize()

        return await self.storage.get_raw_data(
            start_date=start_date,
            end_date=end_date,
            limit=100000,  # 대량 데이터 조회
        )

    # 브랜드 정규화 매핑 (잘린 브랜드명 → 전체 브랜드명)
    BRAND_NORMALIZATION = {
        "burt's": "Burt's Bees",
        "wet": "wet n wild",
        "tree": "Tree Hut",
        "clean": "Clean Skin Club",
        "summer": "Summer Fridays",
        "rare": "Rare Beauty",
        "la": "La Roche-Posay",
        "beauty": "Beauty of Joseon",
        "tower": "Tower 28",
        "drunk": "Drunk Elephant",
        "paula's": "Paula's Choice",
        "the": "The Ordinary",
        "glow": "Glow Recipe",
        "youth": "Youth To The People",
        "first": "First Aid Beauty",
        "charlotte": "Charlotte Tilbury",
        "too": "Too Faced",
        "urban": "Urban Decay",
        "fenty": "Fenty Beauty",
        "huda": "Huda Beauty",
        "anastasia": "Anastasia Beverly Hills",
        "physicians": "Physicians Formula",
        "covergirl": "COVERGIRL",
        "medicube": "MEDICUBE",
    }

    def _normalize_brand(self, brand: str, product_name: str = "") -> str:
        """브랜드명 정규화"""
        if not brand or brand == "Unknown":
            return brand

        brand_lower = brand.lower().strip()

        # 정규화 매핑에서 찾기
        if brand_lower in self.BRAND_NORMALIZATION:
            return self.BRAND_NORMALIZATION[brand_lower]

        # product_name 기반 보정 시도
        if product_name:
            product_lower = product_name.lower()
            for _key, full_brand in self.BRAND_NORMALIZATION.items():
                if full_brand.lower() in product_lower:
                    return full_brand

        return brand

    def _group_by_date(self, data: list[dict]) -> dict[str, list[dict]]:
        """데이터를 날짜별로 그룹핑"""
        grouped = defaultdict(list)
        for item in data:
            date_str = item.get("snapshot_date") or item.get("date", "")
            if date_str:
                # YYYY-MM-DD 형식으로 정규화
                if "T" in date_str:
                    date_str = date_str.split("T")[0]

                # 브랜드명 정규화 적용
                raw_brand = item.get("brand", "")
                product_name = item.get("product_name", "")
                normalized_brand = self._normalize_brand(raw_brand, product_name)

                # 카테고리와 순위 정보 추가 (원본 데이터 필드명 매핑)
                product = {
                    "asin": item.get("asin", ""),
                    "title": product_name,
                    "brand": normalized_brand,
                    "rank": item.get("rank", 0),
                    "category": item.get("category_id", ""),
                    "price": item.get("price"),
                    "rating": item.get("rating"),
                    "reviews_count": item.get("reviews_count"),
                }
                grouped[date_str].append(product)

        return dict(sorted(grouped.items()))

    def _analyze_laneige(self, daily_data: dict[str, list[dict]]) -> dict[str, Any]:
        """LANEIGE 브랜드 심층 분석"""
        daily_sos = []
        daily_product_counts = []
        # 핵심 수정: ASIN+카테고리 복합키로 추적 (서로 다른 카테고리 순위를 혼동하지 않음)
        all_products = defaultdict(list)  # (ASIN, category) -> [순위 리스트]

        for date_str, products in daily_data.items():
            laneige_products = [p for p in products if self._is_laneige(p)]
            total_products = len(products)

            # 일별 SoS
            sos = (len(laneige_products) / total_products * 100) if total_products > 0 else 0
            daily_sos.append({"date": date_str, "sos": sos, "count": len(laneige_products)})
            daily_product_counts.append(len(laneige_products))

            # 제품별 순위 추적 - 카테고리별로 분리
            for p in laneige_products:
                asin = p.get("asin", "")
                rank = p.get("rank", 0)
                category = p.get("category", "")
                if asin and rank and category:
                    # 복합 키 (ASIN, 카테고리)로 순위 추적
                    key = (asin, category)
                    all_products[key].append(
                        {
                            "date": date_str,
                            "rank": rank,
                            "title": p.get("title", ""),
                            "category": category,
                        }
                    )

        # 기간 평균/변화율 계산
        sos_values = [d["sos"] for d in daily_sos]
        start_sos = sos_values[0] if sos_values else 0
        end_sos = sos_values[-1] if sos_values else 0
        avg_sos = statistics.mean(sos_values) if sos_values else 0

        # 제품별 순위 변동 - 동일 카테고리 내에서만 비교
        product_changes = []

        for (asin, category), ranks in all_products.items():
            if len(ranks) >= 2:
                first_rank = ranks[0]["rank"]
                last_rank = ranks[-1]["rank"]
                change = first_rank - last_rank  # 양수 = 순위 상승

                # 카테고리 한글명 변환
                category_name = self._get_category_name(category)

                product_changes.append(
                    {
                        "asin": asin,
                        "title": ranks[-1]["title"],
                        "category": category,
                        "category_name": category_name,
                        "start_rank": first_rank,
                        "end_rank": last_rank,
                        "change": change,
                        "avg_rank": statistics.mean([r["rank"] for r in ranks]),
                    }
                )

        # 순위 변동 기준 정렬
        product_changes.sort(key=lambda x: x["change"], reverse=True)

        return {
            "daily_sos": daily_sos,
            "avg_sos": round(avg_sos, 2),
            "start_sos": round(start_sos, 2),
            "end_sos": round(end_sos, 2),
            "sos_change": round(end_sos - start_sos, 2),
            "sos_change_pct": round((end_sos - start_sos) / start_sos * 100, 1)
            if start_sos > 0
            else 0,
            "avg_product_count": round(statistics.mean(daily_product_counts), 1)
            if daily_product_counts
            else 0,
            "top_products": product_changes[:10],  # Top 10
            "rising_products": [p for p in product_changes if p["change"] > 5][:5],  # 5위 이상 상승
            "falling_products": [p for p in product_changes if p["change"] < -5][
                -5:
            ],  # 5위 이상 하락
        }

    def _analyze_market(self, daily_data: dict[str, list[dict]]) -> dict[str, Any]:
        """시장 전체 지표 분석 (HHI 등)"""
        daily_hhi = []

        for date_str, products in daily_data.items():
            # 브랜드별 점유율 계산 (Unknown/빈 브랜드 제외)
            brand_counts = defaultdict(int)
            for p in products:
                brand = p.get("brand", "")
                # Unknown 및 빈 브랜드는 HHI 계산에서 제외
                if not brand or brand.lower() == "unknown":
                    continue
                brand_counts[brand] += 1

            total = sum(brand_counts.values())
            if total > 0:
                # HHI = Σ(share^2) * 10000
                hhi = sum((count / total * 100) ** 2 for count in brand_counts.values())
                daily_hhi.append({"date": date_str, "hhi": round(hhi, 2)})

        hhi_values = [d["hhi"] for d in daily_hhi]
        avg_hhi = statistics.mean(hhi_values) if hhi_values else 0

        # HHI 해석
        if avg_hhi < 1500:
            hhi_interpretation = "경쟁적 시장 (분산)"
        elif avg_hhi < 2500:
            hhi_interpretation = "중간 집중도"
        else:
            hhi_interpretation = "고집중 시장"

        return {
            "daily_hhi": daily_hhi,
            "avg_hhi": round(avg_hhi, 2),
            "hhi_interpretation": hhi_interpretation,
            "start_hhi": daily_hhi[0]["hhi"] if daily_hhi else 0,
            "end_hhi": daily_hhi[-1]["hhi"] if daily_hhi else 0,
        }

    def _analyze_brands(self, daily_data: dict[str, list[dict]]) -> list[dict[str, Any]]:
        """브랜드별 성과 분석"""
        brand_daily = defaultdict(lambda: defaultdict(int))

        for date_str, products in daily_data.items():
            for p in products:
                brand = p.get("brand", "")
                # Unknown 및 빈 브랜드는 브랜드 분석에서 제외
                if not brand or brand.lower() == "unknown":
                    continue
                brand_daily[brand][date_str] += 1

        # 브랜드별 SoS 계산
        results = []
        dates = sorted(daily_data.keys())

        for brand, daily_counts in brand_daily.items():
            counts = [daily_counts.get(d, 0) for d in dates]

            # 일별 전체 제품 수로 SoS 계산
            daily_sos = []
            for d in dates:
                total = len(daily_data[d])
                sos = (daily_counts.get(d, 0) / total * 100) if total > 0 else 0
                daily_sos.append(sos)

            if daily_sos:
                results.append(
                    {
                        "brand": brand,
                        "avg_sos": round(statistics.mean(daily_sos), 2),
                        "start_sos": round(daily_sos[0], 2),
                        "end_sos": round(daily_sos[-1], 2),
                        "sos_change": round(daily_sos[-1] - daily_sos[0], 2),
                        "avg_count": round(statistics.mean(counts), 1),
                        "total_appearances": sum(counts),
                    }
                )

        # SoS 기준 내림차순 정렬
        results.sort(key=lambda x: x["avg_sos"], reverse=True)
        return results[:20]  # Top 20 브랜드

    def _analyze_categories(self, daily_data: dict[str, list[dict]]) -> dict[str, Any]:
        """카테고리별 분석"""
        category_data = defaultdict(lambda: {"laneige": [], "total": [], "dates": []})

        for date_str, products in daily_data.items():
            cat_counts = defaultdict(lambda: {"laneige": 0, "total": 0})

            for p in products:
                cat = p.get("category", "unknown")
                cat_counts[cat]["total"] += 1
                if self._is_laneige(p):
                    cat_counts[cat]["laneige"] += 1

            for cat, counts in cat_counts.items():
                category_data[cat]["laneige"].append(counts["laneige"])
                category_data[cat]["total"].append(counts["total"])
                category_data[cat]["dates"].append(date_str)

        result = {}
        for cat, data in category_data.items():
            if data["total"]:
                daily_sos = [
                    (laneige_count / total_count * 100) if total_count > 0 else 0
                    for laneige_count, total_count in zip(
                        data["laneige"], data["total"], strict=False
                    )
                ]
                result[cat] = {
                    "avg_sos": round(statistics.mean(daily_sos), 2) if daily_sos else 0,
                    "start_sos": round(daily_sos[0], 2) if daily_sos else 0,
                    "end_sos": round(daily_sos[-1], 2) if daily_sos else 0,
                    "avg_laneige_count": round(statistics.mean(data["laneige"]), 1),
                    "avg_total_count": round(statistics.mean(data["total"]), 1),
                }

        return result

    def _identify_top_movers(self, daily_data: dict[str, list[dict]]) -> dict[str, list[dict]]:
        """급등/급락 제품 식별 - 동일 카테고리 내에서만 비교"""
        # (ASIN, category) 복합 키로 추적
        product_ranks = defaultdict(list)

        for date_str, products in daily_data.items():
            for p in products:
                asin = p.get("asin", "")
                category = p.get("category", "")
                if asin and category:
                    key = (asin, category)
                    product_ranks[key].append(
                        {
                            "date": date_str,
                            "rank": p.get("rank", 0),
                            "title": p.get("title", ""),
                            "brand": p.get("brand", ""),
                            "category": category,
                        }
                    )

        movers = []
        for (asin, category), ranks in product_ranks.items():
            if len(ranks) >= 2:
                first = ranks[0]
                last = ranks[-1]
                change = first["rank"] - last["rank"]  # 양수 = 상승

                category_name = self._get_category_name(category)

                movers.append(
                    {
                        "asin": asin,
                        "title": last["title"],
                        "brand": last["brand"],
                        "category": category,
                        "category_name": category_name,
                        "start_rank": first["rank"],
                        "end_rank": last["rank"],
                        "change": change,
                        "is_laneige": self._is_laneige(last),
                    }
                )

        # 정렬
        movers.sort(key=lambda x: x["change"], reverse=True)

        return {
            "risers": movers[:10],  # Top 10 상승
            "fallers": movers[-10:][::-1],  # Top 10 하락
            "laneige_risers": [m for m in movers if m["is_laneige"] and m["change"] > 0][:5],
            "laneige_fallers": [m for m in movers if m["is_laneige"] and m["change"] < 0][-5:],
        }

    def _analyze_competitive_shifts(self, daily_data: dict[str, list[dict]]) -> dict[str, Any]:
        """경쟁 구도 변화 분석"""
        dates = sorted(daily_data.keys())
        if len(dates) < 2:
            return {
                "new_entrants": [],
                "exits": [],
                "total_brands_start": 0,
                "total_brands_end": 0,
            }

        first_date = dates[0]
        last_date = dates[-1]

        # 첫날/마지막날 브랜드 비교
        first_brands = {p.get("brand", "") for p in daily_data[first_date] if p.get("brand")}
        last_brands = {p.get("brand", "") for p in daily_data[last_date] if p.get("brand")}

        new_entrants = last_brands - first_brands
        exits = first_brands - last_brands

        return {
            "new_entrants": list(new_entrants)[:10],
            "exits": list(exits)[:10],
            "total_brands_start": len(first_brands),
            "total_brands_end": len(last_brands),
        }

    def _build_daily_trends(self, daily_data: dict[str, list[dict]]) -> list[dict[str, Any]]:
        """일별 추이 데이터 생성 (차트용)"""
        trends = []

        for date_str, products in sorted(daily_data.items()):
            laneige_count = sum(1 for p in products if self._is_laneige(p))
            total = len(products)
            sos = (laneige_count / total * 100) if total > 0 else 0

            # HHI 계산
            brand_counts = defaultdict(int)
            for p in products:
                brand = p.get("brand", "Unknown")
                if brand:
                    brand_counts[brand] += 1

            hhi = sum((c / total * 100) ** 2 for c in brand_counts.values()) if total > 0 else 0

            trends.append(
                {
                    "date": date_str,
                    "laneige_sos": round(sos, 2),
                    "laneige_count": laneige_count,
                    "total_products": total,
                    "hhi": round(hhi, 2),
                    "brand_count": len(brand_counts),
                }
            )

        return trends

    def _is_laneige(self, product: dict) -> bool:
        """제품이 LANEIGE 브랜드인지 확인"""
        brand = (product.get("brand") or "").upper()
        title = (product.get("title") or "").upper()
        return "LANEIGE" in brand or "LANEIGE" in title

    def _get_category_name(self, category_id: str) -> str:
        """카테고리 ID를 한글명으로 변환"""
        CATEGORY_MAP = {
            "beauty_personal_care": "Beauty & Personal Care",
            "beauty": "Beauty & Personal Care",
            "skin_care": "Skin Care",
            "lip_care": "Lip Care",
            "lip_makeup": "Lip Makeup",
            "face_powder": "Face Powder",
        }
        return CATEGORY_MAP.get(category_id, category_id)

"""
Metrics Agent
지표 계산 에이전트
"""

import json
from datetime import datetime
from typing import Any

from src.monitoring.logger import AgentLogger
from src.monitoring.metrics import QualityMetrics
from src.monitoring.tracer import ExecutionTracer
from src.tools.calculators.metric_calculator import MetricCalculator


class MetricsAgent:
    """
    지표 계산 에이전트
    Implements MetricsAgentProtocol (src.domain.interfaces.agent)
    """

    def __init__(
        self,
        config_path: str = "./config/thresholds.json",
        logger: AgentLogger | None = None,
        tracer: ExecutionTracer | None = None,
        metrics: QualityMetrics | None = None,
    ):
        """
        Args:
            config_path: 설정 파일 경로
            logger: 로거
            tracer: 추적기
            metrics: 메트릭 수집기
        """
        self.config = self._load_config(config_path)
        self.calculator = MetricCalculator(self.config)
        self.logger = logger or AgentLogger("metrics")
        self.tracer = tracer
        self.metrics = metrics

        self._results: dict[str, Any] = {}

    def _load_config(self, path: str) -> dict:
        """설정 로드"""
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    async def execute(
        self, crawl_data: dict[str, Any], historical_data: dict[str, list] | None = None
    ) -> dict[str, Any]:
        """
        지표 계산 실행

        Args:
            crawl_data: 크롤러 에이전트 결과
            historical_data: 히스토리 데이터 (ASIN -> 순위 기록)

        Returns:
            {
                "status": "completed",
                "brand_metrics": [...],
                "product_metrics": [...],
                "market_metrics": [...],
                "alerts": [...]
            }
        """
        self.logger.agent_start("MetricsAgent", "지표 계산")
        start_time = datetime.now()

        if self.metrics:
            self.metrics.record_agent_start("metrics")

        if self.tracer:
            self.tracer.start_span("metrics_agent")

        try:
            results = {
                "status": "completed",
                "calculated_at": datetime.now().isoformat(),
                "brand_metrics": [],
                "product_metrics": [],
                "market_metrics": [],
                "alerts": [],
                "summary": {},
            }

            historical = historical_data or {}

            # 카테고리별 처리
            for cat_key, cat_data in crawl_data.get("categories", {}).items():
                self.logger.info(f"Calculating metrics for category: {cat_key}")

                if self.tracer:
                    self.tracer.start_span("calculate_category", {"category": cat_key})

                products = cat_data.get("rank_records", [])
                if not products:
                    if self.tracer:
                        self.tracer.end_span("skipped")
                    continue

                # Level 1: 시장/브랜드 지표
                market_metric = self._calculate_market_metrics(cat_key, products)
                results["market_metrics"].append(market_metric)

                brand_metrics = self._calculate_brand_metrics(cat_key, products)
                results["brand_metrics"].extend(brand_metrics)

                # Level 2 & 3: 제품별 지표 (LANEIGE만)
                laneige_products = [p for p in products if self._is_laneige(p)]

                for product in laneige_products:
                    asin = product.get("product_asin")
                    history = historical.get(asin, [])

                    product_metric = self._calculate_product_metrics(product, history, cat_key)
                    results["product_metrics"].append(product_metric)

                    # 알림 체크
                    alerts = self._check_alerts(product_metric, product, history)
                    results["alerts"].extend(alerts)

                if self.tracer:
                    self.tracer.end_span("completed")

            # 전체 요약 생성
            results["summary"] = self._generate_summary(results)

            self._results = results
            duration = (datetime.now() - start_time).total_seconds()

            if self.tracer:
                self.tracer.end_span("completed")

            if self.metrics:
                self.metrics.record_agent_complete(
                    "metrics",
                    {
                        "brand_metrics": len(results["brand_metrics"]),
                        "product_metrics": len(results["product_metrics"]),
                        "alerts": len(results["alerts"]),
                    },
                )

            self.logger.agent_complete(
                "MetricsAgent",
                duration,
                f"{len(results['brand_metrics'])} brands, {len(results['product_metrics'])} products",
            )

            return results

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()

            if self.tracer:
                self.tracer.end_span("failed", str(e))

            if self.metrics:
                self.metrics.record_agent_error("metrics", str(e))

            self.logger.agent_error("MetricsAgent", str(e), duration)
            raise

    def _calculate_market_metrics(self, category_id: str, products: list[dict]) -> dict[str, Any]:
        """시장 지표 계산"""
        # HHI - products 리스트를 직접 전달
        hhi = self.calculator.calculate_hhi(products)

        # 브랜드별 제품 수 (Top 브랜드 계산용)
        brand_counts = {}
        for p in products:
            brand = p.get("brand", "Unknown")
            brand_counts[brand] = brand_counts.get(brand, 0) + 1

        # Top 브랜드
        top_brand = max(brand_counts.items(), key=lambda x: x[1]) if brand_counts else ("", 0)
        top_brand_sos = top_brand[1] / len(products) if products else 0

        # Top 브랜드의 CPI
        cpi = self.calculator.calculate_cpi(products, top_brand[0]) if top_brand[0] else None

        return {
            "category_id": category_id,
            "hhi": hhi,
            "cpi": cpi,
            "churn_rate_7d": None,  # 히스토리 필요
            "avg_rating_gap": self._calc_avg_rating_gap(products),
            "top_brand": top_brand[0],
            "top_brand_sos": round(top_brand_sos, 3),
            "total_products": len(products),
        }

    def _calculate_brand_metrics(
        self, category_id: str, products: list[dict]
    ) -> list[dict[str, Any]]:
        """브랜드별 지표 계산"""
        # 브랜드별 그룹핑
        by_brand = {}
        for p in products:
            brand = p.get("brand", "Unknown")
            if brand not in by_brand:
                by_brand[brand] = []
            by_brand[brand].append(p)

        brand_metrics = []

        for brand, brand_products in by_brand.items():
            ranks = [p.get("rank", 100) for p in brand_products]

            # calculate_sos: 전체 products와 brand명을 전달
            sos = self.calculator.calculate_sos(products, brand)
            # calculate_brand_avg_rank: 전체 products와 brand명을 전달
            avg_rank = self.calculator.calculate_brand_avg_rank(products, brand)

            brand_metrics.append(
                {
                    "brand_name": brand,
                    "category_id": category_id,
                    "share_of_shelf": sos,
                    "avg_rank": avg_rank,
                    "product_count": len(brand_products),
                    "top10_count": sum(1 for r in ranks if r <= 10),
                    "top20_count": sum(1 for r in ranks if r <= 20),
                    "is_laneige": brand.lower() == "laneige",
                }
            )

        return brand_metrics

    def _calculate_product_metrics(
        self, product: dict, history: list[dict], category_id: str
    ) -> dict[str, Any]:
        """제품별 지표 계산"""
        current_rank = product.get("rank", 0)
        asin = product.get("product_asin", product.get("asin", ""))

        # 히스토리 기반 계산
        rank_history = [h.get("rank", 0) for h in history] if history else []

        # 순위 변동
        rank_change_1d = None
        rank_change_7d = None
        if len(rank_history) >= 1:
            rank_change_1d = current_rank - rank_history[-1]
        if len(rank_history) >= 7:
            rank_change_7d = current_rank - rank_history[-7]

        # 변동성 (List[int] 전달)
        volatility = (
            self.calculator.calculate_rank_volatility(rank_history + [current_rank])
            if rank_history
            else 0
        )

        # 연속 기간 (List[Dict], asin, top_n 전달)
        top_n = self.config.get("ranking", {}).get("top_n_tiers", [3, 5, 10, 20])[2]  # Top 10
        streak = self.calculator.calculate_streak_days(history, asin, top_n) if history else 0

        # 평점 트렌드 (히스토리 필요)
        rating_trend = None

        return {
            "asin": asin,
            "product_title": product.get("title", product.get("product_name", "")),
            "category_id": category_id,
            "current_rank": current_rank,
            "rank_change_1d": rank_change_1d,
            "rank_change_7d": rank_change_7d,
            "rank_volatility": volatility,
            "streak_days": streak,
            "rating": product.get("rating"),
            "rating_trend": rating_trend,
            "price": product.get("price"),
            "review_count": product.get("review_count"),
        }

    def _check_alerts(self, product_metric: dict, product: dict, history: list[dict]) -> list[dict]:
        """알림 조건 체크"""
        alerts = []
        thresholds = self.config.get("thresholds", {})

        current_rank = product_metric.get("current_rank", 0)
        rank_change_1d = product_metric.get("rank_change_1d")

        # 순위 급락 알림
        significant_drop = thresholds.get("significant_rank_drop", 5)
        if rank_change_1d and rank_change_1d >= significant_drop:
            alerts.append(
                {
                    "type": "rank_drop",
                    "severity": "warning" if rank_change_1d < 10 else "critical",
                    "asin": product_metric["asin"],
                    "title": product_metric["product_title"],
                    "message": f"순위 {rank_change_1d}단계 하락 (현재 {current_rank}위)",
                    "details": {
                        "previous_rank": current_rank - rank_change_1d,
                        "current_rank": current_rank,
                        "change": rank_change_1d,
                    },
                }
            )

        # Top 10 진입 알림
        if current_rank <= 10:
            previous_rank = current_rank - (rank_change_1d or 0)
            if previous_rank > 10:
                alerts.append(
                    {
                        "type": "top10_entry",
                        "severity": "info",
                        "asin": product_metric["asin"],
                        "title": product_metric["product_title"],
                        "message": f"Top 10 진입 ({current_rank}위)",
                        "details": {
                            "current_rank": current_rank,
                            "category": product_metric["category_id"],
                        },
                    }
                )

        # Rank Shock (급변동)
        if history and len(history) >= 1:
            yesterday_rank = history[-1].get("rank", 0)
            is_shock = self.calculator.calculate_rank_shock(current_rank, yesterday_rank)
            if is_shock:
                rank_change = current_rank - yesterday_rank
                alerts.append(
                    {
                        "type": "rank_shock",
                        "severity": "warning",
                        "asin": product_metric["asin"],
                        "title": product_metric["product_title"],
                        "message": f"순위 급변동 감지 (변동폭: {abs(rank_change)})",
                        "details": {
                            "is_shock": True,
                            "change": rank_change,
                            "previous_rank": yesterday_rank,
                            "current_rank": current_rank,
                        },
                    }
                )

        return alerts

    def _calc_avg_rating_gap(self, products: list[dict]) -> float | None:
        """평균 평점 갭 계산"""
        ratings = [p.get("rating", 0) for p in products if p.get("rating")]
        if not ratings:
            return None

        max_rating = max(ratings)
        min_rating = min(ratings)
        return round(max_rating - min_rating, 2)

    def _is_laneige(self, product: dict) -> bool:
        """LANEIGE 제품 여부"""
        title = str(product.get("title", "")).lower()
        brand = str(product.get("brand", "")).lower()
        return "laneige" in title or "laneige" in brand

    def _generate_summary(self, results: dict) -> dict[str, Any]:
        """결과 요약 생성"""
        brand_metrics = results.get("brand_metrics", [])
        product_metrics = results.get("product_metrics", [])
        alerts = results.get("alerts", [])

        # LANEIGE 브랜드 지표
        laneige_brands = [b for b in brand_metrics if b.get("is_laneige")]

        # 카테고리별 LANEIGE SoS
        sos_by_category = {}
        for b in laneige_brands:
            cat = b.get("category_id")
            sos_by_category[cat] = b.get("share_of_shelf", 0)

        # 베스트 순위 제품
        best_product = None
        if product_metrics:
            best = min(product_metrics, key=lambda x: x.get("current_rank", 100))
            best_product = {
                "asin": best.get("asin"),
                "title": best.get("product_title"),
                "rank": best.get("current_rank"),
                "category": best.get("category_id"),
            }

        return {
            "laneige_products_tracked": len(product_metrics),
            "laneige_sos_by_category": sos_by_category,
            "best_ranking_product": best_product,
            "alert_count": len(alerts),
            "critical_alerts": len([a for a in alerts if a.get("severity") == "critical"]),
            "warning_alerts": len([a for a in alerts if a.get("severity") == "warning"]),
        }

    def get_results(self) -> dict[str, Any]:
        """마지막 실행 결과"""
        return self._results

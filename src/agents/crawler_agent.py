"""
Crawler Agent
Amazon 베스트셀러 크롤링 에이전트
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

# 한국 시간대 (UTC+9)
from src.domain.entities import RankRecord
from src.monitoring.logger import AgentLogger
from src.monitoring.metrics import QualityMetrics
from src.monitoring.tracer import ExecutionTracer
from src.shared.constants import KST
from src.tools.scrapers.amazon_scraper import AmazonScraper


class CrawlerAgent:
    """
    Amazon 크롤링 에이전트
    Implements CrawlerAgentProtocol (src.domain.interfaces.agent)
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
        self.scraper = AmazonScraper()
        self.logger = logger or AgentLogger("crawler")
        self.tracer = tracer
        self.metrics = metrics

        # 결과 저장
        self._results: dict[str, Any] = {}

    def _load_config(self, path: str) -> dict:
        """설정 로드"""
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    async def execute(self, categories: list[str] | None = None) -> dict[str, Any]:
        """
        크롤링 실행

        Args:
            categories: 크롤링할 카테고리 목록 (None이면 전체)

        Returns:
            {
                "status": "completed" | "partial" | "failed",
                "categories": {...},
                "total_products": int,
                "laneige_products": [...],
                "errors": [...]
            }
        """
        self.logger.agent_start("CrawlerAgent", "Amazon 베스트셀러 크롤링")
        start_time = datetime.now()

        if self.metrics:
            self.metrics.record_agent_start("crawler")

        if self.tracer:
            self.tracer.start_span("crawler_agent", {"categories": categories})

        try:
            # 크롤링할 카테고리 결정
            category_config = self.config.get("categories", {})
            target_categories = categories or list(category_config.keys())

            results = {
                "status": "completed",
                "crawled_at": datetime.now().isoformat(),
                "categories": {},
                "all_products": [],
                "laneige_products": [],
                "errors": [],
            }

            # 각 카테고리 크롤링
            for cat_key in target_categories:
                if cat_key not in category_config:
                    self.logger.warning(f"Unknown category: {cat_key}")
                    continue

                cat_info = category_config[cat_key]
                url = cat_info["url"]
                cat_name = cat_info["name"]

                self.logger.info(f"Crawling category: {cat_name}")

                if self.tracer:
                    self.tracer.start_span("crawl_category", {"category": cat_key})

                try:
                    cat_start = datetime.now()
                    crawl_result = await self.scraper.scrape_category(cat_key, url)
                    cat_duration = (datetime.now() - cat_start).total_seconds()

                    if not crawl_result.get("success"):
                        raise Exception(crawl_result.get("error", "Unknown error"))

                    products = crawl_result.get("products", [])

                    # RankRecord 생성
                    rank_records = []
                    for product in products:
                        record = RankRecord(
                            snapshot_date=datetime.now(KST).date(),
                            category_id=cat_key,
                            asin=product.get("asin", ""),
                            product_name=product.get("title", product.get("product_name", "")),
                            brand=product.get("brand", "Unknown"),
                            rank=product.get("rank", 0),
                            price=product.get("price"),
                            rating=product.get("rating"),
                            reviews_count=product.get("reviews_count", product.get("review_count")),
                            badge=product.get("badge", ""),
                            product_url=product.get(
                                "url",
                                product.get(
                                    "product_url",
                                    f"https://www.amazon.com/dp/{product.get('asin', '')}",
                                ),
                            ),
                        )
                        rank_records.append(record)

                        # LANEIGE 제품 필터링
                        if self._is_laneige_product(product):
                            results["laneige_products"].append(
                                {**product, "category": cat_key, "category_name": cat_name}
                            )

                    results["categories"][cat_key] = {
                        "name": cat_name,
                        "product_count": len(products),
                        "rank_records": [r.model_dump() for r in rank_records],
                        "duration_seconds": round(cat_duration, 2),
                    }
                    results["all_products"].extend(products)

                    if self.metrics:
                        self.metrics.record_crawl(
                            category=cat_key,
                            products_count=len(products),
                            duration_seconds=cat_duration,
                            success=True,
                        )

                    if self.tracer:
                        self.tracer.end_span("completed")

                    self.logger.info(
                        f"Category {cat_name}: {len(products)} products",
                        {"laneige_count": sum(1 for p in products if self._is_laneige_product(p))},
                    )

                except Exception as e:
                    error_msg = str(e)
                    results["errors"].append({"category": cat_key, "error": error_msg})
                    results["status"] = "partial"

                    if self.metrics:
                        self.metrics.record_crawl(cat_key, 0, 0, success=False)

                    if self.tracer:
                        self.tracer.end_span("failed", error_msg)

                    self.logger.error(f"Failed to crawl {cat_name}: {error_msg}")

                # Rate limiting
                await asyncio.sleep(2)

            # 경쟁사 제품 추적 (tracked_competitors.json)
            competitor_products = await self._scrape_tracked_competitors()
            if competitor_products:
                results["competitor_products"] = competitor_products
                self.logger.info(f"Tracked {len(competitor_products)} competitor products")

            # 전체 결과 집계
            results["total_products"] = len(results["all_products"])
            results["laneige_count"] = len(results["laneige_products"])
            results["competitor_count"] = len(results.get("competitor_products", []))

            # 모든 카테고리 실패 시
            if not results["categories"]:
                results["status"] = "failed"

            self._results = results
            duration = (datetime.now() - start_time).total_seconds()

            if self.tracer:
                self.tracer.end_span("completed")

            if self.metrics:
                self.metrics.record_agent_complete(
                    "crawler",
                    {
                        "total_products": results["total_products"],
                        "laneige_count": results["laneige_count"],
                    },
                )

            self.logger.agent_complete(
                "CrawlerAgent",
                duration,
                f"{results['total_products']} products, {results['laneige_count']} LANEIGE",
            )

            return results

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()

            if self.tracer:
                self.tracer.end_span("failed", str(e))

            if self.metrics:
                self.metrics.record_agent_error("crawler", str(e))

            self.logger.agent_error("CrawlerAgent", str(e), duration)
            raise

    def _is_laneige_product(self, product: dict) -> bool:
        """LANEIGE 제품 여부 확인"""
        title = product.get("title", "").lower()
        brand = product.get("brand", "").lower()

        return "laneige" in title or "laneige" in brand

    async def _scrape_tracked_competitors(self) -> list[dict[str, Any]]:
        """
        tracked_competitors.json에 정의된 경쟁사 제품 크롤링
        """
        try:
            config_path = Path("./config/tracked_competitors.json")
            if not config_path.exists():
                self.logger.info("No tracked_competitors.json found, skipping competitor tracking")
                return []

            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)

            competitors = config.get("competitors", {})
            all_competitor_products = []

            for brand_name, brand_config in competitors.items():
                self.logger.info(f"Scraping tracked competitor: {brand_name}")

                try:
                    products = await self.scraper.scrape_competitor_products(brand_config)
                    all_competitor_products.extend(products)
                except Exception as e:
                    self.logger.error(f"Failed to scrape competitor {brand_name}: {e}")
                    continue

                # 브랜드 간 딜레이
                await asyncio.sleep(3)

            return all_competitor_products

        except Exception as e:
            self.logger.error(f"Error in competitor tracking: {e}")
            return []

    async def close(self) -> None:
        """리소스 정리"""
        await self.scraper.close()

    def get_results(self) -> dict[str, Any]:
        """마지막 실행 결과 반환"""
        return self._results

    def get_laneige_summary(self) -> dict[str, Any]:
        """LANEIGE 제품 요약"""
        if not self._results:
            return {}

        products = self._results.get("laneige_products", [])

        if not products:
            return {"total_count": 0, "categories": {}, "best_rank": None, "products": []}

        # 카테고리별 집계
        by_category = {}
        for p in products:
            cat = p.get("category", "unknown")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(p)

        category_summary = {}
        for cat, cat_products in by_category.items():
            ranks = [p["rank"] for p in cat_products]
            category_summary[cat] = {
                "count": len(cat_products),
                "best_rank": min(ranks),
                "worst_rank": max(ranks),
                "avg_rank": round(sum(ranks) / len(ranks), 1),
            }

        # 전체 베스트 순위
        all_ranks = [p["rank"] for p in products]
        best = min(products, key=lambda x: x["rank"])

        return {
            "total_count": len(products),
            "categories": category_summary,
            "best_rank": min(all_ranks),
            "best_product": {
                "title": best.get("title"),
                "rank": best.get("rank"),
                "category": best.get("category_name"),
            },
            "products": products,
        }

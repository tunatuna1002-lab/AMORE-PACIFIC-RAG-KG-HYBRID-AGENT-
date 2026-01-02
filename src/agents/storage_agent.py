"""
Storage Agent
Google Sheets 저장 에이전트
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.tools.sheets_writer import SheetsWriter
from src.ontology.schema import Product, RankRecord, BrandMetrics, ProductMetrics, MarketMetrics
from src.monitoring.logger import AgentLogger
from src.monitoring.tracer import ExecutionTracer
from src.monitoring.metrics import QualityMetrics


class StorageAgent:
    """Google Sheets 저장 에이전트"""

    def __init__(
        self,
        spreadsheet_id: Optional[str] = None,
        logger: Optional[AgentLogger] = None,
        tracer: Optional[ExecutionTracer] = None,
        metrics: Optional[QualityMetrics] = None
    ):
        """
        Args:
            spreadsheet_id: Google Sheets 스프레드시트 ID
            logger: 로거
            tracer: 추적기
            metrics: 메트릭 수집기
        """
        self.sheets = SheetsWriter(spreadsheet_id)
        self.logger = logger or AgentLogger("storage")
        self.tracer = tracer
        self.metrics = metrics

        self._results: Dict[str, Any] = {}

    async def execute(self, crawl_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        크롤링 데이터 저장

        Args:
            crawl_data: 크롤러 에이전트 결과

        Returns:
            {
                "status": "completed" | "failed",
                "saved_records": int,
                "updated_products": int,
                ...
            }
        """
        self.logger.agent_start("StorageAgent", "데이터 저장")
        start_time = datetime.now()

        if self.metrics:
            self.metrics.record_agent_start("storage")

        if self.tracer:
            self.tracer.start_span("storage_agent")

        try:
            results = {
                "status": "completed",
                "saved_at": datetime.now().isoformat(),
                "raw_records": 0,
                "products_upserted": 0,
                "errors": []
            }

            # 1. Raw 순위 데이터 저장
            if self.tracer:
                self.tracer.start_span("save_raw_data")

            try:
                all_records = []
                for cat_key, cat_data in crawl_data.get("categories", {}).items():
                    records = cat_data.get("rank_records", [])
                    for record in records:
                        # record가 이미 dict인 경우와 RankRecord인 경우 처리
                        if isinstance(record, dict):
                            all_records.append(record)
                        else:
                            all_records.append(record.model_dump() if hasattr(record, 'model_dump') else record)

                if all_records:
                    await self.sheets.append_rank_records(all_records)
                    results["raw_records"] = len(all_records)
                    self.logger.info(f"Saved {len(all_records)} rank records")

                if self.tracer:
                    self.tracer.end_span("completed")

            except Exception as e:
                results["errors"].append({"step": "raw_data", "error": str(e)})
                self.logger.error(f"Failed to save raw data: {e}")
                if self.tracer:
                    self.tracer.end_span("failed", str(e))

            # 2. 제품 정보 업서트
            if self.tracer:
                self.tracer.start_span("upsert_products")

            try:
                products_data = crawl_data.get("all_products", [])
                for p in products_data:
                    product_dict = {
                        "asin": p.get("asin", ""),
                        "product_name": p.get("title", p.get("product_name", "")),
                        "brand": p.get("brand", "Unknown"),
                        "product_url": p.get("url", f"https://www.amazon.com/dp/{p.get('asin', '')}"),
                        "first_seen_date": datetime.now().date().isoformat(),
                        "launch_date": ""
                    }
                    await self.sheets.upsert_product(product_dict)
                    results["products_upserted"] += 1

                self.logger.info(f"Upserted {results['products_upserted']} products")

                if self.tracer:
                    self.tracer.end_span("completed")

            except Exception as e:
                results["errors"].append({"step": "products", "error": str(e)})
                self.logger.error(f"Failed to upsert products: {e}")
                if self.tracer:
                    self.tracer.end_span("failed", str(e))

            # 상태 결정
            if results["errors"]:
                results["status"] = "partial" if results["raw_records"] > 0 else "failed"

            self._results = results
            duration = (datetime.now() - start_time).total_seconds()

            if self.tracer:
                self.tracer.end_span("completed")

            if self.metrics:
                self.metrics.record_agent_complete("storage", {
                    "raw_records": results["raw_records"],
                    "products": results["products_upserted"]
                })

            self.logger.agent_complete(
                "StorageAgent",
                duration,
                f"{results['raw_records']} records, {results['products_upserted']} products"
            )

            return results

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()

            if self.tracer:
                self.tracer.end_span("failed", str(e))

            if self.metrics:
                self.metrics.record_agent_error("storage", str(e))

            self.logger.agent_error("StorageAgent", str(e), duration)
            raise

    async def save_metrics(
        self,
        brand_metrics: Optional[List[BrandMetrics]] = None,
        product_metrics: Optional[List[ProductMetrics]] = None,
        market_metrics: Optional[List[MarketMetrics]] = None
    ) -> Dict[str, Any]:
        """
        계산된 지표 저장

        Args:
            brand_metrics: 브랜드 지표
            product_metrics: 제품 지표
            market_metrics: 시장 지표

        Returns:
            저장 결과
        """
        self.logger.info("Saving calculated metrics")

        if self.tracer:
            self.tracer.start_span("save_metrics")

        results = {
            "brand_metrics": 0,
            "product_metrics": 0,
            "market_metrics": 0
        }

        try:
            # 브랜드 지표 저장
            if brand_metrics:
                for bm in brand_metrics:
                    row = [
                        datetime.now().isoformat(),
                        bm.brand_name,
                        bm.category_id,
                        bm.share_of_shelf,
                        bm.avg_rank,
                        bm.product_count,
                        bm.top10_count,
                        bm.top20_count
                    ]
                    self.sheets._append_row("BrandMetrics", row)
                results["brand_metrics"] = len(brand_metrics)

            # 제품 지표 저장
            if product_metrics:
                for pm in product_metrics:
                    row = [
                        datetime.now().isoformat(),
                        pm.asin,
                        pm.product_title,
                        pm.category_id,
                        pm.current_rank,
                        pm.rank_change_1d,
                        pm.rank_change_7d,
                        pm.rank_volatility,
                        pm.streak_days,
                        pm.rating_trend
                    ]
                    self.sheets._append_row("ProductMetrics", row)
                results["product_metrics"] = len(product_metrics)

            # 시장 지표 저장
            if market_metrics:
                for mm in market_metrics:
                    row = [
                        datetime.now().isoformat(),
                        mm.category_id,
                        mm.hhi,
                        mm.cpi,
                        mm.churn_rate_7d,
                        mm.avg_rating_gap,
                        mm.top_brand,
                        mm.top_brand_sos
                    ]
                    self.sheets._append_row("MarketMetrics", row)
                results["market_metrics"] = len(market_metrics)

            if self.tracer:
                self.tracer.end_span("completed")

            self.logger.info(f"Saved metrics: {results}")
            return results

        except Exception as e:
            if self.tracer:
                self.tracer.end_span("failed", str(e))
            self.logger.error(f"Failed to save metrics: {e}")
            raise

    def get_historical_data(
        self,
        asin: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        제품 히스토리 조회

        Args:
            asin: 제품 ASIN
            days: 조회 기간

        Returns:
            순위 기록 리스트
        """
        return self.sheets.get_rank_history(asin, days)

    def get_results(self) -> Dict[str, Any]:
        """마지막 실행 결과"""
        return self._results

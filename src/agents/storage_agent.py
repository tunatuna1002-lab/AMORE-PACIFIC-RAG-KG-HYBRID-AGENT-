"""
Storage Agent
Google Sheets + SQLite 이중 저장 에이전트

데이터를 두 곳에 동시 저장:
1. Google Sheets: 클라우드 백업, 비개발자 접근용
2. SQLite (Railway Volume): 빠른 조회, 대시보드/챗봇용
"""

import json
from datetime import datetime
from typing import Any

from src.domain.entities import BrandMetrics, MarketMetrics, ProductMetrics
from src.monitoring.logger import AgentLogger
from src.monitoring.metrics import QualityMetrics
from src.monitoring.tracer import ExecutionTracer
from src.tools.storage.sheets_writer import SheetsWriter
from src.tools.storage.sqlite_storage import SQLiteStorage, get_sqlite_storage


class StorageAgent:
    """
    Google Sheets + SQLite 이중 저장 에이전트
    Implements StorageAgentProtocol (src.domain.interfaces.agent)
    """

    def __init__(
        self,
        spreadsheet_id: str | None = None,
        logger: AgentLogger | None = None,
        tracer: ExecutionTracer | None = None,
        metrics: QualityMetrics | None = None,
        enable_sqlite: bool = True,
    ):
        """
        Args:
            spreadsheet_id: Google Sheets 스프레드시트 ID
            logger: 로거
            tracer: 추적기
            metrics: 메트릭 수집기
            enable_sqlite: SQLite 이중 저장 활성화 (기본값: True)
        """
        self.sheets = SheetsWriter(spreadsheet_id)
        self.sqlite: SQLiteStorage | None = get_sqlite_storage() if enable_sqlite else None
        self.enable_sqlite = enable_sqlite

        self.logger = logger or AgentLogger("storage")
        self.tracer = tracer
        self.metrics = metrics

        self._results: dict[str, Any] = {}

    async def execute(self, crawl_data: dict[str, Any]) -> dict[str, Any]:
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
                "sqlite_records": 0,  # SQLite 저장 결과
                "errors": [],
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
                            all_records.append(
                                record.model_dump() if hasattr(record, "model_dump") else record
                            )

                if all_records:
                    # 1-1. Google Sheets 저장
                    append_result = await self.sheets.append_rank_records(all_records)
                    if append_result.get("success"):
                        results["raw_records"] = len(all_records)
                        self.logger.info(f"Saved {len(all_records)} rank records to Google Sheets")
                    else:
                        error_msg = append_result.get("error", "Unknown error")
                        results["errors"].append({"step": "raw_data_sheets", "error": error_msg})
                        self.logger.error(f"Failed to save rank records to Sheets: {error_msg}")

                    # 1-2. SQLite 저장 (이중 저장)
                    if self.enable_sqlite and self.sqlite:
                        try:
                            await self.sqlite.initialize()
                            sqlite_result = await self.sqlite.append_rank_records(all_records)
                            if sqlite_result.get("success"):
                                results["sqlite_records"] = sqlite_result.get("rows_added", 0)
                                self.logger.info(
                                    f"Saved {results['sqlite_records']} rank records to SQLite"
                                )
                            else:
                                error_msg = sqlite_result.get("error", "Unknown error")
                                results["errors"].append(
                                    {"step": "raw_data_sqlite", "error": error_msg}
                                )
                                self.logger.error(
                                    f"Failed to save rank records to SQLite: {error_msg}"
                                )
                        except Exception as sqlite_err:
                            results["errors"].append(
                                {"step": "raw_data_sqlite", "error": str(sqlite_err)}
                            )
                            self.logger.error(f"SQLite storage error: {sqlite_err}", exc_info=True)
                            # 동기화 불일치 경고
                            self.logger.warning(
                                "⚠️ SQLite 저장 실패로 Google Sheets와 불일치 발생. "
                                "수동 동기화 필요: python scripts/sync_sheets_to_sqlite.py"
                            )

                if self.tracer:
                    self.tracer.end_span("completed")

            except Exception as e:
                results["errors"].append({"step": "raw_data", "error": str(e)})
                self.logger.error(f"Failed to save raw data: {e}")
                if self.tracer:
                    self.tracer.end_span("failed", str(e))

            # 2. 제품 정보 업서트 (배치 처리로 API 호출 최소화)
            if self.tracer:
                self.tracer.start_span("upsert_products")

            try:
                products_data = crawl_data.get("all_products", [])
                products_list = []
                for p in products_data:
                    product_dict = {
                        "asin": p.get("asin", ""),
                        "product_name": p.get("title", p.get("product_name", "")),
                        "brand": p.get("brand", "Unknown"),
                        "product_url": p.get(
                            "url", f"https://www.amazon.com/dp/{p.get('asin', '')}"
                        ),
                        "first_seen_date": datetime.now().date().isoformat(),
                        "launch_date": "",
                    }
                    products_list.append(product_dict)

                # 배치 처리로 API 호출 횟수 최소화 (기존: 제품마다 API 호출 → 변경: 2번만 호출)
                if products_list:
                    batch_result = await self.sheets.upsert_products_batch(products_list)
                    results["products_upserted"] = batch_result.get(
                        "created", 0
                    ) + batch_result.get("updated", 0)
                    self.logger.info(
                        f"Batch upserted products: created={batch_result.get('created', 0)}, updated={batch_result.get('updated', 0)}"
                    )

                if self.tracer:
                    self.tracer.end_span("completed")

            except Exception as e:
                results["errors"].append({"step": "products", "error": str(e)})
                self.logger.error(f"Failed to upsert products: {e}")
                if self.tracer:
                    self.tracer.end_span("failed", str(e))

            # 3. 경쟁사 추적 데이터 저장
            competitor_products = crawl_data.get("competitor_products", [])
            if competitor_products:
                if self.tracer:
                    self.tracer.start_span("save_competitor_data")

                try:
                    # SQLite에 경쟁사 데이터 저장
                    if self.enable_sqlite and self.sqlite:
                        await self.sqlite.initialize()
                        comp_result = await self.sqlite.save_competitor_products(
                            competitor_products
                        )
                        if comp_result.get("success"):
                            results["competitor_products_saved"] = comp_result.get("rows_added", 0)
                            self.logger.info(
                                f"Saved {results['competitor_products_saved']} competitor products to SQLite"
                            )
                        else:
                            results["errors"].append(
                                {
                                    "step": "competitor_sqlite",
                                    "error": comp_result.get("error", "Unknown"),
                                }
                            )

                    # JSON 파일로도 저장 (대시보드용)
                    try:
                        from pathlib import Path

                        comp_json_path = Path("./data/competitor_products.json")
                        with open(comp_json_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "updated_at": datetime.now().isoformat(),
                                    "products": competitor_products,
                                },
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )
                        self.logger.info(f"Saved competitor data to {comp_json_path}")
                    except Exception as json_err:
                        self.logger.warning(f"Failed to save competitor JSON: {json_err}")

                    if self.tracer:
                        self.tracer.end_span("completed")

                except Exception as e:
                    results["errors"].append({"step": "competitor_data", "error": str(e)})
                    self.logger.error(f"Failed to save competitor data: {e}")
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
                self.metrics.record_agent_complete(
                    "storage",
                    {
                        "raw_records": results["raw_records"],
                        "products": results["products_upserted"],
                    },
                )

            sqlite_info = f", SQLite: {results['sqlite_records']}" if self.enable_sqlite else ""
            self.logger.agent_complete(
                "StorageAgent",
                duration,
                f"Sheets: {results['raw_records']} records, {results['products_upserted']} products{sqlite_info}",
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
        brand_metrics: list[BrandMetrics] | None = None,
        product_metrics: list[ProductMetrics] | None = None,
        market_metrics: list[MarketMetrics] | None = None,
    ) -> dict[str, Any]:
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

        results = {"brand_metrics": 0, "product_metrics": 0, "market_metrics": 0}

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
                        bm.top20_count,
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
                        pm.rating_trend,
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
                        mm.top_brand_sos,
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

    def get_historical_data(self, asin: str, days: int = 30) -> list[dict[str, Any]]:
        """
        제품 히스토리 조회

        Args:
            asin: 제품 ASIN
            days: 조회 기간

        Returns:
            순위 기록 리스트
        """
        return self.sheets.get_rank_history(asin, days)

    def get_results(self) -> dict[str, Any]:
        """마지막 실행 결과"""
        return self._results

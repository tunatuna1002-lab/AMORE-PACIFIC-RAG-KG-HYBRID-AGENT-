"""
Crawl Workflow
==============
Orchestrates the crawling workflow.

Flow:
1. Scrape categories
2. Save to storage
3. Calculate metrics
4. Return results

Clean Architecture:
- Depends only on domain interfaces
- No infrastructure dependencies
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.domain.interfaces.metric import MetricCalculatorProtocol
from src.domain.interfaces.scraper import ScraperProtocol
from src.domain.interfaces.storage import StorageProtocol


@dataclass
class CrawlWorkflowResult:
    """Crawl workflow execution result"""

    success: bool = False
    records_count: int = 0
    categories_processed: int = 0
    metrics: dict[str, Any] | None = None
    execution_time: float = 0.0
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "records_count": self.records_count,
            "categories_processed": self.categories_processed,
            "metrics": self.metrics,
            "execution_time": self.execution_time,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class CrawlWorkflow:
    """
    Crawl Workflow

    Orchestrates the crawling process using dependency injection.

    Usage:
        workflow = CrawlWorkflow(
            scraper=scraper,
            storage=storage,
            metric_calculator=calculator
        )
        result = await workflow.execute(categories=["lip_care"])
    """

    def __init__(
        self,
        scraper: ScraperProtocol,
        storage: StorageProtocol,
        metric_calculator: MetricCalculatorProtocol,
    ):
        """
        Args:
            scraper: Scraper implementation
            storage: Storage implementation
            metric_calculator: Metric calculator implementation
        """
        self.scraper = scraper
        self.storage = storage
        self.metric_calculator = metric_calculator

    async def execute(self, categories: list[str]) -> CrawlWorkflowResult:
        """
        Execute crawl workflow.

        Args:
            categories: List of category IDs to crawl

        Returns:
            CrawlWorkflowResult with crawl statistics
        """
        start_time = time.time()
        result = CrawlWorkflowResult()

        if not categories:
            result.execution_time = time.time() - start_time
            return result

        try:
            # Step 1: Initialize scraper
            await self.scraper.initialize()

            # Step 2: Scrape all categories
            all_records = []
            for category_id in categories:
                # Construct URL (protocol expects category_id and url)
                url = f"https://www.amazon.com/gp/bestsellers/beauty/{category_id}"
                records = await self.scraper.scrape_category(category_id=category_id, url=url)
                all_records.extend(records)
                result.categories_processed += 1

            result.records_count = len(all_records)

            if all_records:
                # Step 3: Save to storage
                await self.storage.append_rank_records(all_records)

                # Step 4: Calculate metrics
                brand_metrics = {}
                for category_id in categories:
                    category_records = [
                        r for r in all_records if r.get("category_id") == category_id
                    ]
                    if category_records:
                        metrics = self.metric_calculator.calculate_brand_metrics(
                            records=category_records,
                            brand="LANEIGE",
                            category_id=category_id,
                        )
                        brand_metrics[category_id] = metrics

                # Save metrics
                if brand_metrics:
                    await self.storage.save_brand_metrics([brand_metrics])

                result.metrics = brand_metrics

            result.success = True

        except Exception as e:
            result.error = str(e)
            result.success = False

        finally:
            await self.scraper.close()

        result.execution_time = time.time() - start_time
        return result

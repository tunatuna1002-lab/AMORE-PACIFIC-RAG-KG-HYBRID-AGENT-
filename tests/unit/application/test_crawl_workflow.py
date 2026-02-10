"""
Tests for CrawlWorkflow
=======================
Tests crawling workflow.
"""

import pytest

from src.application.workflows.crawl_workflow import CrawlWorkflow, CrawlWorkflowResult


class TestCrawlWorkflow:
    """Test suite for CrawlWorkflow"""

    @pytest.fixture
    def workflow(self, mock_scraper, mock_storage, mock_metric_calculator):
        """Create CrawlWorkflow instance"""
        return CrawlWorkflow(
            scraper=mock_scraper,
            storage=mock_storage,
            metric_calculator=mock_metric_calculator,
        )

    @pytest.mark.asyncio
    async def test_execute_single_category(self, workflow, mock_scraper):
        """Test executing crawl for single category"""
        result = await workflow.execute(categories=["lip_care"])

        assert isinstance(result, CrawlWorkflowResult)
        assert result.success is True
        assert result.records_count > 0
        mock_scraper.scrape_category.assert_called()

    @pytest.mark.asyncio
    async def test_execute_multiple_categories(self, workflow, mock_scraper):
        """Test executing crawl for multiple categories"""
        result = await workflow.execute(categories=["lip_care", "face_powder"])

        assert result.success is True
        assert result.categories_processed == 2

    @pytest.mark.asyncio
    async def test_metrics_calculated(self, workflow, mock_metric_calculator):
        """Test that metrics are calculated after crawling"""
        result = await workflow.execute(categories=["lip_care"])

        assert result.metrics is not None
        mock_metric_calculator.calculate_brand_metrics.assert_called()

    @pytest.mark.asyncio
    async def test_data_saved(self, workflow, mock_storage):
        """Test that data is saved to storage"""
        result = await workflow.execute(categories=["lip_care"])

        mock_storage.append_rank_records.assert_called()
        mock_storage.save_brand_metrics.assert_called()

    @pytest.mark.asyncio
    async def test_error_handling(self, workflow, mock_scraper):
        """Test error handling during crawl"""
        mock_scraper.scrape_category.side_effect = Exception("Scraping failed")

        result = await workflow.execute(categories=["lip_care"])

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_empty_categories(self, workflow):
        """Test handling empty categories list"""
        result = await workflow.execute(categories=[])

        assert result.records_count == 0

    @pytest.mark.asyncio
    async def test_result_to_dict(self, workflow):
        """Test CrawlWorkflowResult serialization"""
        result = await workflow.execute(categories=["lip_care"])

        result_dict = result.to_dict()
        assert "success" in result_dict
        assert "records_count" in result_dict
        assert "categories_processed" in result_dict
        assert "execution_time" in result_dict

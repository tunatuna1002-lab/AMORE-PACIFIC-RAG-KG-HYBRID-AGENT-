"""
Tests for InsightWorkflow
=========================
Tests insight generation workflow.
"""

import pytest

from src.application.workflows.insight_workflow import (
    InsightWorkflow,
    InsightWorkflowResult,
)


class TestInsightWorkflow:
    """Test suite for InsightWorkflow"""

    @pytest.fixture
    def workflow(self, mock_insight_agent, mock_storage):
        """Create InsightWorkflow instance"""
        return InsightWorkflow(insight_agent=mock_insight_agent, storage=mock_storage)

    @pytest.mark.asyncio
    async def test_execute_with_metrics(self, workflow, sample_metrics, mock_insight_agent):
        """Test executing insight generation with metrics"""
        result = await workflow.execute(metrics_data=sample_metrics)

        assert isinstance(result, InsightWorkflowResult)
        assert result.success is True
        assert result.insights is not None
        mock_insight_agent.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_target_brand(self, workflow, sample_metrics):
        """Test executing with specific target brand"""
        result = await workflow.execute(metrics_data=sample_metrics, target_brand="LANEIGE")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_category(self, workflow, sample_metrics):
        """Test executing with specific category"""
        result = await workflow.execute(metrics_data=sample_metrics, category_id="lip_care")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_error_handling(self, workflow, sample_metrics, mock_insight_agent):
        """Test error handling during insight generation"""
        mock_insight_agent.execute.side_effect = Exception("LLM Error")

        result = await workflow.execute(metrics_data=sample_metrics)

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_result_to_dict(self, workflow, sample_metrics):
        """Test InsightWorkflowResult serialization"""
        result = await workflow.execute(metrics_data=sample_metrics)

        result_dict = result.to_dict()
        assert "success" in result_dict
        assert "insights" in result_dict
        assert "summary" in result_dict
        assert "execution_time" in result_dict

    @pytest.mark.asyncio
    async def test_empty_metrics(self, workflow):
        """Test handling empty metrics"""
        result = await workflow.execute(metrics_data={})

        assert result.insights == []

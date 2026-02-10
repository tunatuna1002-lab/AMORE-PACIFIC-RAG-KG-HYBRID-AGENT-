"""
Tests for AlertWorkflow
=======================
Tests alert processing workflow.
"""

import pytest

from src.application.workflows.alert_workflow import AlertWorkflow, AlertWorkflowResult


class TestAlertWorkflow:
    """Test suite for AlertWorkflow"""

    @pytest.fixture
    def workflow(self, mock_alert_agent):
        """Create AlertWorkflow instance"""
        return AlertWorkflow(alert_agent=mock_alert_agent)

    @pytest.mark.asyncio
    async def test_execute_with_metrics(self, workflow, sample_metrics, mock_alert_agent):
        """Test executing alert processing with metrics"""
        result = await workflow.execute(metrics_data=sample_metrics)

        assert isinstance(result, AlertWorkflowResult)
        assert result.success is True
        mock_alert_agent.process_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_pending_alerts(self, workflow, mock_alert_agent):
        """Test sending pending alerts"""
        result = await workflow.send_pending_alerts()

        assert result.success is True
        mock_alert_agent.send_pending_alerts.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self, workflow, sample_metrics, mock_alert_agent):
        """Test error handling during alert processing"""
        mock_alert_agent.process_metrics.side_effect = Exception("Alert Error")

        result = await workflow.execute(metrics_data=sample_metrics)

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_result_to_dict(self, workflow, sample_metrics):
        """Test AlertWorkflowResult serialization"""
        result = await workflow.execute(metrics_data=sample_metrics)

        result_dict = result.to_dict()
        assert "success" in result_dict
        assert "alerts_created" in result_dict
        assert "execution_time" in result_dict

    @pytest.mark.asyncio
    async def test_empty_metrics(self, workflow):
        """Test handling empty metrics"""
        result = await workflow.execute(metrics_data={})

        assert result.alerts_created == 0

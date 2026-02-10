"""
Tests for ChatWorkflow
======================
Tests chat query processing workflow.
"""

import pytest

from src.application.workflows.chat_workflow import ChatWorkflow, ChatWorkflowResult


class TestChatWorkflow:
    """Test suite for ChatWorkflow"""

    @pytest.fixture
    def workflow(self, mock_chatbot, mock_retriever):
        """Create ChatWorkflow instance"""
        return ChatWorkflow(chatbot=mock_chatbot, retriever=mock_retriever)

    @pytest.mark.asyncio
    async def test_execute_simple_query(self, workflow, mock_chatbot):
        """Test executing a simple query"""
        result = await workflow.execute(query="LANEIGE 순위 알려줘")

        assert isinstance(result, ChatWorkflowResult)
        assert result.response is not None
        assert result.query == "LANEIGE 순위 알려줘"
        assert result.complexity in ["simple", "moderate", "complex"]
        mock_chatbot.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_session_id(self, workflow, mock_chatbot):
        """Test executing query with session ID"""
        result = await workflow.execute(query="LANEIGE 순위는?", session_id="session-123")

        assert result.session_id == "session-123"
        mock_chatbot.chat.assert_called_once()
        call_args = mock_chatbot.chat.call_args
        assert call_args[1]["session_id"] == "session-123"

    @pytest.mark.asyncio
    async def test_execute_with_metrics(self, workflow, mock_chatbot, sample_metrics):
        """Test executing query with current metrics"""
        result = await workflow.execute(query="LANEIGE SoS는?", current_metrics=sample_metrics)

        assert result.response is not None
        mock_chatbot.chat.assert_called_once()
        call_args = mock_chatbot.chat.call_args
        assert call_args[1]["current_metrics"] == sample_metrics

    @pytest.mark.asyncio
    async def test_execute_complex_query(self, workflow, mock_chatbot):
        """Test executing a complex query"""
        result = await workflow.execute(query="LANEIGE 경쟁력 분석해줘")

        assert result.complexity == "complex"
        assert result.response is not None

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, workflow, mock_chatbot):
        """Test error handling during execution"""
        mock_chatbot.chat.side_effect = Exception("LLM Error")

        result = await workflow.execute(query="Test query")

        assert result.error is not None
        assert "LLM Error" in result.error

    @pytest.mark.asyncio
    async def test_result_to_dict(self, workflow):
        """Test ChatWorkflowResult serialization"""
        result = await workflow.execute(query="Test")

        result_dict = result.to_dict()
        assert "query" in result_dict
        assert "response" in result_dict
        assert "complexity" in result_dict
        assert "intent" in result_dict
        assert "sources" in result_dict
        assert "execution_time" in result_dict

    @pytest.mark.asyncio
    async def test_empty_query(self, workflow):
        """Test handling empty query"""
        result = await workflow.execute(query="")

        assert result.query == ""
        assert result.complexity == "simple"

    @pytest.mark.asyncio
    async def test_retrieval_used(self, workflow, mock_retriever, mock_chatbot):
        """Test that retrieval is used for context gathering"""
        await workflow.execute(query="LANEIGE 경쟁력은?")

        # Retriever should be called
        mock_retriever.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_conversation_history(self, workflow, mock_chatbot):
        """Test conversation history is maintained"""
        # First query
        await workflow.execute(query="LANEIGE 순위는?", session_id="session-1")

        # Second query in same session
        await workflow.execute(query="가격은?", session_id="session-1")

        # Chatbot should be called twice
        assert mock_chatbot.chat.call_count == 2

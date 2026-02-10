"""Tests for src.memory.context module."""

import pytest

from src.memory.context import ContextManager, ConversationTurn, DataContext, WorkflowContext


class TestConversationTurn:
    def test_creation_with_defaults(self):
        turn = ConversationTurn(role="user", content="hello")
        assert turn.role == "user"
        assert turn.content == "hello"
        assert turn.timestamp  # auto-generated
        assert turn.metadata == {}

    def test_creation_with_metadata(self):
        turn = ConversationTurn(role="assistant", content="hi", metadata={"key": "val"})
        assert turn.metadata == {"key": "val"}


class TestWorkflowContext:
    def test_defaults(self):
        wf = WorkflowContext()
        assert wf.current_step == ""
        assert wf.completed_steps == []
        assert wf.pending_steps == []
        assert wf.step_results == {}
        assert wf.errors == []


class TestDataContext:
    def test_defaults(self):
        dc = DataContext()
        assert dc.last_crawl_date is None
        assert dc.categories_crawled == []
        assert dc.products_count == 0
        assert dc.laneige_products == []
        assert dc.metrics_calculated is False
        assert dc.insights_generated is False


class TestContextManager:
    @pytest.fixture
    def ctx(self, tmp_path):
        return ContextManager(context_dir=str(tmp_path / "context"))

    def test_add_user_message(self, ctx):
        ctx.add_user_message("hello")
        history = ctx.get_conversation_history(limit=10)
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "hello"

    def test_add_assistant_message(self, ctx):
        ctx.add_assistant_message("hi there", metadata={"source": "rag"})
        history = ctx.get_conversation_history(limit=10)
        assert len(history) == 1
        assert history[0]["role"] == "assistant"
        assert history[0]["metadata"] == {"source": "rag"}

    def test_trim_conversation(self, ctx):
        ctx._max_conversation_turns = 5
        for i in range(10):
            ctx.add_user_message(f"msg {i}")
        history = ctx.get_conversation_history(limit=100)
        assert len(history) == 5
        assert history[0]["content"] == "msg 5"

    def test_get_conversation_summary_empty(self, ctx):
        assert ctx.get_conversation_summary() == "이전 대화 없음"

    def test_get_conversation_summary(self, ctx):
        ctx.add_user_message("hello")
        ctx.add_assistant_message("hi")
        summary = ctx.get_conversation_summary()
        assert "[사용자]: hello" in summary
        assert "[어시스턴트]: hi" in summary

    def test_get_conversation_summary_truncates(self, ctx):
        ctx.add_user_message("x" * 200)
        summary = ctx.get_conversation_summary()
        assert "..." in summary

    def test_workflow_lifecycle(self, ctx):
        ctx.start_workflow(["step1", "step2", "step3"])
        status = ctx.get_workflow_status()
        assert status["current_step"] == "step1"
        # start_workflow copies all steps to pending and sets current_step
        assert status["pending_steps"] == ["step1", "step2", "step3"]

        next_step = ctx.advance_workflow(result={"data": 1})
        assert next_step == "step1"

        next_step = ctx.advance_workflow()
        assert next_step == "step2"

        next_step = ctx.advance_workflow()
        assert next_step == "step3"

        next_step = ctx.advance_workflow()
        assert next_step is None

        status = ctx.get_workflow_status()
        assert status["progress_percent"] == 100.0

    def test_workflow_error_recording(self, ctx):
        ctx.start_workflow(["step1"])
        ctx.record_workflow_error("step1", "Something failed")
        status = ctx.get_workflow_status()
        assert status["has_errors"] is True
        assert status["error_count"] == 1

    def test_empty_workflow_status(self, ctx):
        status = ctx.get_workflow_status()
        assert status["progress"] == "0/0"
        assert status["progress_percent"] == 0

    def test_update_crawl_data(self, ctx):
        ctx.update_crawl_data(
            categories=["lip_care", "lip_makeup"],
            products_count=200,
            laneige_products=[{"asin": "B123", "title": "Lip Mask"}],
        )
        data_status = ctx.get_data_status()
        assert data_status["total_products"] == 200
        assert data_status["laneige_count"] == 1
        assert data_status["last_crawl"] is not None

    def test_set_metrics_and_insights(self, ctx):
        ctx.set_metrics_calculated(True)
        ctx.set_insights_generated(True)
        status = ctx.get_data_status()
        assert status["metrics_ready"] is True
        assert status["insights_ready"] is True

    def test_get_laneige_products(self, ctx):
        products = [{"asin": "B1"}, {"asin": "B2"}]
        ctx.update_crawl_data(["lip_care"], 100, products)
        assert ctx.get_laneige_products() == products

    def test_variable_management(self, ctx):
        ctx.set_variable("key1", "value1")
        assert ctx.get_variable("key1") == "value1"
        assert ctx.get_variable("nonexistent", "default") == "default"

        ctx.clear_variables()
        assert ctx.get_variable("key1") is None

    def test_get_full_context(self, ctx):
        ctx.add_user_message("test")
        ctx.set_variable("k", "v")
        full = ctx.get_full_context()
        assert "conversation" in full
        assert "workflow" in full
        assert "data" in full
        assert "variables" in full
        assert full["variables"] == {"k": "v"}

    def test_build_llm_context_empty(self, ctx):
        assert ctx.build_llm_context() == "컨텍스트 없음"

    def test_build_llm_context_with_data(self, ctx):
        ctx.update_crawl_data(["lip_care"], 100, [])
        ctx.add_user_message("hello")
        llm_ctx = ctx.build_llm_context()
        assert "데이터 현황" in llm_ctx
        assert "lip_care" in llm_ctx

    def test_build_llm_context_with_workflow(self, ctx):
        ctx.start_workflow(["step1", "step2"])
        llm_ctx = ctx.build_llm_context()
        assert "워크플로우 진행" in llm_ctx

    def test_save_and_load_context(self, ctx):
        ctx.add_user_message("saved message")
        ctx.set_variable("saved_key", "saved_val")
        ctx.save_context("test_session")

        new_ctx = ContextManager(context_dir=str(ctx.context_dir))
        assert new_ctx.load_context("test_session") is True
        history = new_ctx.get_conversation_history()
        assert len(history) == 1
        assert history[0]["content"] == "saved message"
        assert new_ctx.get_variable("saved_key") == "saved_val"

    def test_load_nonexistent_context(self, ctx):
        assert ctx.load_context("nonexistent") is False

    def test_load_corrupted_context(self, ctx, tmp_path):
        filepath = ctx.context_dir / "corrupted_context.json"
        filepath.write_text("invalid json{{{", encoding="utf-8")
        assert ctx.load_context("corrupted") is False

    def test_reset(self, ctx):
        ctx.add_user_message("test")
        ctx.set_variable("k", "v")
        ctx.start_workflow(["step1"])
        ctx.update_crawl_data(["cat"], 10, [])

        ctx.reset()

        assert ctx.get_conversation_history() == []
        assert ctx.get_variable("k") is None
        assert ctx.get_workflow_status()["current_step"] == ""
        assert ctx.get_data_status()["total_products"] == 0

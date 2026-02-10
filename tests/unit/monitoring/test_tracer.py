"""Tests for src.monitoring.tracer module."""

import pytest

from src.monitoring.tracer import ExecutionTracer, Span, Trace


class TestSpan:
    def test_creation(self):
        span = Span(span_id="s1", name="test")
        assert span.span_id == "s1"
        assert span.name == "test"
        assert span.parent_id is None
        assert span.status == "running"
        assert span.start_time > 0
        assert span.end_time is None
        assert span.attributes == {}
        assert span.events == []

    def test_duration_ms_not_ended(self):
        span = Span(span_id="s1", name="test")
        assert span.duration_ms is None

    def test_duration_ms_ended(self):
        span = Span(span_id="s1", name="test", start_time=100.0, end_time=100.5)
        assert span.duration_ms == pytest.approx(500.0)


class TestTrace:
    def test_creation(self):
        trace = Trace(trace_id="t1", session_id="s1")
        assert trace.trace_id == "t1"
        assert trace.session_id == "s1"
        assert trace.spans == []
        assert trace.created_at  # auto-generated
        assert trace.metadata == {}


class TestExecutionTracer:
    @pytest.fixture
    def tracer(self, tmp_path):
        return ExecutionTracer(trace_dir=str(tmp_path / "traces"))

    def test_start_trace(self, tracer):
        trace_id = tracer.start_trace("session_1")
        assert trace_id.startswith("trace_")
        assert tracer.get_current_trace_id() == trace_id

    def test_end_trace(self, tracer):
        tracer.start_trace("session_1")
        summary = tracer.end_trace()
        assert summary is not None
        assert summary["session_id"] == "session_1"
        assert summary["total_spans"] == 0
        assert tracer.get_current_trace_id() is None

    def test_end_trace_no_current(self, tracer):
        assert tracer.end_trace() is None

    def test_start_and_end_span(self, tracer):
        tracer.start_trace("s1")
        span = tracer.start_span("operation_1", attributes={"key": "val"})
        assert span.name == "operation_1"
        assert span.attributes == {"key": "val"}

        ended = tracer.end_span(status="completed")
        assert ended.status == "completed"
        assert ended.end_time is not None

    def test_nested_spans(self, tracer):
        tracer.start_trace("s1")

        parent = tracer.start_span("parent")
        child = tracer.start_span("child")
        assert child.parent_id == parent.span_id

        tracer.end_span()  # child
        tracer.end_span()  # parent

        summary = tracer.end_trace()
        assert summary["total_spans"] == 2
        assert summary["completed"] == 2

    def test_span_context_manager(self, tracer):
        tracer.start_trace("s1")

        with tracer.span("context_op", {"detail": "test"}) as s:
            assert s.name == "context_op"

        summary = tracer.end_trace()
        assert summary["completed"] == 1

    def test_span_context_manager_exception(self, tracer):
        tracer.start_trace("s1")

        with pytest.raises(ValueError):
            with tracer.span("failing_op"):
                raise ValueError("boom")

        summary = tracer.end_trace()
        assert summary["failed"] == 1

    def test_add_event(self, tracer):
        tracer.start_trace("s1")
        tracer.start_span("op1")
        tracer.add_event("checkpoint", {"progress": 50})

        current = tracer.get_current_span()
        assert len(current.events) == 1
        assert current.events[0]["name"] == "checkpoint"

    def test_add_event_no_span(self, tracer):
        tracer.start_trace("s1")
        tracer.add_event("orphan")  # should not raise

    def test_set_attribute(self, tracer):
        tracer.start_trace("s1")
        tracer.start_span("op1")
        tracer.set_attribute("result_count", 42)

        current = tracer.get_current_span()
        assert current.attributes["result_count"] == 42

    def test_set_attribute_no_span(self, tracer):
        tracer.start_trace("s1")
        tracer.set_attribute("key", "val")  # should not raise

    def test_end_span_no_stack(self, tracer):
        tracer.start_trace("s1")
        assert tracer.end_span() is None

    def test_end_span_with_error(self, tracer):
        tracer.start_trace("s1")
        tracer.start_span("op1")
        ended = tracer.end_span(status="failed", error="Connection refused")
        assert ended.attributes["error"] == "Connection refused"

    def test_get_current_span(self, tracer):
        tracer.start_trace("s1")
        assert tracer.get_current_span() is None

        tracer.start_span("op1")
        assert tracer.get_current_span().name == "op1"

    def test_save_and_load_trace(self, tracer):
        trace_id = tracer.start_trace("s1")
        tracer.start_span("op1")
        tracer.end_span()
        tracer.end_trace()

        loaded = tracer.load_trace(trace_id)
        assert loaded is not None
        assert loaded["trace_id"] == trace_id
        assert len(loaded["spans"]) == 1

    def test_load_nonexistent_trace(self, tracer):
        assert tracer.load_trace("nonexistent") is None

    def test_list_traces(self, tracer):
        for i in range(3):
            tid = tracer.start_trace(f"session_{i}")
            tracer.start_span(f"op_{i}")
            tracer.end_span()
            tracer.end_trace()

        traces = tracer.list_traces()
        assert len(traces) == 3
        assert all("trace_id" in t for t in traces)

    def test_list_traces_limit(self, tracer):
        for i in range(5):
            tracer.start_trace(f"s{i}")
            tracer.end_trace()

        traces = tracer.list_traces(limit=2)
        assert len(traces) == 2

    def test_get_span_tree(self, tracer):
        tracer.start_trace("s1")

        tracer.start_span("root")
        tracer.start_span("child1")
        tracer.end_span()
        tracer.start_span("child2")
        tracer.end_span()
        tracer.end_span()

        tree = tracer.get_span_tree()
        assert len(tree) == 1
        assert tree[0]["name"] == "root"
        assert len(tree[0]["children"]) == 2

    def test_get_span_tree_no_trace(self, tracer):
        assert tracer.get_span_tree() == []

    def test_format_trace_tree(self, tracer):
        tracer.start_trace("s1")
        tracer.start_span("root")
        tracer.start_span("child")
        tracer.end_span()
        tracer.end_span()

        formatted = tracer.format_trace_tree()
        assert "root" in formatted
        assert "child" in formatted

    def test_incomplete_spans_on_end_trace(self, tracer):
        tracer.start_trace("s1")
        tracer.start_span("incomplete_op")
        # Don't end span - let end_trace clean it up

        summary = tracer.end_trace()
        assert summary["total_spans"] == 1

    def test_trace_with_metadata(self, tracer):
        trace_id = tracer.start_trace("s1", metadata={"version": "1.0"})
        tracer.end_trace()

        loaded = tracer.load_trace(trace_id)
        assert loaded["metadata"]["version"] == "1.0"

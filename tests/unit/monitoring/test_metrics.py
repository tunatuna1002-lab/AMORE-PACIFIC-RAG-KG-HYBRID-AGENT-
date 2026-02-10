"""Tests for src.monitoring.metrics module."""

import pytest

from src.monitoring.metrics import MetricPoint, QualityMetrics


class TestMetricPoint:
    def test_creation(self):
        mp = MetricPoint(name="test", value=1.0)
        assert mp.name == "test"
        assert mp.value == 1.0
        assert mp.timestamp > 0
        assert mp.labels == {}

    def test_creation_with_labels(self):
        mp = MetricPoint(name="test", value=2.0, labels={"env": "prod"})
        assert mp.labels == {"env": "prod"}


class TestQualityMetrics:
    @pytest.fixture
    def metrics(self, tmp_path):
        return QualityMetrics(metrics_dir=str(tmp_path / "metrics"))

    def test_increment(self, metrics):
        metrics.increment("requests")
        metrics.increment("requests")
        metrics.increment("requests", 3)
        assert metrics.get_counter("requests") == 5.0

    def test_increment_with_labels(self, metrics):
        metrics.increment("requests", labels={"method": "GET"})
        metrics.increment("requests", labels={"method": "POST"})
        assert metrics.get_counter("requests", {"method": "GET"}) == 1.0
        assert metrics.get_counter("requests", {"method": "POST"}) == 1.0

    def test_gauge(self, metrics):
        metrics.gauge("cpu_usage", 75.5)
        assert metrics.get_gauge("cpu_usage") == 75.5

        metrics.gauge("cpu_usage", 80.0)
        assert metrics.get_gauge("cpu_usage") == 80.0

    def test_gauge_nonexistent(self, metrics):
        assert metrics.get_gauge("nonexistent") is None

    def test_histogram(self, metrics):
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            metrics.histogram("latency", v)

        stats = metrics.get_histogram_stats("latency")
        assert stats["count"] == 5
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert stats["p50"] == 3.0

    def test_histogram_empty(self, metrics):
        assert metrics.get_histogram_stats("nonexistent") == {}

    def test_histogram_max_size(self, metrics):
        for i in range(1100):
            metrics.histogram("big", float(i))
        stats = metrics.get_histogram_stats("big")
        assert stats["count"] == 1000

    def test_make_key_no_labels(self, metrics):
        assert metrics._make_key("test") == "test"

    def test_make_key_with_labels(self, metrics):
        key = metrics._make_key("test", {"a": "1", "b": "2"})
        assert key == "test{a=1,b=2}"

    def test_timer(self, metrics):
        metrics.start_timer("operation")
        import time

        time.sleep(0.01)
        duration = metrics.stop_timer("operation")
        assert duration is not None
        assert duration > 0

        stats = metrics.get_histogram_stats("operation_seconds")
        assert stats["count"] == 1

    def test_stop_timer_not_started(self, metrics):
        assert metrics.stop_timer("nonexistent") is None

    def test_session_lifecycle(self, metrics):
        metrics.start_session()
        assert metrics._session_start is not None

        summary = metrics.end_session(status="completed")
        assert summary["status"] == "completed"
        assert summary["duration_seconds"] >= 0
        assert metrics.get_counter("sessions_total") == 1.0
        assert metrics.get_counter("sessions_completed") == 1.0

    def test_end_session_not_started(self, metrics):
        assert metrics.end_session() == {}

    def test_agent_metrics(self, metrics):
        metrics.start_session()
        metrics.record_agent_start("crawler_agent")
        metrics.record_agent_complete("crawler_agent", result={"count": 100})

        assert metrics._agent_metrics["crawler_agent"]["status"] == "completed"
        assert metrics._agent_metrics["crawler_agent"]["duration_seconds"] >= 0

    def test_agent_error_metrics(self, metrics):
        metrics.start_session()
        metrics.record_agent_start("crawler_agent")
        metrics.record_agent_error("crawler_agent", "WAF blocked")

        assert metrics._agent_metrics["crawler_agent"]["status"] == "failed"
        assert metrics._agent_metrics["crawler_agent"]["error"] == "WAF blocked"

    def test_record_llm_call(self, metrics):
        metrics.record_llm_call(
            model="gpt-4.1-mini",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=1200.0,
            cost=0.001,
        )
        assert metrics.get_counter("llm_calls_total", {"model": "gpt-4.1-mini"}) == 1.0
        assert metrics.get_counter("llm_prompt_tokens_total", {"model": "gpt-4.1-mini"}) == 100.0

    def test_record_crawl(self, metrics):
        metrics.record_crawl("lip_care", products_count=100, duration_seconds=5.0)
        assert metrics.get_counter("crawls_total", {"category": "lip_care"}) == 1.0
        assert metrics.get_gauge("crawl_products_count", {"category": "lip_care"}) == 100

    def test_record_crawl_failure(self, metrics):
        metrics.record_crawl("lip_care", products_count=0, duration_seconds=30.0, success=False)
        assert metrics.get_counter("crawl_errors_total", {"category": "lip_care"}) == 1.0

    def test_get_all_metrics(self, metrics):
        metrics.increment("test_counter")
        metrics.gauge("test_gauge", 1.0)
        metrics.histogram("test_hist", 1.0)

        all_metrics = metrics.get_all_metrics()
        assert "counters" in all_metrics
        assert "gauges" in all_metrics
        assert "histograms" in all_metrics

    def test_generate_report(self, metrics):
        metrics.start_session()
        metrics.record_agent_start("crawler_agent")
        metrics.record_agent_complete("crawler_agent")
        metrics.record_llm_call("gpt-4.1-mini", 100, 50, 1000.0)
        metrics.end_session()

        report = metrics.generate_report()
        assert "generated_at" in report
        assert "summary" in report
        assert report["summary"]["sessions"]["total"] == 1.0

    def test_save_and_load(self, metrics, tmp_path):
        metrics.increment("saved_counter", 5)
        metrics.gauge("saved_gauge", 42.0)
        metrics.histogram("saved_hist", 1.5)
        metrics.save()

        new_metrics = QualityMetrics(metrics_dir=str(tmp_path / "metrics"))
        assert new_metrics.load() is True
        assert new_metrics.get_counter("saved_counter") == 5.0
        assert new_metrics.get_gauge("saved_gauge") == 42.0

    def test_load_nonexistent(self, metrics):
        assert metrics.load(date="2000-01-01") is False

    def test_load_corrupted(self, tmp_path):
        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir()
        from datetime import datetime

        today = datetime.now().strftime("%Y-%m-%d")
        (metrics_dir / f"metrics_{today}.json").write_text("invalid{json")

        m = QualityMetrics(metrics_dir=str(metrics_dir))
        assert m.load() is False

    def test_reset(self, metrics):
        metrics.increment("counter")
        metrics.gauge("gauge", 1.0)
        metrics.histogram("hist", 1.0)
        metrics.start_timer("timer")
        metrics.start_session()

        metrics.reset()

        assert metrics.get_counter("counter") == 0
        assert metrics.get_gauge("gauge") is None
        assert metrics.get_histogram_stats("hist") == {}
        assert metrics._timers == {}
        assert metrics._session_start is None

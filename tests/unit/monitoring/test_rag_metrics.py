"""Tests for RAGMetricsCollector (4.1)"""

from src.monitoring.rag_metrics import RAGMetricsCollector, RetrievalRecord


class TestRetrievalRecord:
    def test_precision(self):
        r = RetrievalRecord(
            query="test",
            chunks_retrieved=5,
            relevant_count=3,
            total_count=5,
            first_relevant_rank=1,
            retrieval_time_ms=10.0,
        )
        assert r.precision == 0.6

    def test_precision_zero_total(self):
        r = RetrievalRecord(
            query="test",
            chunks_retrieved=0,
            relevant_count=0,
            total_count=0,
            first_relevant_rank=None,
            retrieval_time_ms=0,
        )
        assert r.precision == 0.0

    def test_reciprocal_rank(self):
        r = RetrievalRecord(
            query="test",
            chunks_retrieved=5,
            relevant_count=3,
            total_count=5,
            first_relevant_rank=2,
            retrieval_time_ms=10.0,
        )
        assert r.reciprocal_rank == 0.5

    def test_reciprocal_rank_none(self):
        r = RetrievalRecord(
            query="test",
            chunks_retrieved=5,
            relevant_count=0,
            total_count=5,
            first_relevant_rank=None,
            retrieval_time_ms=10.0,
        )
        assert r.reciprocal_rank == 0.0


class TestRAGMetricsCollector:
    def test_record_and_get_metrics(self):
        collector = RAGMetricsCollector()
        collector.record_retrieval(
            query="test",
            chunks=[{"id": "1"}, {"id": "2"}, {"id": "3"}],
            relevant_chunks=[{"id": "1"}, {"id": "3"}],
            retrieval_time_ms=50.0,
        )
        metrics = collector.get_metrics()
        assert metrics["total_retrievals"] == 1
        assert metrics["records_in_window"] == 1
        assert metrics["avg_chunks_retrieved"] == 3.0
        assert metrics["avg_precision_at_k"] > 0

    def test_empty_metrics(self):
        collector = RAGMetricsCollector()
        metrics = collector.get_metrics()
        assert metrics["total_retrievals"] == 0
        assert metrics["avg_precision_at_k"] == 0.0
        assert metrics["mrr"] == 0.0

    def test_window_size_limit(self):
        collector = RAGMetricsCollector(window_size=3)
        for i in range(5):
            collector.record_retrieval(
                query=f"q{i}", chunks=[{"id": str(i)}], retrieval_time_ms=10.0
            )
        metrics = collector.get_metrics()
        assert metrics["records_in_window"] == 3
        assert metrics["total_retrievals"] == 5

    def test_mrr_calculation(self):
        collector = RAGMetricsCollector()
        # First relevant at position 1
        collector.record_retrieval(
            query="q1",
            chunks=[{"id": "a"}, {"id": "b"}],
            relevant_chunks=[{"id": "a"}],
            retrieval_time_ms=10.0,
        )
        # First relevant at position 2
        collector.record_retrieval(
            query="q2",
            chunks=[{"id": "c"}, {"id": "d"}],
            relevant_chunks=[{"id": "d"}],
            retrieval_time_ms=10.0,
        )
        metrics = collector.get_metrics()
        # MRR = (1/1 + 1/2) / 2 = 0.75
        assert metrics["mrr"] == 0.75

    def test_no_relevant_chunks_defaults(self):
        collector = RAGMetricsCollector()
        collector.record_retrieval(
            query="test",
            chunks=[{"id": "1"}, {"id": "2"}],
            retrieval_time_ms=5.0,
        )
        metrics = collector.get_metrics()
        # Without relevant_chunks, all chunks counted as relevant
        assert metrics["avg_precision_at_k"] == 1.0

    def test_reset(self):
        collector = RAGMetricsCollector()
        collector.record_retrieval(query="test", chunks=[{"id": "1"}], retrieval_time_ms=5.0)
        collector.reset()
        metrics = collector.get_metrics()
        assert metrics["total_retrievals"] == 0
        assert metrics["records_in_window"] == 0

    def test_recent_queries(self):
        collector = RAGMetricsCollector()
        for i in range(3):
            collector.record_retrieval(query=f"query_{i}", chunks=[], retrieval_time_ms=0)
        metrics = collector.get_metrics()
        assert len(metrics["recent_queries"]) == 3

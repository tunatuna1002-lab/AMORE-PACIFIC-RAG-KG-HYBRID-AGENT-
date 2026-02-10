"""Tests for QueryRouter (3.2)"""

from unittest.mock import AsyncMock

import pytest

from src.core.query_router import QueryCategory, QueryRouter, RouteResult, SubQuery


class TestQueryClassification:
    def test_classify_metric(self):
        router = QueryRouter()
        assert router.classify("LANEIGE 점유율 알려줘") == QueryCategory.METRIC

    def test_classify_trend(self):
        router = QueryRouter()
        assert router.classify("최근 순위 변화 추이") == QueryCategory.TREND

    def test_classify_competitive(self):
        router = QueryRouter()
        assert router.classify("경쟁사 비교 분석") == QueryCategory.COMPETITIVE

    def test_classify_diagnostic(self):
        router = QueryRouter()
        assert router.classify("왜 순위가 떨어졌나") == QueryCategory.DIAGNOSTIC

    def test_classify_general(self):
        router = QueryRouter()
        assert router.classify("안녕하세요") == QueryCategory.GENERAL


class TestCompoundDetection:
    def test_compound_with_and(self):
        router = QueryRouter()
        assert router.is_compound("LANEIGE 점유율과 경쟁사 비교 분석") is True

    def test_compound_with_grigo(self):
        router = QueryRouter()
        assert router.is_compound("순위 분석 그리고 트렌드 확인") is True

    def test_simple_query(self):
        router = QueryRouter()
        assert router.is_compound("LANEIGE 순위 알려줘") is False


class TestDecomposition:
    def test_decompose_compound(self):
        router = QueryRouter()
        subs = router.decompose("LANEIGE 점유율과 경쟁사 비교 분석")
        assert len(subs) >= 1
        assert all(isinstance(sq, SubQuery) for sq in subs)

    def test_decompose_simple_returns_original(self):
        router = QueryRouter()
        subs = router.decompose("LANEIGE 순위")
        assert len(subs) == 1
        assert subs[0].query == "LANEIGE 순위"


class TestRouting:
    def test_route_simple(self):
        router = QueryRouter()
        result = router.route("LANEIGE 순위 알려줘")
        assert isinstance(result, RouteResult)
        assert not result.is_compound

    def test_route_compound(self):
        router = QueryRouter()
        result = router.route("LANEIGE 점유율과 경쟁사 비교 분석")
        assert result.is_compound
        assert len(result.sub_queries) >= 1


class TestDispatchAndSynthesize:
    @pytest.mark.asyncio
    async def test_dispatch_parallel(self):
        router = QueryRouter()
        sub_queries = [
            SubQuery(query="A", category=QueryCategory.METRIC),
            SubQuery(query="B", category=QueryCategory.TREND),
        ]
        handler = AsyncMock(side_effect=["result_A", "result_B"])
        results = await router.dispatch_parallel(sub_queries, handler)
        assert len(results) == 2
        assert results[0]["success"]
        assert results[1]["success"]

    @pytest.mark.asyncio
    async def test_dispatch_with_error(self):
        router = QueryRouter()
        sub_queries = [SubQuery(query="A", category=QueryCategory.METRIC)]
        handler = AsyncMock(side_effect=Exception("fail"))
        results = await router.dispatch_parallel(sub_queries, handler)
        assert not results[0]["success"]

    def test_synthesize(self):
        router = QueryRouter()
        results = [
            {"sub_query": "Q1", "category": "metric", "result": "Answer1", "success": True},
            {"sub_query": "Q2", "category": "trend", "result": "Answer2", "success": True},
        ]
        text = router.synthesize("Original Q", results)
        assert "Q1" in text
        assert "Q2" in text

    def test_synthesize_all_failed(self):
        router = QueryRouter()
        results = [{"sub_query": "Q", "category": "metric", "result": "err", "success": False}]
        text = router.synthesize("Q", results)
        assert "오류" in text


class TestStats:
    def test_stats_tracking(self):
        router = QueryRouter()
        router.route("simple query")
        router.route("LANEIGE 점유율과 경쟁사 비교 분석")
        stats = router.get_stats()
        assert stats["total_routes"] == 2

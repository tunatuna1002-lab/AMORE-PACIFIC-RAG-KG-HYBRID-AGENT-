"""Tests for QueryRouter (F2: is_compound only)"""

from src.core.query_router import MAX_QUERY_LENGTH, QueryCategory, QueryRouter


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


class TestReDoSDefense:
    """Task #6: ReDoS defense via MAX_QUERY_LENGTH (5000 chars)."""

    def test_long_query_is_not_compound(self):
        """Queries exceeding MAX_QUERY_LENGTH short-circuit before any regex runs."""
        assert QueryRouter().is_compound("A와 B 비교 " * 1000) is False

    def test_query_at_limit_passes_through(self):
        """Query exactly at MAX_QUERY_LENGTH is still matched normally."""
        suffix = "점유율과 경쟁사 비교"
        query = ("가" * (MAX_QUERY_LENGTH - len(suffix))) + suffix
        assert len(query) == MAX_QUERY_LENGTH
        assert QueryRouter().is_compound(query) is True


class TestDeadMethodsRemoved:
    def test_only_is_compound_remains(self):
        for dead in ("classify", "decompose", "route", "dispatch_parallel", "synthesize"):
            assert not hasattr(QueryRouter, dead), dead

    def test_query_category_values_match_intent_mapping(self):
        assert {c.value for c in QueryCategory} == {
            "metric",
            "trend",
            "competitive",
            "diagnostic",
            "general",
        }

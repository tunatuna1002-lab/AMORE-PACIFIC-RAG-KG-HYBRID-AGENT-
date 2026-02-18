"""
Query Router - 멀티 에이전트 쿼리 라우터
==========================================
복합 쿼리를 분류 → 병렬 서브쿼리 디스패치 → 결과 합성

LangGraph Multi-Agent Router 패턴의 경량 구현.

단일 카테고리 쿼리는 기존 경로를 그대로 유지합니다 (오버헤드 없음).
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class QueryCategory(Enum):
    """쿼리 카테고리"""

    METRIC = "metric"  # 지표 조회/해석 (SoS, HHI, 순위)
    TREND = "trend"  # 트렌드/추이 분석
    COMPETITIVE = "competitive"  # 경쟁사 비교/분석
    DIAGNOSTIC = "diagnostic"  # 원인 분석/진단
    GENERAL = "general"  # 일반 질문


@dataclass
class SubQuery:
    """분해된 서브쿼리"""

    query: str
    category: QueryCategory
    priority: int = 0  # 낮을수록 우선
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteResult:
    """라우팅 결과"""

    original_query: str
    category: QueryCategory
    is_compound: bool = False
    sub_queries: list[SubQuery] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


MAX_QUERY_LENGTH = 5000


class QueryRouter:
    """
    쿼리 분류 및 라우팅

    Usage:
        router = QueryRouter()
        result = router.route("LANEIGE 점유율과 경쟁사 비교 분석")
        if result.is_compound:
            # 서브쿼리 병렬 처리
            ...
    """

    # 복합 쿼리 감지 패턴
    COMPOUND_PATTERNS = [
        r"(.+?)(?:와|과|하고|및|그리고)\s*(.+?)(?:\s*(?:비교|분석|알려|보여))",
        r"(.+?)(?:점유율|순위).*(?:경쟁|비교|대비)",
        r"(.+?)\s*(?:그리고|또한|더불어)\s*(.+)",
    ]

    def __init__(self):
        self._stats = {"total_routes": 0, "compound_queries": 0}

    def classify(self, query: str) -> QueryCategory:
        """쿼리 카테고리 분류 - delegates to unified classifier."""
        from src.core.intent import classify_intent as _unified_classify
        from src.core.intent import to_query_category as _to_query_category

        unified = _unified_classify(query)
        cat_value = _to_query_category(unified)
        try:
            return QueryCategory(cat_value)
        except ValueError:
            return QueryCategory.GENERAL

    def is_compound(self, query: str) -> bool:
        """복합 쿼리 여부 판단"""
        for pattern in self.COMPOUND_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False

    def decompose(self, query: str) -> list[SubQuery]:
        """복합 쿼리를 서브쿼리로 분해"""
        sub_queries = []

        for pattern in self.COMPOUND_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                groups = match.groups()
                for i, part in enumerate(groups):
                    part = part.strip()
                    if part and len(part) > 2:
                        category = self.classify(part)
                        sub_queries.append(
                            SubQuery(
                                query=part,
                                category=category,
                                priority=i,
                            )
                        )
                break

        # 분해 실패 시 원본 반환
        if not sub_queries:
            sub_queries.append(
                SubQuery(
                    query=query,
                    category=self.classify(query),
                    priority=0,
                )
            )

        return sub_queries

    def route(self, query: str) -> RouteResult:
        """쿼리 라우팅"""
        self._stats["total_routes"] += 1

        # ReDoS 방어: 과도하게 긴 쿼리는 regex 전에 차단
        if len(query) > MAX_QUERY_LENGTH:
            logger.warning(
                "Query exceeds MAX_QUERY_LENGTH (%d > %d), returning GENERAL",
                len(query),
                MAX_QUERY_LENGTH,
            )
            return RouteResult(
                original_query=query[:MAX_QUERY_LENGTH],
                category=QueryCategory.GENERAL,
            )

        category = self.classify(query)
        compound = self.is_compound(query)

        result = RouteResult(
            original_query=query,
            category=category,
            is_compound=compound,
        )

        if compound:
            self._stats["compound_queries"] += 1
            result.sub_queries = self.decompose(query)
            logger.info(f"Compound query decomposed: {len(result.sub_queries)} sub-queries")

        return result

    async def dispatch_parallel(
        self,
        sub_queries: list[SubQuery],
        handler,  # async callable(query: str) -> Any
    ) -> list[dict[str, Any]]:
        """서브쿼리 병렬 디스패치"""
        tasks = [handler(sq.query) for sq in sub_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [
            {
                "sub_query": sq.query,
                "category": sq.category.value,
                "result": r if not isinstance(r, Exception) else str(r),
                "success": not isinstance(r, Exception),
            }
            for sq, r in zip(sub_queries, results, strict=False)
        ]

    def synthesize(self, query: str, results: list[dict[str, Any]]) -> str:
        """서브쿼리 결과 합성"""
        successful = [r for r in results if r["success"]]

        if not successful:
            return "서브쿼리 처리 중 오류가 발생했습니다."

        parts = [f"## {query}\n"]
        for r in successful:
            parts.append(f"### {r['sub_query']} ({r['category']})")
            result_text = r["result"]
            if hasattr(result_text, "text"):
                result_text = result_text.text
            elif isinstance(result_text, dict):
                result_text = str(result_text)
            parts.append(str(result_text))
            parts.append("")

        return "\n".join(parts)

    def get_stats(self) -> dict[str, Any]:
        """통계 반환"""
        return self._stats

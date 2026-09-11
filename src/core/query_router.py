"""
Query Router - 복합 쿼리 감지
==============================
``QueryGraph`` 가 MEDIUM/LOW 신뢰도 질의를 ReAct로 보낼지 판단할 때 사용하는
복합 쿼리(두 가지 이상 요청) 감지기.

F2: 호출자가 없던 classify/decompose/route/dispatch_parallel/synthesize 는 삭제됨.
의도 분류는 ``src.core.intent`` 가 단일 출처입니다.
"""

import logging
import re
from enum import Enum

logger = logging.getLogger(__name__)


class QueryCategory(Enum):
    """쿼리 카테고리 (``src.core.intent.to_query_category`` 가 반환하는 값의 집합)"""

    METRIC = "metric"  # 지표 조회/해석 (SoS, HHI, 순위)
    TREND = "trend"  # 트렌드/추이 분석
    COMPETITIVE = "competitive"  # 경쟁사 비교/분석
    DIAGNOSTIC = "diagnostic"  # 원인 분석/진단
    GENERAL = "general"  # 일반 질문


MAX_QUERY_LENGTH = 5000


class QueryRouter:
    """
    복합 쿼리 감지

    Usage:
        router = QueryRouter()
        if router.is_compound("LANEIGE 점유율과 경쟁사 비교 분석"):
            ...
    """

    # 복합 쿼리 감지 패턴
    COMPOUND_PATTERNS = [
        r"(.+?)(?:와|과|하고|및|그리고)\s*(.+?)(?:\s*(?:비교|분석|알려|보여))",
        r"(.+?)(?:점유율|순위).*(?:경쟁|비교|대비)",
        r"(.+?)\s*(?:그리고|또한|더불어)\s*(.+)",
    ]

    def is_compound(self, query: str) -> bool:
        """복합 쿼리 여부 판단"""
        # ReDoS 방어: 과도하게 긴 쿼리는 regex 전에 차단
        if len(query) > MAX_QUERY_LENGTH:
            logger.warning(
                "Query exceeds MAX_QUERY_LENGTH (%d > %d), not treated as compound",
                len(query),
                MAX_QUERY_LENGTH,
            )
            return False
        for pattern in self.COMPOUND_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False

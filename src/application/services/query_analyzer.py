"""
Query Analyzer Service
======================
Analyzes user queries to determine complexity and intent.
Extracted from brain.py for reusability in Application layer.

This service has NO dependencies on infrastructure - pure business logic.
"""

import re
from enum import Enum
from typing import Any


class ComplexityLevel(str, Enum):
    """Query complexity levels"""

    SIMPLE = "simple"  # Direct lookup (rank, metric)
    MODERATE = "moderate"  # Comparison, trend
    COMPLEX = "complex"  # Analysis, multi-step reasoning


class QueryIntent(str, Enum):
    """Query intent types"""

    RANK_QUERY = "rank_query"  # 순위 조회
    METRIC_QUERY = "metric_query"  # 메트릭 조회 (SoS, HHI, CPI)
    COMPARISON = "comparison"  # 브랜드/제품 비교
    TREND_ANALYSIS = "trend_analysis"  # 추세 분석
    RECOMMENDATION = "recommendation"  # 전략 제안
    PRODUCT_DETAIL = "product_detail"  # 제품 상세 정보
    GENERAL = "general"  # 일반 질문


class QueryAnalyzer:
    """
    Query Analyzer Service

    Analyzes user queries to determine:
    1. Complexity level (simple/moderate/complex)
    2. Intent (what the user wants)
    3. Keywords (extracted entities)

    Usage:
        analyzer = QueryAnalyzer()
        result = analyzer.analyze("LANEIGE 경쟁력 분석해줘")
        # {"complexity": "complex", "intent": "recommendation", "keywords": ["LANEIGE", "경쟁력"]}
    """

    # Keywords for complexity detection
    ANALYSIS_KEYWORDS = [
        "분석",
        "평가",
        "진단",
        "검토",
        "분석해",
        "평가해",
        "진단해",
        "분석하고",
        "analyze",
        "evaluate",
        "assess",
    ]

    STRATEGY_KEYWORDS = [
        "전략",
        "제안",
        "추천",
        "방안",
        "계획",
        "제안해",
        "추천해",
        "recommend",
        "strategy",
        "suggest",
        "plan",
    ]

    COMPARISON_KEYWORDS = [
        "비교",
        "차이",
        "대비",
        "vs",
        "versus",
        "compare",
        "비교해",
        "차이점",
    ]

    TREND_KEYWORDS = [
        "추세",
        "변화",
        "트렌드",
        "흐름",
        "변동",
        "변했",
        "trend",
        "change",
        "변화",
    ]

    # Keywords for intent detection
    RANK_KEYWORDS = ["순위", "랭킹", "등수", "rank", "ranking"]

    METRIC_KEYWORDS = [
        "sos",
        "hhi",
        "cpi",
        "점유율",
        "지표",
        "메트릭",
        "metric",
        "집중도",
    ]

    PRODUCT_KEYWORDS = ["제품", "상품", "아이템", "product", "item", "정보", "스펙"]

    def analyze_complexity(self, query: str) -> ComplexityLevel:
        """
        Analyze query complexity.

        Args:
            query: User query string

        Returns:
            ComplexityLevel (SIMPLE/MODERATE/COMPLEX)
        """
        if not query or not query.strip():
            return ComplexityLevel.SIMPLE

        query_lower = query.lower()

        # Complex: Analysis keywords
        if any(keyword in query_lower for keyword in self.ANALYSIS_KEYWORDS):
            return ComplexityLevel.COMPLEX

        # Complex: Strategy/recommendation keywords
        if any(keyword in query_lower for keyword in self.STRATEGY_KEYWORDS):
            return ComplexityLevel.COMPLEX

        # Complex: Multiple questions (contains multiple question separators)
        # Count comma-separated questions or multiple question marks
        question_separators = query.count(",") + query.count("?")
        if question_separators >= 2:
            return ComplexityLevel.COMPLEX

        # Complex: Multi-step reasoning indicators
        multistep_indicators = [" 후 ", " 다음 ", "먼저", "그리고", "이어서", " and ", " then "]
        if any(indicator in query_lower for indicator in multistep_indicators):
            return ComplexityLevel.COMPLEX

        # Moderate: Comparison
        if any(keyword in query_lower for keyword in self.COMPARISON_KEYWORDS):
            return ComplexityLevel.MODERATE

        # Moderate: Trend analysis
        if any(keyword in query_lower for keyword in self.TREND_KEYWORDS):
            return ComplexityLevel.MODERATE

        # Moderate: Long query (>50 chars)
        if len(query.strip()) > 50:
            return ComplexityLevel.MODERATE

        # Default: Simple
        return ComplexityLevel.SIMPLE

    def detect_intent(self, query: str) -> QueryIntent:
        """
        Detect query intent.

        Args:
            query: User query string

        Returns:
            QueryIntent enum value
        """
        if not query or not query.strip():
            return QueryIntent.GENERAL

        query_lower = query.lower()

        # Priority 1: Recommendation (highest priority for complex queries)
        if any(keyword in query_lower for keyword in self.STRATEGY_KEYWORDS):
            return QueryIntent.RECOMMENDATION

        # Priority 2: Comparison
        if any(keyword in query_lower for keyword in self.COMPARISON_KEYWORDS):
            return QueryIntent.COMPARISON

        # Priority 3: Trend Analysis
        if any(keyword in query_lower for keyword in self.TREND_KEYWORDS):
            return QueryIntent.TREND_ANALYSIS

        # Priority 4: Metric Query
        if any(keyword in query_lower for keyword in self.METRIC_KEYWORDS):
            return QueryIntent.METRIC_QUERY

        # Priority 5: Rank Query
        if any(keyword in query_lower for keyword in self.RANK_KEYWORDS):
            return QueryIntent.RANK_QUERY

        # Priority 6: Product Detail
        if any(keyword in query_lower for keyword in self.PRODUCT_KEYWORDS):
            return QueryIntent.PRODUCT_DETAIL

        # Default: General
        return QueryIntent.GENERAL

    def extract_keywords(self, query: str) -> list[str]:
        """
        Extract key entities/keywords from query.

        Args:
            query: User query string

        Returns:
            List of extracted keywords
        """
        if not query or not query.strip():
            return []

        keywords = []

        # Extract brand names (capitalized words or known brands)
        brand_pattern = r"\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b"
        brands = re.findall(brand_pattern, query)
        keywords.extend(brands)

        # Extract metric names
        for keyword in self.METRIC_KEYWORDS:
            if keyword.upper() in query.upper():
                keywords.append(keyword.upper())

        # Extract Korean keywords (nouns-like patterns)
        korean_pattern = r"[가-힣]{2,}"
        korean_words = re.findall(korean_pattern, query)
        keywords.extend([w for w in korean_words if len(w) >= 2])

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords[:10]  # Limit to 10 keywords

    def analyze(self, query: str) -> dict[str, Any]:
        """
        Full query analysis.

        Args:
            query: User query string

        Returns:
            Analysis result dictionary
            {
                "complexity": ComplexityLevel,
                "intent": QueryIntent,
                "keywords": list[str],
                "query": str
            }
        """
        return {
            "query": query,
            "complexity": self.analyze_complexity(query),
            "intent": self.detect_intent(query),
            "keywords": self.extract_keywords(query),
        }

"""Query expansion
===============

Everything that rewrites or widens a query *before* it hits the index:

* :class:`QueryEnhancer` / :class:`EnhancedQuery` — dictionary-based domain
  synonym expansion and rule-based decomposition of compound questions
  (absorbed from the former ``src/rag/query_enhancer.py``).
* :func:`expand_query` — appends interpretation keywords derived from the
  ontology inferences and the extracted indicators.
* :func:`rewrite_for_relevance` — the one-shot retry rewrite used when
  relevance grading finds too few relevant documents.

LLM-driven expansion is a different thing and stays where it is used:
``DocumentRetriever.expand_query`` (bilingual paraphrases) and
``src/rag/query_rewriter.py`` (conversational deixis).

Moved out of ``hybrid_retriever.py`` / ``query_enhancer.py`` (F3 split).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from src.domain.entities.relations import InsightType

if TYPE_CHECKING:
    from src.domain.entities.relations import InferenceResult

logger = logging.getLogger(__name__)


# 도메인 동의어 사전 (LLM 불필요)
DOMAIN_SYNONYMS: dict[str, list[str]] = {
    # 지표
    "점유율": ["SoS", "Share of Shelf", "market share", "셸프 점유율"],
    "sos": ["Share of Shelf", "점유율", "market share"],
    "집중도": ["HHI", "Herfindahl", "시장집중도", "허핀달"],
    "hhi": ["Herfindahl", "시장집중도", "집중도"],
    "가격지수": ["CPI", "Competitive Price Index", "경쟁가격지수"],
    "cpi": ["Competitive Price Index", "가격지수", "경쟁가격지수"],
    # 카테고리
    "립케어": ["Lip Care", "lip balm", "lip mask", "립밤", "립마스크"],
    "립메이크업": ["Lip Makeup", "lipstick", "lip gloss", "립스틱", "립글로스"],
    "스킨케어": ["Skin Care", "skincare", "스킨케어"],
    "파우더": ["Face Powder", "powder", "파운데이션 파우더"],
    # 브랜드
    "라네즈": ["LANEIGE", "laneige"],
    "laneige": ["라네즈", "LANEIGE"],
    # 제품
    "립슬리핑마스크": ["Lip Sleeping Mask", "LSM", "립 슬리핑 마스크"],
    # 분석 용어
    "경쟁사": ["competitor", "경쟁 브랜드", "rival brand"],
    "트렌드": ["trend", "추이", "동향", "변화"],
    "순위": ["rank", "ranking", "BSR", "Best Seller Rank"],
}


class QueryEnhancer:
    """
    사전 검색 쿼리 강화

    도메인 동의어 확장 및 복합 질문 분해를 통해
    검색 품질을 향상시킵니다.

    Usage:
        enhancer = QueryEnhancer()
        enhanced = enhancer.enhance(query, entities)
        # enhanced.expanded_query: 동의어 확장된 쿼리
        # enhanced.sub_queries: 분해된 서브쿼리 (복합 질문인 경우)
    """

    def __init__(
        self,
        synonyms: dict[str, list[str]] | None = None,
        max_expansions: int = 3,
    ):
        """
        Args:
            synonyms: 커스텀 동의어 사전 (None이면 기본 사전 사용)
            max_expansions: 쿼리당 최대 동의어 확장 수
        """
        self.synonyms = synonyms or DOMAIN_SYNONYMS
        self.max_expansions = max_expansions
        self._stats = {"queries_enhanced": 0, "synonyms_added": 0, "queries_decomposed": 0}

    def enhance(
        self,
        query: str,
        entities: dict[str, list[str]] | None = None,
    ) -> EnhancedQuery:
        """
        쿼리 강화 (메인 메서드)

        Args:
            query: 원본 쿼리
            entities: 추출된 엔티티 (있으면 활용)

        Returns:
            EnhancedQuery 객체
        """
        self._stats["queries_enhanced"] += 1

        # 1. 동의어 확장
        expanded_query, added_synonyms = self._expand_synonyms(query)

        # 2. 복합 질문 분해 (간단한 규칙 기반)
        sub_queries = self._decompose_if_complex(query, entities)

        return EnhancedQuery(
            original_query=query,
            expanded_query=expanded_query,
            sub_queries=sub_queries,
            added_synonyms=added_synonyms,
        )

    def _expand_synonyms(self, query: str) -> tuple[str, list[str]]:
        """
        도메인 동의어 확장

        쿼리에 포함된 키워드의 동의어를 추가합니다.
        LLM 호출 없이 딕셔너리 기반으로 수행합니다.

        Args:
            query: 원본 쿼리

        Returns:
            (확장된 쿼리, 추가된 동의어 리스트)
        """
        query_lower = query.lower()
        additions = []
        count = 0

        for keyword, synonyms in self.synonyms.items():
            if count >= self.max_expansions:
                break

            if keyword.lower() in query_lower:
                # 이미 쿼리에 포함되지 않은 동의어만 추가
                for syn in synonyms[:2]:  # 키워드당 최대 2개
                    if syn.lower() not in query_lower:
                        additions.append(syn)
                        count += 1
                        if count >= self.max_expansions:
                            break

        if additions:
            self._stats["synonyms_added"] += len(additions)
            expanded = f"{query} ({', '.join(additions)})"
            logger.debug(f"Query expanded: '{query}' → '{expanded}'")
            return expanded, additions

        return query, []

    def _decompose_if_complex(
        self,
        query: str,
        entities: dict[str, list[str]] | None = None,
    ) -> list[str]:
        """
        복합 질문 분해

        "A와 B를 비교해줘" 같은 복합 질문을 서브쿼리로 분해합니다.
        규칙 기반으로 수행 (LLM 호출 최소화).

        Args:
            query: 원본 쿼리
            entities: 추출된 엔티티

        Returns:
            서브쿼리 리스트 (단일 질문이면 빈 리스트)
        """
        # 복합 질문 패턴
        comparison_patterns = [
            r"(.+?)(?:와|과|하고)\s+(.+?)(?:를?\s*)?비교",
            r"(.+?)(?:vs|VS|versus)\s+(.+)",
        ]

        for pattern in comparison_patterns:
            match = re.search(pattern, query)
            if match:
                part1 = match.group(1).strip()
                part2 = match.group(2).strip()
                self._stats["queries_decomposed"] += 1
                return [
                    f"{part1} 현황 분석",
                    f"{part2} 현황 분석",
                ]

        # "A점유율과 B추이" 패턴
        and_patterns = [
            r"(.+?)(?:와|과|하고|,)\s+(.+?)(?:알려|보여|분석)",
        ]

        for pattern in and_patterns:
            match = re.search(pattern, query)
            if match:
                part1 = match.group(1).strip()
                part2 = match.group(2).strip()
                if len(part1) > 3 and len(part2) > 3:
                    self._stats["queries_decomposed"] += 1
                    return [part1, part2]

        return []

    def get_stats(self) -> dict[str, Any]:
        """통계 반환"""
        return {**self._stats}


class EnhancedQuery:
    """
    강화된 쿼리 결과

    Attributes:
        original_query: 원본 쿼리
        expanded_query: 동의어 확장된 쿼리
        sub_queries: 분해된 서브쿼리 (복합 질문인 경우)
        added_synonyms: 추가된 동의어 목록
    """

    def __init__(
        self,
        original_query: str,
        expanded_query: str,
        sub_queries: list[str] | None = None,
        added_synonyms: list[str] | None = None,
    ):
        self.original_query = original_query
        self.expanded_query = expanded_query
        self.sub_queries = sub_queries or []
        self.added_synonyms = added_synonyms or []

    @property
    def is_complex(self) -> bool:
        """복합 질문 여부"""
        return len(self.sub_queries) > 0

    @property
    def search_query(self) -> str:
        """검색에 사용할 최종 쿼리"""
        return self.expanded_query

    def __repr__(self) -> str:
        return (
            f"EnhancedQuery(original='{self.original_query}', "
            f"expanded='{self.expanded_query}', "
            f"sub_queries={self.sub_queries})"
        )


# ---------------------------------------------------------------------------
# Inference-driven expansion
# ---------------------------------------------------------------------------

# 추론된 인사이트 유형 → 검색에 덧붙일 해석 키워드
_INSIGHT_EXPANSIONS: list[tuple[set[InsightType], str]] = [
    ({InsightType.MARKET_POSITION, InsightType.MARKET_DOMINANCE}, "시장 포지션 해석"),
    ({InsightType.RISK_ALERT}, "위험 신호 대응"),
    ({InsightType.COMPETITIVE_THREAT}, "경쟁 위협 분석"),
    ({InsightType.GROWTH_OPPORTUNITY, InsightType.GROWTH_MOMENTUM}, "성장 기회 전략"),
    ({InsightType.PRICE_QUALITY_GAP, InsightType.PRICE_POSITION}, "가격 전략 해석"),
]

# 지표 → 검색에 덧붙일 해석 키워드
_INDICATOR_EXPANSIONS: dict[str, str] = {
    "sos": "SoS 점유율 해석",
    "hhi": "HHI 시장집중도 해석",
    "cpi": "CPI 가격지수 해석",
}

# 재작성 시 사용할 지표·카테고리의 전체 이름
_INDICATOR_FULL_NAMES: dict[str, str] = {
    "sos": "Share of Shelf 점유율",
    "hhi": "HHI 시장집중도",
    "cpi": "CPI 가격지수",
}

_CATEGORY_FULL_NAMES: dict[str, str] = {
    "lip_care": "Lip Care 립케어",
    "lip_makeup": "Lip Makeup 립메이크업",
    "face_powder": "Face Powder 파우더",
}


def expand_query(
    query: str,
    inferences: list[InferenceResult],
    entities: dict[str, list[str]],
) -> str:
    """추론 결과 기반 쿼리 확장.

    Args:
        query: 원본 쿼리
        inferences: 추론 결과
        entities: 엔티티

    Returns:
        확장된 쿼리 (덧붙일 것이 없으면 원본 그대로)
    """
    expansion_terms: list[str] = []

    insight_types = {inf.insight_type for inf in inferences}
    for trigger_types, term in _INSIGHT_EXPANSIONS:
        if insight_types & trigger_types:
            expansion_terms.append(term)

    for indicator in entities.get("indicators", []):
        term = _INDICATOR_EXPANSIONS.get(indicator)
        if term:
            expansion_terms.append(term)

    if expansion_terms:
        return f"{query} {' '.join(expansion_terms)}"
    return query


def rewrite_for_relevance(query: str, entities: dict) -> str:
    """관련성 부족 시 쿼리 재작성.

    엔티티 정보를 활용하여 더 구체적인 검색 쿼리를 생성합니다.

    Args:
        query: 원본 쿼리
        entities: 추출된 엔티티

    Returns:
        재작성된 쿼리
    """
    parts = [query]

    # 브랜드 추가
    brands = entities.get("brands", [])
    if brands and brands[0].lower() not in query.lower():
        parts.append(brands[0])

    # 지표 추가
    for ind in entities.get("indicators", [])[:2]:
        full_name = _INDICATOR_FULL_NAMES.get(ind, ind)
        if full_name.lower() not in query.lower():
            parts.append(full_name)

    # 카테고리 추가
    for cat in entities.get("categories", [])[:1]:
        full_name = _CATEGORY_FULL_NAMES.get(cat, cat)
        if full_name.lower() not in query.lower():
            parts.append(full_name)

    rewritten = " ".join(parts)
    if rewritten != query:
        logger.info(f"Query rewritten for relevance: '{query}' → '{rewritten}'")
    return rewritten

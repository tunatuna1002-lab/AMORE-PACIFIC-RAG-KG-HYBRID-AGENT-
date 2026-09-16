"""
Follow-up suggestions
=====================
Dynamic follow-up question suggestions for the v1 chat route.
"""

from __future__ import annotations

import re

from src.rag.router import QueryType


def generate_dynamic_suggestions(
    query_type: QueryType, entities: dict, response: str, user_query: str = ""
) -> list[str]:
    """동적 후속 질문 제안 (v2 - 개선 버전)"""
    suggestions = []

    # 엔티티 추출
    brands = entities.get("brands", [])
    indicators = entities.get("indicators", [])
    categories = entities.get("categories", [])

    # 1순위: 응답 키워드 기반 제안
    if response:
        keyword_suggestions = _extract_response_keywords(response)
        suggestions.extend(keyword_suggestions)

    # 2순위: 엔티티 기반 제안
    if len(suggestions) < 3:
        entity_suggestions = _generate_entity_suggestions(brands, categories, indicators)
        suggestions.extend(entity_suggestions)

    # 3순위: 쿼리 유형 기반 제안 (폴백)
    if len(suggestions) < 3:
        type_suggestions = _generate_type_suggestions(query_type, brands, indicators)
        suggestions.extend(type_suggestions)

    # 중복 제거 및 상위 3개
    unique = list(dict.fromkeys(suggestions))
    return unique[:3]


def _extract_response_keywords(response: str) -> list[str]:
    """응답에서 후속 질문 관련 키워드 추출"""
    keywords = []

    patterns = {
        r"순위.{0,10}(하락|급락|떨어)": "순위 하락 원인을 분석해주세요",
        r"순위.{0,10}(상승|급등|올라)": "상승 요인을 상세 분석해주세요",
        r"경쟁사|경쟁 브랜드|competitor": "경쟁사 상세 비교를 해주세요",
        r"가격.{0,10}(인상|인하|변동)": "가격 전략을 분석해주세요",
        r"리뷰|평점|rating": "소비자 피드백을 상세 분석해주세요",
        r"트렌드|유행|trend": "트렌드 상세 분석을 해주세요",
        r"성장.{0,5}(기회|가능|potential)": "성장 전략을 제안해주세요",
        r"위험|리스크|위협|risk": "리스크 대응 전략은?",
        r"SoS|점유율|share": "점유율 개선 전략은?",
        r"Top.{0,3}(10|5)|상위": "Top 10 진입 전략은?",
    }

    for pattern, suggestion in patterns.items():
        if re.search(pattern, response, re.IGNORECASE):
            keywords.append(suggestion)
            if len(keywords) >= 2:
                break

    return keywords


def _generate_entity_suggestions(
    brands: list[str], categories: list[str], indicators: list[str]
) -> list[str]:
    """엔티티 기반 동적 제안 생성"""
    suggestions = []

    if brands:
        brand = brands[0]
        suggestions.append(f"{brand} 경쟁사 비교 분석")
        if len(brands) > 1:
            suggestions.append(f"{brands[0]} vs {brands[1]} 비교")

    if categories:
        cat = categories[0]
        suggestions.append(f"{cat} 시장 트렌드 분석")

    if indicators:
        ind = indicators[0].upper()
        suggestions.append(f"{ind} 개선 전략")

    return suggestions


def _generate_type_suggestions(
    query_type: QueryType, brands: list[str], indicators: list[str]
) -> list[str]:
    """쿼리 유형 기반 폴백 제안"""
    suggestions = []

    if query_type == QueryType.DEFINITION:
        if indicators:
            ind = indicators[0].upper()
            suggestions.append(f"{ind}가 높으면 어떤 의미인가요?")
        suggestions.extend(["관련된 다른 지표는?", "실제 데이터에 적용해주세요"])

    elif query_type == QueryType.INTERPRETATION:
        suggestions.extend(
            ["현재 LANEIGE 수치 분석", "경쟁사와 비교해주세요", "개선 액션 아이템은?"]
        )

    elif query_type == QueryType.DATA_QUERY:
        suggestions.extend(["이 수치가 좋은 건가요?", "최근 7일 추이 분석", "경쟁사 대비 현황"])

    elif query_type == QueryType.ANALYSIS:
        suggestions.extend(["가장 시급한 액션은?", "Top 10 진입 전략", "리스크 요인 분석"])

    elif query_type == QueryType.COMBINATION:
        suggestions.extend(["다른 시나리오 분석", "현재 해당 상황 존재 여부"])

    else:
        suggestions = ["SoS(점유율) 설명해주세요", "LANEIGE 현재 순위는?", "전략적 권고사항"]

    return suggestions

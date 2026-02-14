"""
Suggestion Engine
후속 질문 제안 생성 엔진

4-tier priority system:
1. Keyword-based suggestions (response 분석)
2. Entity-based suggestions (KG 경쟁사 활용)
3. Inference-based suggestions (추론 결과 기반)
4. Type-based suggestions (쿼리 유형 기반 폴백)
"""

import re
from typing import Any

from src.domain.entities.relations import InferenceResult
from src.rag.router import QueryType


class SuggestionEngine:
    """Generates follow-up question suggestions based on context.

    Uses a 4-tier priority system:
    1. Keyword-based suggestions
    2. Entity-based suggestions
    3. Inference-based suggestions
    4. Type-based suggestions (fallback)
    """

    def __init__(self, knowledge_graph=None, config: dict[str, Any] | None = None):
        """
        Args:
            knowledge_graph: 지식 그래프 (KG 경쟁사 조회용)
            config: 설정 딕셔너리
        """
        self.knowledge_graph = knowledge_graph
        self.config = config or {}

    def generate(
        self,
        query_type: QueryType,
        entities: dict[str, list[str]],
        inferences: list[InferenceResult],
        response: str = "",
    ) -> list[str]:
        """Generate follow-up suggestions using 4-tier priority.

        Args:
            query_type: 질문 유형
            entities: 추출된 엔티티
            inferences: 온톨로지 추론 결과
            response: AI 응답 내용 (키워드 분석용)

        Returns:
            3개의 후속 질문 리스트
        """
        from src.shared.constants import SUGGESTION_MAX_COUNT

        suggestions = []

        # 1순위: 응답 키워드 기반 제안
        if response:
            keyword_suggestions = self._extract_response_keywords(response)
            suggestions.extend(keyword_suggestions)

        # 2순위: 엔티티 기반 제안 (KG 경쟁사 활용)
        if len(suggestions) < SUGGESTION_MAX_COUNT:
            entity_suggestions = self._generate_entity_suggestions(entities)
            suggestions.extend(entity_suggestions)

        # 3순위: 추론 결과 기반 제안
        if len(suggestions) < SUGGESTION_MAX_COUNT and inferences:
            inference_suggestions = self._generate_inference_suggestions(inferences)
            suggestions.extend(inference_suggestions)

        # 4순위: 쿼리 유형 기반 제안 (폴백)
        if len(suggestions) < SUGGESTION_MAX_COUNT:
            type_suggestions = self._generate_type_suggestions(query_type, entities)
            suggestions.extend(type_suggestions)

        # 중복 제거 및 상위 3개
        unique_suggestions = list(dict.fromkeys(suggestions))
        return unique_suggestions[:SUGGESTION_MAX_COUNT]

    def _extract_response_keywords(self, response: str) -> list[str]:
        """응답에서 후속 질문 관련 키워드 추출 (Phase 3)"""
        keywords = []

        # 패턴 매칭 - 응답 내용에 따라 관련 후속 질문 생성
        patterns = {
            r"순위.{0,10}(하락|급락|떨어)": "순위 하락 원인 분석",
            r"순위.{0,10}(상승|급등|올라)": "상승 요인 상세 분석",
            r"경쟁사|경쟁 브랜드|competitor": "경쟁사 상세 비교",
            r"가격.{0,10}(인상|인하|변동)": "가격 전략 분석",
            r"리뷰|평점|rating": "소비자 피드백 상세 분석",
            r"트렌드|유행|trend": "트렌드 상세 분석",
            r"성장.{0,5}(기회|가능|potential)": "성장 전략 제안",
            r"위험|리스크|위협|risk": "리스크 대응 전략은?",
            r"SoS|점유율|share": "점유율 개선 전략은?",
            r"Top.{0,3}(10|5)|상위": "Top 10 진입 전략은?",
        }

        for pattern, suggestion in patterns.items():
            if re.search(pattern, response, re.IGNORECASE):
                keywords.append(suggestion)
                if len(keywords) >= 2:  # 최대 2개
                    break

        return keywords

    def _generate_entity_suggestions(self, entities: dict[str, list[str]]) -> list[str]:
        """엔티티 기반 동적 제안 생성 (Phase 2 - KG 경쟁사 활용)"""
        suggestions = []

        brands = entities.get("brands", [])
        categories = entities.get("categories", [])
        indicators = entities.get("indicators", [])

        # 브랜드 기반 (KG에서 경쟁사 조회)
        if brands:
            brand = brands[0]
            # KG에서 경쟁사 조회 시도
            if self.knowledge_graph:
                try:
                    competitors = self.knowledge_graph.get_related_brands(brand, limit=2)
                    if competitors:
                        comp = (
                            competitors[0]
                            if isinstance(competitors[0], str)
                            else competitors[0].get("name", "")
                        )
                        if comp:
                            suggestions.append(f"{brand} vs {comp} 비교 분석")
                except Exception:
                    pass  # KG 없으면 스킵

            suggestions.append(f"{brand} 제품별 성과 분석")

            # 다중 브랜드 비교
            if len(brands) > 1:
                suggestions.append(f"{brands[0]} vs {brands[1]} 비교")

        # 카테고리 기반
        if categories:
            cat = categories[0]
            suggestions.append(f"{cat} 시장 트렌드 분석")
            suggestions.append(f"{cat} Top 5 브랜드 현황")

        # 지표 기반
        if indicators:
            ind = indicators[0].upper()
            suggestions.append(f"{ind} 개선 전략")
            suggestions.append(f"{ind} 경쟁사 비교")

        return suggestions

    def _generate_inference_suggestions(self, inferences: list[InferenceResult]) -> list[str]:
        """추론 결과 기반 제안"""
        suggestions = []

        for inf in inferences[:2]:
            insight_lower = inf.insight.lower()
            insight_type_val = (
                inf.insight_type.value
                if hasattr(inf.insight_type, "value")
                else str(inf.insight_type)
            )

            if "경쟁" in insight_lower or "COMPETITIVE" in insight_type_val:
                suggestions.append("주요 경쟁사 분석")
            if "가격" in insight_lower or "PRICE" in insight_type_val:
                suggestions.append("가격 전략 상세 분석")
            if "성장" in insight_lower or "GROWTH" in insight_type_val:
                suggestions.append("성장 기회 구체화")
            if inf.recommendation:
                # 권장 액션이 있으면 관련 질문
                suggestions.append(f"'{inf.recommendation}' 실행 방법")

        return suggestions

    def _generate_type_suggestions(
        self, query_type: QueryType, entities: dict[str, list[str]]
    ) -> list[str]:
        """쿼리 유형 기반 폴백 제안"""
        suggestions = []
        indicators = entities.get("indicators", [])

        if query_type == QueryType.DEFINITION:
            if indicators:
                ind = indicators[0].upper()
                suggestions.append(f"{ind}가 높으면 어떤 의미?")
            suggestions.extend(["관련된 다른 지표는?", "실제 데이터에 적용해주세요"])

        elif query_type == QueryType.INTERPRETATION:
            suggestions.extend(["이 수치가 좋은 건가요?", "개선을 위한 액션은?"])

        elif query_type == QueryType.ANALYSIS:
            suggestions.extend(["시계열 트렌드 분석", "경쟁사와 비교해주세요"])

        elif query_type == QueryType.DATA_QUERY:
            suggestions.extend(["최근 7일 추이 분석", "경쟁사 대비 현황"])

        elif query_type == QueryType.COMBINATION:
            suggestions.extend(["다른 시나리오 분석", "현재 해당 상황 존재 여부"])

        else:
            # 기본 제안
            suggestions = ["SoS(점유율) 설명", "LANEIGE 현재 순위", "전략적 권고사항"]

        return suggestions

    def get_fallback_suggestions(self) -> list[str]:
        """폴백 제안"""
        return ["SoS(점유율)에 대해 알려주세요", "오늘의 주요 인사이트는?", "LANEIGE 현재 순위는?"]

"""
Suggestion Generator
====================
HybridChatbotAgent에서 분리된 후속 질문 생성 서비스

책임:
- 응답 키워드 기반 후속 질문 생성
- 엔티티 기반 동적 제안 생성
- 추론 결과 기반 제안 생성
- 쿼리 유형 기반 폴백 제안 생성
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from src.domain.entities import InferenceResult
from src.rag.router import QueryType
from src.shared.constants import SUGGESTION_MAX_COUNT

if TYPE_CHECKING:
    from src.ontology.knowledge_graph import KnowledgeGraph


class SuggestionGenerator:
    """
    후속 질문 생성 서비스

    HybridChatbotAgent의 제안 생성 로직을 분리하여 SRP 준수.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph | None = None):
        """
        Args:
            knowledge_graph: KnowledgeGraph 인스턴스 (선택)
        """
        self._kg = knowledge_graph

    def generate(
        self,
        query_type: QueryType,
        entities: dict[str, list[str]],
        inferences: list[InferenceResult],
        response: str = "",
    ) -> list[str]:
        """
        후속 질문 제안 생성 (v2 - 개선 버전)

        우선순위:
        1. 응답 키워드 기반 (response 분석)
        2. 엔티티 기반 (KG 경쟁사 활용)
        3. 추론 결과 기반
        4. 쿼리 유형 기반 (폴백)

        Args:
            query_type: 질문 유형
            entities: 추출된 엔티티
            inferences: 온톨로지 추론 결과
            response: AI 응답 내용 (키워드 분석용)

        Returns:
            3개의 후속 질문 리스트
        """
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
        """응답에서 후속 질문 관련 키워드 추출"""
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
        """엔티티 기반 동적 제안 생성 (KG 경쟁사 활용)"""
        suggestions = []

        brands = entities.get("brands", [])
        categories = entities.get("categories", [])
        indicators = entities.get("indicators", [])

        # 브랜드 기반 (KG에서 경쟁사 조회)
        if brands:
            brand = brands[0]
            # KG에서 경쟁사 조회 시도
            if self._kg:
                try:
                    competitors = self._kg.get_related_brands(brand, limit=2)
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
        return [
            "SoS(점유율)에 대해 알려주세요",
            "오늘의 주요 인사이트는?",
            "LANEIGE 현재 순위는?",
        ]

    async def generate_with_llm(
        self,
        user_query: str,
        response_summary: str,
        entities: dict[str, list[str]],
        llm_client: Any = None,
    ) -> list[str]:
        """
        LLM 기반 후속 질문 생성

        비용: ~$0.0002/호출 (GPT-4.1-mini 기준)

        Args:
            user_query: 사용자 질문
            response_summary: AI 응답 요약 (300자 제한)
            entities: 추출된 엔티티
            llm_client: LLM 클라이언트

        Returns:
            3개의 후속 질문 리스트 (실패 시 빈 리스트)
        """
        if not llm_client:
            return []

        try:
            brands = entities.get("brands", [])
            categories = entities.get("categories", [])
            indicators = entities.get("indicators", [])

            context_str = f"""
브랜드: {', '.join(brands) if brands else '없음'}
카테고리: {', '.join(categories) if categories else '없음'}
지표: {', '.join(indicators) if indicators else '없음'}
"""

            prompt = f"""사용자 질문: {user_query}
AI 응답 요약: {response_summary[:300]}
컨텍스트: {context_str}

위 대화를 바탕으로 사용자가 물어볼 수 있는 후속 질문 3개를 생성하세요.

규칙:
1. 각 질문은 20자 이내
2. 한국어로
3. 물음표로 끝내기
4. 데이터 분석 관련 (아마존 시장 분석)

출력 형식 (3줄):
1. 첫번째 질문?
2. 두번째 질문?
3. 세번째 질문?
"""

            response = await llm_client.acompletion(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7,
            )

            content = response.choices[0].message.content
            # 파싱: "1. 질문?" 형식에서 추출
            lines = content.strip().split("\n")
            suggestions = []
            for line in lines:
                # "1. 질문?" 또는 "- 질문?" 또는 "질문?" 형식
                line = line.strip()
                if line:
                    # 번호 제거
                    import re

                    cleaned = re.sub(r"^[\d\-\.\)]+\s*", "", line)
                    if cleaned:
                        suggestions.append(cleaned)

            return suggestions[:SUGGESTION_MAX_COUNT]

        except Exception:
            return []

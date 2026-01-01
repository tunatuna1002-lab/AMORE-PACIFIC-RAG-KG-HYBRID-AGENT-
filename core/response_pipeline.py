"""
응답 생성 파이프라인
====================
RAG + KG 컨텍스트 기반 LLM 응답 생성

역할:
- 컨텍스트 기반 프롬프트 구성
- LLM API 호출
- 응답 후처리 및 검증
- Response 객체 생성

연결 파일:
- core/models.py: Context, Response, ConfidenceLevel
- core/context_gatherer.py: 컨텍스트 수집
- rag/templates.py: 프롬프트 템플릿
- utils/openai_client.py: OpenAI API
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

from .models import Context, Response, ConfidenceLevel, Decision, ToolResult

logger = logging.getLogger(__name__)


class ResponsePipeline:
    """
    응답 생성 파이프라인

    컨텍스트를 기반으로 LLM 응답을 생성하고 후처리.

    Usage:
        pipeline = ResponsePipeline(openai_client)
        response = await pipeline.generate(query, context)
    """

    # 시스템 프롬프트
    SYSTEM_PROMPT = """당신은 아모레퍼시픽 라네즈 브랜드의 Amazon 마켓 분석 전문가입니다.

역할:
- 라네즈(LANEIGE) 브랜드의 Amazon 판매 성과 분석
- 경쟁사 대비 포지셔닝 평가
- SoS(Share of Shelf), HHI, CPI 등 핵심 지표 해석
- 데이터 기반 전략 제언

응답 원칙:
1. 정확성: 제공된 데이터와 컨텍스트에 기반하여 응답
2. 구체성: 수치와 근거를 명시
3. 실행가능성: 구체적인 액션 제안
4. 간결성: 핵심 위주로 명확하게

언어: 한국어로 응답"""

    def __init__(
        self,
        openai_client: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1500,
        temperature: float = 0.3
    ):
        """
        Args:
            openai_client: OpenAI 클라이언트
            model: 사용할 모델
            max_tokens: 최대 토큰 수
            temperature: 생성 온도
        """
        self.client = openai_client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    # =========================================================================
    # 클라이언트 설정
    # =========================================================================

    def set_client(self, client: Any) -> None:
        """OpenAI 클라이언트 설정 (지연 주입용)"""
        self.client = client

    # =========================================================================
    # 메인 생성 메서드
    # =========================================================================

    async def generate(
        self,
        query: str,
        context: Context,
        decision: Optional[Decision] = None,
        tool_result: Optional[ToolResult] = None
    ) -> Response:
        """
        컨텍스트 기반 응답 생성

        Args:
            query: 사용자 질문
            context: 수집된 컨텍스트
            decision: LLM 판단 결과 (있으면 활용)
            tool_result: 도구 실행 결과 (있으면 포함)

        Returns:
            Response 객체
        """
        start_time = datetime.now()

        try:
            # 프롬프트 구성
            messages = self._build_messages(query, context, decision, tool_result)

            # LLM 호출
            if self.client:
                response_text = await self._call_llm(messages)
            else:
                # 클라이언트 없으면 컨텍스트 기반 기본 응답
                response_text = self._generate_fallback_response(query, context)

            # 응답 후처리
            processed_text = self._post_process(response_text, context)

            # 제안 질문 생성
            suggestions = self._generate_suggestions(query, context)

            # 출처 추출
            sources = self._extract_sources(context)

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return Response(
                text=processed_text,
                query_type=self._infer_query_type(query, context),
                confidence_level=self._assess_confidence(context),
                confidence_score=self._calculate_confidence_score(context),
                sources=sources,
                entities=context.entities,
                tools_called=[tool_result.tool_name] if tool_result else [],
                suggestions=suggestions,
                processing_time_ms=processing_time
            )

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return Response.fallback(
                f"응답 생성 중 오류가 발생했습니다: {str(e)}"
            )

    async def generate_with_tool_result(
        self,
        query: str,
        context: Context,
        tool_result: ToolResult
    ) -> Response:
        """
        도구 실행 결과를 포함한 응답 생성

        Args:
            query: 원본 질문
            context: 컨텍스트
            tool_result: 도구 실행 결과

        Returns:
            Response
        """
        # 도구 결과 요약 추가
        tool_summary = tool_result.to_summary()

        # 도구 결과를 컨텍스트에 반영
        enhanced_context = Context(
            query=context.query,
            entities=context.entities,
            rag_docs=context.rag_docs,
            kg_facts=context.kg_facts,
            kg_inferences=context.kg_inferences,
            system_state=context.system_state,
            summary=f"{context.summary}\n\n[도구 실행 결과] {tool_summary}"
        )

        return await self.generate(query, enhanced_context, tool_result=tool_result)

    # =========================================================================
    # 프롬프트 구성
    # =========================================================================

    def _build_messages(
        self,
        query: str,
        context: Context,
        decision: Optional[Decision] = None,
        tool_result: Optional[ToolResult] = None
    ) -> List[Dict[str, str]]:
        """
        LLM 메시지 구성

        Args:
            query: 질문
            context: 컨텍스트
            decision: 판단 결과
            tool_result: 도구 결과

        Returns:
            메시지 리스트
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

        # 컨텍스트 메시지
        context_content = self._format_context(context)
        messages.append({
            "role": "system",
            "content": f"[분석 컨텍스트]\n{context_content}"
        })

        # 판단 결과 (있으면)
        if decision and decision.key_points:
            key_points_str = "\n".join(f"- {p}" for p in decision.key_points)
            messages.append({
                "role": "system",
                "content": f"[응답 핵심 포인트]\n{key_points_str}"
            })

        # 도구 결과 (있으면)
        if tool_result and tool_result.success:
            tool_content = self._format_tool_result(tool_result)
            messages.append({
                "role": "system",
                "content": f"[도구 실행 결과]\n{tool_content}"
            })

        # 사용자 질문
        messages.append({
            "role": "user",
            "content": query
        })

        return messages

    def _format_context(self, context: Context) -> str:
        """컨텍스트 포맷팅"""
        if context.summary:
            return context.summary

        parts = []

        # 시스템 상태
        if context.system_state:
            state = context.system_state
            parts.append(f"데이터 상태: {state.data_freshness}")
            if state.kg_initialized:
                parts.append(f"KG: {state.kg_triple_count} 트리플")

        # KG 추론
        if context.kg_inferences:
            parts.append("\n인사이트:")
            for inf in context.kg_inferences[:3]:
                parts.append(f"- {inf.get('insight', '')}")

        # KG 사실
        if context.kg_facts:
            parts.append("\n관련 정보:")
            for fact in context.kg_facts[:3]:
                parts.append(f"- {fact.fact_type}: {fact.entity}")

        return "\n".join(parts) if parts else "컨텍스트 없음"

    def _format_tool_result(self, tool_result: ToolResult) -> str:
        """도구 결과 포맷팅"""
        if not tool_result.success:
            return f"실행 실패: {tool_result.error}"

        data = tool_result.data

        if tool_result.tool_name == "crawl_amazon":
            total = data.get("total_products", 0)
            laneige = data.get("laneige_count", 0)
            return f"크롤링 완료: 총 {total}개 제품, LANEIGE {laneige}개"

        elif tool_result.tool_name == "calculate_metrics":
            brands = len(data.get("brand_metrics", []))
            products = len(data.get("product_metrics", []))
            alerts = len(data.get("alerts", []))
            return f"지표 계산 완료: {brands}개 브랜드, {products}개 제품, {alerts}개 알림"

        elif tool_result.tool_name == "query_data":
            return json.dumps(data, ensure_ascii=False, indent=2)[:500]

        return tool_result.to_summary()

    # =========================================================================
    # LLM 호출
    # =========================================================================

    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        LLM API 호출

        Args:
            messages: 메시지 리스트

        Returns:
            응답 텍스트
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    # =========================================================================
    # 후처리
    # =========================================================================

    def _post_process(self, text: str, context: Context) -> str:
        """
        응답 후처리

        Args:
            text: 원본 응답
            context: 컨텍스트

        Returns:
            후처리된 응답
        """
        # 빈 응답 처리
        if not text or not text.strip():
            return "응답을 생성할 수 없습니다. 다시 질문해주세요."

        # 불필요한 prefix 제거
        prefixes_to_remove = [
            "답변:",
            "응답:",
            "분석 결과:",
        ]
        for prefix in prefixes_to_remove:
            if text.strip().startswith(prefix):
                text = text.strip()[len(prefix):].strip()

        return text.strip()

    def _generate_fallback_response(
        self,
        query: str,
        context: Context
    ) -> str:
        """
        클라이언트 없을 때 기본 응답 생성

        Args:
            query: 질문
            context: 컨텍스트

        Returns:
            기본 응답
        """
        parts = []

        # RAG 문서가 있으면 문서 내용 기반 응답
        if context.rag_docs:
            for doc in context.rag_docs[:2]:
                content = doc.get("content", "")
                if content:
                    parts.append(content[:1000])  # 최대 1000자
            if parts:
                return "\n\n".join(parts)

        # 인사이트 있으면 표시
        if context.kg_inferences:
            parts.append("분석 인사이트:")
            for inf in context.kg_inferences[:3]:
                insight = inf.get("insight", "")
                rec = inf.get("recommendation", "")
                parts.append(f"• {insight}")
                if rec:
                    parts.append(f"  → {rec}")
            parts.append("")

        # KG 사실 있으면 표시
        if context.kg_facts:
            parts.append("관련 정보:")
            for fact in context.kg_facts[:3]:
                parts.append(f"• {fact.entity}: {fact.fact_type}")
            parts.append("")

        if not parts:
            parts.append("추가 데이터가 필요합니다. 크롤링 및 지표 계산을 실행해주세요.")

        return "\n".join(parts)

    # =========================================================================
    # 메타데이터 생성
    # =========================================================================

    def _generate_suggestions(
        self,
        query: str,
        context: Context
    ) -> List[str]:
        """후속 질문 제안 생성"""
        suggestions = []

        # 엔티티 기반 제안
        brands = context.entities.get("brands", [])
        if brands:
            brand = brands[0]
            suggestions.append(f"{brand} 경쟁사 분석해줘")
            suggestions.append(f"{brand} 순위 변동 추이 보여줘")

        # 시스템 상태 기반 제안
        if context.system_state:
            if not context.system_state.kg_initialized:
                suggestions.append("지식 그래프 초기화해줘")
            if context.system_state.data_freshness != "fresh":
                suggestions.append("최신 데이터 크롤링해줘")

        # 기본 제안
        if not suggestions:
            suggestions = [
                "라네즈 현재 순위 알려줘",
                "SoS가 뭐야?",
                "오늘 크롤링 해줘"
            ]

        return suggestions[:3]

    def _extract_sources(self, context: Context) -> List[str]:
        """출처 추출"""
        sources = []

        # RAG 문서 출처
        for doc in context.rag_docs:
            title = doc.get("metadata", {}).get("title", "")
            if title and title not in sources:
                sources.append(title)

        # KG 출처
        if context.kg_facts:
            sources.append("Knowledge Graph")

        if context.kg_inferences:
            sources.append("Ontology Reasoning")

        return sources[:5]

    def _infer_query_type(self, query: str, context: Context) -> str:
        """질문 유형 추론"""
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["뭐야", "무엇", "정의", "어떻게 계산"]):
            return "definition"
        elif any(kw in query_lower for kw in ["해석", "의미", "높으면", "낮으면"]):
            return "interpretation"
        elif any(kw in query_lower for kw in ["순위", "랭킹", "현재", "지금"]):
            return "data_query"
        elif any(kw in query_lower for kw in ["분석", "비교", "전략", "리포트"]):
            return "analysis"
        elif any(kw in query_lower for kw in ["크롤링", "수집", "업데이트"]):
            return "action"
        else:
            return "general"

    def _assess_confidence(self, context: Context) -> ConfidenceLevel:
        """신뢰도 레벨 평가"""
        score = self._calculate_confidence_score(context)

        if score >= 5.0:
            return ConfidenceLevel.HIGH
        elif score >= 3.0:
            return ConfidenceLevel.MEDIUM
        elif score >= 1.5:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNKNOWN

    def _calculate_confidence_score(self, context: Context) -> float:
        """신뢰도 점수 계산"""
        score = 0.0

        # RAG 문서 있음
        if context.rag_docs:
            score += min(len(context.rag_docs), 3) * 1.0

        # KG 사실 있음
        if context.kg_facts:
            score += min(len(context.kg_facts), 3) * 1.0

        # KG 추론 있음
        if context.kg_inferences:
            score += min(len(context.kg_inferences), 2) * 1.5

        # 시스템 상태 양호
        if context.system_state:
            if context.system_state.data_freshness == "fresh":
                score += 1.0
            if context.system_state.kg_initialized:
                score += 0.5

        return min(score, 10.0)

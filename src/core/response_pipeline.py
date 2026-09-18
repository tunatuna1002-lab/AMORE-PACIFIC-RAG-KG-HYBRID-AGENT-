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

import json
import logging
from dataclasses import replace
from datetime import datetime
from typing import Any

from src.rag.evidence_assembly import evidence_source_labels
from src.rag.evidence_renderer import CITATION_INSTRUCTION, render_for_prompt
from src.shared.constants import DEFAULT_MODEL

from .confidence import ConfidenceAssessor
from .hallucination_detector import HallucinationDetector
from .models import ConfidenceLevel, Context, Decision, Response, ToolResult
from .numeric_verifier import MODE_OFF, apply_numeric_verification, skipped_summary

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
        openai_client: Any | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1500,
        temperature: float = 0.3,
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
        self._tracer = None  # Set externally via set_tracer()
        self._hallucination_detector = HallucinationDetector()
        # 신뢰도 사다리는 ConfidenceAssessor 하나만 쓴다 (§4.2 이중화 제거)
        self._confidence_assessor = ConfidenceAssessor()

    def _get_system_prompt(self) -> str:
        """시스템 프롬프트 — PromptRegistry 경유 (플래그 off이면 인라인 폴백).

        대시보드 기본 경로(v4)가 이 파이프라인을 쓰는데 프롬프트만 별도로
        하드코딩돼 있어, 근거 인용 규칙(v1b) 등 registry 개선이 v4에 반영되지
        않았다. context_builder와 동일하게 `prompts.use_centralized_prompts`
        플래그를 따른다.
        """
        from src.infrastructure.feature_flags import FeatureFlags

        if not FeatureFlags.get_instance().use_centralized_prompts():
            return self.SYSTEM_PROMPT

        try:
            from prompts.registry import PromptRegistry

            return PromptRegistry.get_instance().get_system_prompt("chatbot")
        except Exception:
            logger.warning("PromptRegistry load failed, using inline prompt", exc_info=True)
            return self.SYSTEM_PROMPT

    # =========================================================================
    # 클라이언트 설정
    # =========================================================================

    def set_client(self, client: Any) -> None:
        """OpenAI 클라이언트 설정 (지연 주입용)"""
        self.client = client

    def set_tracer(self, tracer) -> None:
        """ExecutionTracer 설정"""
        self._tracer = tracer

    # =========================================================================
    # 메인 생성 메서드
    # =========================================================================

    async def generate(
        self,
        query: str,
        context: Context,
        decision: Decision | None = None,
        tool_result: ToolResult | None = None,
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
            # 도구 결과도 증거 카드다 — 프롬프트·인용·출처·수치 검증이 검색 카드와 같은 경로를
            # 타도록 컨텍스트에 합친다 (트랙 4-A). 호출자의 Context는 바꾸지 않는다.
            context = self._with_tool_evidence(context, tool_result)

            # 프롬프트 구성
            messages = self._build_messages(query, context, decision, tool_result)

            # HIGH confidence fast path - lighter LLM call
            is_high_confidence = (
                decision
                and hasattr(decision, "confidence")
                and decision.confidence >= 0.85
                and decision.tool == "direct_answer"
                and hasattr(decision, "reason")
                and "HIGH confidence" in (decision.reason or "")
            )

            # LLM 사용 가능 여부 (litellm은 OPENAI_API_KEY로 직접 호출)
            import os

            llm_available = self.client or os.environ.get("OPENAI_API_KEY")

            if is_high_confidence and llm_available:
                # Use faster, shorter prompt for HIGH confidence
                response_text = await self._call_llm_fast(query, context)
            elif llm_available:
                response_text = await self._call_llm(messages)
            else:
                # LLM 없으면 컨텍스트 기반 기본 응답
                response_text = self._generate_fallback_response(query, context)

            # 응답 후처리
            processed_text = self._post_process(response_text, context)

            # 답변 수치 검증 (설계 E8) — 이 답변 프롬프트에 실린 카드와 대조
            processed_text, numeric_verification = self._verify_numbers(processed_text, context)

            # 환각 감지 (low confidence 응답만)
            hallucination_penalty = 1.0
            grounding_warning = False
            if decision and hasattr(decision, "confidence") and decision.confidence < 0.8:
                try:
                    context_text = context.summary or ""
                    groundedness = await self._hallucination_detector.check(
                        processed_text, context_text
                    )
                    if not groundedness.is_grounded:
                        logger.warning(f"Hallucination warning: score={groundedness.score:.2f}")
                        # 로깅만 하지 않고 응답 신뢰도에 반영 (근거 부족 → 신뢰도 하향)
                        hallucination_penalty = 0.6
                        grounding_warning = True
                except Exception as e:
                    logger.debug(f"Hallucination check skipped: {e}")

            # 제안 질문 생성
            suggestions = self._generate_suggestions(query, context)

            # 출처 추출
            sources = self._extract_sources(context)

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # 신뢰도 계산 — 근거 점수가 상한, LLM 자기보고는 감쇠만 한다.
            # 과거에는 max()를 써서 근거 기반 점수가 "바닥 올리기"로만 작동했다.
            # 두 값은 스케일도 달라(근거 0-10 vs LLM 0-1) 근거가 거의 없을 때만
            # LLM 자신감이 이겼고, 그 결과 무근거 답변이 신뢰도를 얻었다.
            calculated_confidence = self._calculate_confidence_score(context)
            decision_confidence = getattr(decision, "confidence", None) if decision else None
            if decision_confidence:
                final_confidence = calculated_confidence * min(float(decision_confidence), 1.0)
            else:
                final_confidence = calculated_confidence

            return Response(
                text=processed_text,
                query_type=self._infer_query_type(query, context),
                confidence_level=self._assess_confidence(context),
                confidence_score=final_confidence * hallucination_penalty,
                grounding_warning=grounding_warning,
                sources=sources,
                entities=context.entities,
                tools_called=[tool_result.tool_name]
                if tool_result and hasattr(tool_result, "tool_name")
                else [],
                suggestions=suggestions,
                processing_time_ms=processing_time,
                metadata=(
                    {"numeric_verification": numeric_verification}
                    if numeric_verification is not None
                    else {}
                ),
            )

        except Exception as e:
            logger.error(f"Response generation failed: {e}", exc_info=True)
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return Response.fallback(f"응답 생성 중 오류가 발생했습니다: {str(e)}")

    async def generate_with_tool_result(
        self, query: str, context: Context, tool_result: ToolResult
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
            summary=f"{context.summary}\n\n[도구 실행 결과] {tool_summary}",
            evidence=context.evidence,
            prompt_evidence=context.prompt_evidence,
        )

        return await self.generate(query, enhanced_context, tool_result=tool_result)

    # =========================================================================
    # 프롬프트 구성
    # =========================================================================

    def _build_messages(
        self,
        query: str,
        context: Context,
        decision: Decision | None = None,
        tool_result: ToolResult | None = None,
    ) -> list[dict[str, str]]:
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
        messages = [{"role": "system", "content": self._get_system_prompt()}]

        # 컨텍스트 메시지 (+ 카드가 있으면 인용 규칙 한 번)
        context_content = self._format_context(context)
        messages.append(
            {
                "role": "system",
                "content": f"[분석 컨텍스트]\n{context_content}{self._citation_block(context)}",
            }
        )

        # 판단 결과 (있으면) - None-safety 추가
        if decision:
            key_points = (
                decision.key_points
                if hasattr(decision, "key_points") and decision.key_points
                else []
            )
            if key_points:
                key_points_str = "\n".join(f"- {p}" for p in key_points)
                messages.append(
                    {"role": "system", "content": f"[응답 핵심 포인트]\n{key_points_str}"}
                )

        # 도구 결과 (있으면)
        if tool_result and tool_result.success:
            tool_content = self._format_tool_result(tool_result)
            messages.append({"role": "system", "content": f"[도구 실행 결과]\n{tool_content}"})

        # 사용자 질문
        messages.append({"role": "user", "content": query})

        return messages

    def _format_context(self, context: Context) -> str:
        """컨텍스트 포맷팅 — summary(카드 렌더링)가 없으면 상태 한 줄과 카드만 싣는다 (E1)."""
        if context.summary:
            return context.summary

        parts = []

        # 시스템 상태
        if context.system_state:
            state = context.system_state
            parts.append(f"데이터 상태: {state.data_freshness}")
            if state.kg_initialized:
                parts.append(f"KG: {state.kg_triple_count} 트리플")

        cards = render_for_prompt(context.prompt_evidence)
        if cards:
            parts.append(cards)

        return "\n\n".join(parts) if parts else "컨텍스트 없음"

    @staticmethod
    def _citation_block(context: Context) -> str:
        """답변 프롬프트의 인용 규칙 (설계 E8 앞부분). 카드가 없으면 인용할 id도 없다."""
        if not context.prompt_evidence:
            return ""
        return f"\n\n[인용 규칙]\n{CITATION_INSTRUCTION}"

    @staticmethod
    def _with_tool_evidence(context: Context, tool_result: ToolResult | None) -> Context:
        """도구 결과의 증거 카드를 컨텍스트 카드에 합친 사본을 돌려준다 (트랙 4-A).

        레지스트리 도구(``src/core/tool_registry.py``)의 결과는 전부 카드다. 카드를 합치면
        답변 프롬프트의 인용 규칙·출처 표시(``_extract_sources``)·답변 수치 검증(E8)이
        검색 카드와 똑같이 도구 카드에도 적용된다. 카드가 없는 결과(옛 도구·실패)는 그대로 둔다.
        """
        from src.core.tool_registry import tool_evidence
        from src.domain.entities.evidence import EvidenceSet

        cards = tool_evidence(tool_result)
        if not cards:
            return context

        return replace(
            context,
            evidence=EvidenceSet([*context.evidence, *cards]).to_list(),
            prompt_evidence=EvidenceSet([*context.prompt_evidence, *cards]).to_list(),
        )

    def _format_tool_result(self, tool_result: ToolResult) -> str:
        """도구 결과 포맷팅"""
        if not tool_result.success:
            return f"실행 실패: {tool_result.error}"

        from src.core.tool_registry import render_tool_observation, tool_evidence

        if tool_evidence(tool_result):
            # 레지스트리 도구: 카드를 프롬프트와 같은 형식(``[id] 내용 (as_of, source)``)으로
            return render_tool_observation(tool_result)

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

    async def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """
        LLM API 호출 (litellm 사용)

        Args:
            messages: 메시지 리스트

        Returns:
            응답 텍스트
        """
        try:
            from litellm import acompletion

            if self._tracer and self._tracer.get_current_trace_id():
                with self._tracer.llm_span(
                    "response_llm",
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ) as span:
                    response = await acompletion(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                    if hasattr(response, "usage") and response.usage:
                        span.attributes["llm.prompt_tokens"] = getattr(
                            response.usage, "prompt_tokens", 0
                        )
                        span.attributes["llm.completion_tokens"] = getattr(
                            response.usage, "completion_tokens", 0
                        )
                        span.attributes["llm.total_tokens"] = getattr(
                            response.usage, "total_tokens", 0
                        )
            else:
                response = await acompletion(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    async def _call_llm_fast(self, query: str, context: Context) -> str:
        """
        HIGH 신뢰도용 빠른 LLM 응답 생성

        컨텍스트가 충분할 때 간결한 프롬프트로 빠르게 응답.
        max_tokens를 줄이고 temperature를 낮춰 결정적 응답.

        Args:
            query: 사용자 질문
            context: 수집된 컨텍스트

        Returns:
            응답 텍스트
        """
        from litellm import acompletion

        # 간결한 시스템 프롬프트
        fast_system = (
            "아모레퍼시픽 LANEIGE 브랜드 Amazon 마켓 분석 전문가입니다. "
            "제공된 데이터를 바탕으로 간결하고 정확하게 한국어로 답변하세요. "
            "수치와 근거를 명시하세요."
        )

        # 컨텍스트 요약을 사용자 메시지에 직접 포함
        user_msg = (
            f"## 질문\n{query}\n\n## 데이터\n{context.summary or '데이터 없음'}"
            f"{self._citation_block(context)}"
        )

        try:
            response = await acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": fast_system},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=800,  # 절반으로 줄임
                temperature=0.2,  # 더 결정적
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Fast LLM call failed, falling back: {e}")
            return self._generate_fallback_response(query, context)

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
                text = text.strip()[len(prefix) :].strip()

        return text.strip()

    def _generate_fallback_response(self, query: str, context: Context) -> str:
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

    @staticmethod
    def _verify_numbers(text: str, context: Context) -> tuple[str, dict[str, Any] | None]:
        """플래그 모드로 답변 수치를 검증한다 → (답변, ``metadata["numeric_verification"]``).

        ``generate``의 모든 생성 분기(일반 LLM·HIGH 신뢰도 fast path·LLM 없는 기본 응답)와
        ``generate_with_tool_result``가 후처리 직후 이 한 곳을 지난다. 이 파이프라인에는
        스트리밍 생성이 없다 — ``UnifiedBrain.process_query_stream``은 ``generate``로 답을 다
        만든 뒤 한 번에 내보내므로 enforce 치환도 스트림 텍스트에 그대로 반영된다.

        스키마는 ``src/core/numeric_verifier.py`` 모듈 docstring. ``off``면 메타데이터 없음.
        검증기 오류는 답변을 막지 않는다(``skipped: "error"``).
        """
        from src.infrastructure.feature_flags import FeatureFlags

        mode = FeatureFlags.get_instance().numeric_verification_mode()
        if mode == MODE_OFF:
            return text, None
        try:
            return apply_numeric_verification(text, context.prompt_evidence, mode)
        except Exception:
            logger.warning("Numeric verification failed; answer kept as generated", exc_info=True)
            return text, skipped_summary(mode, "error")

    # =========================================================================
    # 메타데이터 생성
    # =========================================================================

    def _generate_suggestions(self, query: str, context: Context) -> list[str]:
        """후속 질문 제안 생성"""
        suggestions = []

        # 엔티티 기반 제안
        brands = context.entities.get("brands", [])
        if brands:
            brand = brands[0]
            suggestions.append(f"{brand} 경쟁사 분석해줘")
            suggestions.append(f"{brand} 순위 변동 추이 보여줘")

        # 시스템 상태 기반 제안 (분석 질문만 — 시스템 명령 제외)
        if context.system_state:
            if not context.system_state.kg_initialized:
                suggestions.append("LANEIGE 브랜드 현황 요약해줘")
            if context.system_state.data_freshness != "fresh":
                suggestions.append("최근 시장 점유율 변화 분석해줘")

        # 기본 제안 (분석 질문만)
        if not suggestions:
            suggestions = ["라네즈 현재 순위 알려줘", "SoS가 뭐야?", "오늘 주요 인사이트는?"]

        return suggestions[:3]

    def _extract_sources(self, context: Context) -> list[str]:
        """출처 = 답변 프롬프트에 실린 증거 카드의 출처 (설계 E1).

        문서 제목·``sqlite:<table> (as_of)``·``KG``·``rule:<이름>``을 카드 순서대로 중복 없이
        돌려준다. ``Response.sources``의 계약(list[str])은 그대로다. 프롬프트에 싣지 않은
        원자료(rag_docs·kg_facts)는 출처로 내지 않는다 — 모델이 보지 않은 근거이기 때문이다.
        """
        return evidence_source_labels(context.prompt_evidence)

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
        """신뢰도 레벨 평가 — 사다리는 ConfidenceAssessor에 위임한다.

        과거에는 confidence.py의 5.0/3.0/1.5 사다리를 여기서 재구현해,
        임계값이 갈라질 수 있었다 (§4.2).
        """
        score = self._calculate_confidence_score(context)
        return self._confidence_assessor.assess({"max_score": score})

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

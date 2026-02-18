"""
Hybrid Chatbot Agent
Ontology-RAG 하이브리드 챗봇 에이전트

Flow:
1. 쿼리에서 엔티티/의도 추출
2. Knowledge Graph에서 관련 사실 조회
3. Ontology Reasoner로 추론
4. RAG로 가이드라인 검색
5. 통합 컨텍스트로 LLM 응답 생성
"""

import logging
from datetime import datetime
from typing import Any

from litellm import acompletion

from src.agents.base_hybrid_agent import BaseHybridAgent
from src.domain.entities.relations import InferenceResult
from src.memory.context import ContextManager
from src.monitoring.logger import AgentLogger
from src.monitoring.metrics import QualityMetrics
from src.monitoring.tracer import ExecutionTracer
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.rag.context_builder import CompactContextBuilder
from src.rag.query_rewriter import QueryRewriter, RewriteResult, create_rewrite_result_no_change
from src.rag.router import QueryType, RAGRouter

logger = logging.getLogger(__name__)


class HybridChatbotAgent(BaseHybridAgent):
    """
    Ontology-RAG 하이브리드 챗봇 에이전트
    Implements ChatbotAgentProtocol (src.domain.interfaces.chatbot)

    기존 ChatbotAgent와의 차이점:
    - 온톨로지 추론 결과 기반 응답
    - 지식 그래프에서 관련 사실 조회
    - 추론 과정 설명 제공

    사용 예:
        agent = HybridChatbotAgent()
        result = await agent.chat("LANEIGE Lip Care 경쟁력 분석해줘")
    """

    AGENT_NAME = "hybrid_chatbot"

    # 설정 파일 경로
    CONFIG_PATH = "config/thresholds.json"

    # 브랜드 정규화 매핑 (잘린 브랜드명 → 전체 브랜드명)
    BRAND_NORMALIZATION = {
        "burt's": "Burt's Bees",
        "wet": "wet n wild",
        "tree": "Tree Hut",
        "clean": "Clean Skin Club",
        "summer": "Summer Fridays",
        "rare": "Rare Beauty",
        "la": "La Roche-Posay",
        "beauty": "Beauty of Joseon",
        "tower": "Tower 28",
        "drunk": "Drunk Elephant",
        "paula's": "Paula's Choice",
        "the": "The Ordinary",
        "glow": "Glow Recipe",
        "youth": "Youth To The People",
        "first": "First Aid Beauty",
        "charlotte": "Charlotte Tilbury",
        "too": "Too Faced",
        "urban": "Urban Decay",
        "fenty": "Fenty Beauty",
        "huda": "Huda Beauty",
        "anastasia": "Anastasia Beverly Hills",
        "physicians": "Physicians Formula",
        "covergirl": "COVERGIRL",
        "medicube": "MEDICUBE",
    }

    @classmethod
    def _load_config(cls) -> dict:
        """설정 파일에서 chatbot 관련 설정 로드"""
        import json
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / cls.CONFIG_PATH

        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    config = json.load(f)
                    return config.get("system", {}).get("chatbot", {})
            except Exception:
                logger.warning("Suppressed Exception", exc_info=True)

        return {}  # 설정 없으면 기본값 사용

    def __init__(
        self,
        model: str = None,
        docs_dir: str = ".",
        knowledge_graph: KnowledgeGraph | None = None,
        reasoner: OntologyReasoner | None = None,
        logger: AgentLogger | None = None,
        tracer: ExecutionTracer | None = None,
        metrics: QualityMetrics | None = None,
        context_manager: ContextManager | None = None,
    ):
        """
        Args:
            model: LLM 모델명 (None이면 설정 파일에서 로드)
            docs_dir: RAG 문서 디렉토리
            knowledge_graph: 지식 그래프 (공유 가능)
            reasoner: 추론기 (공유 가능)
            logger: 로거
            tracer: 추적기
            metrics: 메트릭 수집기
            context_manager: 컨텍스트 관리자
        """
        import os

        # 설정 파일에서 chatbot 설정 로드
        config = self._load_config()
        resolved_model = model or config.get("model", "gpt-4.1-mini")

        # Temperature: 챗봇 전용 환경변수 > 일반 환경변수 > 설정파일 > 기본값(0.4)
        # 챗봇은 사실적/일관된 답변을 위해 낮은 temperature 사용 (E2E Audit - 2026-01-27)
        from src.shared.constants import CHATBOT_TEMPERATURE

        self.temperature = float(
            os.getenv(
                "LLM_CHATBOT_TEMPERATURE",
                os.getenv("LLM_TEMPERATURE", config.get("temperature", CHATBOT_TEMPERATURE)),
            )
        )
        self.max_context_tokens = config.get("max_context_tokens", 8000)

        # Base class initialisation (KG, reasoner, retriever, context_builder, etc.)
        super().__init__(
            model=resolved_model,
            docs_dir=docs_dir,
            knowledge_graph=knowledge_graph,
            reasoner=reasoner,
            agent_logger=logger,
            tracer=tracer,
            metrics=metrics,
            context_builder_max_tokens=3000,
        )

        # Chatbot-specific: RAG router
        self.router = RAGRouter()

        # Chatbot-specific: compact context builder
        self.compact_builder = CompactContextBuilder(max_tokens=1500)

        # 메모리
        self.context = context_manager or ContextManager()

        # 현재 데이터 컨텍스트
        self._current_data: dict[str, Any] = {}

        # Query Rewriter (대화 맥락 기반 질문 재구성)
        self.query_rewriter = QueryRewriter(model=model)

        # 외부 신호 수집기 (Tavily + RSS + Reddit)
        self._external_signal_collector = None
        self._last_external_signals: list[Any] = []

        # 응답 검증 파이프라인 (지연 초기화)
        self._verification_pipeline: Any = None
        self._enable_verification = config.get("enable_verification", True)

        # 분해된 컴포넌트 (always active)
        from src.infrastructure.container import Container

        self.suggestion_engine = Container.get_suggestion_engine()
        self.source_provider = Container.get_source_provider()
        self.signal_manager = Container.get_external_signal_manager()

    @property
    def verification_pipeline(self) -> Any:
        """검증 파이프라인 (지연 초기화)"""
        if self._verification_pipeline is None:
            from src.core.verification_pipeline import VerificationPipelineFactory

            self._verification_pipeline = VerificationPipelineFactory.get_instance()
        return self._verification_pipeline

    def set_data_context(self, data: dict[str, Any]) -> None:
        """
        현재 데이터 컨텍스트 설정

        Args:
            data: 지표/인사이트 데이터
        """
        self._current_data = data

        # 지식 그래프 업데이트
        if data:
            self.hybrid_retriever.update_knowledge_graph(metrics_data=data)

    async def chat(
        self, user_message: str, session_id: str | None = None, include_reasoning: bool = True
    ) -> dict[str, Any]:
        """
        사용자 질문에 응답

        Args:
            user_message: 사용자 메시지
            session_id: 세션 ID
            include_reasoning: 추론 과정 포함 여부

        Returns:
            {
                "response": "...",
                "query_type": "...",
                "inferences": [...],
                "sources": [...],
                "suggestions": [...]
            }
        """
        # 감사 로깅 시작
        audit_context = self.logger.chat_request(query=user_message, session_id=session_id)
        start_time = datetime.now()

        if self.tracer:
            self.tracer.start_span("hybrid_chatbot_response", {"query_length": len(user_message)})

        try:
            # 1. 쿼리 라우팅 (의도 분류)
            route_result = self.router.route(user_message)
            query_type = route_result.get("query_type")

            self.logger.debug(f"Query type: {query_type}")

            # 2. Fallback 처리 (의도 불명)
            if query_type == QueryType.UNKNOWN:
                fallback_response = route_result.get("fallback_message", "")
                return {
                    "response": fallback_response,
                    "query_type": "unknown",
                    "is_fallback": True,
                    "inferences": [],
                    "sources": [],
                    "suggestions": self.suggestion_engine.get_fallback_suggestions(),
                }

            # 2.5 질문 재구성 (대화 맥락 기반)
            rewrite_result = await self._maybe_rewrite_query(user_message)

            # 명확화 필요시 바로 반환
            if rewrite_result.needs_clarification:
                self.context.add_user_message(user_message)
                self.context.add_assistant_message(rewrite_result.clarification_message)
                return {
                    "response": rewrite_result.clarification_message,
                    "query_type": "clarification",
                    "is_fallback": True,
                    "inferences": [],
                    "sources": [],
                    "suggestions": [
                        "특정 브랜드를 지정해주세요",
                        "어떤 지표가 궁금하신가요?",
                        "제품명을 알려주세요",
                    ],
                    "query_info": {
                        "original": user_message,
                        "rewritten": None,
                        "was_rewritten": False,
                        "needs_clarification": True,
                    },
                }

            # 재구성된 쿼리 사용 (검색용)
            search_query = rewrite_result.rewritten_query

            if rewrite_result.was_rewritten:
                self.logger.info(f"Query rewritten: '{user_message}' -> '{search_query}'")

            # 3. 하이브리드 검색 (추론 + RAG)
            if self.tracer:
                self.tracer.start_span("hybrid_retrieval")

            hybrid_context = await self.hybrid_retriever.retrieve(
                query=search_query,  # 재구성된 쿼리 사용
                current_metrics=self._current_data,
                include_explanations=include_reasoning,
            )
            self._last_hybrid_context = hybrid_context

            if self.tracer:
                self.tracer.end_span("completed")

            # 3.5. 외부 신호 수집 (Tavily 뉴스, RSS, Reddit)
            external_signals = await self.signal_manager.collect(
                query=search_query, entities=hybrid_context.entities
            )
            self._last_external_signals = external_signals
            failed_signals = self.signal_manager.get_failed_collectors()

            # 4. 컨텍스트 구성
            if self.tracer:
                self.tracer.start_span("build_context")

            # 쿼리 유형에 따라 빌더 선택
            if query_type in [QueryType.DEFINITION, QueryType.INTERPRETATION]:
                # 간단한 질문은 컴팩트 빌더
                context = self.compact_builder.build(
                    hybrid_context=hybrid_context,
                    current_metrics=self._current_data,
                    query=user_message,
                    knowledge_graph=self.kg,
                )
            else:
                # 분석 질문은 풀 빌더 (카테고리 계층 인식 포함)
                context = self.context_builder.build(
                    hybrid_context=hybrid_context,
                    current_metrics=self._current_data,
                    query=user_message,
                    knowledge_graph=self.kg,
                )

            if self.tracer:
                self.tracer.end_span("completed")

            # 5. LLM 응답 생성
            if self.tracer:
                self.tracer.start_span("llm_response")

            response = await self._generate_response(
                user_message=user_message,
                query_type=query_type,
                context=context,
                inferences=hybrid_context.inferences,
            )

            if self.tracer:
                self.tracer.end_span("completed")

            # 6. 출처 정보 추출 및 포맷팅 (외부 신호 포함)
            sources = self.source_provider.extract_sources(
                hybrid_context=hybrid_context,
                current_data=self._current_data,
                external_signals=external_signals,
                model=self.model,
            )
            formatted_sources = self.source_provider.format_sources_for_display(sources)

            # 실패한 신호 수집기 경고 추가
            failed_signal_warning = ""
            if failed_signals:
                failed_signal_warning = (
                    f"\n\n> ⚠️ **외부 신호 수집 실패**: {', '.join(failed_signals)}"
                )
                failed_signal_warning += (
                    "\n> *(위 데이터 소스는 현재 사용할 수 없습니다."
                    " 응답은 나머지 데이터를 기반으로 생성되었습니다.)*"
                )

            # 7. 응답에 출처 섹션 및 경고 추가
            full_response = response + failed_signal_warning + formatted_sources

            # 8. 대화 기록 저장
            self.context.add_user_message(user_message)
            self.context.add_assistant_message(full_response)

            # 9. 후속 질문 제안 (v2 - 응답 내용 분석 포함)
            suggestions = self.suggestion_engine.generate(
                query_type=query_type,
                entities=hybrid_context.entities,
                inferences=hybrid_context.inferences,
                response=full_response,
            )

            duration = (datetime.now() - start_time).total_seconds()

            if self.tracer:
                self.tracer.end_span("completed")

            # 감사 로깅 완료 (상세 메트릭 포함)
            self.logger.chat_response(
                request_context=audit_context,
                response=full_response,
                model=self.model,
                entities_extracted=hybrid_context.entities,
                intent_detected=query_type.value
                if hasattr(query_type, "value")
                else str(query_type),
                kg_facts_count=len(hybrid_context.ontology_facts),
                rag_chunks_count=len(hybrid_context.rag_chunks),
                inferences_count=len(hybrid_context.inferences),
                success=True,
            )

            # 응답 검증 (선택적)
            verification_result = None
            if self._enable_verification:
                try:
                    verification_context = {
                        "category": hybrid_context.entities.get("category")
                        if hybrid_context.entities
                        else None,
                        "brand": hybrid_context.entities.get("brand")
                        if hybrid_context.entities
                        else None,
                    }
                    verified = await self.verification_pipeline.verify(
                        full_response, context=verification_context, include_details=True
                    )
                    verification_result = self.verification_pipeline.get_verification_summary(
                        verified
                    )
                    self.logger.debug(
                        f"Verification: {verified.grade.value} ({verified.score:.0%})"
                    )
                except Exception as ve:
                    self.logger.warning(f"Verification failed: {ve}")
                    verification_result = None

            result = {
                "response": full_response,
                "query_type": query_type.value if hasattr(query_type, "value") else str(query_type),
                "is_fallback": False,
                "inferences": [inf.to_dict() for inf in hybrid_context.inferences],
                "sources": sources,
                "suggestions": suggestions,
                "entities": hybrid_context.entities,
                "query_info": {
                    "original": user_message,
                    "rewritten": search_query if rewrite_result.was_rewritten else None,
                    "was_rewritten": rewrite_result.was_rewritten,
                },
                "stats": {
                    "inferences_count": len(hybrid_context.inferences),
                    "rag_chunks_count": len(hybrid_context.rag_chunks),
                    "kg_facts_count": len(hybrid_context.ontology_facts),
                    "response_time_ms": duration * 1000,
                },
            }

            # 검증 결과 추가
            if verification_result:
                result["verification"] = verification_result

            return result

        except Exception as e:
            if self.tracer:
                self.tracer.end_span("failed", str(e))

            # 감사 로깅 (에러)
            self.logger.chat_response(
                request_context=audit_context,
                response="",
                model=self.model,
                success=False,
                error=str(e),
            )

            return {
                "response": "죄송합니다. 응답 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "query_type": "error",
                "is_fallback": True,
                "error": str(e),
                "inferences": [],
                "sources": [],
                "suggestions": self.suggestion_engine.get_fallback_suggestions(),
            }

    async def _generate_response(
        self,
        user_message: str,
        query_type: QueryType,
        context: str,
        inferences: list[InferenceResult],
    ) -> str:
        """LLM 응답 생성"""
        # 시스템 프롬프트 (카테고리 계층 인식 추가)
        system_prompt = self.context_builder.build_system_prompt(include_guardrails=True)

        # 카테고리 계층 및 순위 비교 규칙 추가
        system_prompt += """

## 카테고리 계층 구조 인식
- 제품은 여러 계층의 카테고리에 동시에 소속될 수 있습니다
- 예: 특정 립케어 제품이 "Lip Care"에서 4위이면서, 상위 카테고리인 "Beauty & Personal Care"에서는 73위일 수 있습니다
- 순위를 언급할 때는 반드시 어느 카테고리에서의 순위인지 명시하세요
- 카테고리 간 순위 차이가 있는 경우, 이는 자연스러운 현상입니다 (하위 카테고리가 더 세분화되어 경쟁 범위가 좁기 때문)

## ⚠️ 순위 비교 규칙 (중요)
- 순위 변동 분석은 **반드시 동일 카테고리 내에서만** 유효합니다
- 예시 (올바름): "Lip Care 4위 → Lip Care 6위 = 2단계 하락"
- 예시 (잘못됨): "Lip Care 4위 → Beauty 67위 = 63단계 하락" ← 서로 다른 카테고리이므로 비교 불가
- 30위 이상의 급격한 순위 변동이 감지되면, 카테고리 혼동이 아닌지 먼저 확인하세요
- 순위 변동을 보고할 때는 항상 [카테고리명]을 명시하세요

## 브랜드명 정규화 규칙
다음 브랜드명은 잘린 이름이므로 정식 명칭으로 사용하세요:
- "Burt's" → "Burt's Bees"
- "wet" → "wet n wild"
- "Tree" → "Tree Hut"
- "Summer" → "Summer Fridays"
- "Rare" → "Rare Beauty"
- "La" → "La Roche-Posay"
- "Beauty" (단독 사용 시) → "Beauty of Joseon"
- "Tower" → "Tower 28"
- "Drunk" → "Drunk Elephant"
- "Paula's" → "Paula's Choice"
- "The" (단독 사용 시) → "The Ordinary"
- 주요 브랜드 외 브랜드는 "소규모/신흥 브랜드" 또는 "Non-major Brands"로 표현
- ⚠️ "Unknown", "기타 브랜드(Unknown)", "미확인 브랜드" 표현 절대 금지
"""

        # 대화 히스토리
        conversation = self.context.get_conversation_summary()

        # 추론 결과 강조
        inference_summary = ""
        if inferences:
            inference_lines = []
            for inf in inferences[:3]:
                inference_lines.append(f"- [{inf.insight_type.value}] {inf.insight}")
            inference_summary = "\n".join(inference_lines)

        # 카테고리 계층 컨텍스트 추출 (마지막 하이브리드 컨텍스트에서)
        category_hierarchy_context = ""
        if self._last_hybrid_context and self._last_hybrid_context.entities:
            category_hierarchy_context = self._build_category_hierarchy_context(
                self._last_hybrid_context.entities
            )

        user_prompt = f"""
{context}

---

## 카테고리 계층 정보
{category_hierarchy_context if category_hierarchy_context else "카테고리 계층 정보 없음"}

## 온톨로지 추론 결과 (우선 참고)
{inference_summary if inference_summary else "관련 추론 결과 없음"}

## 이전 대화
{conversation if conversation else "없음"}

## 사용자 질문
{user_message}

---

요구사항:
1. 온톨로지 추론 결과가 있으면 이를 기반으로 답변
2. 구체적인 수치를 인용하여 답변
3. 순위를 언급할 때는 카테고리를 명시 (예: "Lip Care에서 4위", "Beauty & Personal Care 전체에서는 73위")
4. 불확실한 부분은 명확히 밝힘
5. 단정적 표현 대신 가능성 표현 사용
6. 간결하고 명확하게 답변
7. 외부 뉴스/기사를 인용할 때는 반드시 [출처명, 날짜] 형식으로 표시
   예: "LANEIGE가 글래스 스킨 트렌드를 선도하고 있습니다 [Allure, 2026-01-20]"
8. Reddit/YouTube 등 소셜 데이터도 [Reddit r/서브레딧, 날짜] 형식으로 인용
"""

        try:
            response = await acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=800,
            )

            if response.choices:
                answer = response.choices[0].message.content
            else:
                answer = "죄송합니다. 응답을 생성하지 못했습니다."

            # 토큰 사용량 기록
            if self.metrics and hasattr(response, "usage"):
                self.metrics.record_llm_call(
                    model=self.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    latency_ms=0,
                    cost=self._estimate_cost(
                        response.usage.prompt_tokens, response.usage.completion_tokens
                    ),
                )

            # 가드레일 적용
            answer = self.templates.apply_guardrails(answer)

            # 브랜드명 정규화 적용
            answer = self._normalize_response_brands(answer)

            return answer

        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return self._generate_fallback_response(inferences)

    def _generate_fallback_response(self, inferences: list[InferenceResult]) -> str:
        """폴백 응답 생성"""
        if inferences:
            lines = ["분석 결과를 바탕으로 답변드립니다:\n"]
            for inf in inferences[:2]:
                lines.append(f"- {inf.insight}")
                if inf.recommendation:
                    lines.append(f"  → 권장: {inf.recommendation}")
            return "\n".join(lines)

        return "죄송합니다. 현재 응답을 생성할 수 없습니다. 잠시 후 다시 시도해주세요."

    def _normalize_brand(self, brand: str) -> str:
        """브랜드명 정규화"""
        if not brand or brand == "Unknown":
            return brand

        brand_lower = brand.lower().strip()

        # 정규화 매핑에서 찾기
        if brand_lower in self.BRAND_NORMALIZATION:
            return self.BRAND_NORMALIZATION[brand_lower]

        return brand

    def _normalize_response_brands(self, response: str) -> str:
        """응답 내 브랜드명 정규화"""
        import re

        # 특수 케이스: 아포스트로피가 포함된 브랜드명
        special_brands = {
            "Burt's": ("Burt's Bees", r"(?i)\bBurt's(?!\s*Bees)"),
            "Paula's": ("Paula's Choice", r"(?i)\bPaula's(?!\s*Choice)"),
        }

        for _truncated, (full, pattern) in special_brands.items():
            if full.lower() not in response.lower():
                response = re.sub(pattern, full, response)

        # 일반 브랜드명 정규화
        for truncated, full in self.BRAND_NORMALIZATION.items():
            # 아포스트로피 브랜드는 위에서 처리했으므로 스킵
            if "'" in truncated:
                continue

            # 단어 경계를 사용하여 정확히 매칭 (대소문자 무시)
            pattern = rf"\b{re.escape(truncated)}\b"
            # 이미 전체 브랜드명이 포함된 경우는 제외
            if full.lower() not in response.lower():
                response = re.sub(pattern, full, response, flags=re.IGNORECASE)

        return response

    def _build_category_hierarchy_context(self, entities: dict[str, list[str]]) -> str:
        """
        카테고리 계층 컨텍스트 생성

        Args:
            entities: 추출된 엔티티 (카테고리, 제품 등)

        Returns:
            카테고리 계층 정보 문자열
        """
        if not self.kg:
            return ""

        context_parts = []

        # 카테고리 엔티티에서 계층 정보 추출
        if not entities:
            return ""

        categories = entities.get("categories", [])
        for category in categories:
            hierarchy = self.kg.get_category_hierarchy(category)
            if "error" in hierarchy:
                continue

            # 현재 카테고리 정보
            context_parts.append(f"**{hierarchy['name']}** (Level {hierarchy['level']})")

            # 상위 카테고리 경로
            if hierarchy.get("ancestors"):
                path = " > ".join([a["name"] for a in reversed(hierarchy["ancestors"])])
                context_parts.append(f"  - 상위 경로: {path} > {hierarchy['name']}")

            # 하위 카테고리
            if hierarchy.get("descendants"):
                children = ", ".join([d["name"] for d in hierarchy["descendants"][:5]])
                context_parts.append(f"  - 하위 카테고리: {children}")

            context_parts.append("")

        # 제품의 카테고리 컨텍스트 (순위 관련 질문 시)
        products = entities.get("products", [])
        for product_asin in products:
            product_ctx = self.kg.get_product_category_context(product_asin)
            if product_ctx.get("categories"):
                context_parts.append(f"**제품 {product_asin}의 카테고리별 순위:**")
                for cat_info in product_ctx["categories"]:
                    hierarchy = cat_info.get("hierarchy", {})
                    cat_name = hierarchy.get("name", cat_info.get("category_id"))
                    rank = cat_info.get("rank", "N/A")
                    context_parts.append(f"  - {cat_name}: {rank}위")
                context_parts.append("")

        return "\n".join(context_parts) if context_parts else ""

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """비용 추정"""
        input_cost = (prompt_tokens / 1_000_000) * 0.40
        output_cost = (completion_tokens / 1_000_000) * 1.60
        return round(input_cost + output_cost, 6)

    def get_conversation_history(self, limit: int = 10) -> list[dict]:
        """대화 기록 조회"""
        return self.context.get_conversation_history(limit)

    def clear_conversation(self) -> None:
        """대화 기록 초기화"""
        self.context.reset()
        self.query_rewriter.clear_cache()

    async def _maybe_rewrite_query(self, query: str) -> RewriteResult:
        """
        필요시 질문 재구성 (대화 맥락 기반)

        후속 질문에서 지시어(그것, 그 제품, 해당 등)를 이전 대화 맥락을 참조하여
        구체적인 대상으로 치환합니다.

        최적화:
        1. 대화 히스토리가 없으면 스킵
        2. 지시어가 없으면 스킵 (LLM 호출 절약)

        Args:
            query: 사용자 질문

        Returns:
            RewriteResult 객체
        """
        # 대화 히스토리가 없으면 스킵
        history = self.context.get_conversation_history(limit=3)
        if not history:
            return create_rewrite_result_no_change(query)

        # 지시어가 없으면 스킵 (LLM 호출 절약)
        if not self.query_rewriter.needs_rewrite(query):
            return create_rewrite_result_no_change(query)

        # LLM으로 재구성
        return await self.query_rewriter.rewrite(query, history)

    async def explain_last_response(self) -> str:
        """마지막 응답의 추론 과정 설명"""
        if not self._last_hybrid_context or not self._last_hybrid_context.inferences:
            return "설명할 추론 결과가 없습니다."

        return self.reasoner.explain_all(self._last_hybrid_context.inferences)


class HybridChatbotSession:
    """하이브리드 챗봇 세션 관리 (멀티 유저 지원)"""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph | None = None,
        reasoner: OntologyReasoner | None = None,
    ):
        """
        Args:
            knowledge_graph: 공유 지식 그래프
            reasoner: 공유 추론기
        """
        self._sessions: dict[str, HybridChatbotAgent] = {}
        self._shared_kg = knowledge_graph
        self._shared_reasoner = reasoner

    def get_or_create(self, session_id: str, **kwargs) -> HybridChatbotAgent:
        """세션별 챗봇 인스턴스 반환"""
        if session_id not in self._sessions:
            self._sessions[session_id] = HybridChatbotAgent(
                knowledge_graph=self._shared_kg, reasoner=self._shared_reasoner, **kwargs
            )
        return self._sessions[session_id]

    def close_session(self, session_id: str) -> None:
        """세션 종료"""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def list_sessions(self) -> list[str]:
        """활성 세션 목록"""
        return list(self._sessions.keys())

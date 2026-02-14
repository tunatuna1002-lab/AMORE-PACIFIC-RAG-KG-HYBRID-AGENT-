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

from src.domain.entities.relations import InferenceResult
from src.memory.context import ContextManager
from src.monitoring.logger import AgentLogger
from src.monitoring.metrics import QualityMetrics
from src.monitoring.tracer import ExecutionTracer
from src.ontology.business_rules import register_all_rules
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.rag.context_builder import CompactContextBuilder, ContextBuilder
from src.rag.hybrid_retriever import HybridContext, HybridRetriever
from src.rag.query_rewriter import QueryRewriter, RewriteResult, create_rewrite_result_no_change
from src.rag.retriever import DocumentRetriever
from src.rag.router import QueryType, RAGRouter
from src.rag.templates import ResponseTemplates

logger = logging.getLogger(__name__)


class HybridChatbotAgent:
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
        self.model = model or config.get("model", "gpt-4.1-mini")

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

        # 온톨로지 컴포넌트
        self.kg = knowledge_graph or KnowledgeGraph()
        self.reasoner = reasoner or OntologyReasoner(self.kg)

        # 비즈니스 규칙 등록
        if not self.reasoner.rules:
            register_all_rules(self.reasoner)

        # RAG 컴포넌트
        self.doc_retriever = DocumentRetriever(docs_dir)
        self.router = RAGRouter()

        # 하이브리드 검색기
        self.hybrid_retriever = HybridRetriever(
            knowledge_graph=self.kg,
            reasoner=self.reasoner,
            doc_retriever=self.doc_retriever,
            auto_init_rules=False,
        )

        # 컨텍스트 빌더
        self.context_builder = ContextBuilder(max_tokens=3000)
        self.compact_builder = CompactContextBuilder(max_tokens=1500)

        # 템플릿
        self.templates = ResponseTemplates()

        # 메모리
        self.context = context_manager or ContextManager()

        # 모니터링
        self.logger = logger or AgentLogger("hybrid_chatbot")
        self.tracer = tracer
        self.metrics = metrics

        # 현재 데이터 컨텍스트
        self._current_data: dict[str, Any] = {}

        # 마지막 하이브리드 컨텍스트
        self._last_hybrid_context: HybridContext | None = None

        # Query Rewriter (대화 맥락 기반 질문 재구성)
        self.query_rewriter = QueryRewriter(model=model)

        # 외부 신호 수집기 (Tavily + RSS + Reddit)
        self._external_signal_collector = None
        self._last_external_signals: list[Any] = []

        # 응답 검증 파이프라인 (지연 초기화)
        self._verification_pipeline: Any = None
        self._enable_verification = config.get("enable_verification", True)

        # 분해된 컴포넌트 (feature-flag-guarded)
        from src.infrastructure.feature_flags import FeatureFlags

        flags = FeatureFlags.get_instance()
        if flags.use_decomposed_chatbot():
            from src.agents.external_signal_manager import ExternalSignalManager
            from src.agents.source_provider import SourceProvider
            from src.agents.suggestion_engine import SuggestionEngine

            self.suggestion_engine = SuggestionEngine(knowledge_graph=self.kg, config=config)
            self.source_provider = SourceProvider(config=config, knowledge_graph=self.kg)
            self.signal_manager = ExternalSignalManager(config=config)

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
                    "suggestions": self.suggestion_engine.get_fallback_suggestions()
                    if hasattr(self, "suggestion_engine")
                    else self._get_fallback_suggestions(),
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
            if hasattr(self, "signal_manager"):
                external_signals = await self.signal_manager.collect(
                    query=search_query, entities=hybrid_context.entities
                )
                self._last_external_signals = external_signals
                failed_signals = self.signal_manager.get_failed_collectors()
            else:
                external_signals = await self._collect_external_signals(
                    query=search_query, entities=hybrid_context.entities
                )
                self._last_external_signals = external_signals
                failed_signals = self._get_failed_signal_collectors()

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
            if hasattr(self, "source_provider"):
                sources = self.source_provider.extract_sources(
                    hybrid_context=hybrid_context,
                    current_data=self._current_data,
                    external_signals=external_signals,
                    model=self.model,
                )
                formatted_sources = self.source_provider.format_sources_for_display(sources)
            else:
                sources = self._extract_sources(hybrid_context, external_signals)
                formatted_sources = self._format_sources_for_response(sources)

            # 실패한 신호 수집기 경고 추가
            failed_signal_warning = ""
            if failed_signals:
                failed_signal_warning = (
                    f"\n\n> ⚠️ **외부 신호 수집 실패**: {', '.join(failed_signals)}"
                )
                failed_signal_warning += "\n> *(위 데이터 소스는 현재 사용할 수 없습니다. 응답은 나머지 데이터를 기반으로 생성되었습니다.)*"

            # 7. 응답에 출처 섹션 및 경고 추가
            full_response = response + failed_signal_warning + formatted_sources

            # 8. 대화 기록 저장
            self.context.add_user_message(user_message)
            self.context.add_assistant_message(full_response)

            # 9. 후속 질문 제안 (v2 - 응답 내용 분석 포함)
            if hasattr(self, "suggestion_engine"):
                suggestions = self.suggestion_engine.generate(
                    query_type=query_type,
                    entities=hybrid_context.entities,
                    inferences=hybrid_context.inferences,
                    response=full_response,
                )
            else:
                suggestions = self._generate_suggestions(
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
                "suggestions": self.suggestion_engine.get_fallback_suggestions()
                if hasattr(self, "suggestion_engine")
                else self._get_fallback_suggestions(),
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

    def _generate_suggestions(
        self,
        query_type: QueryType,
        entities: dict[str, list[str]],
        inferences: list[InferenceResult],
        response: str = "",
    ) -> list[str]:
        """
        후속 질문 제안 (v2 - 개선 버전)

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
        import re

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
            try:
                competitors = self.kg.get_related_brands(brand, limit=2)
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

    def _get_fallback_suggestions(self) -> list[str]:
        """폴백 제안"""
        return ["SoS(점유율)에 대해 알려주세요", "오늘의 주요 인사이트는?", "LANEIGE 현재 순위는?"]

    async def _generate_llm_suggestions(
        self, user_query: str, response_summary: str, entities: dict[str, list[str]]
    ) -> list[str]:
        """
        LLM 기반 후속 질문 생성 (Phase 4)

        비용: ~$0.0002/호출 (GPT-4.1-mini 기준)

        Args:
            user_query: 사용자 질문
            response_summary: AI 응답 요약 (300자 제한)
            entities: 추출된 엔티티

        Returns:
            3개의 후속 질문 리스트 (실패 시 빈 리스트)
        """
        import json

        from src.shared.constants import SUGGESTION_MAX_TOKENS, SUGGESTION_TEMPERATURE

        prompt = f"""당신은 AMORE Pacific 시장 분석 챗봇입니다.
사용자와의 대화를 이어가기 위한 후속 질문 3개를 생성하세요.

[사용자 질문]
{user_query}

[AI 응답 요약]
{response_summary[:300]}

[추출된 엔티티]
- 브랜드: {", ".join(entities.get("brands", [])) or "없음"}
- 카테고리: {", ".join(entities.get("categories", [])) or "없음"}
- 지표: {", ".join(entities.get("indicators", [])) or "없음"}

[규칙]
1. 대화 흐름에 자연스럽게 이어지는 질문
2. 구체적이고 실행 가능한 질문
3. 20자 이내로 간결하게
4. JSON 배열 형식으로만 응답

응답 형식: ["질문1", "질문2", "질문3"]"""

        try:
            response = await acompletion(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=SUGGESTION_TEMPERATURE,
                max_tokens=SUGGESTION_MAX_TOKENS,
            )

            content = response.choices[0].message.content.strip()
            # JSON 파싱
            suggestions = json.loads(content)
            if isinstance(suggestions, list):
                return [str(s) for s in suggestions[:3]]
            return []

        except Exception as e:
            self.logger.warning(f"LLM suggestion generation failed: {e}")
            return []  # 폴백은 기존 로직으로

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

    def _extract_sources(
        self, hybrid_context: HybridContext, external_signals: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        출처 정보 추출 (Perplexity/Liner 스타일 상세 출처 제공)

        Args:
            hybrid_context: 하이브리드 검색 컨텍스트
            external_signals: 외부 신호 리스트 (Tavily 뉴스, RSS, Reddit 등)

        Returns:
            출처 정보 리스트 (유형별 상세 정보 포함)
        """
        sources = []

        # 1. 크롤링 데이터 출처 - URL 및 상세 정보 추가 (ASIN 포함 - E2E Audit 2026-01-27)
        if self._current_data:
            metadata = self._current_data.get("metadata", {})
            data_date = metadata.get("data_date", "")
            categories = self._current_data.get("categories", {})

            total_products = (
                sum(len(cat_data.get("rank_records", [])) for cat_data in categories.values())
                if categories
                else 0
            )

            # 질의에서 언급된 제품의 ASIN 추출 (provenance chain 강화)
            mentioned_asins = self._extract_mentioned_asins(hybrid_context, categories)

            crawled_source = {
                "type": "crawled_data",
                "icon": "📊",
                "description": "Amazon Best Sellers 크롤링 데이터",
                "collected_at": data_date,
                "url": "https://www.amazon.com/gp/bestsellers/beauty",
                "details": {
                    "categories": list(categories.keys()) if categories else [],
                    "total_products": total_products,
                    "snapshot_date": data_date,
                },
            }

            # 관련 제품의 ASIN 정보 추가
            if mentioned_asins:
                crawled_source["mentioned_products"] = mentioned_asins

            sources.append(crawled_source)

        # 2. Knowledge Graph 출처 - 엔티티 및 관계 정보 추가
        if hybrid_context.ontology_facts:
            sources.append(
                {
                    "type": "knowledge_graph",
                    "icon": "🔗",
                    "description": "지식 그래프 관계 데이터",
                    "fact_count": len(hybrid_context.ontology_facts),
                    "entities": self._extract_entity_names(hybrid_context.ontology_facts),
                    "relations": self._extract_relation_types(hybrid_context.ontology_facts),
                    "details": {
                        "source": "Amazon US 실시간 데이터 기반 지식 그래프",
                        "fact_count": len(hybrid_context.ontology_facts),
                    },
                }
            )

        # 3. 온톨로지 추론 출처 - 규칙 상세 정보
        if hybrid_context.inferences:
            for inf in hybrid_context.inferences:
                sources.append(
                    {
                        "type": "ontology_inference",
                        "icon": "🧠",
                        "description": f"온톨로지 규칙: {inf.rule_name}",
                        "rule_name": inf.rule_name,
                        "confidence": inf.confidence,
                        "evidence": inf.evidence,
                        "insight_type": inf.insight_type.value
                        if hasattr(inf.insight_type, "value")
                        else str(inf.insight_type),
                        "details": {"insight": inf.insight, "recommendation": inf.recommendation},
                    }
                )

        # 4. RAG 문서 출처 - 파일 경로 및 관련성 점수
        rag_sources_map = {}
        for chunk in hybrid_context.rag_chunks:
            metadata = chunk.get("metadata", {})
            doc_id = metadata.get("doc_id", "")
            title = metadata.get("title", "")
            file_path = metadata.get("file_path", "")
            score = chunk.get("score", 0)
            section = metadata.get("section", "")

            if doc_id or title:
                doc_key = doc_id or title
                # 같은 문서의 여러 청크 중 가장 높은 점수만 유지
                if doc_key not in rag_sources_map or score > rag_sources_map[doc_key].get(
                    "relevance_score", 0
                ):
                    rag_sources_map[doc_key] = {
                        "type": "rag_document",
                        "icon": "📄",
                        "description": title or doc_id,
                        "file_path": file_path,
                        "section": section,
                        "relevance_score": score,
                        "details": {"doc_id": doc_id, "title": title},
                    }

        sources.extend(rag_sources_map.values())

        # 5. 카테고리 계층 출처 (신규)
        if hybrid_context.entities and hybrid_context.entities.get("categories"):
            for category in hybrid_context.entities["categories"][:3]:  # 최대 3개
                hierarchy = self.kg.get_category_hierarchy(category)
                if "error" not in hierarchy:
                    path = []
                    if hierarchy.get("ancestors"):
                        path = [a["name"] for a in reversed(hierarchy["ancestors"])]
                    path.append(hierarchy.get("name", category))

                    sources.append(
                        {
                            "type": "category_hierarchy",
                            "icon": "🗂️",
                            "description": "카테고리 계층 구조",
                            "path": path,
                            "level": hierarchy.get("level", 0),
                            "url": hierarchy.get("url", ""),
                            "details": {"category": category, "hierarchy_depth": len(path)},
                        }
                    )

        # 6. 외부 신호 출처 (Tavily 뉴스, RSS, Reddit 등)
        if external_signals:
            for signal in external_signals[:5]:  # 상위 5개만
                # ExternalSignal 객체에서 정보 추출
                signal_source = getattr(signal, "source", "unknown")
                reliability = 0.7  # 기본값

                # 메타데이터에서 신뢰도 추출
                if hasattr(signal, "metadata") and signal.metadata:
                    reliability = signal.metadata.get("reliability_score", 0.7)

                # 소스 유형에 따라 아이콘 결정
                if "tavily" in signal_source.lower() or "news" in signal_source.lower():
                    icon = "📰"
                    source_type = "external_news"
                elif "reddit" in signal_source.lower():
                    icon = "💬"
                    source_type = "social_media"
                elif "rss" in signal_source.lower():
                    icon = "📡"
                    source_type = "rss_feed"
                elif "youtube" in signal_source.lower():
                    icon = "📺"
                    source_type = "social_media"
                else:
                    icon = "🌐"
                    source_type = "external_source"

                sources.append(
                    {
                        "type": source_type,
                        "icon": icon,
                        "description": getattr(signal, "title", "Unknown"),
                        "source": signal_source,
                        "url": getattr(signal, "url", ""),
                        "published_at": getattr(signal, "published_at", ""),
                        "reliability_score": reliability,
                        "relevance_score": getattr(signal, "relevance_score", 0.5),
                        "details": {
                            "content_preview": getattr(signal, "content", "")[:200]
                            if hasattr(signal, "content")
                            else "",
                            "tier": getattr(signal, "tier", "unknown"),
                        },
                    }
                )

        # 7. AI 모델 출처 (항상 포함)
        sources.append(
            {
                "type": "ai_model",
                "icon": "🤖",
                "description": f"AI 분석: {self.model}",
                "model": self.model,
                "disclaimer": "AI가 생성한 분석입니다. 중요한 의사결정 시 추가 검증을 권장합니다.",
                "generated_at": datetime.now().isoformat(),
            }
        )

        return sources

    def _extract_entity_names(self, ontology_facts) -> list[str]:
        """
        KG facts에서 엔티티 이름 추출

        Args:
            ontology_facts: 온톨로지 사실 리스트 또는 딕셔너리

        Returns:
            엔티티 이름 리스트 (최대 5개)
        """
        entities = set()

        if isinstance(ontology_facts, list):
            for fact in ontology_facts:
                if isinstance(fact, dict):
                    subject = fact.get("subject", "")
                    obj = fact.get("object", "")
                    if subject:
                        entities.add(subject)
                    if obj:
                        entities.add(obj)
        elif isinstance(ontology_facts, dict):
            # 단일 fact인 경우
            subject = ontology_facts.get("subject", "")
            obj = ontology_facts.get("object", "")
            if subject:
                entities.add(subject)
            if obj:
                entities.add(obj)

        # None이나 빈 문자열 제거 후 최대 5개 반환
        return list(filter(None, entities))[:5]

    def _extract_relation_types(self, ontology_facts) -> list[str]:
        """
        KG facts에서 관계 유형 추출

        Args:
            ontology_facts: 온톨로지 사실 리스트 또는 딕셔너리

        Returns:
            관계 유형 리스트 (중복 제거)
        """
        relations = set()

        if isinstance(ontology_facts, list):
            for fact in ontology_facts:
                if isinstance(fact, dict):
                    predicate = fact.get("predicate", "")
                    if predicate:
                        relations.add(predicate)
        elif isinstance(ontology_facts, dict):
            # 단일 fact인 경우
            predicate = ontology_facts.get("predicate", "")
            if predicate:
                relations.add(predicate)

        # None이나 빈 문자열 제거
        return list(filter(None, relations))

    def _extract_mentioned_asins(
        self, hybrid_context: HybridContext, categories: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        질의에서 언급된 제품의 ASIN 정보 추출 (E2E Audit - 2026-01-27)

        Args:
            hybrid_context: 하이브리드 검색 컨텍스트
            categories: 크롤링된 카테고리 데이터

        Returns:
            제품 ASIN 정보 리스트 [{asin, name, brand, rank, category, url}]
        """
        mentioned_products = []
        seen_asins = set()

        # 1. KG 엔티티에서 제품명/브랜드 추출
        mentioned_brands = set()
        if hybrid_context.entities:
            mentioned_brands = set(hybrid_context.entities.get("brands", []))

        # 2. 카테고리 데이터에서 관련 제품 ASIN 추출
        for category_id, cat_data in categories.items():
            rank_records = cat_data.get("rank_records", [])

            for record in rank_records:
                asin = record.get("asin", "")
                brand = record.get("brand", "")
                product_name = record.get("product_name", record.get("title", ""))
                rank = record.get("rank", 0)

                # 이미 처리된 ASIN 스킵
                if asin in seen_asins:
                    continue

                # 언급된 브랜드의 제품만 포함 (최대 5개)
                if brand in mentioned_brands:
                    seen_asins.add(asin)
                    mentioned_products.append(
                        {
                            "asin": asin,
                            "name": product_name,
                            "brand": brand,
                            "rank": rank,
                            "category": category_id,
                            "url": f"https://www.amazon.com/dp/{asin}" if asin else "",
                        }
                    )

                    if len(mentioned_products) >= 5:
                        break

            if len(mentioned_products) >= 5:
                break

        # 순위 기준 정렬
        mentioned_products.sort(key=lambda x: x.get("rank", 999))
        return mentioned_products[:5]

    def _format_sources_for_response(self, sources: list[dict[str, Any]]) -> str:
        """
        출처를 응답에 포함할 형식으로 변환 (Perplexity 스타일)

        Args:
            sources: 출처 정보 리스트

        Returns:
            마크다운 형식의 출처 섹션
        """
        if not sources:
            return ""

        lines = ["\n\n---"]

        # 데이터 출처 시점을 명확히 표시 (사용자 요청)
        crawled_source = next((s for s in sources if s["type"] == "crawled_data"), None)
        if crawled_source:
            collected_at = crawled_source.get("collected_at", "")
            if collected_at:
                lines.append(f"📅 **데이터 기준: Amazon US Best Sellers {collected_at} 수집**")
                lines.append("*(Amazon은 Best Sellers 순위를 매 시간 업데이트합니다)*")
                lines.append("")

        lines.extend(["**📚 출처 및 참고자료:**", ""])

        for i, source in enumerate(sources, 1):
            icon = source.get("icon", "•")
            desc = source.get("description", "알 수 없는 출처")

            if source["type"] == "crawled_data":
                collected = source.get("collected_at", "")
                url = source.get("url", "")
                details = source.get("details", {})
                total = details.get("total_products", 0)
                mentioned_products = source.get("mentioned_products", [])

                lines.append(f"{i}. {icon} **{desc}**")
                lines.append(f"   - 수집일: {collected}")
                if url:
                    lines.append(f"   - URL: {url}")
                if total > 0:
                    lines.append(f"   - 총 제품 수: {total}개")

                # ASIN 기반 제품 추적 정보 표시 (E2E Audit - 2026-01-27)
                if mentioned_products:
                    lines.append("   - 📦 관련 제품 (ASIN 기준):")
                    for prod in mentioned_products[:3]:  # 최대 3개 표시
                        asin = prod.get("asin", "")
                        name = prod.get("name", "")
                        rank = prod.get("rank", "")
                        category = prod.get("category", "")
                        lines.append(f"     • [{asin}] {name} (#{rank} in {category})")

                lines.append("")

            elif source["type"] == "knowledge_graph":
                fact_count = source.get("fact_count", 0)
                entities = source.get("entities", [])
                relations = source.get("relations", [])
                lines.append(f"{i}. {icon} **{desc}** ({fact_count}개 관계)")
                if entities:
                    lines.append(f"   - 주요 엔티티: {', '.join(entities[:3])}")
                if relations:
                    lines.append(f"   - 관계 유형: {', '.join(relations[:3])}")
                lines.append("")

            elif source["type"] == "ontology_inference":
                conf = source.get("confidence", 0) * 100
                rule_name = source.get("rule_name", "알 수 없음")
                lines.append(f"{i}. {icon} **{desc}**")
                lines.append(f"   - 신뢰도: {conf:.0f}%")
                lines.append(f"   - 규칙: {rule_name}")
                lines.append("")

            elif source["type"] == "rag_document":
                file_path = source.get("file_path", "")
                section = source.get("section", "")
                score = source.get("relevance_score", 0)
                file_name = file_path.split("/")[-1] if file_path else ""
                lines.append(f"{i}. {icon} **{desc}**")
                if file_name:
                    lines.append(f"   - 파일: {file_name}")
                if section:
                    lines.append(f"   - 섹션: {section}")
                if score > 0:
                    lines.append(f"   - 관련도: {score:.2f}")
                lines.append("")

            elif source["type"] == "category_hierarchy":
                path = source.get("path", [])
                level = source.get("level", 0)
                url = source.get("url", "")
                lines.append(f"{i}. {icon} **{desc}**")
                if path:
                    lines.append(f"   - 계층: {' > '.join(path)}")
                lines.append(f"   - 레벨: {level}")
                if url:
                    lines.append(f"   - URL: {url}")
                lines.append("")

            elif source["type"] in ["external_news", "rss_feed"]:
                # 외부 뉴스 / RSS 피드 (Tavily, Allure, WWD 등)
                url = source.get("url", "")
                published_at = source.get("published_at", "")
                reliability = source.get("reliability_score", 0.7) * 100
                source_name = source.get("source", "")
                lines.append(f"{i}. {icon} **{desc}** (신뢰도: {reliability:.0f}%)")
                if source_name:
                    lines.append(f"   - 출처: {source_name}")
                if published_at:
                    lines.append(f"   - 날짜: {published_at}")
                if url:
                    lines.append(f"   - URL: {url}")
                lines.append("")

            elif source["type"] == "social_media":
                # 소셜 미디어 (Reddit, YouTube 등)
                url = source.get("url", "")
                published_at = source.get("published_at", "")
                reliability = source.get("reliability_score", 0.5) * 100
                source_name = source.get("source", "")
                relevance = source.get("relevance_score", 0)
                lines.append(f"{i}. {icon} **{desc}** (신뢰도: {reliability:.0f}%)")
                if source_name:
                    lines.append(f"   - 플랫폼: {source_name}")
                if published_at:
                    lines.append(f"   - 날짜: {published_at}")
                if relevance > 0:
                    lines.append(f"   - 관련도: {relevance:.2f}")
                if url:
                    lines.append(f"   - URL: {url}")
                lines.append("")

            elif source["type"] == "ai_model":
                model = source.get("model", "")
                disclaimer = source.get("disclaimer", "")
                lines.append(f"{i}. {icon} **{desc}**")
                if model:
                    lines.append(f"   - 모델: {model}")
                if disclaimer:
                    lines.append(f"   - 참고: {disclaimer}")
                lines.append("")

        return "\n".join(lines)

    async def _collect_external_signals(
        self, query: str, entities: dict[str, list[str]] | None = None
    ) -> list[Any]:
        """
        외부 신호 수집 (Tavily 뉴스, RSS, Reddit)

        Args:
            query: 사용자 질문
            entities: 추출된 엔티티 (브랜드, 카테고리 등)

        Returns:
            ExternalSignal 리스트
        """
        try:
            # 외부 신호 수집기 lazy initialization
            if self._external_signal_collector is None:
                try:
                    from src.tools.collectors.external_signal_collector import (
                        ExternalSignalCollector,
                    )

                    self._external_signal_collector = ExternalSignalCollector()
                    await self._external_signal_collector.initialize()
                except ImportError as e:
                    self.logger.warning(f"ExternalSignalCollector not available: {e}")
                    return []
                except Exception as e:
                    self.logger.warning(f"Failed to initialize ExternalSignalCollector: {e}")
                    return []

            # 엔티티에서 브랜드/토픽 추출
            brands = []
            topics = []

            if entities:
                brands = entities.get("brands", [])
                categories = entities.get("categories", [])
                # 카테고리를 토픽으로 변환
                topics = [cat.replace("_", " ") for cat in categories]

            # 기본값 설정
            if not brands:
                brands = ["LANEIGE", "K-Beauty"]
            if not topics:
                topics = ["skincare trends", "beauty news"]

            # Tavily 뉴스 검색 (비동기) - 검색 기간 확장
            all_signals = []

            try:
                tavily_signals = await self._external_signal_collector.fetch_tavily_news(
                    brands=brands[:3],  # 최대 3개 브랜드
                    topics=topics[:2],  # 최대 2개 토픽
                    days=14,  # 2주로 확장 (더 많은 뉴스 수집)
                    max_results=8,  # 최대 8개로 증가
                )
                all_signals.extend(tavily_signals)
                self.logger.info(f"Collected {len(tavily_signals)} Tavily news signals")
            except Exception as e:
                self.logger.warning(f"Tavily news fetch failed: {e}")

            # RSS 피드 수집 (선택적)
            try:
                keywords = brands + topics
                rss_signals = await self._external_signal_collector.fetch_all_rss_feeds(
                    keywords=keywords[:5]
                )
                # 상위 3개만 추가
                all_signals.extend(rss_signals[:3])
                self.logger.debug(f"Collected {len(rss_signals)} RSS signals")
            except Exception as e:
                self.logger.debug(f"RSS fetch skipped: {e}")

            # 신뢰도 * 관련성 점수로 정렬하여 상위 8개 반환
            all_signals.sort(
                key=lambda s: (
                    getattr(s, "metadata", {}).get("reliability_score", 0.7)
                    * getattr(s, "relevance_score", 0.5)
                ),
                reverse=True,
            )

            return all_signals[:8]

        except Exception as e:
            self.logger.error(f"External signal collection failed: {e}")
            return []

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

    def _get_failed_signal_collectors(self) -> list[str]:
        """
        사용 불가능한 외부 신호 수집기 목록 반환

        Returns:
            실패한 수집기 이름 리스트
        """
        failed = []

        # ExternalSignalCollector 체크
        if self._external_signal_collector is None:
            try:
                import importlib.util

                if importlib.util.find_spec("src.tools.external_signal_collector") is None:
                    failed.append("External Signals (Tavily/RSS/Reddit)")
            except ImportError:
                failed.append("External Signals (Tavily/RSS/Reddit)")

        return failed

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

    def get_last_hybrid_context(self) -> HybridContext | None:
        """마지막 하이브리드 컨텍스트"""
        return self._last_hybrid_context

    def get_knowledge_graph(self) -> KnowledgeGraph:
        """지식 그래프 반환"""
        return self.kg

    def get_reasoner(self) -> OntologyReasoner:
        """추론기 반환"""
        return self.reasoner

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

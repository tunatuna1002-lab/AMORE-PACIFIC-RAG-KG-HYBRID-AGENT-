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

from datetime import datetime
from typing import Dict, Any, List, Optional

from litellm import acompletion

from ontology.knowledge_graph import KnowledgeGraph
from ontology.reasoner import OntologyReasoner
from ontology.business_rules import register_all_rules
from ontology.relations import InferenceResult

from rag.hybrid_retriever import HybridRetriever, HybridContext, EntityExtractor
from rag.context_builder import ContextBuilder, CompactContextBuilder
from rag.router import RAGRouter, QueryType
from rag.retriever import DocumentRetriever
from rag.templates import ResponseTemplates

from memory.context import ContextManager

from monitoring.logger import AgentLogger
from monitoring.tracer import ExecutionTracer
from monitoring.metrics import QualityMetrics


class HybridChatbotAgent:
    """
    Ontology-RAG 하이브리드 챗봇 에이전트

    기존 ChatbotAgent와의 차이점:
    - 온톨로지 추론 결과 기반 응답
    - 지식 그래프에서 관련 사실 조회
    - 추론 과정 설명 제공

    사용 예:
        agent = HybridChatbotAgent()
        result = await agent.chat("LANEIGE Lip Care 경쟁력 분석해줘")
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        docs_dir: str = ".",
        knowledge_graph: Optional[KnowledgeGraph] = None,
        reasoner: Optional[OntologyReasoner] = None,
        logger: Optional[AgentLogger] = None,
        tracer: Optional[ExecutionTracer] = None,
        metrics: Optional[QualityMetrics] = None,
        context_manager: Optional[ContextManager] = None
    ):
        """
        Args:
            model: LLM 모델명
            docs_dir: RAG 문서 디렉토리
            knowledge_graph: 지식 그래프 (공유 가능)
            reasoner: 추론기 (공유 가능)
            logger: 로거
            tracer: 추적기
            metrics: 메트릭 수집기
            context_manager: 컨텍스트 관리자
        """
        self.model = model

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
            auto_init_rules=False
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
        self._current_data: Dict[str, Any] = {}

        # 마지막 하이브리드 컨텍스트
        self._last_hybrid_context: Optional[HybridContext] = None

    def set_data_context(self, data: Dict[str, Any]) -> None:
        """
        현재 데이터 컨텍스트 설정

        Args:
            data: 지표/인사이트 데이터
        """
        self._current_data = data

        # 지식 그래프 업데이트
        if data:
            self.hybrid_retriever.update_knowledge_graph(
                metrics_data=data
            )

    async def chat(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        include_reasoning: bool = True
    ) -> Dict[str, Any]:
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
        self.logger.info(f"User query: {user_message[:50]}...")
        start_time = datetime.now()

        if self.tracer:
            self.tracer.start_span("hybrid_chatbot_response", {
                "query_length": len(user_message)
            })

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
                    "suggestions": self._get_fallback_suggestions()
                }

            # 3. 하이브리드 검색 (추론 + RAG)
            if self.tracer:
                self.tracer.start_span("hybrid_retrieval")

            hybrid_context = await self.hybrid_retriever.retrieve(
                query=user_message,
                current_metrics=self._current_data,
                include_explanations=include_reasoning
            )
            self._last_hybrid_context = hybrid_context

            if self.tracer:
                self.tracer.end_span("completed")

            # 4. 컨텍스트 구성
            if self.tracer:
                self.tracer.start_span("build_context")

            # 쿼리 유형에 따라 빌더 선택
            if query_type in [QueryType.DEFINITION, QueryType.INTERPRETATION]:
                # 간단한 질문은 컴팩트 빌더
                context = self.compact_builder.build(
                    hybrid_context=hybrid_context,
                    current_metrics=self._current_data,
                    query=user_message
                )
            else:
                # 분석 질문은 풀 빌더
                context = self.context_builder.build(
                    hybrid_context=hybrid_context,
                    current_metrics=self._current_data,
                    query=user_message
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
                inferences=hybrid_context.inferences
            )

            if self.tracer:
                self.tracer.end_span("completed")

            # 6. 대화 기록 저장
            self.context.add_user_message(user_message)
            self.context.add_assistant_message(response)

            # 7. 후속 질문 제안
            suggestions = self._generate_suggestions(
                query_type=query_type,
                entities=hybrid_context.entities,
                inferences=hybrid_context.inferences
            )

            duration = (datetime.now() - start_time).total_seconds()

            if self.tracer:
                self.tracer.end_span("completed")

            self.logger.info(
                f"Response generated in {duration:.2f}s",
                {"query_type": query_type.value if hasattr(query_type, 'value') else str(query_type)}
            )

            return {
                "response": response,
                "query_type": query_type.value if hasattr(query_type, 'value') else str(query_type),
                "is_fallback": False,
                "inferences": [inf.to_dict() for inf in hybrid_context.inferences],
                "sources": self._extract_sources(hybrid_context),
                "suggestions": suggestions,
                "entities": hybrid_context.entities,
                "stats": {
                    "inferences_count": len(hybrid_context.inferences),
                    "rag_chunks_count": len(hybrid_context.rag_chunks),
                    "response_time_ms": duration * 1000
                }
            }

        except Exception as e:
            if self.tracer:
                self.tracer.end_span("failed", str(e))

            self.logger.error(f"Chat error: {e}")

            return {
                "response": "죄송합니다. 응답 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "query_type": "error",
                "is_fallback": True,
                "error": str(e),
                "inferences": [],
                "sources": [],
                "suggestions": self._get_fallback_suggestions()
            }

    async def _generate_response(
        self,
        user_message: str,
        query_type: QueryType,
        context: str,
        inferences: List[InferenceResult]
    ) -> str:
        """LLM 응답 생성"""
        # 시스템 프롬프트
        system_prompt = self.context_builder.build_system_prompt(
            include_guardrails=True
        )

        # 대화 히스토리
        conversation = self.context.get_conversation_summary()

        # 추론 결과 강조
        inference_summary = ""
        if inferences:
            inference_lines = []
            for inf in inferences[:3]:
                inference_lines.append(f"- [{inf.insight_type.value}] {inf.insight}")
            inference_summary = "\n".join(inference_lines)

        user_prompt = f"""
{context}

---

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
3. 불확실한 부분은 명확히 밝힘
4. 단정적 표현 대신 가능성 표현 사용
5. 간결하고 명확하게 답변
"""

        try:
            response = await acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )

            answer = response.choices[0].message.content

            # 토큰 사용량 기록
            if self.metrics and hasattr(response, 'usage'):
                self.metrics.record_llm_call(
                    model=self.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    latency_ms=0,
                    cost=self._estimate_cost(
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens
                    )
                )

            # 가드레일 적용
            answer = self.templates.apply_guardrails(answer)

            return answer

        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return self._generate_fallback_response(inferences)

    def _generate_fallback_response(
        self,
        inferences: List[InferenceResult]
    ) -> str:
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
        entities: Dict[str, List[str]],
        inferences: List[InferenceResult]
    ) -> List[str]:
        """후속 질문 제안"""
        suggestions = []

        # 추론 결과 기반 제안
        if inferences:
            for inf in inferences[:2]:
                if "경쟁" in inf.insight or "COMPETITIVE" in inf.insight_type.value:
                    suggestions.append("주요 경쟁사 분석을 해주세요")
                if "가격" in inf.insight or "PRICE" in inf.insight_type.value:
                    suggestions.append("가격 전략에 대해 더 알려주세요")
                if "성장" in inf.insight or "GROWTH" in inf.insight_type.value:
                    suggestions.append("성장 기회를 더 구체적으로 설명해주세요")

        # 쿼리 유형별 제안
        if query_type == QueryType.DEFINITION:
            suggestions.extend([
                "이 지표의 해석 기준은 어떻게 되나요?",
                "관련된 다른 지표는 무엇이 있나요?"
            ])
        elif query_type == QueryType.INTERPRETATION:
            suggestions.extend([
                "이 수치가 좋은 건가요?",
                "개선을 위한 액션이 있나요?"
            ])
        elif query_type == QueryType.ANALYSIS:
            suggestions.extend([
                "시계열 트렌드를 알려주세요",
                "경쟁사와 비교해주세요"
            ])

        # 엔티티 기반 제안
        if entities.get("brands"):
            brand = entities["brands"][0]
            suggestions.append(f"{brand}의 최근 순위 변동은?")

        if entities.get("categories"):
            category = entities["categories"][0]
            suggestions.append(f"{category} 카테고리 Top 5는?")

        # 중복 제거 및 상위 3개
        unique_suggestions = list(dict.fromkeys(suggestions))
        return unique_suggestions[:3]

    def _get_fallback_suggestions(self) -> List[str]:
        """폴백 제안"""
        return [
            "SoS(점유율)에 대해 알려주세요",
            "오늘의 주요 인사이트는?",
            "LANEIGE 현재 순위는?"
        ]

    def _extract_sources(self, hybrid_context: HybridContext) -> List[Dict[str, Any]]:
        """출처 추출"""
        sources = []

        # 온톨로지 추론 출처
        if hybrid_context.inferences:
            sources.append({
                "type": "ontology",
                "name": "Ontology Reasoning",
                "count": len(hybrid_context.inferences)
            })

        # 지식 그래프 출처
        if hybrid_context.ontology_facts:
            sources.append({
                "type": "knowledge_graph",
                "name": "Knowledge Graph",
                "count": len(hybrid_context.ontology_facts)
            })

        # RAG 문서 출처
        for chunk in hybrid_context.rag_chunks:
            doc_id = chunk.get("metadata", {}).get("doc_id", "")
            title = chunk.get("metadata", {}).get("title", "")
            if doc_id or title:
                sources.append({
                    "type": "rag",
                    "name": title or doc_id,
                    "doc_id": doc_id
                })

        return sources

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """비용 추정"""
        input_cost = (prompt_tokens / 1_000_000) * 0.40
        output_cost = (completion_tokens / 1_000_000) * 1.60
        return round(input_cost + output_cost, 6)

    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """대화 기록 조회"""
        return self.context.get_conversation_history(limit)

    def clear_conversation(self) -> None:
        """대화 기록 초기화"""
        self.context.reset()

    def get_last_hybrid_context(self) -> Optional[HybridContext]:
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
        knowledge_graph: Optional[KnowledgeGraph] = None,
        reasoner: Optional[OntologyReasoner] = None
    ):
        """
        Args:
            knowledge_graph: 공유 지식 그래프
            reasoner: 공유 추론기
        """
        self._sessions: Dict[str, HybridChatbotAgent] = {}
        self._shared_kg = knowledge_graph
        self._shared_reasoner = reasoner

    def get_or_create(
        self,
        session_id: str,
        **kwargs
    ) -> HybridChatbotAgent:
        """세션별 챗봇 인스턴스 반환"""
        if session_id not in self._sessions:
            self._sessions[session_id] = HybridChatbotAgent(
                knowledge_graph=self._shared_kg,
                reasoner=self._shared_reasoner,
                **kwargs
            )
        return self._sessions[session_id]

    def close_session(self, session_id: str) -> None:
        """세션 종료"""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def list_sessions(self) -> List[str]:
        """활성 세션 목록"""
        return list(self._sessions.keys())

"""
Chatbot Agent
대시보드 챗봇 에이전트 (RAG 기반 질의응답)
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from litellm import acompletion
from src.rag.router import RAGRouter, QueryType
from src.rag.retriever import DocumentRetriever
from src.rag.templates import ResponseTemplates
from src.memory.context import ContextManager
from src.monitoring.logger import AgentLogger
from src.monitoring.tracer import ExecutionTracer
from src.monitoring.metrics import QualityMetrics


class ChatbotAgent:
    """대시보드 챗봇 에이전트"""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        docs_dir: str = "./docs",
        logger: Optional[AgentLogger] = None,
        tracer: Optional[ExecutionTracer] = None,
        metrics: Optional[QualityMetrics] = None,
        context_manager: Optional[ContextManager] = None
    ):
        """
        Args:
            model: LLM 모델명
            docs_dir: RAG 문서 디렉토리
            logger: 로거
            tracer: 추적기
            metrics: 메트릭 수집기
            context_manager: 컨텍스트 관리자
        """
        self.model = model
        self.router = RAGRouter()
        self.retriever = DocumentRetriever(docs_dir)
        self.templates = ResponseTemplates()
        self.context = context_manager or ContextManager()
        self.logger = logger or AgentLogger("chatbot")
        self.tracer = tracer
        self.metrics = metrics

        # 현재 데이터 컨텍스트
        self._current_data: Dict[str, Any] = {}

    def set_data_context(self, data: Dict[str, Any]) -> None:
        """
        현재 데이터 컨텍스트 설정

        Args:
            data: 지표/인사이트 데이터
        """
        self._current_data = data

    async def chat(
        self,
        user_message: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        사용자 질문에 응답

        Args:
            user_message: 사용자 메시지
            session_id: 세션 ID

        Returns:
            {
                "response": "...",
                "query_type": "...",
                "sources": [...],
                "suggestions": [...]
            }
        """
        self.logger.info(f"User query: {user_message[:50]}...")
        start_time = datetime.now()

        if self.tracer:
            self.tracer.start_span("chatbot_response", {"query_length": len(user_message)})

        try:
            # 1. 쿼리 라우팅
            route_result = self.router.route(user_message)
            query_type = route_result.get("query_type")
            entities = route_result.get("entities", {})

            self.logger.debug(f"Query type: {query_type}", {"entities": entities})

            # 2. Fallback 처리
            if query_type == QueryType.UNKNOWN:
                fallback_response = route_result.get("fallback_message", "")
                suggestions = route_result.get("suggestions", [])

                return {
                    "response": fallback_response,
                    "query_type": "unknown",
                    "is_fallback": True,
                    "suggestions": suggestions,
                    "sources": []
                }

            # 3. RAG 컨텍스트 검색
            if self.tracer:
                self.tracer.start_span("retrieve_documents")

            rag_context = self.router.get_context_for_query(user_message, query_type)

            if self.tracer:
                self.tracer.end_span("completed")

            # 4. 데이터 컨텍스트 구성
            data_context = self._build_data_context(query_type, entities)

            # 5. LLM 응답 생성
            if self.tracer:
                self.tracer.start_span("llm_response")

            response = await self._generate_response(
                user_message,
                query_type,
                rag_context,
                data_context
            )

            if self.tracer:
                self.tracer.end_span("completed")

            # 6. 대화 기록 저장
            self.context.add_user_message(user_message)
            self.context.add_assistant_message(response)

            # 7. 후속 질문 제안
            suggestions = self._generate_suggestions(query_type, entities)

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
                "sources": self._extract_sources(rag_context),
                "suggestions": suggestions,
                "entities": entities
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
                "sources": [],
                "suggestions": [
                    "질문을 다시 입력해주세요",
                    "'SoS란 무엇인가요?'와 같이 구체적으로 질문해주세요"
                ]
            }

    def _build_data_context(
        self,
        query_type: QueryType,
        entities: Dict
    ) -> str:
        """데이터 컨텍스트 구성"""
        if not self._current_data:
            return "현재 분석 데이터가 없습니다."

        context_parts = []

        # 쿼리 타입별 관련 데이터 추출
        if query_type in [QueryType.INTERPRETATION, QueryType.COMBINATION]:
            # 지표 데이터
            summary = self._current_data.get("summary", {})
            if summary:
                context_parts.append(f"""
현재 데이터 요약:
- LANEIGE 추적 제품: {summary.get('laneige_products_tracked', 0)}개
- 알림: {summary.get('alert_count', 0)}건
""")

            # 카테고리별 SoS
            sos_data = summary.get("laneige_sos_by_category", {})
            if sos_data:
                sos_lines = [f"  - {cat}: {sos*100:.1f}%" for cat, sos in sos_data.items()]
                context_parts.append("카테고리별 LANEIGE SoS:\n" + "\n".join(sos_lines))

        # 엔티티 기반 필터링
        category = entities.get("category")
        product = entities.get("product")

        if category:
            # 특정 카테고리 데이터
            brand_metrics = self._current_data.get("brand_metrics", [])
            cat_metrics = [b for b in brand_metrics if b.get("category_id") == category]
            if cat_metrics:
                laneige = next((b for b in cat_metrics if b.get("is_laneige")), None)
                if laneige:
                    context_parts.append(f"""
{category} 카테고리 LANEIGE 현황:
- SoS: {laneige.get('share_of_shelf', 0)*100:.1f}%
- 평균 순위: {laneige.get('avg_rank', '-')}
- Top 10 제품: {laneige.get('top10_count', 0)}개
""")

        if product:
            # 특정 제품 데이터
            product_metrics = self._current_data.get("product_metrics", [])
            prod_data = next(
                (p for p in product_metrics if product.lower() in p.get("product_title", "").lower()),
                None
            )
            if prod_data:
                context_parts.append(f"""
제품 '{prod_data.get('product_title', '')[:30]}' 현황:
- 현재 순위: {prod_data.get('current_rank')}위
- 1일 변동: {prod_data.get('rank_change_1d', '데이터 없음')}
- 7일 변동: {prod_data.get('rank_change_7d', '데이터 없음')}
""")

        return "\n".join(context_parts) if context_parts else "관련 데이터가 없습니다."

    async def _generate_response(
        self,
        user_message: str,
        query_type: QueryType,
        rag_context: str,
        data_context: str
    ) -> str:
        """LLM 응답 생성"""
        system_prompt = self.templates.get_system_prompt()

        # 대화 히스토리
        conversation = self.context.get_conversation_summary()

        user_prompt = f"""
## 사용자 질문
{user_message}

## 관련 가이드라인
{rag_context if rag_context else "관련 가이드라인 없음"}

## 현재 데이터
{data_context}

## 이전 대화
{conversation}

요구사항:
1. 질문에 직접적으로 답변하세요
2. 데이터가 있으면 구체적인 수치를 인용하세요
3. 불확실한 부분은 명확히 밝히세요
4. 단정적 표현을 피하세요
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
            return "죄송합니다. 현재 응답을 생성할 수 없습니다. 잠시 후 다시 시도해주세요."

    def _generate_suggestions(
        self,
        query_type: QueryType,
        entities: Dict
    ) -> List[str]:
        """후속 질문 제안"""
        suggestions = []

        if query_type == QueryType.DEFINITION:
            suggestions = [
                "이 지표의 해석 기준은 어떻게 되나요?",
                "관련된 다른 지표는 무엇이 있나요?",
                "실제 데이터에서 이 지표를 확인할 수 있나요?"
            ]
        elif query_type == QueryType.INTERPRETATION:
            suggestions = [
                "이 수치가 좋은 건가요, 나쁜 건가요?",
                "경쟁사와 비교하면 어떤가요?",
                "개선을 위한 액션 아이템이 있나요?"
            ]
        elif query_type == QueryType.COMBINATION:
            suggestions = [
                "다른 지표와 함께 분석해주세요",
                "시계열 트렌드를 알려주세요",
                "우선순위가 높은 액션은 무엇인가요?"
            ]
        elif query_type == QueryType.PRODUCT:
            product = entities.get("product", "해당 제품")
            suggestions = [
                f"{product}의 최근 순위 변동은 어떤가요?",
                f"{product}의 경쟁 제품은 무엇인가요?",
                f"{product}의 리뷰 트렌드를 알려주세요"
            ]
        elif query_type == QueryType.CATEGORY:
            category = entities.get("category", "해당 카테고리")
            suggestions = [
                f"{category}의 Top 5 브랜드는 어디인가요?",
                f"{category}에서 LANEIGE 포지션은 어떤가요?",
                f"{category}의 시장 집중도는 어떤가요?"
            ]
        else:
            suggestions = [
                "SoS(점유율)에 대해 알려주세요",
                "오늘의 주요 인사이트는 무엇인가요?",
                "순위가 하락한 제품이 있나요?"
            ]

        return suggestions[:3]

    def _extract_sources(self, rag_context: str) -> List[str]:
        """출처 추출"""
        sources = []

        # 문서명 패턴 매칭 (간소화)
        doc_names = [
            "Strategic Indicators Definition",
            "Metric Interpretation Guide",
            "Indicator Combination Playbook",
            "Home Page Insight Rules"
        ]

        for doc in doc_names:
            if doc.lower().replace(" ", "") in rag_context.lower().replace(" ", ""):
                sources.append(doc)

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


class ChatbotSession:
    """챗봇 세션 관리 (멀티 유저 지원)"""

    def __init__(self):
        self._sessions: Dict[str, ChatbotAgent] = {}

    def get_or_create(self, session_id: str, **kwargs) -> ChatbotAgent:
        """세션별 챗봇 인스턴스 반환"""
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatbotAgent(**kwargs)
        return self._sessions[session_id]

    def close_session(self, session_id: str) -> None:
        """세션 종료"""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def list_sessions(self) -> List[str]:
        """활성 세션 목록"""
        return list(self._sessions.keys())

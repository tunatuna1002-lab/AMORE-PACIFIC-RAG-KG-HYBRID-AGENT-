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

from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.ontology.business_rules import register_all_rules
from src.domain.entities.relations import InferenceResult

from src.rag.hybrid_retriever import HybridRetriever, HybridContext, EntityExtractor
from src.rag.context_builder import ContextBuilder, CompactContextBuilder
from src.rag.router import RAGRouter, QueryType
from src.rag.retriever import DocumentRetriever
from src.rag.templates import ResponseTemplates

from src.memory.context import ContextManager

from src.monitoring.logger import AgentLogger
from src.monitoring.tracer import ExecutionTracer
from src.monitoring.metrics import QualityMetrics


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
                    query=user_message,
                    knowledge_graph=self.kg
                )
            else:
                # 분석 질문은 풀 빌더 (카테고리 계층 인식 포함)
                context = self.context_builder.build(
                    hybrid_context=hybrid_context,
                    current_metrics=self._current_data,
                    query=user_message,
                    knowledge_graph=self.kg
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

            # 6. 출처 정보 추출 및 포맷팅
            sources = self._extract_sources(hybrid_context)
            formatted_sources = self._format_sources_for_response(sources)

            # 7. 응답에 출처 섹션 추가
            full_response = response + formatted_sources

            # 8. 대화 기록 저장
            self.context.add_user_message(user_message)
            self.context.add_assistant_message(full_response)

            # 9. 후속 질문 제안
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
                "response": full_response,
                "query_type": query_type.value if hasattr(query_type, 'value') else str(query_type),
                "is_fallback": False,
                "inferences": [inf.to_dict() for inf in hybrid_context.inferences],
                "sources": sources,
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
        # 시스템 프롬프트 (카테고리 계층 인식 추가)
        system_prompt = self.context_builder.build_system_prompt(
            include_guardrails=True
        )

        # 카테고리 계층 설명 추가
        system_prompt += """

## 카테고리 계층 구조 인식
- 제품은 여러 계층의 카테고리에 동시에 소속될 수 있습니다
- 예: 특정 립케어 제품이 "Lip Care"에서 4위이면서, 상위 카테고리인 "Beauty & Personal Care"에서는 73위일 수 있습니다
- 순위를 언급할 때는 반드시 어느 카테고리에서의 순위인지 명시하세요
- 카테고리 간 순위 차이가 있는 경우, 이는 자연스러운 현상입니다 (하위 카테고리가 더 세분화되어 경쟁 범위가 좁기 때문)
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

            if response.choices:
                answer = response.choices[0].message.content
            else:
                answer = "죄송합니다. 응답을 생성하지 못했습니다."

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

    def _build_category_hierarchy_context(
        self,
        entities: Dict[str, List[str]]
    ) -> str:
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
            if hierarchy.get('ancestors'):
                path = " > ".join([a['name'] for a in reversed(hierarchy['ancestors'])])
                context_parts.append(f"  - 상위 경로: {path} > {hierarchy['name']}")

            # 하위 카테고리
            if hierarchy.get('descendants'):
                children = ", ".join([d['name'] for d in hierarchy['descendants'][:5]])
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

    def _extract_sources(self, hybrid_context: HybridContext) -> List[Dict[str, Any]]:
        """
        출처 정보 추출 (Perplexity/Liner 스타일 상세 출처 제공)

        Args:
            hybrid_context: 하이브리드 검색 컨텍스트

        Returns:
            출처 정보 리스트 (유형별 상세 정보 포함)
        """
        sources = []

        # 1. 크롤링 데이터 출처 - URL 및 상세 정보 추가
        if self._current_data:
            metadata = self._current_data.get("metadata", {})
            data_date = metadata.get("data_date", "")
            categories = self._current_data.get("categories", {})

            total_products = sum(
                len(cat_data.get("rank_records", []))
                for cat_data in categories.values()
            ) if categories else 0

            sources.append({
                "type": "crawled_data",
                "icon": "📊",
                "description": "Amazon Best Sellers 크롤링 데이터",
                "collected_at": data_date,
                "url": "https://www.amazon.com/gp/bestsellers/beauty",
                "details": {
                    "categories": list(categories.keys()) if categories else [],
                    "total_products": total_products,
                    "snapshot_date": data_date
                }
            })

        # 2. Knowledge Graph 출처 - 엔티티 및 관계 정보 추가
        if hybrid_context.ontology_facts:
            sources.append({
                "type": "knowledge_graph",
                "icon": "🔗",
                "description": "지식 그래프 관계 데이터",
                "fact_count": len(hybrid_context.ontology_facts),
                "entities": self._extract_entity_names(hybrid_context.ontology_facts),
                "relations": self._extract_relation_types(hybrid_context.ontology_facts),
                "details": {
                    "source": "Amazon US 실시간 데이터 기반 지식 그래프",
                    "fact_count": len(hybrid_context.ontology_facts)
                }
            })

        # 3. 온톨로지 추론 출처 - 규칙 상세 정보
        if hybrid_context.inferences:
            for inf in hybrid_context.inferences:
                sources.append({
                    "type": "ontology_inference",
                    "icon": "🧠",
                    "description": f"온톨로지 규칙: {inf.rule_name}",
                    "rule_name": inf.rule_name,
                    "confidence": inf.confidence,
                    "evidence": inf.evidence,
                    "insight_type": inf.insight_type.value if hasattr(inf.insight_type, 'value') else str(inf.insight_type),
                    "details": {
                        "insight": inf.insight,
                        "recommendation": inf.recommendation
                    }
                })

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
                if doc_key not in rag_sources_map or score > rag_sources_map[doc_key].get("relevance_score", 0):
                    rag_sources_map[doc_key] = {
                        "type": "rag_document",
                        "icon": "📄",
                        "description": title or doc_id,
                        "file_path": file_path,
                        "section": section,
                        "relevance_score": score,
                        "details": {
                            "doc_id": doc_id,
                            "title": title
                        }
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

                    sources.append({
                        "type": "category_hierarchy",
                        "icon": "🗂️",
                        "description": "카테고리 계층 구조",
                        "path": path,
                        "level": hierarchy.get("level", 0),
                        "url": hierarchy.get("url", ""),
                        "details": {
                            "category": category,
                            "hierarchy_depth": len(path)
                        }
                    })

        # 6. AI 모델 출처 (항상 포함)
        sources.append({
            "type": "ai_model",
            "icon": "🤖",
            "description": f"AI 분석: {self.model}",
            "model": self.model,
            "disclaimer": "AI가 생성한 분석입니다. 중요한 의사결정 시 추가 검증을 권장합니다.",
            "generated_at": datetime.now().isoformat()
        })

        return sources

    def _extract_entity_names(self, ontology_facts) -> List[str]:
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

    def _extract_relation_types(self, ontology_facts) -> List[str]:
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

    def _format_sources_for_response(self, sources: List[Dict[str, Any]]) -> str:
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
                lines.append(f"{i}. {icon} **{desc}**")
                lines.append(f"   - 수집일: {collected}")
                if url:
                    lines.append(f"   - URL: {url}")
                if total > 0:
                    lines.append(f"   - 총 제품 수: {total}개")
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

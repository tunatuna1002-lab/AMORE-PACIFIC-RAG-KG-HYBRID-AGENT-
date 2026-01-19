"""
Hybrid Insight Agent
Ontology-RAG 하이브리드 인사이트 생성 에이전트

Flow:
1. 현재 데이터로 Knowledge Graph 업데이트
2. Ontology Reasoner로 규칙 기반 추론
3. RAG로 관련 가이드라인 검색
4. 추론 결과 + RAG 컨텍스트로 LLM 인사이트 생성
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from litellm import acompletion

from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.ontology.business_rules import register_all_rules
from src.ontology.relations import InferenceResult, InsightType

from src.rag.hybrid_retriever import HybridRetriever, HybridContext
from src.rag.context_builder import ContextBuilder
from src.rag.retriever import DocumentRetriever
from src.rag.templates import ResponseTemplates

from src.monitoring.logger import AgentLogger
from src.monitoring.tracer import ExecutionTracer
from src.monitoring.metrics import QualityMetrics


class HybridInsightAgent:
    """
    Ontology-RAG 하이브리드 인사이트 생성 에이전트

    기존 InsightAgent와의 차이점:
    - 온톨로지 추론 결과를 기반으로 인사이트 생성
    - 규칙 기반 추론으로 일관성 보장
    - 추론 과정 설명 가능 (Explainability)

    사용 예:
        agent = HybridInsightAgent(model="gpt-4.1-mini")
        result = await agent.execute(metrics_data)
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        docs_dir: str = ".",
        knowledge_graph: Optional[KnowledgeGraph] = None,
        reasoner: Optional[OntologyReasoner] = None,
        logger: Optional[AgentLogger] = None,
        tracer: Optional[ExecutionTracer] = None,
        metrics: Optional[QualityMetrics] = None
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

        # 하이브리드 검색기
        self.hybrid_retriever = HybridRetriever(
            knowledge_graph=self.kg,
            reasoner=self.reasoner,
            doc_retriever=self.doc_retriever,
            auto_init_rules=False  # 이미 등록됨
        )

        # 컨텍스트 빌더
        self.context_builder = ContextBuilder(max_tokens=4000)

        # 템플릿
        self.templates = ResponseTemplates()

        # 모니터링
        self.logger = logger or AgentLogger("hybrid_insight")
        self.tracer = tracer
        self.metrics = metrics

        # 결과 캐시
        self._results: Dict[str, Any] = {}
        self._last_hybrid_context: Optional[HybridContext] = None

    async def execute(
        self,
        metrics_data: Dict[str, Any],
        crawl_data: Optional[Dict[str, Any]] = None,
        crawl_summary: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        하이브리드 인사이트 생성

        Args:
            metrics_data: 지표 에이전트 결과
            crawl_data: 크롤링 원본 데이터 (KG 업데이트용)
            crawl_summary: 크롤링 요약

        Returns:
            {
                "status": "completed",
                "daily_insight": "...",
                "action_items": [...],
                "highlights": [...],
                "inferences": [...],
                "explanations": [...]
            }
        """
        self.logger.agent_start("HybridInsightAgent", "하이브리드 인사이트 생성")
        start_time = datetime.now()

        if self.metrics:
            self.metrics.record_agent_start("hybrid_insight")

        if self.tracer:
            self.tracer.start_span("hybrid_insight_agent")

        try:
            # 데이터 출처 정보 추출
            data_source = self._extract_data_source_info(metrics_data, crawl_data)

            results = {
                "status": "completed",
                "generated_at": datetime.now().isoformat(),
                "data_source": data_source,  # 데이터 출처 정보 추가
                "daily_insight": "",
                "action_items": [],
                "highlights": [],
                "warnings": [],
                "inferences": [],
                "explanations": [],
                "hybrid_stats": {}
            }

            # 1. Knowledge Graph 업데이트
            if self.tracer:
                self.tracer.start_span("update_knowledge_graph")

            kg_stats = self._update_knowledge_graph(crawl_data, metrics_data)
            results["hybrid_stats"]["kg_update"] = kg_stats

            if self.tracer:
                self.tracer.end_span("completed")

            # 2. 하이브리드 검색 (추론 + RAG)
            if self.tracer:
                self.tracer.start_span("hybrid_retrieval")

            hybrid_context = await self._run_hybrid_retrieval(metrics_data)
            self._last_hybrid_context = hybrid_context
            results["inferences"] = [inf.to_dict() for inf in hybrid_context.inferences]

            if self.tracer:
                self.tracer.end_span("completed")

            # 3. 추론 설명 생성
            if self.tracer:
                self.tracer.start_span("generate_explanations")

            explanations = self._generate_explanations(hybrid_context.inferences)
            results["explanations"] = explanations

            if self.tracer:
                self.tracer.end_span("completed")

            # 4. 일일 인사이트 생성 (LLM)
            if self.tracer:
                self.tracer.start_span("generate_daily_insight")

            daily_insight = await self._generate_daily_insight(
                hybrid_context, metrics_data, crawl_summary
            )
            results["daily_insight"] = daily_insight

            if self.tracer:
                self.tracer.end_span("completed")

            # 5. 액션 아이템 추출
            if self.tracer:
                self.tracer.start_span("extract_actions")

            action_items = self._extract_action_items(
                hybrid_context.inferences, metrics_data
            )
            results["action_items"] = action_items

            if self.tracer:
                self.tracer.end_span("completed")

            # 6. 하이라이트 추출
            results["highlights"] = self._extract_highlights(
                hybrid_context.inferences, metrics_data
            )

            # 7. 경고 수집
            alerts = metrics_data.get("alerts", [])
            results["warnings"] = [
                a for a in alerts
                if a.get("severity") in ["warning", "critical"]
            ]

            # 8. 통계
            results["hybrid_stats"].update({
                "inferences_count": len(hybrid_context.inferences),
                "rag_chunks_count": len(hybrid_context.rag_chunks),
                "ontology_facts_count": len(hybrid_context.ontology_facts)
            })

            self._results = results
            duration = (datetime.now() - start_time).total_seconds()

            if self.tracer:
                self.tracer.end_span("completed")

            if self.metrics:
                self.metrics.record_agent_complete("hybrid_insight", {
                    "action_items": len(results["action_items"]),
                    "inferences": len(results["inferences"])
                })

            self.logger.agent_complete(
                "HybridInsightAgent",
                duration,
                f"{len(results['inferences'])} inferences, "
                f"{len(results['action_items'])} actions"
            )

            return results

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()

            if self.tracer:
                self.tracer.end_span("failed", str(e))

            if self.metrics:
                self.metrics.record_agent_error("hybrid_insight", str(e))

            self.logger.agent_error("HybridInsightAgent", str(e), duration)
            raise

    def _update_knowledge_graph(
        self,
        crawl_data: Optional[Dict],
        metrics_data: Dict
    ) -> Dict[str, int]:
        """Knowledge Graph 업데이트"""
        stats = {"crawl_relations": 0, "metrics_relations": 0}

        if crawl_data:
            stats["crawl_relations"] = self.kg.load_from_crawl_data(crawl_data)
            self.logger.debug(f"KG updated from crawl: {stats['crawl_relations']} relations")

        if metrics_data:
            stats["metrics_relations"] = self.kg.load_from_metrics_data(metrics_data)
            self.logger.debug(f"KG updated from metrics: {stats['metrics_relations']} relations")

        return stats

    async def _run_hybrid_retrieval(
        self,
        metrics_data: Dict
    ) -> HybridContext:
        """하이브리드 검색 수행"""
        # 일일 인사이트용 쿼리
        query = "LANEIGE 오늘의 Amazon 베스트셀러 성과 분석"

        # 하이브리드 검색
        context = await self.hybrid_retriever.retrieve(
            query=query,
            current_metrics=metrics_data,
            include_explanations=True
        )

        self.logger.info(
            f"Hybrid retrieval: {len(context.inferences)} inferences, "
            f"{len(context.rag_chunks)} RAG chunks"
        )

        return context

    def _generate_explanations(
        self,
        inferences: List[InferenceResult]
    ) -> List[Dict[str, Any]]:
        """추론 설명 생성"""
        explanations = []

        for inf in inferences:
            explanation = {
                "rule": inf.rule_name,
                "type": inf.insight_type.value,
                "insight": inf.insight,
                "explanation": self.reasoner.explain_inference(inf),
                "confidence": inf.confidence
            }
            explanations.append(explanation)

        return explanations

    async def _generate_daily_insight(
        self,
        hybrid_context: HybridContext,
        metrics_data: Dict,
        crawl_summary: Optional[Dict]
    ) -> str:
        """일일 인사이트 생성 (LLM)"""
        # 컨텍스트 구성
        context = self.context_builder.build(
            hybrid_context=hybrid_context,
            current_metrics=metrics_data,
            query="오늘의 LANEIGE Amazon 베스트셀러 인사이트"
        )

        # 시스템 프롬프트
        system_prompt = self.context_builder.build_system_prompt(
            include_guardrails=True
        )

        # 사용자 프롬프트
        user_prompt = f"""
{context}

---

## 요청사항

위 분석 결과와 데이터를 바탕으로 오늘의 LANEIGE Amazon 베스트셀러 인사이트를 작성해주세요.

포함 내용:
1. **핵심 요약** (3-5문장): 오늘의 가장 중요한 인사이트
2. **주요 발견**: 온톨로지 추론 결과 기반 핵심 발견 사항
3. **주의 필요 사항**: 경고 또는 모니터링이 필요한 부분
4. **권장 액션**: 구체적인 다음 단계 제안

형식:
- 마크다운 형식
- 구체적인 수치 인용
- 단정적 표현 대신 가능성 표현 사용
"""

        try:
            response = await acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1200
            )

            insight = response.choices[0].message.content

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
            insight = self.templates.apply_guardrails(insight)

            return insight

        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return self._generate_fallback_insight(hybrid_context, metrics_data)

    def _generate_fallback_insight(
        self,
        hybrid_context: HybridContext,
        metrics_data: Dict
    ) -> str:
        """폴백 인사이트 생성"""
        summary = metrics_data.get("summary", {})
        inferences = hybrid_context.inferences

        insight_parts = [
            f"## 오늘의 LANEIGE Amazon 베스트셀러 분석\n",
            f"- 추적 중인 제품: {summary.get('laneige_products_tracked', 0)}개",
            f"- 알림: {summary.get('alert_count', 0)}건"
        ]

        # 추론 결과 추가
        if inferences:
            insight_parts.append("\n### 주요 분석 결과")
            for inf in inferences[:3]:
                insight_parts.append(f"- {inf.insight}")

        insight_parts.append(
            "\n\n_※ 상세 인사이트 생성 중 오류가 발생하여 기본 요약을 제공합니다._"
        )

        return "\n".join(insight_parts)

    def _extract_action_items(
        self,
        inferences: List[InferenceResult],
        metrics_data: Dict
    ) -> List[Dict]:
        """액션 아이템 추출"""
        actions = []

        # 추론 결과에서 액션 추출
        for inf in inferences:
            if inf.recommendation:
                priority = self._get_priority_from_insight(inf)
                actions.append({
                    "priority": priority,
                    "type": inf.insight_type.value,
                    "action": inf.recommendation,
                    "source": "ontology_inference",
                    "rule": inf.rule_name,
                    "confidence": inf.confidence
                })

        # 알림 기반 액션 추가
        for alert in metrics_data.get("alerts", []):
            if alert.get("severity") == "critical":
                actions.append({
                    "priority": "high",
                    "type": alert.get("type"),
                    "action": f"[긴급] {alert.get('message')} - 즉시 확인 필요",
                    "source": "alert",
                    "product": alert.get("title"),
                    "asin": alert.get("asin")
                })
            elif alert.get("severity") == "warning":
                actions.append({
                    "priority": "medium",
                    "type": alert.get("type"),
                    "action": f"[주의] {alert.get('message')} - 모니터링 강화",
                    "source": "alert",
                    "product": alert.get("title"),
                    "asin": alert.get("asin")
                })

        # 우선순위 정렬
        priority_order = {"high": 0, "medium": 1, "low": 2}
        actions.sort(key=lambda x: priority_order.get(x.get("priority"), 3))

        return actions[:10]

    def _get_priority_from_insight(self, inference: InferenceResult) -> str:
        """인사이트 유형에서 우선순위 결정"""
        high_priority = {
            InsightType.RISK_ALERT,
            InsightType.COMPETITIVE_THREAT,
            InsightType.RANK_SHOCK
        }
        medium_priority = {
            InsightType.PRICE_QUALITY_GAP,
            InsightType.COMPETITIVE_ADVANTAGE,
            InsightType.GROWTH_OPPORTUNITY
        }

        if inference.insight_type in high_priority:
            return "high"
        elif inference.insight_type in medium_priority:
            return "medium"
        else:
            return "low"

    def _extract_highlights(
        self,
        inferences: List[InferenceResult],
        metrics_data: Dict
    ) -> List[Dict]:
        """하이라이트 추출"""
        highlights = []

        # 긍정적 추론 결과
        positive_types = {
            InsightType.MARKET_DOMINANCE,
            InsightType.GROWTH_MOMENTUM,
            InsightType.STABILITY,
            InsightType.COMPETITIVE_ADVANTAGE
        }

        for inf in inferences:
            if inf.insight_type in positive_types:
                highlights.append({
                    "type": inf.insight_type.value,
                    "title": inf.insight_type.value.replace("_", " ").title(),
                    "detail": inf.insight,
                    "source": "ontology"
                })

        # 제품 메트릭에서 하이라이트
        product_metrics = metrics_data.get("product_metrics", [])

        # Top 10 진입
        for p in product_metrics:
            if p.get("current_rank", 100) <= 10:
                highlights.append({
                    "type": "top_rank",
                    "title": f"Top 10: {p.get('product_title', '')[:30]}...",
                    "detail": f"{p.get('category_id')} 카테고리 {p.get('current_rank')}위",
                    "asin": p.get("asin")
                })

        # 순위 상승
        improving = [
            p for p in product_metrics
            if p.get("rank_change_1d") and p.get("rank_change_1d") < -3
        ]
        for p in improving[:3]:
            highlights.append({
                "type": "rank_up",
                "title": f"순위 상승: {p.get('product_title', '')[:30]}...",
                "detail": f"{abs(p.get('rank_change_1d'))}단계 상승 → 현재 {p.get('current_rank')}위",
                "asin": p.get("asin")
            })

        return highlights[:10]

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """비용 추정"""
        input_cost = (prompt_tokens / 1_000_000) * 0.40
        output_cost = (completion_tokens / 1_000_000) * 1.60
        return round(input_cost + output_cost, 6)

    def get_results(self) -> Dict[str, Any]:
        """마지막 실행 결과"""
        return self._results

    def get_last_hybrid_context(self) -> Optional[HybridContext]:
        """마지막 하이브리드 컨텍스트"""
        return self._last_hybrid_context

    def get_knowledge_graph(self) -> KnowledgeGraph:
        """지식 그래프 반환"""
        return self.kg

    def get_reasoner(self) -> OntologyReasoner:
        """추론기 반환"""
        return self.reasoner

    def _extract_data_source_info(
        self,
        metrics_data: Optional[Dict],
        crawl_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        데이터 출처 정보 추출

        Args:
            metrics_data: 지표 데이터
            crawl_data: 크롤링 데이터

        Returns:
            데이터 출처 정보 딕셔너리
        """
        source_info = {
            "platform": "Amazon US Best Sellers",
            "collected_at": None,
            "snapshot_date": None,
            "categories": [],
            "total_products": 0,
            "disclaimer": "Amazon은 Best Sellers 순위를 매 시간 업데이트합니다. 표시된 데이터는 수집 시점의 스냅샷입니다."
        }

        # 크롤링 데이터에서 수집 시점 추출
        if crawl_data:
            collected_at = crawl_data.get("collected_at")
            if collected_at:
                source_info["collected_at"] = collected_at

            # 크롤링 요약에서 정보 추출
            if "summary" in crawl_data:
                summary = crawl_data["summary"]
                source_info["total_products"] = summary.get("total_products", 0)
                source_info["categories"] = summary.get("categories", [])

        # 지표 데이터에서 날짜 정보 추출
        if metrics_data:
            metadata = metrics_data.get("metadata", {})
            if metadata:
                data_date = metadata.get("data_date")
                if data_date:
                    source_info["snapshot_date"] = data_date
                if not source_info["collected_at"]:
                    source_info["collected_at"] = metadata.get("generated_at")

            # 카테고리 정보
            categories = metrics_data.get("categories", {})
            if categories and not source_info["categories"]:
                source_info["categories"] = list(categories.keys())

            # 제품 수
            if not source_info["total_products"]:
                total = sum(
                    len(cat_data.get("rank_records", []))
                    for cat_data in categories.values()
                ) if categories else 0
                source_info["total_products"] = total

        return source_info

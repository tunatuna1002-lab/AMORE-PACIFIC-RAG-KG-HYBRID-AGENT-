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
from src.domain.entities.relations import (
    InferenceResult,
    InsightType,
    RelationType,
    Relation,
)

from src.rag.hybrid_retriever import HybridRetriever, HybridContext
from src.rag.context_builder import ContextBuilder
from src.rag.retriever import DocumentRetriever
from src.rag.templates import ResponseTemplates

from src.monitoring.logger import AgentLogger
from src.monitoring.tracer import ExecutionTracer
from src.monitoring.metrics import QualityMetrics
from src.tools.external_signal_collector import ExternalSignalCollector
from src.tools.market_intelligence import MarketIntelligenceEngine, DataLayer
from src.tools.source_manager import SourceManager, InsightSourceBuilder

# New collectors (Phase 1 & 2)
try:
    from src.tools.google_trends_collector import GoogleTrendsCollector

    GOOGLE_TRENDS_AVAILABLE = True
except ImportError as e:
    from src.monitoring.logger import get_logger

    _logger = get_logger("hybrid_insight")
    _logger.warning(
        f"GoogleTrendsCollector not available - Google Trends signals will be skipped: {e}"
    )
    GOOGLE_TRENDS_AVAILABLE = False

try:
    from src.tools.youtube_collector import YouTubeCollector

    YOUTUBE_AVAILABLE = True
except ImportError as e:
    from src.monitoring.logger import get_logger

    _logger = get_logger("hybrid_insight")
    _logger.warning(
        f"YouTubeCollector not available - YouTube signals will be skipped: {e}"
    )
    YOUTUBE_AVAILABLE = False


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
        metrics: Optional[QualityMetrics] = None,
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
        import os

        self.model = model
        # Temperature: 인사이트 전용 환경변수 > 일반 환경변수 > 기본값(0.6)
        # 인사이트는 창의적 분석/전략 제안을 위해 약간 높은 temperature 사용 (E2E Audit - 2026-01-27)
        from src.shared.constants import INSIGHT_TEMPERATURE

        self.temperature = float(
            os.getenv(
                "LLM_INSIGHT_TEMPERATURE",
                os.getenv("LLM_TEMPERATURE", str(INSIGHT_TEMPERATURE)),
            )
        )

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
            auto_init_rules=False,  # 이미 등록됨
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

        # External Signal Collector
        self._signal_collector: Optional[ExternalSignalCollector] = None

        # Market Intelligence Engine
        self._market_intelligence: Optional[MarketIntelligenceEngine] = None
        self._insight_source_builder: Optional[InsightSourceBuilder] = None

        # New collectors (Phase 1 & 2)
        self._google_trends: Optional[GoogleTrendsCollector] = None
        self._youtube_collector: Optional[YouTubeCollector] = None

    async def execute(
        self,
        metrics_data: Dict[str, Any],
        crawl_data: Optional[Dict[str, Any]] = None,
        crawl_summary: Optional[Dict] = None,
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
                "hybrid_stats": {},
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

            # 3. RAG → KG 지식 추출
            rag_kg_stats = self._ingest_rag_knowledge(hybrid_context.rag_chunks)
            results["hybrid_stats"]["rag_to_kg"] = rag_kg_stats

            # 4. 추론 설명 생성
            if self.tracer:
                self.tracer.start_span("generate_explanations")

            explanations = self._generate_explanations(hybrid_context.inferences)
            results["explanations"] = explanations

            if self.tracer:
                self.tracer.end_span("completed")

            # 5. External Signal 수집 (LLM 호출 전에 수행)
            if self.tracer:
                self.tracer.start_span("collect_external_signals")

            external_signals = await self._collect_external_signals()
            results["external_signals"] = external_signals
            signal_kg_stats = self._ingest_external_signals(external_signals)
            results["hybrid_stats"]["signal_to_kg"] = signal_kg_stats

            # 실패한 신호 수집기 추적
            results["failed_signals"] = self._get_failed_signal_collectors()

            if self.tracer:
                self.tracer.end_span("completed")

            # 5.5. Market Intelligence 수집 (Layer 2-4)
            if self.tracer:
                self.tracer.start_span("collect_market_intelligence")

            market_intelligence = await self._collect_market_intelligence()
            results["market_intelligence"] = market_intelligence

            if self.tracer:
                self.tracer.end_span("completed")

            # 6. 일일 인사이트 생성 (LLM + External Signal + Market Intelligence 포함)
            if self.tracer:
                self.tracer.start_span("generate_daily_insight")

            daily_insight = await self._generate_daily_insight(
                hybrid_context,
                metrics_data,
                crawl_summary,
                external_signals,
                market_intelligence,
                results.get("failed_signals", []),
            )
            results["daily_insight"] = daily_insight

            if self.tracer:
                self.tracer.end_span("completed")

            # 7. 액션 아이템 추출
            if self.tracer:
                self.tracer.start_span("extract_actions")

            action_items = self._extract_action_items(
                hybrid_context.inferences, metrics_data
            )
            results["action_items"] = action_items

            if self.tracer:
                self.tracer.end_span("completed")

            # 8. 하이라이트 추출
            results["highlights"] = self._extract_highlights(
                hybrid_context.inferences, metrics_data
            )

            # 9. 경고 수집
            alerts = metrics_data.get("alerts", [])
            results["warnings"] = [
                a for a in alerts if a.get("severity") in ["warning", "critical"]
            ]

            if self.tracer:
                self.tracer.end_span("completed")

            # 10. 통계
            results["hybrid_stats"].update(
                {
                    "inferences_count": len(hybrid_context.inferences),
                    "rag_chunks_count": len(hybrid_context.rag_chunks),
                    "ontology_facts_count": len(hybrid_context.ontology_facts),
                    "external_signals_count": len(external_signals.get("signals", [])),
                    "market_intelligence_sources": len(
                        market_intelligence.get("sources", [])
                    ),
                }
            )

            self._results = results
            duration = (datetime.now() - start_time).total_seconds()

            if self.tracer:
                self.tracer.end_span("completed")

            if self.metrics:
                self.metrics.record_agent_complete(
                    "hybrid_insight",
                    {
                        "action_items": len(results["action_items"]),
                        "inferences": len(results["inferences"]),
                    },
                )

            self.logger.agent_complete(
                "HybridInsightAgent",
                duration,
                f"{len(results['inferences'])} inferences, "
                f"{len(results['action_items'])} actions",
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
        self, crawl_data: Optional[Dict], metrics_data: Dict
    ) -> Dict[str, int]:
        """Knowledge Graph 업데이트"""
        stats = {"crawl_relations": 0, "metrics_relations": 0}

        if crawl_data:
            stats["crawl_relations"] = self.kg.load_from_crawl_data(crawl_data)
            self.logger.debug(
                f"KG updated from crawl: {stats['crawl_relations']} relations"
            )

        if metrics_data:
            stats["metrics_relations"] = self.kg.load_from_metrics_data(metrics_data)
            self.logger.debug(
                f"KG updated from metrics: {stats['metrics_relations']} relations"
            )

        return stats

    async def _run_hybrid_retrieval(self, metrics_data: Dict) -> HybridContext:
        """하이브리드 검색 수행"""
        # 일일 인사이트용 쿼리
        query = "LANEIGE 오늘의 Amazon 베스트셀러 성과 분석"

        # 하이브리드 검색
        context = await self.hybrid_retriever.retrieve(
            query=query, current_metrics=metrics_data, include_explanations=True
        )

        self.logger.info(
            f"Hybrid retrieval: {len(context.inferences)} inferences, "
            f"{len(context.rag_chunks)} RAG chunks"
        )

        return context

    def _generate_explanations(
        self, inferences: List[InferenceResult]
    ) -> List[Dict[str, Any]]:
        """추론 설명 생성"""
        explanations = []

        for inf in inferences:
            explanation = {
                "rule": inf.rule_name,
                "type": inf.insight_type.value,
                "insight": inf.insight,
                "explanation": self.reasoner.explain_inference(inf),
                "confidence": inf.confidence,
            }
            explanations.append(explanation)

        return explanations

    async def _generate_daily_insight(
        self,
        hybrid_context: HybridContext,
        metrics_data: Dict,
        crawl_summary: Optional[Dict],
        external_signals: Optional[Dict] = None,
        market_intelligence: Optional[Dict] = None,
        failed_signals: Optional[List[str]] = None,
    ) -> str:
        """일일 인사이트 생성 (LLM)"""
        # 컨텍스트 구성
        context = self.context_builder.build(
            hybrid_context=hybrid_context,
            current_metrics=metrics_data,
            query="오늘의 LANEIGE Amazon 베스트셀러 인사이트",
            knowledge_graph=self.kg,
        )

        # External Signal 컨텍스트 추가
        external_context = ""
        if external_signals and external_signals.get("report_section"):
            external_context = f"""

## 외부 트렌드 신호

{external_signals["report_section"]}

_※ 위 외부 신호는 전문 매체(Allure, Byrdie 등), Reddit, TikTok 등에서 수집되었습니다._
"""

        # Market Intelligence 컨텍스트 추가
        market_context = ""
        if market_intelligence and market_intelligence.get("insight_section"):
            market_context = f"""

## 시장 인텔리전스 (4-Layer 분석)

{market_intelligence["insight_section"]}

"""

        # 시스템 프롬프트
        system_prompt = self.context_builder.build_system_prompt(
            include_guardrails=True
        )

        # 사용자 프롬프트 (4-Layer Why 분석 템플릿)
        reference_section = self._build_reference_section(
            hybrid_context, external_signals, market_intelligence
        )
        user_prompt = f"""
{context}
{external_context}
{market_context}
{reference_section}
---

## 요청사항

위 분석 결과와 데이터를 바탕으로 오늘의 LANEIGE Amazon US 일일 인사이트를 작성해주세요.

### 출력 형식 (반드시 이 구조를 따르세요):

```markdown
# LANEIGE Amazon US 일일 인사이트

## 📌 오늘의 핵심
[가장 중요한 변화 1-2가지 + 그 원인을 연결하여 설명]
예: "Lip Sleeping Mask 순위 상승은 Q3 Americas 매출 +6.9% 성장[2]과 최근 TikTok 바이럴[3]의 복합 효과로 판단됩니다."

## 🔍 원인 분석 (Why?)

### Layer 4: 거시경제/무역
• [관세청 수출입 데이터 기반 분석] [1]
• [환율/관세 영향 분석]

### Layer 3: 산업/기업 동향
• [아모레퍼시픽 IR 실적 기반 분석] [2]
• [브랜드 전략/캠페인 영향]

### Layer 2: 소비자 트렌드
• [Reddit/SNS 트렌드 분석] [3]
• [뷰티 매체 보도 내용]

### Layer 1: Amazon 성과
• [순위 변동, SoS, 가격 등 핵심 지표]
• [경쟁사 동향]

## ⚠️ 주의 사항
• [리스크 또는 모니터링 필요 사항]

## 💡 권장 액션
1. [즉시 실행] 구체적 액션 1
2. [모니터링] 구체적 액션 2
3. [검토 필요] 구체적 액션 3

## 📚 참고자료
[제공된 참고자료 목록을 그대로 사용]
```

### 작성 원칙:
1. **인과관계 중심**: "A가 발생했다" → "A는 B 때문에 발생한 것으로 판단된다"
2. **출처 필수 인용**: 모든 사실 주장에 [1], [2] 형태로 출처 인용
3. **계층적 분석**: Layer 4(거시) → Layer 1(Amazon)으로 원인-결과 연결
4. **정량적 표현**: "증가" 대신 "+12%", "많음" 대신 "2,400 업보트"
5. **가설적 표현**: 확실하지 않은 내용은 "~로 판단됩니다", "~가능성이 있습니다" 사용
"""

        try:
            response = await acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=1200,
            )

            insight = response.choices[0].message.content

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
            insight = self.templates.apply_guardrails(insight)

            # 실패한 신호 수집기 경고 추가
            if failed_signals:
                warning_section = "\n\n---\n"
                warning_section += f"⚠️ **외부 트렌드 정보 일부 미반영**\n"
                warning_section += f"- **수집 실패**: {', '.join(failed_signals)}\n"
                warning_section += (
                    "- **영향**: 본 리포트는 크롤링/KG 데이터 기준으로 작성됨\n"
                )
                warning_section += (
                    "- **권장**: 1-2시간 후 재시도 또는 수동으로 트렌드 확인"
                )
                insight += warning_section

            return insight

        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return self._generate_fallback_insight(hybrid_context, metrics_data)

    def _generate_fallback_insight(
        self, hybrid_context: HybridContext, metrics_data: Dict
    ) -> str:
        """폴백 인사이트 생성"""
        summary = metrics_data.get("summary", {})
        inferences = hybrid_context.inferences

        insight_parts = [
            f"## 오늘의 LANEIGE Amazon 베스트셀러 분석\n",
            f"- 추적 중인 제품: {summary.get('laneige_products_tracked', 0)}개",
            f"- 알림: {summary.get('alert_count', 0)}건",
        ]

        # 추론 결과 추가
        if inferences:
            insight_parts.append("\n### 주요 분석 결과")
            for inf in inferences[:3]:
                insight_parts.append(f"- {inf.insight}")

        insight_parts.append(
            "\n\n_※ 상세 인사이트 생성 중 오류가 발생하여 기본 요약을 제공합니다._"
        )

        reference_section = self._build_reference_section(hybrid_context, {}, None)
        if reference_section:
            insight_parts.append("\n" + reference_section)

        return "\n".join(insight_parts)

    def _extract_action_items(
        self, inferences: List[InferenceResult], metrics_data: Dict
    ) -> List[Dict]:
        """액션 아이템 추출"""
        actions = []

        # 추론 결과에서 액션 추출
        for inf in inferences:
            if inf.recommendation:
                priority = self._get_priority_from_insight(inf)
                actions.append(
                    {
                        "priority": priority,
                        "type": inf.insight_type.value,
                        "action": inf.recommendation,
                        "source": "ontology_inference",
                        "rule": inf.rule_name,
                        "confidence": inf.confidence,
                    }
                )

        # 알림 기반 액션 추가
        for alert in metrics_data.get("alerts", []):
            if alert.get("severity") == "critical":
                actions.append(
                    {
                        "priority": "high",
                        "type": alert.get("type"),
                        "action": f"[긴급] {alert.get('message')} - 즉시 확인 필요",
                        "source": "alert",
                        "product": alert.get("title"),
                        "asin": alert.get("asin"),
                    }
                )
            elif alert.get("severity") == "warning":
                actions.append(
                    {
                        "priority": "medium",
                        "type": alert.get("type"),
                        "action": f"[주의] {alert.get('message')} - 모니터링 강화",
                        "source": "alert",
                        "product": alert.get("title"),
                        "asin": alert.get("asin"),
                    }
                )

        # 우선순위 정렬
        priority_order = {"high": 0, "medium": 1, "low": 2}
        actions.sort(key=lambda x: priority_order.get(x.get("priority"), 3))

        return actions[:10]

    def _build_reference_section(
        self,
        hybrid_context: HybridContext,
        external_signals: Optional[Dict[str, Any]],
        market_intelligence: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        참고자료 섹션 생성 (숫자 인용용)

        출처 우선순위:
        1. Market Intelligence (Layer 4 → Layer 3 → Layer 2)
        2. External Signals
        3. RAG Documents
        4. Knowledge Graph
        """
        entries = []
        idx = 1

        # 1. Market Intelligence 출처 (Layer 4: 거시경제/무역)
        if market_intelligence:
            sources = market_intelligence.get("sources", [])
            for source in sources[:6]:  # 최대 6개
                publisher = source.get("publisher", "")
                title = source.get("title", "")
                date = source.get("date", "")
                source_type = source.get("source_type", "")
                reliability = source.get("reliability_score", 0)

                # 출처 유형별 포맷
                if source_type == "government":
                    entries.append(f"[{idx}] {publisher}, {title}, {date}")
                elif source_type == "ir":
                    url = source.get("url", "")
                    entries.append(f'[{idx}] {publisher}, "{title}", {date}')
                elif source_type == "news":
                    entries.append(f'[{idx}] {publisher}, "{title}", {date}')
                else:
                    entries.append(f"[{idx}] {publisher}: {title} ({date})")
                idx += 1

        # 2. External Signal 출처 (Reddit, TikTok 등)
        for signal in (external_signals or {}).get("signals", [])[:3]:
            source = signal.get("source", "").replace("_", " ").title()
            title = signal.get("title", "")[:50]
            collected_at = signal.get("collected_at", "") or signal.get(
                "published_at", ""
            )
            if collected_at:
                collected_at = collected_at[:10]  # YYYY-MM-DD만
            entries.append(f'[{idx}] {source}, "{title}...", {collected_at}')
            idx += 1

        # 3. RAG 문서 출처
        for chunk in (hybrid_context.rag_chunks or [])[:2]:
            metadata = chunk.get("metadata", {})
            title = metadata.get("title") or metadata.get("doc_id", "가이드 문서")
            source_filename = metadata.get("source_filename", "")
            if source_filename:
                entries.append(f"[{idx}] 내부 가이드: {title} ({source_filename})")
            else:
                entries.append(f"[{idx}] 내부 가이드: {title}")
            idx += 1

        # 4. KG 근거 (요약)
        if hybrid_context.ontology_facts:
            fact_types = sorted(
                {
                    fact.get("type")
                    for fact in hybrid_context.ontology_facts
                    if fact.get("type")
                }
            )
            if fact_types:
                entries.append(
                    f"[{idx}] KnowledgeGraph: 온톨로지 추론 ({', '.join(fact_types)})"
                )

        if not entries:
            return ""

        return "## 참고자료\n" + "\n".join(entries)

    def _get_priority_from_insight(self, inference: InferenceResult) -> str:
        """인사이트 유형에서 우선순위 결정"""
        high_priority = {
            InsightType.RISK_ALERT,
            InsightType.COMPETITIVE_THREAT,
            InsightType.RANK_SHOCK,
        }
        medium_priority = {
            InsightType.PRICE_QUALITY_GAP,
            InsightType.COMPETITIVE_ADVANTAGE,
            InsightType.GROWTH_OPPORTUNITY,
        }

        if inference.insight_type in high_priority:
            return "high"
        elif inference.insight_type in medium_priority:
            return "medium"
        else:
            return "low"

    def _extract_highlights(
        self, inferences: List[InferenceResult], metrics_data: Dict
    ) -> List[Dict]:
        """하이라이트 추출"""
        highlights = []

        # 긍정적 추론 결과
        positive_types = {
            InsightType.MARKET_DOMINANCE,
            InsightType.GROWTH_MOMENTUM,
            InsightType.STABILITY,
            InsightType.COMPETITIVE_ADVANTAGE,
        }

        for inf in inferences:
            if inf.insight_type in positive_types:
                highlights.append(
                    {
                        "type": inf.insight_type.value,
                        "title": inf.insight_type.value.replace("_", " ").title(),
                        "detail": inf.insight,
                        "source": "ontology",
                    }
                )

        # 제품 메트릭에서 하이라이트
        product_metrics = metrics_data.get("product_metrics", [])

        # Top 10 진입
        for p in product_metrics:
            if p.get("current_rank", 100) <= 10:
                highlights.append(
                    {
                        "type": "top_rank",
                        "title": f"Top 10: {p.get('product_title', '')[:30]}...",
                        "detail": f"{p.get('category_id')} 카테고리 {p.get('current_rank')}위",
                        "asin": p.get("asin"),
                    }
                )

        # 순위 상승
        improving = [
            p
            for p in product_metrics
            if p.get("rank_change_1d") and p.get("rank_change_1d") < -3
        ]
        for p in improving[:3]:
            highlights.append(
                {
                    "type": "rank_up",
                    "title": f"순위 상승: {p.get('product_title', '')[:30]}...",
                    "detail": f"{abs(p.get('rank_change_1d'))}단계 상승 → 현재 {p.get('current_rank')}위",
                    "asin": p.get("asin"),
                }
            )

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

    async def _collect_external_signals(self) -> Dict[str, Any]:
        """
        External Signal 수집

        Returns:
            {
                "signals": [...],
                "report_section": "■ 전문 매체 근거: ...",
                "stats": {"by_tier": {...}, "by_source": {...}}
            }
        """
        result = {"signals": [], "report_section": "", "stats": {}}

        try:
            if not self._signal_collector:
                self._signal_collector = ExternalSignalCollector()
                await self._signal_collector.initialize()

            # 기존 수집된 신호 확인
            if self._signal_collector.signals:
                result["signals"] = [
                    s.to_dict() for s in self._signal_collector.signals[-20:]
                ]
                result["report_section"] = (
                    self._signal_collector.generate_report_section(days=7)
                )
                result["stats"] = self._signal_collector.get_stats()

            self.logger.debug(
                f"External signals: {len(result['signals'])} signals loaded"
            )

        except Exception as e:
            self.logger.warning(f"External signal collection failed: {e}")

        return result

    def _get_failed_signal_collectors(self) -> List[str]:
        """
        사용 불가능한 외부 신호 수집기 목록 반환

        Returns:
            실패한 수집기 이름 리스트
        """
        failed = []

        if not GOOGLE_TRENDS_AVAILABLE:
            failed.append("Google Trends")

        if not YOUTUBE_AVAILABLE:
            failed.append("YouTube")

        # ExternalSignalCollector 체크
        try:
            from src.tools.external_signal_collector import ExternalSignalCollector
        except ImportError:
            failed.append("External Signals (Tavily/RSS/Reddit)")

        # Market Intelligence 체크
        try:
            from src.tools.market_intelligence import MarketIntelligenceEngine
        except ImportError:
            failed.append("Market Intelligence")

        return failed

    async def _collect_market_intelligence(self) -> Dict[str, Any]:
        """
        Market Intelligence 데이터 수집 (Layer 2-4)

        Returns:
            {
                "layer_4": {...},  # 거시경제/무역
                "layer_3": {...},  # 산업/기업
                "layer_2": {...},  # 소비자 트렌드
                "sources": [...],  # 출처 목록
                "insight_section": "..."  # 생성된 인사이트 섹션
            }
        """
        result = {
            "layer_4": {},
            "layer_3": {},
            "layer_2": {},
            "sources": [],
            "insight_section": "",
        }

        try:
            if not self._market_intelligence:
                self._market_intelligence = MarketIntelligenceEngine()
                await self._market_intelligence.initialize()

            # 모든 레이어 병렬 수집
            await self._market_intelligence.collect_all_layers()

            # 레이어별 데이터 추출
            layer_data = self._market_intelligence.layer_data

            if DataLayer.LAYER_4_MACRO in layer_data:
                result["layer_4"] = layer_data[DataLayer.LAYER_4_MACRO].data
                result["sources"].extend(layer_data[DataLayer.LAYER_4_MACRO].sources)

            if DataLayer.LAYER_3_INDUSTRY in layer_data:
                result["layer_3"] = layer_data[DataLayer.LAYER_3_INDUSTRY].data
                result["sources"].extend(layer_data[DataLayer.LAYER_3_INDUSTRY].sources)

            if DataLayer.LAYER_2_CONSUMER in layer_data:
                result["layer_2"] = layer_data[DataLayer.LAYER_2_CONSUMER].data
                result["sources"].extend(layer_data[DataLayer.LAYER_2_CONSUMER].sources)

            # 인사이트 섹션 생성
            result["insight_section"] = (
                self._market_intelligence.generate_layered_insight()
            )

            # Google Trends 수집 (Phase 1)
            google_trends = await self._collect_google_trends()
            if google_trends.get("trends"):
                result["google_trends"] = google_trends["trends"]
                if google_trends.get("insight_section"):
                    result["insight_section"] += (
                        "\n\n" + google_trends["insight_section"]
                    )

            # YouTube 리뷰 수집 (Phase 2)
            youtube_reviews = await self._collect_youtube_reviews()
            if youtube_reviews.get("videos"):
                result["youtube_reviews"] = youtube_reviews["videos"]
                result["youtube_signals"] = youtube_reviews["signals"]
                if youtube_reviews.get("insight_section"):
                    result["insight_section"] += (
                        "\n\n" + youtube_reviews["insight_section"]
                    )

            self.logger.info(
                f"Market Intelligence collected: {len(result['sources'])} sources"
            )

        except Exception as e:
            self.logger.warning(f"Market Intelligence collection failed: {e}")

        return result

    async def _collect_google_trends(self) -> Dict[str, Any]:
        """
        Google Trends 데이터 수집

        Returns:
            {
                "trends": [...],
                "insight_section": "### Google Trends 검색 관심도\n...",
                "collected_at": str
            }
        """
        result = {"trends": [], "insight_section": "", "collected_at": ""}

        if not GOOGLE_TRENDS_AVAILABLE:
            self.logger.debug("Google Trends collector not available")
            return result

        try:
            if not self._google_trends:
                self._google_trends = GoogleTrendsCollector(
                    geo="US", timeframe="today 3-m"
                )

            # 뷰티 트렌드 수집
            trends = await self._google_trends.fetch_beauty_trends()

            if trends:
                result["trends"] = [t.to_dict() for t in trends]
                result["insight_section"] = (
                    self._google_trends.generate_insight_section(trends)
                )
                result["collected_at"] = trends[0].collected_at if trends else ""

                # 데이터 저장
                await self._google_trends.save_trends(trends)

            self.logger.info(f"Google Trends collected: {len(trends)} keywords")

        except Exception as e:
            self.logger.warning(f"Google Trends collection failed: {e}")

        return result

    async def _collect_youtube_reviews(self) -> Dict[str, Any]:
        """
        YouTube 리뷰 데이터 수집

        Returns:
            {
                "videos": [...],
                "signals": [...],  # ExternalSignal 형식
                "insight_section": "### YouTube 리뷰 트렌드\n...",
                "collected_at": str
            }
        """
        result = {
            "videos": [],
            "signals": [],
            "insight_section": "",
            "collected_at": "",
        }

        if not YOUTUBE_AVAILABLE:
            self.logger.debug("YouTube collector not available")
            return result

        try:
            if not self._youtube_collector:
                self._youtube_collector = YouTubeCollector()

            # LANEIGE 리뷰 수집
            videos = await self._youtube_collector.fetch_laneige_reviews(max_results=30)

            if videos:
                result["videos"] = [v.to_dict() for v in videos]
                result["signals"] = self._youtube_collector.to_external_signals(videos)
                result["insight_section"] = (
                    self._youtube_collector.generate_insight_section(videos)
                )
                result["collected_at"] = videos[0].collected_at if videos else ""

                # 데이터 저장
                await self._youtube_collector.save_videos(videos)

            self.logger.info(f"YouTube reviews collected: {len(videos)} videos")

        except Exception as e:
            self.logger.warning(f"YouTube collection failed: {e}")

        return result

    def _extract_data_source_info(
        self, metrics_data: Optional[Dict], crawl_data: Optional[Dict]
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
            "disclaimer": "Amazon은 Best Sellers 순위를 매 시간 업데이트합니다. 표시된 데이터는 수집 시점의 스냅샷입니다.",
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
                total = (
                    sum(
                        len(cat_data.get("rank_records", []))
                        for cat_data in categories.values()
                    )
                    if categories
                    else 0
                )
                source_info["total_products"] = total

        return source_info

    def _ingest_rag_knowledge(self, rag_chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """RAG 청크에서 지식 추출 후 KG에 적재"""
        stats = {"trend_relations": 0, "action_relations": 0}

        for chunk in rag_chunks or []:
            metadata = chunk.get("metadata", {})
            doc_type = metadata.get("doc_type", "")
            doc_id = metadata.get("doc_id", "")
            chunk_id = metadata.get("chunk_id") or chunk.get("id")
            target_brand = metadata.get("target_brand")
            brands_covered = metadata.get("brands_covered", [])

            subject = (
                self._normalize_brand_name(target_brand) if target_brand else "MARKET"
            )
            if not target_brand and brands_covered:
                subject = self._normalize_brand_name(brands_covered[0])

            # 트렌드 키워드 추출 (인텔리전스 문서 우선)
            if doc_type in {"intelligence", "knowledge_base"}:
                trend_keywords = metadata.get("keywords", [])
                for keyword in trend_keywords:
                    if len(keyword) < 3:
                        continue
                    relation = Relation(
                        subject=subject,
                        predicate=RelationType.HAS_TREND,
                        object=keyword,
                        properties={
                            "source": "rag",
                            "doc_id": doc_id,
                            "chunk_id": chunk_id,
                            "doc_type": doc_type,
                            "source_filename": metadata.get("source_filename", ""),
                        },
                    )
                    if self.kg.add_relation(relation):
                        stats["trend_relations"] += 1

            # 액션 아이템 추출 (플레이북/대응 가이드)
            if doc_type in {"playbook", "response_guide"}:
                action_lines = self._extract_action_lines(chunk.get("content", ""))
                for action in action_lines:
                    relation = Relation(
                        subject=subject,
                        predicate=RelationType.REQUIRES_ACTION,
                        object=action,
                        properties={
                            "source": "rag",
                            "doc_id": doc_id,
                            "chunk_id": chunk_id,
                            "doc_type": doc_type,
                            "source_filename": metadata.get("source_filename", ""),
                        },
                    )
                    if self.kg.add_relation(relation):
                        stats["action_relations"] += 1

        return stats

    def _ingest_external_signals(
        self, external_signals: Dict[str, Any]
    ) -> Dict[str, int]:
        """External Signal을 KG에 적재"""
        stats = {"trend_relations": 0}
        signals = external_signals.get("signals", []) if external_signals else []

        for signal in signals:
            keywords = signal.get("keywords", [])
            if not keywords:
                continue

            subject = self._infer_signal_subject(keywords)
            for keyword in keywords:
                relation = Relation(
                    subject=subject,
                    predicate=RelationType.HAS_TREND,
                    object=keyword,
                    properties={
                        "source": "external_signal",
                        "signal_id": signal.get("signal_id"),
                        "source_name": signal.get("source"),
                        "url": signal.get("url"),
                        "published_at": signal.get("published_at"),
                        "collected_at": signal.get("collected_at"),
                    },
                )
                if self.kg.add_relation(relation):
                    stats["trend_relations"] += 1

        return stats

    def _extract_action_lines(self, content: str) -> List[str]:
        """문서 본문에서 액션 항목 추출"""
        actions = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("- ") or stripped.startswith("* "):
                actions.append(stripped[2:].strip())
            elif stripped[:2].isdigit() and stripped[1:3] == ". ":
                actions.append(stripped[3:].strip())
        return [a for a in actions if 5 <= len(a) <= 140]

    def _infer_signal_subject(self, keywords: List[str]) -> str:
        """External Signal 키워드에서 대상 엔티티 추정"""
        brand_keywords = {
            "laneige": "LANEIGE",
            "cosrx": "COSRX",
            "tirtir": "TIRTIR",
            "rare beauty": "RARE BEAUTY",
            "innisfree": "INNISFREE",
            "etude": "ETUDE",
            "sulwhasoo": "SULWHASOO",
            "hera": "HERA",
        }
        for keyword in keywords:
            normalized = keyword.lower()
            if normalized in brand_keywords:
                return brand_keywords[normalized]
        return "MARKET"

    def _normalize_brand_name(self, brand: Optional[str]) -> str:
        if not brand:
            return "MARKET"
        return brand.upper() if brand.isalpha() else brand

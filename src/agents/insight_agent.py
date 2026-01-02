"""
Insight Agent
인사이트 생성 에이전트 (LLM 기반)

.. deprecated::
    이 모듈은 `HybridInsightAgent`로 대체되었습니다.
    새 코드에서는 `from src.agents import HybridInsightAgent`를 사용하세요.

    HybridInsightAgent는 Ontology 기반 추론과 RAG를 결합하여
    더 정확하고 맥락에 맞는 인사이트를 생성합니다.
"""
import warnings

warnings.warn(
    "InsightAgent is deprecated. Use HybridInsightAgent instead. "
    "See: from src.agents import HybridInsightAgent",
    DeprecationWarning,
    stacklevel=2
)

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from litellm import acompletion
from src.rag.retriever import DocumentRetriever
from src.rag.templates import ResponseTemplates
from src.monitoring.logger import AgentLogger
from src.monitoring.tracer import ExecutionTracer
from src.monitoring.metrics import QualityMetrics


class InsightAgent:
    """인사이트 생성 에이전트"""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        docs_dir: str = "./docs",
        logger: Optional[AgentLogger] = None,
        tracer: Optional[ExecutionTracer] = None,
        metrics: Optional[QualityMetrics] = None
    ):
        """
        Args:
            model: LLM 모델명
            docs_dir: RAG 문서 디렉토리
            logger: 로거
            tracer: 추적기
            metrics: 메트릭 수집기
        """
        self.model = model
        self.retriever = DocumentRetriever(docs_dir)
        self.templates = ResponseTemplates()
        self.logger = logger or AgentLogger("insight")
        self.tracer = tracer
        self.metrics = metrics

        self._results: Dict[str, Any] = {}

    async def execute(
        self,
        metrics_data: Dict[str, Any],
        crawl_summary: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        인사이트 생성

        Args:
            metrics_data: 지표 에이전트 결과
            crawl_summary: 크롤링 요약 (선택)

        Returns:
            {
                "status": "completed",
                "daily_insight": "...",
                "action_items": [...],
                "highlights": [...]
            }
        """
        self.logger.agent_start("InsightAgent", "인사이트 생성")
        start_time = datetime.now()

        if self.metrics:
            self.metrics.record_agent_start("insight")

        if self.tracer:
            self.tracer.start_span("insight_agent")

        try:
            results = {
                "status": "completed",
                "generated_at": datetime.now().isoformat(),
                "daily_insight": "",
                "action_items": [],
                "highlights": [],
                "warnings": []
            }

            # 1. RAG 컨텍스트 수집
            if self.tracer:
                self.tracer.start_span("retrieve_context")

            rag_context = await self._build_rag_context(metrics_data)

            if self.tracer:
                self.tracer.end_span("completed")

            # 2. 일일 인사이트 생성
            if self.tracer:
                self.tracer.start_span("generate_daily_insight")

            daily_insight = await self._generate_daily_insight(
                metrics_data, crawl_summary, rag_context
            )
            results["daily_insight"] = daily_insight

            if self.tracer:
                self.tracer.end_span("completed")

            # 3. 액션 아이템 추출
            if self.tracer:
                self.tracer.start_span("extract_actions")

            action_items = await self._extract_action_items(
                metrics_data, daily_insight
            )
            results["action_items"] = action_items

            if self.tracer:
                self.tracer.end_span("completed")

            # 4. 하이라이트 추출
            results["highlights"] = self._extract_highlights(metrics_data)

            # 5. 경고 사항 수집
            alerts = metrics_data.get("alerts", [])
            results["warnings"] = [
                a for a in alerts
                if a.get("severity") in ["warning", "critical"]
            ]

            self._results = results
            duration = (datetime.now() - start_time).total_seconds()

            if self.tracer:
                self.tracer.end_span("completed")

            if self.metrics:
                self.metrics.record_agent_complete("insight", {
                    "action_items": len(results["action_items"]),
                    "highlights": len(results["highlights"])
                })

            self.logger.agent_complete(
                "InsightAgent",
                duration,
                f"{len(results['action_items'])} actions, {len(results['highlights'])} highlights"
            )

            return results

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()

            if self.tracer:
                self.tracer.end_span("failed", str(e))

            if self.metrics:
                self.metrics.record_agent_error("insight", str(e))

            self.logger.agent_error("InsightAgent", str(e), duration)
            raise

    async def _build_rag_context(self, metrics_data: Dict) -> str:
        """RAG 컨텍스트 구축"""
        contexts = []

        # retriever 초기화 확인
        if not self.retriever._initialized:
            await self.retriever.initialize()

        # 지표 해석 가이드 검색
        summary = metrics_data.get("summary", {})

        # SoS 관련 컨텍스트
        sos_data = summary.get("laneige_sos_by_category", {})
        if sos_data:
            sos_context = await self.retriever.get_relevant_context(
                f"SoS 점유율 해석 {list(sos_data.values())}"
            )
            if sos_context:
                contexts.append(sos_context)

        # 알림 관련 컨텍스트
        alerts = metrics_data.get("alerts", [])
        if alerts:
            alert_types = set(a.get("type") for a in alerts)
            for alert_type in alert_types:
                alert_context = await self.retriever.get_relevant_context(
                    f"{alert_type} 알림 대응"
                )
                if alert_context:
                    contexts.append(alert_context)

        return "\n\n---\n\n".join(contexts) if contexts else ""

    async def _generate_daily_insight(
        self,
        metrics_data: Dict,
        crawl_summary: Optional[Dict],
        rag_context: str
    ) -> str:
        """일일 인사이트 생성"""
        # 프롬프트 구성
        system_prompt = self.templates.get_system_prompt()

        # 데이터 요약
        summary = metrics_data.get("summary", {})
        brand_metrics = metrics_data.get("brand_metrics", [])
        product_metrics = metrics_data.get("product_metrics", [])
        alerts = metrics_data.get("alerts", [])

        # LANEIGE 지표 추출
        laneige_brands = [b for b in brand_metrics if b.get("is_laneige")]

        data_summary = f"""
## 오늘의 데이터 요약

### LANEIGE 제품 현황
- 추적 중인 제품 수: {summary.get('laneige_products_tracked', 0)}개
- 카테고리별 점유율(SoS):
{self._format_sos(summary.get('laneige_sos_by_category', {}))}

### 베스트 순위 제품
{self._format_best_product(summary.get('best_ranking_product'))}

### 알림 현황
- 전체 알림: {summary.get('alert_count', 0)}건
- 심각(Critical): {summary.get('critical_alerts', 0)}건
- 경고(Warning): {summary.get('warning_alerts', 0)}건

{self._format_alerts(alerts[:5]) if alerts else "- 특이사항 없음"}
"""

        user_prompt = f"""
아래 데이터를 바탕으로 오늘의 LANEIGE Amazon 베스트셀러 인사이트를 생성해주세요.

{data_summary}

## 참고 가이드라인
{rag_context if rag_context else "- 기본 해석 기준 적용"}

요구사항:
1. 3-5문장의 핵심 인사이트 요약
2. 주목해야 할 순위 변동 사항
3. 카테고리별 LANEIGE 포지션 평가
4. 경쟁사 대비 시사점 (있는 경우)

주의: 단정적 표현을 피하고, 데이터 기반의 객관적 분석을 제공하세요.
"""

        try:
            response = await acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            insight = response.choices[0].message.content

            # 토큰 사용량 기록
            if self.metrics and hasattr(response, 'usage'):
                self.metrics.record_llm_call(
                    model=self.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    latency_ms=0,  # TODO: 실제 latency 측정
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
            return self._generate_fallback_insight(metrics_data)

    async def _extract_action_items(
        self,
        metrics_data: Dict,
        daily_insight: str
    ) -> List[Dict]:
        """액션 아이템 추출"""
        alerts = metrics_data.get("alerts", [])
        actions = []

        # 알림 기반 액션
        for alert in alerts:
            if alert.get("severity") == "critical":
                actions.append({
                    "priority": "high",
                    "type": alert.get("type"),
                    "action": f"[긴급] {alert.get('message')} - 즉시 확인 필요",
                    "product": alert.get("title"),
                    "asin": alert.get("asin")
                })
            elif alert.get("severity") == "warning":
                actions.append({
                    "priority": "medium",
                    "type": alert.get("type"),
                    "action": f"[주의] {alert.get('message')} - 모니터링 강화",
                    "product": alert.get("title"),
                    "asin": alert.get("asin")
                })

        # 정렬 (우선순위)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        actions.sort(key=lambda x: priority_order.get(x.get("priority"), 3))

        return actions[:10]  # 최대 10개

    def _extract_highlights(self, metrics_data: Dict) -> List[Dict]:
        """하이라이트 추출"""
        highlights = []
        product_metrics = metrics_data.get("product_metrics", [])
        summary = metrics_data.get("summary", {})

        # Top 10 진입 제품
        top10_products = [
            p for p in product_metrics
            if p.get("current_rank", 100) <= 10
        ]
        for p in top10_products:
            highlights.append({
                "type": "top_rank",
                "title": f"Top 10 진입: {p.get('product_title', '')[:30]}",
                "detail": f"{p.get('category_id')} 카테고리 {p.get('current_rank')}위",
                "asin": p.get("asin")
            })

        # 순위 상승 제품
        improving = [
            p for p in product_metrics
            if p.get("rank_change_1d") and p.get("rank_change_1d") < -3
        ]
        for p in improving[:3]:
            highlights.append({
                "type": "rank_up",
                "title": f"순위 상승: {p.get('product_title', '')[:30]}",
                "detail": f"{abs(p.get('rank_change_1d'))}단계 상승 → 현재 {p.get('current_rank')}위",
                "asin": p.get("asin")
            })

        # 높은 SoS 카테고리
        sos_data = summary.get("laneige_sos_by_category", {})
        for cat, sos in sos_data.items():
            if sos >= 0.05:  # 5% 이상
                highlights.append({
                    "type": "high_sos",
                    "title": f"높은 점유율: {cat}",
                    "detail": f"SoS {sos*100:.1f}%",
                    "category": cat
                })

        return highlights[:10]

    def _format_sos(self, sos_data: Dict) -> str:
        """SoS 포맷팅"""
        if not sos_data:
            return "  - 데이터 없음"

        lines = []
        for cat, sos in sos_data.items():
            lines.append(f"  - {cat}: {sos*100:.1f}%")
        return "\n".join(lines)

    def _format_best_product(self, product: Optional[Dict]) -> str:
        """베스트 제품 포맷팅"""
        if not product:
            return "- 데이터 없음"

        return f"""- 제품: {product.get('title', '')[:50]}
- 순위: {product.get('rank')}위
- 카테고리: {product.get('category')}"""

    def _format_alerts(self, alerts: List[Dict]) -> str:
        """알림 포맷팅"""
        if not alerts:
            return ""

        lines = ["### 주요 알림"]
        for a in alerts:
            severity = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(
                a.get("severity"), "⚪"
            )
            lines.append(f"- {severity} {a.get('message')}")

        return "\n".join(lines)

    def _generate_fallback_insight(self, metrics_data: Dict) -> str:
        """폴백 인사이트 생성 (LLM 실패 시)"""
        summary = metrics_data.get("summary", {})

        insight = f"""오늘 LANEIGE Amazon 베스트셀러 분석 결과입니다.

- 추적 중인 제품: {summary.get('laneige_products_tracked', 0)}개
- 알림: {summary.get('alert_count', 0)}건 (Critical: {summary.get('critical_alerts', 0)}, Warning: {summary.get('warning_alerts', 0)})

※ 상세 인사이트 생성 중 오류가 발생하여 기본 요약을 제공합니다."""

        return insight

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """비용 추정 (GPT-4.1-mini 기준)"""
        # $0.40/1M input, $1.60/1M output
        input_cost = (prompt_tokens / 1_000_000) * 0.40
        output_cost = (completion_tokens / 1_000_000) * 1.60
        return round(input_cost + output_cost, 6)

    def get_results(self) -> Dict[str, Any]:
        """마지막 실행 결과"""
        return self._results

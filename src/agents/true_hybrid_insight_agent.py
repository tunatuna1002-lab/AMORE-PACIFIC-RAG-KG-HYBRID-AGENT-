"""
True Hybrid Insight Agent
=========================
진정한 RAG-Ontology 하이브리드 인사이트 생성 에이전트

기존 HybridInsightAgent 대비 개선점:
1. TrueHybridRetriever: 필수 벡터 검색 + Semantic Chunking + Re-ranking
2. OWLReasoner: OWL DL 추론 (Python IF-THEN → SWRL 규칙)
3. EntityLinker: NER 기반 온톨로지 개념 연결
4. ConfidenceFusion: 다중 소스 신뢰도 통합 (벡터 40% + 온톨로지 35% + 엔티티 25%)

Flow:
1. 현재 데이터로 OWL Ontology 업데이트
2. OWL Reasoner로 Description Logic 추론
3. TrueHybridRetriever로 검색 (EntityLinker + Vector + Reranking + Fusion)
4. 추론 결과 + RAG 컨텍스트 + 신뢰도로 LLM 인사이트 생성
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from litellm import acompletion

# True Hybrid 컴포넌트
from src.rag.true_hybrid_retriever import TrueHybridRetriever, HybridResult, get_true_hybrid_retriever
from src.rag.entity_linker import EntityLinker, LinkedEntity, get_entity_linker
from src.rag.confidence_fusion import ConfidenceFusion, FusedResult
from src.ontology.owl_reasoner import OWLReasoner, get_owl_reasoner

# 기존 컴포넌트 (호환성 유지)
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.ontology.business_rules import register_all_rules
from src.domain.entities.relations import InferenceResult, InsightType, RelationType, Relation

from src.rag.context_builder import ContextBuilder
from src.rag.retriever import DocumentRetriever
from src.rag.templates import ResponseTemplates

from src.monitoring.logger import AgentLogger
from src.monitoring.tracer import ExecutionTracer
from src.monitoring.metrics import QualityMetrics
from src.tools.external_signal_collector import ExternalSignalCollector


class TrueHybridInsightAgent:
    """
    진정한 RAG-Ontology 하이브리드 인사이트 생성 에이전트

    핵심 개선점:
    - OWL DL 추론: Python 규칙 → owlready2 + Pellet/HermiT
    - 필수 벡터 검색: 선택적 → 의무화 + Semantic Chunking
    - Cross-Encoder Reranking: Two-stage retrieval
    - Entity Linking: 텍스트 → 온톨로지 개념 연결
    - Confidence Fusion: 다중 소스 신뢰도 통합

    사용 예:
        agent = TrueHybridInsightAgent(model="gpt-4.1-mini")
        result = await agent.execute(metrics_data)
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        docs_dir: str = ".",
        owl_reasoner: Optional[OWLReasoner] = None,
        true_hybrid_retriever: Optional[TrueHybridRetriever] = None,
        entity_linker: Optional[EntityLinker] = None,
        # 호환성: 기존 컴포넌트도 지원
        knowledge_graph: Optional[KnowledgeGraph] = None,
        legacy_reasoner: Optional[OntologyReasoner] = None,
        logger: Optional[AgentLogger] = None,
        tracer: Optional[ExecutionTracer] = None,
        metrics: Optional[QualityMetrics] = None
    ):
        """
        Args:
            model: LLM 모델명
            docs_dir: RAG 문서 디렉토리
            owl_reasoner: OWL 추론기 (새로운)
            true_hybrid_retriever: True Hybrid 검색기 (새로운)
            entity_linker: 엔티티 링커 (새로운)
            knowledge_graph: 기존 지식 그래프 (호환성)
            legacy_reasoner: 기존 추론기 (호환성)
            logger: 로거
            tracer: 추적기
            metrics: 메트릭 수집기
        """
        self.model = model
        self.docs_dir = docs_dir

        # True Hybrid 컴포넌트 (새로운)
        self.owl_reasoner = owl_reasoner
        self.true_hybrid_retriever = true_hybrid_retriever
        self.entity_linker = entity_linker
        self._true_hybrid_initialized = False

        # 기존 컴포넌트 (호환성 유지)
        self.kg = knowledge_graph or KnowledgeGraph()
        self.legacy_reasoner = legacy_reasoner or OntologyReasoner(self.kg)
        if not self.legacy_reasoner.rules:
            register_all_rules(self.legacy_reasoner)

        # 컨텍스트 빌더 & 템플릿
        self.context_builder = ContextBuilder(max_tokens=4000)
        self.templates = ResponseTemplates()

        # 모니터링
        self.logger = logger or AgentLogger("true_hybrid_insight")
        self.tracer = tracer
        self.metrics = metrics

        # 결과 캐시
        self._results: Dict[str, Any] = {}
        self._last_hybrid_result: Optional[HybridResult] = None
        self._last_owl_inferences: List[Dict[str, Any]] = []

        # External Signal Collector
        self._signal_collector: Optional[ExternalSignalCollector] = None

    def _initialize_true_hybrid(self) -> bool:
        """True Hybrid 컴포넌트 지연 초기화"""
        if self._true_hybrid_initialized:
            return True

        try:
            # OWL Reasoner
            if not self.owl_reasoner:
                self.owl_reasoner = get_owl_reasoner()
            self.logger.info("OWL Reasoner initialized")

            # Entity Linker
            if not self.entity_linker:
                self.entity_linker = get_entity_linker()
            self.logger.info("Entity Linker initialized")

            # True Hybrid Retriever
            if not self.true_hybrid_retriever:
                self.true_hybrid_retriever = get_true_hybrid_retriever(
                    owl_reasoner=self.owl_reasoner,
                    knowledge_graph=self.kg,
                    docs_path=self.docs_dir
                )
            self.logger.info("True Hybrid Retriever initialized")

            self._true_hybrid_initialized = True
            return True

        except Exception as e:
            self.logger.warning(f"True Hybrid initialization failed: {e}")
            self.logger.warning("Falling back to legacy mode")
            return False

    async def execute(
        self,
        metrics_data: Dict[str, Any],
        crawl_data: Optional[Dict[str, Any]] = None,
        crawl_summary: Optional[Dict] = None,
        use_true_hybrid: bool = True
    ) -> Dict[str, Any]:
        """
        하이브리드 인사이트 생성

        Args:
            metrics_data: 지표 에이전트 결과
            crawl_data: 크롤링 원본 데이터
            crawl_summary: 크롤링 요약
            use_true_hybrid: True Hybrid 모드 사용 여부

        Returns:
            {
                "status": "completed",
                "mode": "true_hybrid" | "legacy",
                "daily_insight": "...",
                "confidence": 0.0~1.0,
                "action_items": [...],
                "highlights": [...],
                "owl_inferences": [...],
                "entity_links": [...],
                "explanations": [...]
            }
        """
        self.logger.agent_start("TrueHybridInsightAgent", "인사이트 생성")
        start_time = datetime.now()

        if self.metrics:
            self.metrics.record_agent_start("true_hybrid_insight")

        if self.tracer:
            self.tracer.start_span("true_hybrid_insight_agent")

        try:
            # True Hybrid 모드 초기화 시도
            true_hybrid_available = use_true_hybrid and self._initialize_true_hybrid()
            mode = "true_hybrid" if true_hybrid_available else "legacy"

            # 데이터 출처 정보
            data_source = self._extract_data_source_info(metrics_data, crawl_data)

            results = {
                "status": "completed",
                "mode": mode,
                "generated_at": datetime.now().isoformat(),
                "data_source": data_source,
                "daily_insight": "",
                "confidence": 0.0,
                "action_items": [],
                "highlights": [],
                "warnings": [],
                "owl_inferences": [],
                "entity_links": [],
                "explanations": [],
                "hybrid_stats": {}
            }

            if true_hybrid_available:
                # === True Hybrid 모드 ===
                results = await self._execute_true_hybrid_mode(
                    results, metrics_data, crawl_data, crawl_summary
                )
            else:
                # === Legacy 모드 ===
                results = await self._execute_legacy_mode(
                    results, metrics_data, crawl_data, crawl_summary
                )

            self._results = results
            duration = (datetime.now() - start_time).total_seconds()

            if self.tracer:
                self.tracer.end_span("completed")

            if self.metrics:
                self.metrics.record_agent_complete("true_hybrid_insight", {
                    "mode": mode,
                    "confidence": results["confidence"],
                    "action_items": len(results["action_items"])
                })

            self.logger.agent_complete(
                "TrueHybridInsightAgent",
                duration,
                f"mode={mode}, confidence={results['confidence']:.2f}"
            )

            return results

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()

            if self.tracer:
                self.tracer.end_span("failed", str(e))

            if self.metrics:
                self.metrics.record_agent_error("true_hybrid_insight", str(e))

            self.logger.agent_error("TrueHybridInsightAgent", str(e), duration)
            raise

    async def _execute_true_hybrid_mode(
        self,
        results: Dict[str, Any],
        metrics_data: Dict[str, Any],
        crawl_data: Optional[Dict[str, Any]],
        crawl_summary: Optional[Dict]
    ) -> Dict[str, Any]:
        """True Hybrid 모드 실행"""

        # 1. OWL Ontology 업데이트
        if self.tracer:
            self.tracer.start_span("update_owl_ontology")

        owl_stats = self._update_owl_ontology(crawl_data, metrics_data)
        results["hybrid_stats"]["owl_update"] = owl_stats

        if self.tracer:
            self.tracer.end_span("completed")

        # 2. OWL 추론 실행
        if self.tracer:
            self.tracer.start_span("owl_reasoning")

        owl_inferences = self._run_owl_reasoning()
        results["owl_inferences"] = owl_inferences
        self._last_owl_inferences = owl_inferences

        if self.tracer:
            self.tracer.end_span("completed")

        # 3. True Hybrid 검색 (Entity Linking + Vector + Reranking + Fusion)
        if self.tracer:
            self.tracer.start_span("true_hybrid_retrieval")

        query = "LANEIGE 오늘의 Amazon 베스트셀러 성과 분석"
        hybrid_result = await self.true_hybrid_retriever.retrieve(query)
        self._last_hybrid_result = hybrid_result

        results["confidence"] = hybrid_result.confidence
        results["entity_links"] = [
            {
                "text": el.text,
                "concept": el.concept_name,
                "concept_type": el.concept_type,
                "confidence": el.confidence
            }
            for el in hybrid_result.entity_links
        ]

        results["hybrid_stats"]["retrieval"] = {
            "documents_count": len(hybrid_result.documents),
            "entity_links_count": len(hybrid_result.entity_links),
            "confidence": hybrid_result.confidence
        }

        if self.tracer:
            self.tracer.end_span("completed")

        # 4. 추론 설명 생성
        if self.tracer:
            self.tracer.start_span("generate_explanations")

        explanations = self._generate_owl_explanations(owl_inferences)
        results["explanations"] = explanations

        if self.tracer:
            self.tracer.end_span("completed")

        # 5. External Signal 수집
        if self.tracer:
            self.tracer.start_span("collect_external_signals")

        external_signals = await self._collect_external_signals()
        results["external_signals"] = external_signals

        if self.tracer:
            self.tracer.end_span("completed")

        # 6. 일일 인사이트 생성 (LLM)
        if self.tracer:
            self.tracer.start_span("generate_daily_insight")

        daily_insight = await self._generate_true_hybrid_insight(
            hybrid_result, owl_inferences, metrics_data, crawl_summary, external_signals
        )
        results["daily_insight"] = daily_insight

        if self.tracer:
            self.tracer.end_span("completed")

        # 7. 액션 아이템 추출
        action_items = self._extract_action_items_from_owl(owl_inferences, metrics_data)
        results["action_items"] = action_items

        # 8. 하이라이트 추출
        results["highlights"] = self._extract_highlights_from_owl(owl_inferences, metrics_data)

        # 9. 경고 수집
        alerts = metrics_data.get("alerts", [])
        results["warnings"] = [
            a for a in alerts
            if a.get("severity") in ["warning", "critical"]
        ]

        return results

    async def _execute_legacy_mode(
        self,
        results: Dict[str, Any],
        metrics_data: Dict[str, Any],
        crawl_data: Optional[Dict[str, Any]],
        crawl_summary: Optional[Dict]
    ) -> Dict[str, Any]:
        """Legacy 모드 실행 (기존 HybridInsightAgent 로직)"""
        from src.rag.hybrid_retriever import HybridRetriever

        # 기존 로직 재사용
        doc_retriever = DocumentRetriever(self.docs_dir)
        hybrid_retriever = HybridRetriever(
            knowledge_graph=self.kg,
            reasoner=self.legacy_reasoner,
            doc_retriever=doc_retriever,
            auto_init_rules=False
        )

        # 1. KG 업데이트
        if crawl_data:
            self.kg.load_from_crawl_data(crawl_data)
        if metrics_data:
            self.kg.load_from_metrics_data(metrics_data)

        # 2. 하이브리드 검색
        query = "LANEIGE 오늘의 Amazon 베스트셀러 성과 분석"
        hybrid_context = await hybrid_retriever.retrieve(
            query=query,
            current_metrics=metrics_data,
            include_explanations=True
        )

        results["confidence"] = 0.5  # Legacy 모드 기본 신뢰도

        # 3. 인사이트 생성
        context = self.context_builder.build(
            hybrid_context=hybrid_context,
            current_metrics=metrics_data,
            query=query,
            knowledge_graph=self.kg
        )

        system_prompt = self.context_builder.build_system_prompt(include_guardrails=True)
        user_prompt = f"""
{context}

## 요청사항
위 분석 결과를 바탕으로 오늘의 LANEIGE Amazon 베스트셀러 인사이트를 작성해주세요.
마크다운 형식, 구체적 수치 인용, 단정적 표현 지양.
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
            results["daily_insight"] = response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            results["daily_insight"] = "인사이트 생성 중 오류 발생"

        # 4. 액션 아이템
        for inf in hybrid_context.inferences:
            if inf.recommendation:
                results["action_items"].append({
                    "priority": "medium",
                    "type": inf.insight_type.value,
                    "action": inf.recommendation,
                    "source": "legacy_inference",
                    "confidence": inf.confidence
                })

        return results

    def _update_owl_ontology(
        self,
        crawl_data: Optional[Dict],
        metrics_data: Dict
    ) -> Dict[str, int]:
        """OWL Ontology 업데이트"""
        stats = {"brands_added": 0, "competitors_added": 0}

        if not self.owl_reasoner:
            return stats

        # 카테고리별 SoS 데이터에서 브랜드 추출
        categories = metrics_data.get("categories", {})
        brand_sos_map: Dict[str, List[float]] = {}

        for cat_id, cat_data in categories.items():
            sos_data = cat_data.get("sos", {}).get("daily", {})
            for brand, sos in sos_data.items():
                if brand not in brand_sos_map:
                    brand_sos_map[brand] = []
                brand_sos_map[brand].append(sos)

        # 평균 SoS로 브랜드 추가
        for brand, sos_list in brand_sos_map.items():
            avg_sos = sum(sos_list) / len(sos_list)
            if self.owl_reasoner.add_brand(brand, avg_sos):
                stats["brands_added"] += 1

        # 경쟁사 관계 추가 (동일 카테고리 내 상위 5개 브랜드)
        for cat_id, cat_data in categories.items():
            sos_data = cat_data.get("sos", {}).get("daily", {})
            top_brands = sorted(sos_data.items(), key=lambda x: x[1], reverse=True)[:5]

            for i, (brand1, _) in enumerate(top_brands):
                for brand2, _ in top_brands[i+1:]:
                    if self.owl_reasoner.add_competitor_relation(brand1, brand2):
                        stats["competitors_added"] += 1

        self.logger.info(f"OWL updated: {stats['brands_added']} brands, {stats['competitors_added']} competitors")
        return stats

    def _run_owl_reasoning(self) -> List[Dict[str, Any]]:
        """OWL 추론 실행"""
        if not self.owl_reasoner:
            return []

        inferences = []

        # Market Position 추론
        self.owl_reasoner.infer_market_positions()

        # 모든 브랜드 정보 조회
        brands = self.owl_reasoner.get_all_brands()

        for brand_info in brands:
            brand_name = brand_info.get("name", "")
            position = brand_info.get("market_position", "")
            sos = brand_info.get("share_of_shelf", 0)

            if position:
                inferences.append({
                    "type": "market_position",
                    "brand": brand_name,
                    "inferred_class": position,
                    "share_of_shelf": sos,
                    "confidence": 0.95,
                    "reasoning": f"OWL DL inference: SoS {sos:.1%} → {position}"
                })

            # 경쟁사 관계
            competitors = brand_info.get("competitors", [])
            if competitors:
                inferences.append({
                    "type": "competitive_relation",
                    "brand": brand_name,
                    "competitors": competitors,
                    "confidence": 0.90,
                    "reasoning": f"OWL symmetric property: competitorOf"
                })

        self.logger.info(f"OWL reasoning: {len(inferences)} inferences")
        return inferences

    def _generate_owl_explanations(
        self,
        owl_inferences: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """OWL 추론 설명 생성"""
        explanations = []

        for inf in owl_inferences:
            inf_type = inf.get("type", "")

            if inf_type == "market_position":
                explanation = {
                    "rule": "OWL_MarketPosition",
                    "type": inf_type,
                    "insight": f"{inf['brand']}은(는) {inf['inferred_class']} 등급입니다.",
                    "explanation": (
                        f"OWL Description Logic 추론: "
                        f"Share of Shelf {inf['share_of_shelf']:.1%} 기준으로 "
                        f"{inf['inferred_class']} 클래스에 분류됨. "
                        f"(DominantBrand: ≥30%, StrongBrand: 15-30%, NicheBrand: <15%)"
                    ),
                    "confidence": inf["confidence"]
                }
            elif inf_type == "competitive_relation":
                explanation = {
                    "rule": "OWL_SymmetricCompetitor",
                    "type": inf_type,
                    "insight": f"{inf['brand']}의 경쟁사: {', '.join(inf['competitors'])}",
                    "explanation": (
                        f"OWL Symmetric Property 추론: "
                        f"competitorOf 관계는 양방향 자동 추론됨. "
                        f"A가 B의 경쟁사면 B도 A의 경쟁사."
                    ),
                    "confidence": inf["confidence"]
                }
            else:
                explanation = {
                    "rule": "OWL_Generic",
                    "type": inf_type,
                    "insight": str(inf),
                    "explanation": "OWL 추론 결과",
                    "confidence": inf.get("confidence", 0.5)
                }

            explanations.append(explanation)

        return explanations

    async def _generate_true_hybrid_insight(
        self,
        hybrid_result: HybridResult,
        owl_inferences: List[Dict[str, Any]],
        metrics_data: Dict,
        crawl_summary: Optional[Dict],
        external_signals: Optional[Dict] = None
    ) -> str:
        """True Hybrid 인사이트 생성 (LLM)"""

        # 컨텍스트 구성
        context_parts = []

        # 1. OWL 추론 결과
        context_parts.append("## OWL 온톨로지 추론 결과\n")
        for inf in owl_inferences[:5]:
            if inf["type"] == "market_position":
                context_parts.append(
                    f"- **{inf['brand']}**: {inf['inferred_class']} (SoS {inf['share_of_shelf']:.1%})"
                )
            elif inf["type"] == "competitive_relation":
                context_parts.append(
                    f"- **{inf['brand']}** 경쟁사: {', '.join(inf['competitors'][:3])}"
                )

        # 2. Entity Links
        if hybrid_result.entity_links:
            context_parts.append("\n## 엔티티 링킹 결과\n")
            for el in hybrid_result.entity_links[:5]:
                context_parts.append(
                    f"- {el.text} → {el.concept_name} ({el.concept_type}, 신뢰도 {el.confidence:.0%})"
                )

        # 3. 검색 결과 (상위 3개)
        if hybrid_result.documents:
            context_parts.append("\n## 관련 문서 (Re-ranked)\n")
            for i, doc in enumerate(hybrid_result.documents[:3], 1):
                content = doc.get("content", "")[:200]
                score = doc.get("score", 0)
                context_parts.append(f"[{i}] (score: {score:.2f}) {content}...")

        # 4. 메트릭 요약
        summary = metrics_data.get("summary", {})
        context_parts.append(f"""
## 오늘의 메트릭 요약
- LANEIGE 추적 제품: {summary.get('laneige_products_tracked', 0)}개
- 알림: {summary.get('alert_count', 0)}건
""")

        # 5. External Signal
        external_context = ""
        if external_signals and external_signals.get("report_section"):
            external_context = f"""
## 외부 트렌드 신호
{external_signals['report_section']}
"""

        # 6. Confidence Score
        context_parts.append(f"""
## 통합 신뢰도
- **Confidence Score**: {hybrid_result.confidence:.0%}
  - 벡터 유사도 기여: 40%
  - 온톨로지 추론 기여: 35%
  - 엔티티 연결 기여: 25%
""")

        full_context = "\n".join(context_parts)

        # 시스템 프롬프트
        system_prompt = """당신은 AMORE Pacific의 시장 분석 전문가입니다.
OWL 온톨로지 추론 결과와 True Hybrid RAG 검색 결과를 바탕으로
정확하고 신뢰할 수 있는 인사이트를 생성합니다.

가이드라인:
- OWL 추론 결과를 우선 인용 (Description Logic 기반)
- Confidence Score를 명시하여 신뢰도 투명성 확보
- 단정적 표현 대신 "~로 분석됩니다", "~할 가능성이 있습니다" 사용
- 구체적 수치 인용
"""

        user_prompt = f"""
{full_context}
{external_context}

---

## 요청사항

위 True Hybrid 분석 결과를 바탕으로 오늘의 LANEIGE Amazon 베스트셀러 인사이트를 작성해주세요.

포함 내용:
1. **핵심 요약** (3-5문장): 가장 중요한 인사이트 + 신뢰도 명시
2. **OWL 추론 기반 발견**: Market Position, 경쟁 관계 등
3. **외부 트렌드 연계** (있는 경우)
4. **주의 필요 사항**
5. **권장 액션**

형식: 마크다운, 신뢰도 표기 (예: "[신뢰도 95%]")
"""

        try:
            response = await acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1500
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
            return self._generate_fallback_insight(owl_inferences, metrics_data)

    def _generate_fallback_insight(
        self,
        owl_inferences: List[Dict[str, Any]],
        metrics_data: Dict
    ) -> str:
        """폴백 인사이트 생성"""
        summary = metrics_data.get("summary", {})

        insight_parts = [
            "## 오늘의 LANEIGE Amazon 베스트셀러 분석\n",
            f"- 추적 중인 제품: {summary.get('laneige_products_tracked', 0)}개",
            f"- 알림: {summary.get('alert_count', 0)}건"
        ]

        if owl_inferences:
            insight_parts.append("\n### OWL 추론 결과")
            for inf in owl_inferences[:3]:
                if inf["type"] == "market_position":
                    insight_parts.append(
                        f"- {inf['brand']}: {inf['inferred_class']} (SoS {inf['share_of_shelf']:.1%})"
                    )

        insight_parts.append(
            "\n\n_※ 상세 인사이트 생성 중 오류가 발생하여 기본 요약을 제공합니다._"
        )

        return "\n".join(insight_parts)

    def _extract_action_items_from_owl(
        self,
        owl_inferences: List[Dict[str, Any]],
        metrics_data: Dict
    ) -> List[Dict]:
        """OWL 추론 결과에서 액션 아이템 추출"""
        actions = []

        for inf in owl_inferences:
            if inf["type"] == "market_position":
                position = inf.get("inferred_class", "")
                brand = inf.get("brand", "")
                sos = inf.get("share_of_shelf", 0)

                if position == "DominantBrand":
                    actions.append({
                        "priority": "low",
                        "type": "market_dominance",
                        "action": f"{brand} 시장 지배력 유지 전략 검토",
                        "source": "owl_inference",
                        "confidence": inf["confidence"]
                    })
                elif position == "NicheBrand" and brand == "LANEIGE":
                    actions.append({
                        "priority": "high",
                        "type": "growth_opportunity",
                        "action": f"{brand} 카테고리 확장 및 SoS 개선 필요 (현재 {sos:.1%})",
                        "source": "owl_inference",
                        "confidence": inf["confidence"]
                    })

        # 알림 기반 액션
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

        # 우선순위 정렬
        priority_order = {"high": 0, "medium": 1, "low": 2}
        actions.sort(key=lambda x: priority_order.get(x.get("priority"), 3))

        return actions[:10]

    def _extract_highlights_from_owl(
        self,
        owl_inferences: List[Dict[str, Any]],
        metrics_data: Dict
    ) -> List[Dict]:
        """OWL 추론 결과에서 하이라이트 추출"""
        highlights = []

        for inf in owl_inferences:
            if inf["type"] == "market_position":
                position = inf.get("inferred_class", "")
                if position in ["DominantBrand", "StrongBrand"]:
                    highlights.append({
                        "type": "owl_inference",
                        "title": f"{inf['brand']}: {position}",
                        "detail": f"OWL 추론 (SoS {inf['share_of_shelf']:.1%})",
                        "source": "owl_reasoner"
                    })

        # 제품 메트릭에서 하이라이트
        product_metrics = metrics_data.get("product_metrics", [])

        for p in product_metrics:
            if p.get("current_rank", 100) <= 10:
                highlights.append({
                    "type": "top_rank",
                    "title": f"Top 10: {p.get('product_title', '')[:30]}...",
                    "detail": f"{p.get('category_id')} 카테고리 {p.get('current_rank')}위",
                    "asin": p.get("asin")
                })

        return highlights[:10]

    async def _collect_external_signals(self) -> Dict[str, Any]:
        """External Signal 수집"""
        result = {
            "signals": [],
            "report_section": "",
            "stats": {}
        }

        try:
            if not self._signal_collector:
                self._signal_collector = ExternalSignalCollector()
                await self._signal_collector.initialize()

            if self._signal_collector.signals:
                result["signals"] = [s.to_dict() for s in self._signal_collector.signals[-20:]]
                result["report_section"] = self._signal_collector.generate_report_section(days=7)
                result["stats"] = self._signal_collector.get_stats()

        except Exception as e:
            self.logger.warning(f"External signal collection failed: {e}")

        return result

    def _extract_data_source_info(
        self,
        metrics_data: Optional[Dict],
        crawl_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """데이터 출처 정보 추출"""
        source_info = {
            "platform": "Amazon US Best Sellers",
            "collected_at": None,
            "snapshot_date": None,
            "categories": [],
            "total_products": 0,
            "disclaimer": "Amazon은 Best Sellers 순위를 매 시간 업데이트합니다."
        }

        if crawl_data:
            source_info["collected_at"] = crawl_data.get("collected_at")

        if metrics_data:
            metadata = metrics_data.get("metadata", {})
            source_info["snapshot_date"] = metadata.get("data_date")
            if not source_info["collected_at"]:
                source_info["collected_at"] = metadata.get("generated_at")

            categories = metrics_data.get("categories", {})
            source_info["categories"] = list(categories.keys())

        return source_info

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """비용 추정"""
        input_cost = (prompt_tokens / 1_000_000) * 0.40
        output_cost = (completion_tokens / 1_000_000) * 1.60
        return round(input_cost + output_cost, 6)

    # === 접근자 메서드 ===

    def get_results(self) -> Dict[str, Any]:
        """마지막 실행 결과"""
        return self._results

    def get_last_hybrid_result(self) -> Optional[HybridResult]:
        """마지막 True Hybrid 결과"""
        return self._last_hybrid_result

    def get_last_owl_inferences(self) -> List[Dict[str, Any]]:
        """마지막 OWL 추론 결과"""
        return self._last_owl_inferences

    def get_owl_reasoner(self) -> Optional[OWLReasoner]:
        """OWL 추론기 반환"""
        return self.owl_reasoner


# 싱글톤 인스턴스
_agent_instance: Optional[TrueHybridInsightAgent] = None


def get_true_hybrid_insight_agent(**kwargs) -> TrueHybridInsightAgent:
    """TrueHybridInsightAgent 싱글톤 인스턴스 반환"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = TrueHybridInsightAgent(**kwargs)
    return _agent_instance

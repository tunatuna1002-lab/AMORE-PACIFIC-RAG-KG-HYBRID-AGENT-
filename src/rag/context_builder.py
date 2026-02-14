"""
Context Builder
LLM 프롬프트용 컨텍스트 조립기

기능:
1. 하이브리드 검색 결과를 LLM 프롬프트로 변환
2. 토큰 제한 고려한 컨텍스트 압축
3. 우선순위 기반 정보 선택
4. 다양한 출력 포맷 지원
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.domain.entities.relations import InferenceResult


class OutputFormat(str, Enum):
    """출력 포맷"""

    MARKDOWN = "markdown"
    PLAIN = "plain"
    STRUCTURED = "structured"


class ContextPriority(str, Enum):
    """컨텍스트 우선순위"""

    CRITICAL = "critical"  # 반드시 포함
    HIGH = "high"  # 높은 우선순위
    MEDIUM = "medium"  # 중간 우선순위
    LOW = "low"  # 낮은 우선순위


@dataclass
class ContextSection:
    """컨텍스트 섹션"""

    title: str
    content: str
    priority: ContextPriority
    token_estimate: int = 0
    source: str = "unknown"

    def __post_init__(self):
        # 토큰 추정 (대략 4글자 = 1토큰)
        if self.token_estimate == 0:
            self.token_estimate = len(self.content) // 4


class ContextBuilder:
    """
    LLM 프롬프트용 컨텍스트 빌더

    역할:
    1. 온톨로지 추론 결과 포맷팅
    2. RAG 검색 결과 포맷팅
    3. 현재 데이터 포맷팅
    4. 토큰 제한 내 최적 조합

    사용 예:
        builder = ContextBuilder(max_tokens=4000)
        context = builder.build(hybrid_context, current_metrics)
    """

    # 섹션 템플릿
    SECTION_TEMPLATES = {
        "inference": """## {title}

{content}
""",
        "rag": """## 참고 가이드라인

{content}
""",
        "data": """## 현재 데이터

{content}
""",
        "entity": """## 관련 정보

{content}
""",
    }

    def __init__(self, max_tokens: int = 4000, output_format: OutputFormat = OutputFormat.MARKDOWN):
        """
        Args:
            max_tokens: 최대 토큰 수
            output_format: 출력 포맷
        """
        self.max_tokens = max_tokens
        self.output_format = output_format

    def build(
        self,
        hybrid_context: Any,  # HybridContext
        current_metrics: dict[str, Any] | None = None,
        query: str | None = None,
        knowledge_graph: Any = None,
    ) -> str:
        """
        통합 컨텍스트 구성

        Args:
            hybrid_context: HybridRetriever 결과
            current_metrics: 현재 지표 데이터
            query: 원본 쿼리
            knowledge_graph: KnowledgeGraph 인스턴스 (카테고리 계층 조회용)

        Returns:
            LLM 프롬프트용 컨텍스트 문자열
        """
        sections: list[ContextSection] = []

        # 1. 카테고리 계층 섹션 (순위 관련 질문 시 우선 배치)
        entities = hybrid_context.entities if hasattr(hybrid_context, "entities") else {}
        if knowledge_graph and (entities.get("categories") or entities.get("products")):
            # 순위 관련 키워드 감지
            is_ranking_query = False
            if query:
                ranking_keywords = ["순위", "rank", "위", "ranking", "등수"]
                is_ranking_query = any(kw in query.lower() for kw in ranking_keywords)

            if is_ranking_query:
                hierarchy_section = self._build_category_hierarchy_section(
                    entities, knowledge_graph
                )
                if hierarchy_section.content.strip():
                    sections.append(hierarchy_section)

        # 2. 온톨로지 추론 결과 섹션
        if hasattr(hybrid_context, "inferences") and hybrid_context.inferences:
            inference_section = self._build_inference_section(hybrid_context.inferences)
            sections.append(inference_section)

        # 3. 현재 데이터 섹션
        if current_metrics:
            data_section = self._build_data_section(current_metrics, entities)
            sections.append(data_section)

        # 4. 지식 그래프 사실 섹션
        if hasattr(hybrid_context, "ontology_facts") and hybrid_context.ontology_facts:
            facts_section = self._build_facts_section(hybrid_context.ontology_facts)
            sections.append(facts_section)

        # 5. RAG 가이드라인 섹션
        if hasattr(hybrid_context, "rag_chunks") and hybrid_context.rag_chunks:
            rag_section = self._build_rag_section(hybrid_context.rag_chunks)
            sections.append(rag_section)

        # 6. 토큰 제한 내 조합
        selected_sections = self._select_within_limit(sections)

        # 7. 최종 조립
        return self._assemble(selected_sections, query)

    def _build_category_hierarchy_section(
        self, entities: dict[str, list[str]], knowledge_graph: Any
    ) -> ContextSection:
        """카테고리 계층 구조 섹션 구성"""
        lines = []

        # 카테고리 계층 정보
        categories = entities.get("categories", [])
        for category in categories:
            hierarchy = knowledge_graph.get_category_hierarchy(category)
            if "error" in hierarchy:
                continue

            lines.append(f"### {hierarchy.get('name', category)}")
            lines.append(f"- **레벨**: {hierarchy.get('level', 0)}")

            # 상위 카테고리 경로
            if hierarchy.get("ancestors"):
                path_names = [a["name"] for a in reversed(hierarchy["ancestors"])]
                path_names.append(hierarchy["name"])
                lines.append(f"- **전체 경로**: {' > '.join(path_names)}")

            # 하위 카테고리
            if hierarchy.get("descendants"):
                children = ", ".join([d["name"] for d in hierarchy["descendants"][:5]])
                if len(hierarchy["descendants"]) > 5:
                    children += f" 외 {len(hierarchy['descendants']) - 5}개"
                lines.append(f"- **하위 카테고리**: {children}")

            lines.append("")

        # 제품별 카테고리 순위 컨텍스트
        products = entities.get("products", [])
        if products:
            lines.append("### 제품별 카테고리 순위")

            for product_asin in products[:5]:  # 최대 5개 제품
                product_ctx = knowledge_graph.get_product_category_context(product_asin)
                if product_ctx.get("categories"):
                    # 제품 메타데이터 조회
                    product_meta = knowledge_graph.get_entity_metadata(product_asin)
                    product_name = product_meta.get("product_name", product_asin)

                    lines.append(f"\n**{product_name}** (ASIN: {product_asin}):")

                    # 카테고리별 순위 정렬 (계층 레벨 순)
                    cat_infos = sorted(
                        product_ctx["categories"],
                        key=lambda x: x.get("hierarchy", {}).get("level", 99),
                    )

                    for cat_info in cat_infos:
                        hierarchy = cat_info.get("hierarchy", {})
                        cat_name = hierarchy.get("name", cat_info.get("category_id"))
                        rank = cat_info.get("rank", "N/A")
                        level = hierarchy.get("level", 0)
                        lines.append(f"  - [{cat_name}] {rank}위 (Level {level})")

                    lines.append("")

        content = "\n".join(lines)

        return ContextSection(
            title="카테고리 계층 구조",
            content=content,
            priority=ContextPriority.HIGH,
            source="category_hierarchy",
        )

    def _build_inference_section(self, inferences: list[InferenceResult]) -> ContextSection:
        """온톨로지 추론 결과 섹션 구성"""
        lines = []

        for i, inf in enumerate(inferences, 1):
            insight_type = inf.insight_type.value.replace("_", " ").title()

            lines.append(f"### 인사이트 {i}: {insight_type}")
            lines.append(f"- **분석 결과**: {inf.insight}")

            if inf.recommendation:
                lines.append(f"- **권장 액션**: {inf.recommendation}")

            lines.append(f"- **신뢰도**: {inf.confidence:.0%}")

            # 근거 조건
            if inf.evidence and inf.evidence.get("satisfied_conditions"):
                conditions = inf.evidence["satisfied_conditions"]
                lines.append(f"- **근거**: {', '.join(conditions)}")

            lines.append("")

        content = "\n".join(lines)

        return ContextSection(
            title="분석 결과 (Ontology Reasoning)",
            content=content,
            priority=ContextPriority.CRITICAL,
            source="ontology",
        )

    def _build_data_section(
        self, metrics: dict[str, Any], entities: dict[str, list[str]]
    ) -> ContextSection:
        """현재 데이터 섹션 구성"""
        lines = []

        summary = metrics.get("summary", {})

        # 전체 요약
        lines.append("### 전체 현황")
        lines.append(f"- 추적 제품 수: {summary.get('laneige_products_tracked', 0)}개")
        lines.append(
            f"- 알림: {summary.get('alert_count', 0)}건 "
            f"(Critical: {summary.get('critical_alerts', 0)}, "
            f"Warning: {summary.get('warning_alerts', 0)})"
        )

        # 카테고리별 SoS
        sos_data = summary.get("laneige_sos_by_category", {})
        if sos_data:
            lines.append("\n### 카테고리별 점유율 (SoS)")
            for cat, sos in sos_data.items():
                lines.append(f"- {cat}: {sos*100:.1f}%")

        # 베스트 제품
        best = summary.get("best_ranking_product")
        if best:
            lines.append("\n### 베스트 순위 제품")
            lines.append(f"- {best.get('title', '')[:40]}...")
            lines.append(f"- 순위: {best.get('rank')}위 ({best.get('category')})")

        # 특정 엔티티 상세 (요청된 경우)
        brands = entities.get("brands", [])
        categories = entities.get("categories", [])

        if brands or categories:
            lines.append("\n### 요청 엔티티 상세")

            # 브랜드 메트릭
            for brand_metric in metrics.get("brand_metrics", []):
                brand_name = brand_metric.get("brand_name", "").lower()
                if brand_name in [b.lower() for b in brands]:
                    cat = brand_metric.get("category_id", "")
                    if not categories or cat in categories:
                        lines.append(f"\n**{brand_metric.get('brand_name')}** ({cat}):")
                        lines.append(f"  - SoS: {brand_metric.get('share_of_shelf', 0)*100:.1f}%")
                        if brand_metric.get("avg_rank"):
                            lines.append(f"  - 평균 순위: {brand_metric['avg_rank']:.1f}")
                        lines.append(f"  - 제품 수: {brand_metric.get('product_count', 0)}개")
                        lines.append(f"  - Top 10: {brand_metric.get('top10_count', 0)}개")

            # 마켓 메트릭 (카테고리)
            for market_metric in metrics.get("market_metrics", []):
                cat = market_metric.get("category_id", "")
                if not categories or cat in categories:
                    lines.append(f"\n**{cat} 카테고리**:")
                    if market_metric.get("hhi"):
                        lines.append(f"  - HHI: {market_metric['hhi']:.3f}")
                    if market_metric.get("cpi"):
                        lines.append(f"  - CPI: {market_metric['cpi']:.1f}")

        content = "\n".join(lines)

        return ContextSection(
            title="현재 데이터", content=content, priority=ContextPriority.HIGH, source="data"
        )

    def _build_facts_section(self, facts: list[dict[str, Any]]) -> ContextSection:
        """지식 그래프 사실 섹션 구성"""
        lines = []

        for fact in facts[:5]:  # 상위 5개
            fact_type = fact.get("type", "")
            entity = fact.get("entity", "")
            data = fact.get("data", {})

            if fact_type == "brand_info":
                lines.append(f"**{entity}**:")
                if data.get("sos"):
                    lines.append(f"  - 점유율: {data['sos']*100:.1f}%")
                if data.get("avg_rank"):
                    lines.append(f"  - 평균 순위: {data['avg_rank']:.1f}")
                if data.get("product_count"):
                    lines.append(f"  - 제품 수: {data['product_count']}개")

            elif fact_type == "brand_products":
                lines.append(f"**{entity}** 제품: {data.get('product_count', 0)}개")

            elif fact_type == "competitors":
                if isinstance(data, list):
                    competitors = [c.get("brand", "") for c in data[:3]]
                    lines.append(f"**{entity}** 경쟁사: {', '.join(competitors)}")

            elif fact_type == "category_brands":
                top_brands = [b.get("brand", "") for b in data.get("top_brands", [])[:3]]
                if top_brands:
                    lines.append(f"**{entity}** Top 브랜드: {', '.join(top_brands)}")

            lines.append("")

        content = "\n".join(lines)

        return ContextSection(
            title="관련 정보 (Knowledge Graph)",
            content=content,
            priority=ContextPriority.MEDIUM,
            source="knowledge_graph",
        )

    def _build_rag_section(self, chunks: list[dict[str, Any]]) -> ContextSection:
        """RAG 검색 결과 섹션 구성"""
        lines = []

        for chunk in chunks[:3]:  # 상위 3개
            title = chunk.get("metadata", {}).get("title", "")
            content = chunk.get("content", "")
            doc_id = chunk.get("metadata", {}).get("doc_id", "")

            if title:
                lines.append(f"### {title}")
            elif doc_id:
                lines.append(f"### {doc_id}")

            # 내용 축약
            if len(content) > 400:
                content = content[:400] + "..."

            lines.append(content)
            lines.append("")

        content = "\n".join(lines)

        return ContextSection(
            title="참고 가이드라인 (RAG)",
            content=content,
            priority=ContextPriority.MEDIUM,
            source="rag",
        )

    def _select_within_limit(self, sections: list[ContextSection]) -> list[ContextSection]:
        """토큰 제한 내 섹션 선택"""
        # 우선순위 정렬
        priority_order = {
            ContextPriority.CRITICAL: 0,
            ContextPriority.HIGH: 1,
            ContextPriority.MEDIUM: 2,
            ContextPriority.LOW: 3,
        }
        sorted_sections = sorted(sections, key=lambda s: priority_order.get(s.priority, 4))

        selected = []
        total_tokens = 0

        for section in sorted_sections:
            if total_tokens + section.token_estimate <= self.max_tokens:
                selected.append(section)
                total_tokens += section.token_estimate
            elif section.priority == ContextPriority.CRITICAL:
                # CRITICAL은 반드시 포함 (다른 것 제거)
                selected.append(section)
                total_tokens += section.token_estimate

        return selected

    def _assemble(self, sections: list[ContextSection], query: str | None = None) -> str:
        """최종 조립"""
        parts = []

        # 쿼리 정보 (있으면)
        if query:
            parts.append(f"## 사용자 질문\n{query}\n")

        # 섹션들
        for section in sections:
            if self.output_format == OutputFormat.MARKDOWN:
                parts.append(f"## {section.title}\n\n{section.content}")
            elif self.output_format == OutputFormat.PLAIN:
                parts.append(f"[{section.title}]\n{section.content}")
            else:
                parts.append(section.content)

        return "\n\n".join(parts)

    def build_system_prompt(
        self,
        include_guardrails: bool = True,
        data_date: str | None = None,
    ) -> str:
        """
        시스템 프롬프트 생성

        Args:
            include_guardrails: 안전장치 포함 여부
            data_date: 데이터 수집일 (날짜 컨텍스트용)

        Returns:
            시스템 프롬프트
        """
        from src.infrastructure.feature_flags import FeatureFlags

        flags = FeatureFlags.get_instance()
        if flags.use_centralized_prompts():
            from prompts.registry import PromptRegistry

            registry = PromptRegistry.get_instance()
            return registry.get_system_prompt(
                "chatbot", include_guardrails=include_guardrails, data_date=data_date
            )

        # ── Legacy inline prompt logic (unchanged) ──
        from prompts.components import (
            build_date_context,
            get_hallucination_prevention,
            get_security_rules,
        )

        prompt = """당신은 Amazon 베스트셀러 순위 분석 전문가입니다.

## 역할
- 순위 데이터 기반 인사이트 제공
- 온톨로지 추론 결과 해석 및 설명
- 지표 해석 및 전략적 시사점 도출
- 마케터의 의사결정 보조

## 응답 원칙
1. 제공된 "분석 결과 (Ontology Reasoning)" 섹션의 인사이트를 우선 활용
2. "참고 가이드라인 (RAG)" 섹션의 해석 기준 적용
3. "현재 데이터" 섹션의 수치를 구체적으로 인용
4. 추론 근거와 함께 설명
"""

        # 날짜 컨텍스트 추가
        prompt += build_date_context(data_date=data_date)

        if include_guardrails:
            prompt += """
## 주의사항 (반드시 준수)
1. **단정 금지**: "원인은 ~입니다", "확실히" 등 단정적 표현 금지
2. **가능성 표현**: "~일 수 있습니다", "~로 보입니다" 등 완곡 표현 사용
3. **원인 확정 금지**: 순위 변동 원인(재고, 광고, 품질)을 단정하지 않음
4. **매출 예측 금지**: 판매량, 매출, ROI 등 수치 예측 금지
5. **추이 확인 권장**: 단기 노이즈 가능성 언급

## 응답 형식
- 한국어로 응답
- 비즈니스 문서 톤 유지
- 구조화된 형식 (제목, 불릿 포인트)
- 인사이트 → 근거 → 권장 순서
"""
            # 환각 방지 규칙 추가
            prompt += get_hallucination_prevention()

            # 보안 규칙 추가
            prompt += get_security_rules()

        return prompt

    def build_user_prompt(
        self, query: str, context: str, additional_instructions: str | None = None
    ) -> str:
        """
        사용자 프롬프트 생성

        Args:
            query: 사용자 질문
            context: 빌드된 컨텍스트
            additional_instructions: 추가 지시사항

        Returns:
            사용자 프롬프트
        """
        prompt = f"""아래 컨텍스트를 참고하여 질문에 답변해주세요.

{context}

---

## 질문
{query}
"""

        if additional_instructions:
            prompt += f"""
## 추가 지시사항
{additional_instructions}
"""

        prompt += """
## 요청사항
1. 온톨로지 추론 결과가 있으면 이를 기반으로 답변
2. 구체적인 수치와 함께 설명
3. 불확실한 부분은 명확히 밝힘
4. 권장 액션이 있으면 포함
"""

        return prompt


class CompactContextBuilder(ContextBuilder):
    """
    토큰 효율적인 컴팩트 컨텍스트 빌더

    짧은 응답이 필요한 경우 사용
    """

    def __init__(self, max_tokens: int = 2000):
        super().__init__(max_tokens=max_tokens)

    def build(
        self,
        hybrid_context: Any,
        current_metrics: dict[str, Any] | None = None,
        query: str | None = None,
        knowledge_graph: Any = None,
    ) -> str:
        """컴팩트 컨텍스트 구성"""
        parts = []

        # 1. 카테고리 계층 정보 (순위 관련 질문 시)
        entities = hybrid_context.entities if hasattr(hybrid_context, "entities") else {}
        if knowledge_graph and query:
            ranking_keywords = ["순위", "rank", "위", "ranking", "등수"]
            if any(kw in query.lower() for kw in ranking_keywords):
                # 제품별 카테고리 순위
                products = entities.get("products", [])
                if products:
                    parts.append("[카테고리별 순위]")
                    for product_asin in products[:3]:
                        product_ctx = knowledge_graph.get_product_category_context(product_asin)
                        if product_ctx.get("categories"):
                            for cat_info in product_ctx["categories"][:2]:
                                hierarchy = cat_info.get("hierarchy", {})
                                cat_name = hierarchy.get("name", "")
                                rank = cat_info.get("rank", "N/A")
                                if cat_name:
                                    parts.append(f"- {cat_name}: {rank}위")

        # 2. 핵심 추론 결과만
        if hasattr(hybrid_context, "inferences") and hybrid_context.inferences:
            parts.append("\n[추론 결과]")
            for inf in hybrid_context.inferences[:3]:
                parts.append(f"- {inf.insight}")
                if inf.recommendation:
                    parts.append(f"  → {inf.recommendation}")

        # 3. 핵심 데이터만
        if current_metrics:
            summary = current_metrics.get("summary", {})
            parts.append("\n[현재 데이터]")
            parts.append(f"- 추적 제품: {summary.get('laneige_products_tracked', 0)}개")

            sos = summary.get("laneige_sos_by_category", {})
            if sos:
                sos_str = ", ".join(f"{k}: {v*100:.1f}%" for k, v in list(sos.items())[:2])
                parts.append(f"- SoS: {sos_str}")

        # 4. RAG는 제목만
        if hasattr(hybrid_context, "rag_chunks") and hybrid_context.rag_chunks:
            parts.append("\n[참고 문서]")
            for chunk in hybrid_context.rag_chunks[:2]:
                title = chunk.get("metadata", {}).get("title", "")
                if title:
                    parts.append(f"- {title}")

        return "\n".join(parts)

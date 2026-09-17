"""
Context Builder
LLM 프롬프트용 컨텍스트 조립기

기능:
1. 하이브리드 검색 결과를 증거 카드로 렌더링 (v4 ``HybridRetriever._combine_contexts``와 같은
   조립기 ``evidence_assembly``·렌더러 ``evidence_renderer``, 설계 E1·E11)
2. 카드 밖 정보(사용자 질문·응답 가이드·일일 인사이트용 현재 데이터) 섹션 조립
3. 토큰 제한 고려한 섹션 선택 (카드 섹션은 잘리지 않는다)
4. 다양한 출력 포맷 지원
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.domain.entities.evidence import Evidence
from src.rag.evidence_assembly import assemble_evidence
from src.rag.evidence_renderer import render_for_prompt


def prompt_cards_for(hybrid_context: Any) -> list[Evidence]:
    """프롬프트에 실을 카드.

    검색기가 카드를 조립한 컨텍스트(``evidence``나 ``prompt_evidence``가 채워짐)는
    ``prompt_evidence`` 그대로 쓴다. 카드 조립 전 컨텍스트(직접 만든 HybridContext 등)는
    원자료 필드로 같은 조립기를 돌린다 — 입력이 같으면 결과도 같다.
    """
    cards = getattr(hybrid_context, "prompt_evidence", None)
    evidence = getattr(hybrid_context, "evidence", None)
    if isinstance(cards, list) and (cards or evidence):
        return cards
    return assemble_evidence(
        entities=getattr(hybrid_context, "entities", None) or {},
        metric_facts=getattr(hybrid_context, "metric_facts", None) or [],
        ontology_facts=getattr(hybrid_context, "ontology_facts", None) or [],
        inferences=getattr(hybrid_context, "inferences", None) or [],
        rag_chunks=getattr(hybrid_context, "rag_chunks", None) or [],
    ).prompt_evidence


@dataclass
class SourceReference:
    """출처 참조"""

    index: int
    source_type: str  # "rag", "kg", "ontology", "data"
    title: str
    detail: str


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
    1. 증거 카드(추론·DB 수치·KG 관계·문서) 렌더링
    2. 현재 데이터 포맷팅 (current_metrics를 넘기는 호출자만)
    3. 토큰 제한 내 섹션 조합

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

    # Korean stopwords for AIS citation matching
    _KOREAN_STOPWORDS = frozenset(
        {
            "은",
            "는",
            "이",
            "가",
            "을",
            "를",
            "의",
            "에",
            "에서",
            "로",
            "으로",
            "와",
            "과",
            "도",
            "만",
            "까지",
            "부터",
            "하고",
            "그리고",
            "또는",
            "하지만",
            "그러나",
            "입니다",
            "합니다",
            "있습니다",
            "없습니다",
            "됩니다",
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "and",
            "or",
            "but",
            "with",
            "from",
            "by",
            "as",
            "it",
        }
    )

    def __init__(
        self,
        max_tokens: int = 4000,
        output_format: OutputFormat = OutputFormat.MARKDOWN,
        enable_ais: bool = True,
    ):
        """
        Args:
            max_tokens: 최대 토큰 수
            output_format: 출력 포맷
            enable_ais: AIS 인라인 인용 활성화 여부
        """
        self.max_tokens = max_tokens
        self.output_format = output_format
        self.enable_ais = enable_ais
        self._sources: list[SourceReference] = []
        self._citation_stats: dict[str, int] = {
            "total_sentences": 0,
            "cited_sentences": 0,
            "uncited_sentences": 0,
        }

    def _register_source(self, source_type: str, title: str, detail: str) -> int:
        """출처 등록 및 인덱스 반환"""
        idx = len(self._sources) + 1
        self._sources.append(SourceReference(idx, source_type, title, detail))
        return idx

    def get_source_references(self) -> list[SourceReference]:
        """등록된 출처 목록 반환"""
        return self._sources.copy()

    def build_source_appendix(self) -> str:
        """출처 부록 생성"""
        if not self._sources:
            return ""
        lines = ["\n---\n## 출처"]
        for ref in self._sources:
            lines.append(f"[{ref.index}] [{ref.source_type}] {ref.title}: {ref.detail}")
        return "\n".join(lines)

    # =========================================================================
    # AIS (Attributed Information Synthesis) Inline Citation
    # =========================================================================

    def _split_sentences(self, text: str) -> list[str]:
        """
        텍스트를 문장 단위로 분리 (한국어 + 영어 지원)

        Korean endings: 다., 요., 니다., 세요.
        Also splits on newlines and '. '
        """
        import re

        if not text or not text.strip():
            return []

        # Split on Korean sentence endings and standard period-space
        # Pattern: sentence-ending punctuation followed by space or newline
        parts = re.split(
            r"(?<=[다요죠음임])\.\s+|(?<=니다)\.\s*|(?<=세요)\.\s*|(?<=습니다)\.\s*"
            r"|(?<=합니다)\.\s*|(?<=됩니다)\.\s*|(?<=있습니다)\.\s*"
            r"|(?<=없습니다)\.\s*"
            r"|(?<=[.!?])\s+|\n+",
            text,
        )

        # Filter empty strings and whitespace-only
        return [s.strip() for s in parts if s and s.strip()]

    def _extract_keywords(self, text: str) -> set[str]:
        """텍스트에서 키워드 추출 (불용어 제거)"""
        import re

        # Split on non-alphanumeric (keep Korean characters)
        tokens = re.findall(r"[\w가-힣]+", text.lower())
        return {t for t in tokens if len(t) > 1 and t not in self._KOREAN_STOPWORDS}

    def _match_sentence_to_sources(self, sentence: str) -> list[int]:
        """
        문장과 매칭되는 출처 인덱스 목록 반환

        Args:
            sentence: 대상 문장

        Returns:
            매칭되는 SourceReference 인덱스 리스트
        """
        if not self._sources or not sentence.strip():
            return []

        sentence_keywords = self._extract_keywords(sentence)
        if not sentence_keywords:
            return []

        matched_indices: list[int] = []

        for ref in self._sources:
            source_text = f"{ref.title} {ref.detail}"
            source_keywords = self._extract_keywords(source_text)
            if not source_keywords:
                continue

            # Jaccard-like overlap ratio
            overlap = sentence_keywords & source_keywords
            # Use ratio relative to sentence keywords (not union)
            ratio = len(overlap) / len(sentence_keywords) if sentence_keywords else 0

            if ratio >= 0.15:
                matched_indices.append(ref.index)

        return matched_indices

    def build_ais_response(self, response_text: str) -> str:
        """
        응답 텍스트에 AIS 인라인 인용 태그 추가

        Args:
            response_text: LLM 응답 텍스트

        Returns:
            [출처N] 태그가 추가된 응답 텍스트
        """
        if not self.enable_ais or not self._sources:
            self._citation_stats = {
                "total_sentences": 0,
                "cited_sentences": 0,
                "uncited_sentences": 0,
            }
            return response_text

        sentences = self._split_sentences(response_text)
        if not sentences:
            self._citation_stats = {
                "total_sentences": 0,
                "cited_sentences": 0,
                "uncited_sentences": 0,
            }
            return response_text

        cited_count = 0
        annotated_parts: list[str] = []

        for sentence in sentences:
            matched = self._match_sentence_to_sources(sentence)
            if matched:
                cited_count += 1
                tags = "".join(f"[출처{idx}]" for idx in matched)
                annotated_parts.append(f"{sentence} {tags}")
            else:
                annotated_parts.append(sentence)

        total = len(sentences)
        self._citation_stats = {
            "total_sentences": total,
            "cited_sentences": cited_count,
            "uncited_sentences": total - cited_count,
        }

        return " ".join(annotated_parts)

    def get_citation_stats(self) -> dict[str, int | float]:
        """
        인용 통계 반환

        Returns:
            {
                "total_sentences": int,
                "cited_sentences": int,
                "uncited_sentences": int,
                "citation_rate": float  (0.0 ~ 1.0)
            }
        """
        total = self._citation_stats.get("total_sentences", 0)
        cited = self._citation_stats.get("cited_sentences", 0)
        return {
            **self._citation_stats,
            "citation_rate": cited / total if total > 0 else 0.0,
        }

    def build(
        self,
        hybrid_context: Any,  # HybridContext
        current_metrics: dict[str, Any] | None = None,
        query: str | None = None,
        knowledge_graph: Any = None,
    ) -> str:
        """
        통합 컨텍스트 구성 — 증거 카드 렌더링 (설계 E1, v4 ``_combine_contexts``와 같은 렌더러)

        추론·DB 수치·KG 관계·문서는 ``prompt_evidence`` 카드로만 싣는다. ``hybrid_context``에
        카드 필드가 없으면(카드 조립 전 객체) 같은 조립기(``assemble_evidence``)로 만든다.
        인용 지시(``CITATION_INSTRUCTION``)는 여기서 붙이지 않는다 — 답변 LLM을 부르는
        프롬프트 조립부가 한 번만 붙인다.

        Args:
            hybrid_context: HybridRetriever 결과
            current_metrics: 현재 지표 데이터. 주면 카드 밖 "현재 데이터" 섹션을 싣는다
                (일일 인사이트 경로용 — 챗봇 답변 경로는 넘기지 않는다)
            query: 원본 쿼리
            knowledge_graph: 하위 호환용 인자 (카테고리 계층은 relation 카드로 싣는다)

        Returns:
            LLM 프롬프트용 컨텍스트 문자열
        """
        self._sources = []
        sections: list[ContextSection] = []
        entities = getattr(hybrid_context, "entities", None) or {}

        evidence_block = render_for_prompt(prompt_cards_for(hybrid_context))
        if evidence_block:
            # 카드는 잘리면 prompt_evidence와 렌더 결과가 어긋나므로 토큰 제한에서 빼지 않는다
            sections.append(
                ContextSection(
                    title="증거 카드",
                    content=evidence_block,
                    priority=ContextPriority.CRITICAL,
                    source="evidence",
                )
            )

        if current_metrics:
            sections.append(self._build_data_section(current_metrics, entities))

        return self._assemble(self._select_within_limit(sections), query)

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
                lines.append(f"- {cat}: {sos * 100:.1f}%")

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
                        lines.append(f"  - SoS: {brand_metric.get('share_of_shelf', 0) * 100:.1f}%")
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

        # 출처 부록 추가
        source_appendix = self.build_source_appendix()
        if source_appendix:
            parts.append(source_appendix)

        # 응답 포맷 가이드 (Grounding 강화)
        parts.append(
            "## 응답 가이드\n"
            "- 위 컨텍스트에 **포함된 데이터만** 사용하여 답변하세요.\n"
            "- 수치 인용 형식: [지표명] [수치]% ([스냅샷 날짜/카테고리])\n"
            # 예시에 구체 수치를 두지 않는다. 예전 예시의 "5.0%"는 컨텍스트에 없는 숫자인데
            # 수치가 없는 질문에서 답변으로 새어 나올 수 있었다 (사이클 10).
            '- 예시 형식: "[브랜드]의 [카테고리] SoS는 [수치]%입니다 ([스냅샷 날짜] 기준)"'
            " — 수치는 반드시 위 컨텍스트에서 가져오세요.\n"
            "- 컨텍스트에 없는 브랜드, 제품, 수치를 생성하지 마세요."
        )

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
1. [규칙 추론] 카드의 인사이트를 우선 활용
2. [문서] 카드의 해석 기준 적용
3. [DB 수치] 카드의 수치를 날짜와 함께 구체적으로 인용
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
                sos_str = ", ".join(f"{k}: {v * 100:.1f}%" for k, v in list(sos.items())[:2])
                parts.append(f"- SoS: {sos_str}")

        # 4. RAG는 제목만
        if hasattr(hybrid_context, "rag_chunks") and hybrid_context.rag_chunks:
            parts.append("\n[참고 문서]")
            for chunk in hybrid_context.rag_chunks[:2]:
                title = chunk.get("metadata", {}).get("title", "")
                if title:
                    parts.append(f"- {title}")

        return "\n".join(parts)

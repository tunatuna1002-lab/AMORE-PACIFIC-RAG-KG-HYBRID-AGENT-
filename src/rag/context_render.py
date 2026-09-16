"""KG-fact → prompt rendering
===========================

Shared rendering primitives plus the legacy ``HybridContext`` renderer.

Three renderers feed LLM prompts in this package and they are pinned to three
*different* output formats by three test suites:

* :func:`combine_contexts` — the ``HybridRetriever`` path. One bullet per fact,
  document bodies truncated at :data:`LEGACY_CHUNK_CHARS`.
* ``ContextBuilder`` (``context_builder.py``) — token-budgeted sections with
  numbered source citations, bodies truncated at
  :data:`CITED_CHUNK_CHARS`. **This is the renderer to converge on**: it is the
  only one that registers sources, so it is the only one whose output can be
  cited back to the user.
* ``OWLRetrievalStrategy._build_combined_context`` — entity links + inference
  dicts; it renders no KG facts at all, only documents.

What is genuinely shared — the caps, the percent formatting, the truncation and
the competitor/brand-name extraction — lives here and every renderer calls it.
The remaining divergence is the *wording*, and collapsing that changes prompt
text, so it needs a golden-set run rather than a refactor (see the F3 report).

Moved out of ``hybrid_retriever.py`` (F3 split). The context is duck-typed so
this module does not import back into ``hybrid_retriever``.
"""

from __future__ import annotations

from typing import Any

# 프롬프트에 싣는 항목 수 상한 (세 렌더러 공통)
MAX_FACTS = 5
MAX_CHUNKS = 3

# 문서 본문 절단 길이. 두 값이 다른 것은 의도된 상태가 아니라 미수렴 잔여물이다.
LEGACY_CHUNK_CHARS = 500  # HybridRetriever / OWLRetrievalStrategy
CITED_CHUNK_CHARS = 400  # ContextBuilder (인용 표기 [n] 만큼 예산을 더 쓴다)


def truncate(content: str, max_chars: int) -> str:
    """본문 축약 — 한계를 넘으면 자르고 말줄임표를 붙인다."""
    if len(content) > max_chars:
        return content[:max_chars] + "..."
    return content


def as_percent(fraction: float, digits: int = 1) -> str:
    """내부 표현(분수 0~1)을 표시용 퍼센트 문자열로."""
    return f"{fraction * 100:.{digits}f}%"


def brand_names(items: Any, limit: int = 3) -> list[str]:
    """``[{"brand": ...}, ...]`` 목록에서 브랜드 이름만 뽑는다.

    Callers that may receive a non-list keep their own ``isinstance`` guard —
    this helper is deliberately as strict as the code it replaced.
    """
    return [item.get("brand", "") for item in items[:limit]]


def hierarchy_path(path: list) -> str:
    """카테고리 계층 경로를 ``a > b > c`` 문자열로."""
    return " > ".join(
        [node.get("name", node.get("id", "")) if isinstance(node, dict) else node for node in path]
    )


# ---------------------------------------------------------------------------
# Legacy HybridContext renderer
# ---------------------------------------------------------------------------


def _render_inferences(inferences: list, include_explanations: bool) -> list[str]:
    """온톨로지 추론 결과 (구조화된 인사이트)."""
    parts = ["## 분석 결과 (Ontology Reasoning)\n"]

    for i, inf in enumerate(inferences, 1):
        parts.append(f"### 인사이트 {i}: {inf.insight_type.value.replace('_', ' ').title()}")
        parts.append(f"- **결론**: {inf.insight}")

        if inf.recommendation:
            parts.append(f"- **권장 액션**: {inf.recommendation}")

        parts.append(f"- **신뢰도**: {inf.confidence:.0%}")

        if include_explanations and inf.evidence:
            conditions = inf.evidence.get("satisfied_conditions", [])
            if conditions:
                parts.append(f"- **근거 조건**: {', '.join(conditions)}")

        parts.append("")

    return parts


def _render_fact(fact: dict[str, Any]) -> list[str]:
    """지식 그래프 사실 1건 → 불릿 줄들."""
    fact_type = fact.get("type", "unknown")
    entity = fact.get("entity", "")
    data = fact.get("data", {})
    lines: list[str] = []

    if fact_type == "brand_info":
        sos = data.get("sos", 0)
        if sos:
            lines.append(f"- **{entity}** SoS: {as_percent(sos)}")
        if data.get("avg_rank"):
            lines.append(f"  - 평균 순위: {data['avg_rank']:.1f}")

    elif fact_type == "brand_products":
        lines.append(f"- **{entity}** 제품 수: {data.get('product_count', 0)}개")

    elif fact_type == "competitors":
        lines.append(f"- **{entity}** 주요 경쟁사: {', '.join(brand_names(data))}")

    elif fact_type == "category_brands":
        top_brands = brand_names(data.get("top_brands", []))
        lines.append(f"- **{entity}** Top 브랜드: {', '.join(top_brands)}")

    elif fact_type == "category_hierarchy":
        level = data.get("level", 0)
        path = data.get("path", [])
        ancestors = data.get("ancestors", [])
        name = data.get("name", entity)
        if path:
            lines.append(f"- **{name}** 계층: {hierarchy_path(path)} (Level {level})")
        if ancestors:
            parent_names = [a.get("name", "") for a in ancestors[:2]]
            lines.append(f"  - 상위 카테고리: {', '.join(parent_names)}")

    return lines


def _render_facts(facts: list[dict[str, Any]]) -> list[str]:
    """지식 그래프 사실 (관련 정보) — 상위 :data:`MAX_FACTS` 개."""
    parts = ["## 관련 정보 (Knowledge Graph)\n"]
    for fact in facts[:MAX_FACTS]:
        parts.extend(_render_fact(fact))
    parts.append("")
    return parts


def _render_chunks(chunks: list[dict[str, Any]]) -> list[str]:
    """RAG 가이드라인 (비구조화 문서) — 상위 :data:`MAX_CHUNKS` 개."""
    parts = ["## 참고 가이드라인 (RAG)\n"]
    for chunk in chunks[:MAX_CHUNKS]:
        title = chunk.get("metadata", {}).get("title", "")
        if title:
            parts.append(f"### {title}")
        parts.append(truncate(chunk.get("content", ""), LEGACY_CHUNK_CHARS))
        parts.append("")
    return parts


def combine_contexts(context: Any, include_explanations: bool = True) -> str:
    """온톨로지 + RAG 컨텍스트 통합.

    Args:
        context: HybridContext (duck-typed)
        include_explanations: 추론 설명 포함

    Returns:
        통합된 컨텍스트 문자열
    """
    parts: list[str] = []

    if context.inferences:
        parts.extend(_render_inferences(context.inferences, include_explanations))

    if context.ontology_facts:
        parts.extend(_render_facts(context.ontology_facts))

    if context.rag_chunks:
        parts.extend(_render_chunks(context.rag_chunks))

    return "\n".join(parts)

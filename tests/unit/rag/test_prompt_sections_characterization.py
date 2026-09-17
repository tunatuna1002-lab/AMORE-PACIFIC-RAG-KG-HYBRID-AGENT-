"""v4·v1 답변 프롬프트 컨텍스트의 섹션 구성 특성화 (트랙 2-B 변경 전 고정).

실제 HybridRetriever(임시 KG·운영 스키마 SQLite·규칙 추론기)와 가짜 문서 검색기로
v4 ``_combine_contexts``(=``retrieve``의 ``combined_context``)와 v1 ``ContextBuilder.build``의
섹션 머리글·핵심 내용을 고정한다.
"""

from src.rag.context_builder import ContextBuilder

from .evidence_pipeline_fixtures import (
    CURRENT_METRICS,
    QUERY,
    make_retriever,
    markdown_headers,
)


async def test_v4_combined_context_sections(tmp_path):
    retriever = make_retriever(tmp_path)

    ctx = await retriever.retrieve(QUERY, current_metrics=CURRENT_METRICS)
    text = ctx.combined_context

    assert markdown_headers(text) == [
        "## 분석 결과 (Ontology Reasoning)",
        "## 관련 정보 (Knowledge Graph)",
        "## 참고 가이드라인 (RAG)",
    ]
    # F8: DB 수치 사실은 모이지만(metric_facts) 렌더되지 않는다
    assert ctx.metric_facts
    assert "0.0681" not in text
    # E2 위반: KG 엔티티 메타데이터의 날짜 없는 SoS가 렌더된다
    assert "SoS: 42.4%" in text


async def test_v1_context_builder_sections(tmp_path):
    retriever = make_retriever(tmp_path)
    ctx = await retriever.retrieve(QUERY, current_metrics=CURRENT_METRICS)

    text = ContextBuilder(max_tokens=3000).build(
        ctx,
        current_metrics=CURRENT_METRICS,
        query="LANEIGE Lip Care 순위와 HHI",
        knowledge_graph=retriever.kg,
    )

    assert markdown_headers(text) == [
        "## 사용자 질문",
        "## 분석 결과 (Ontology Reasoning)",
        "## 카테고리 계층 구조",
        "## 크롤 DB 지표",
        "## 현재 데이터",
        "## 관련 정보 (Knowledge Graph)",
        "## 참고 가이드라인 (RAG)",
        "## 출처",
        "## 응답 가이드",
    ]
    assert "HHI 0.0681" in text  # v1만 DB 수치를 싣는다
    assert "점유율: 42.4%" in text  # KG 메타데이터 SoS (E2 위반)

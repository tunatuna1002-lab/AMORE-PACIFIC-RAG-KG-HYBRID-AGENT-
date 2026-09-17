"""v4·v1 답변 프롬프트 컨텍스트의 섹션 구성 특성화 (트랙 2-B).

실제 HybridRetriever(임시 KG·운영 스키마 SQLite·규칙 추론기)와 가짜 문서 검색기로
v4 ``_combine_contexts``(=``retrieve``의 ``combined_context``)와 v1 ``ContextBuilder.build``의
섹션 구성을 고정한다. 변경 전 구성은 커밋 8d2e84d의 이 파일에 있다.

섹션 매핑 (변경 전 → 변경 후)
- v4 "## 분석 결과 (Ontology Reasoning)" → [규칙 추론] 카드 (신뢰도·근거 조건 줄은 삭제:
  근거는 카드 derived_from으로 표시, 3-B가 채운다)
- v4 "## 관련 정보 (Knowledge Graph)" → [관계] 카드. brand_info의 SoS·평균 순위·제품 수는
  날짜 없는 KG 메타데이터라 삭제 (E2)
- v4 "## 참고 가이드라인 (RAG)" (상위 3개 × 500자) → [문서] 카드 (검색 top_k 전부 × 1500자)
- v4 (없음) → [DB 수치] 카드 (F8)
- v1 "## 분석 결과 (Ontology Reasoning)" → [규칙 추론] 카드
- v1 "## 카테고리 계층 구조" → [관계] 카드(parentCategory·hasSubcategory). KG 제품별 카테고리
  순위는 날짜 없는 수치라 삭제 (E2) — DB [DB 수치] BSR 순위 카드가 대신한다
- v1 "## 크롤 DB 지표" → [DB 수치] 카드
- v1 "## 관련 정보 (Knowledge Graph)" → [관계] 카드 (KG 메타데이터 수치 삭제, E2)
- v1 "## 참고 가이드라인 (RAG)" → [문서] 카드
- v1 "## 출처" ([N] 번호 부록) → 삭제: 카드 id가 인용 단위다
- v1 "## 현재 데이터" → 유지 (카드 밖 정보). current_metrics를 넘기는 일일 인사이트 경로만
  싣는다. 챗봇 답변 경로는 넘기지 않는다
- v1 "## 사용자 질문", "## 응답 가이드" → 유지 (카드 밖 정보)
"""

from src.rag.context_builder import ContextBuilder

from .evidence_pipeline_fixtures import (
    CURRENT_METRICS,
    QUERY,
    make_retriever,
    markdown_headers,
)

CARD_SECTIONS = ["[DB 수치]", "[관계]", "[규칙 추론]", "[문서]"]


def _card_sections(text: str) -> list[str]:
    return [line for line in text.splitlines() if line in CARD_SECTIONS]


async def test_v4_combined_context_sections(tmp_path):
    retriever = make_retriever(tmp_path)

    ctx = await retriever.retrieve(QUERY, current_metrics=CURRENT_METRICS)
    text = ctx.combined_context

    assert markdown_headers(text) == []
    assert _card_sections(text) == CARD_SECTIONS
    assert "lip_care HHI 0.0681" in text  # F8 해결
    assert "42.4" not in text  # KG 메타데이터 SoS 삭제 (E2)


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
        "## 증거 카드",
        "## 현재 데이터",
        "## 응답 가이드",
    ]
    assert _card_sections(text) == CARD_SECTIONS
    assert "lip_care HHI 0.0681" in text
    assert "42.4" not in text

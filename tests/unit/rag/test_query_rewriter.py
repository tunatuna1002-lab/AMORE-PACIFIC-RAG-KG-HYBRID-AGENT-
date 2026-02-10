"""
QueryRewriter 단위 테스트
========================
지시어 감지, 캐시, 후처리, 편의 함수 검증
(LLM 호출이 필요한 rewrite()는 mock 사용)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.query_rewriter import (
    QueryRewriter,
    RewriteResult,
    create_rewrite_result_no_change,
)

# ---------------------------------------------------------------------------
# RewriteResult dataclass
# ---------------------------------------------------------------------------


class TestRewriteResult:
    """RewriteResult 데이터클래스"""

    def test_fields(self):
        r = RewriteResult(
            original_query="그 제품 가격은?",
            rewritten_query="LANEIGE Lip Sleeping Mask 가격은?",
            was_rewritten=True,
            needs_clarification=False,
            clarification_message="",
            confidence=0.9,
            resolved_entities=["그 제품"],
        )
        assert r.was_rewritten is True
        assert r.confidence == 0.9


# ---------------------------------------------------------------------------
# needs_rewrite (regex-based, no LLM)
# ---------------------------------------------------------------------------


class TestNeedsRewrite:
    """지시어 감지 (LLM 없이)"""

    @pytest.fixture
    def rewriter(self):
        return QueryRewriter()

    # 한국어 지시어
    @pytest.mark.parametrize(
        "query",
        [
            "그것 가격은?",
            "그 제품 분석해줘",
            "이 브랜드 SoS는?",
            "해당 카테고리 순위",
        ],
    )
    def test_korean_demonstratives(self, rewriter, query):
        assert rewriter.needs_rewrite(query) is True

    # 영어 지시어
    @pytest.mark.parametrize(
        "query",
        [
            "What is it?",
            "Compare this with that",
            "The same product review",
        ],
    )
    def test_english_demonstratives(self, rewriter, query):
        assert rewriter.needs_rewrite(query) is True

    # 생략 패턴
    @pytest.mark.parametrize(
        "query",
        [
            "왜 떨어졌어?",
            "어떻게 계산해?",
            "비교해줘",
        ],
    )
    def test_omission_patterns(self, rewriter, query):
        assert rewriter.needs_rewrite(query) is True

    # 독립 질문 (재구성 불필요)
    @pytest.mark.parametrize(
        "query",
        [
            "LANEIGE Lip Care SoS 분석",
            "COSRX 순위 알려줘",
            "HHI 정의가 무엇인가요?",
        ],
    )
    def test_independent_queries(self, rewriter, query):
        assert rewriter.needs_rewrite(query) is False


# ---------------------------------------------------------------------------
# rewrite (LLM-based, mocked)
# ---------------------------------------------------------------------------


class TestRewrite:
    """LLM 기반 재구성 (mock)"""

    @pytest.fixture
    def rewriter(self):
        return QueryRewriter()

    @pytest.mark.asyncio
    async def test_no_history_returns_original(self, rewriter):
        result = await rewriter.rewrite("그 제품 가격은?", [])
        assert result.was_rewritten is False
        assert result.rewritten_query == "그 제품 가격은?"

    @pytest.mark.asyncio
    async def test_successful_rewrite(self, rewriter):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "LANEIGE Lip Sleeping Mask의 가격은?"

        with patch(
            "src.rag.query_rewriter.acompletion", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await rewriter.rewrite(
                query="그 제품 가격은?",
                conversation_history=[
                    {"role": "user", "content": "LANEIGE Lip Sleeping Mask 분석해줘"},
                    {"role": "assistant", "content": "LANEIGE Lip Sleeping Mask는..."},
                ],
            )
        assert result.was_rewritten is True
        assert "LANEIGE" in result.rewritten_query

    @pytest.mark.asyncio
    async def test_rewrite_caching(self, rewriter):
        """동일 쿼리+히스토리는 캐시에서 반환"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "LANEIGE 가격?"

        history = [{"role": "user", "content": "LANEIGE 분석"}]

        with patch(
            "src.rag.query_rewriter.acompletion", new_callable=AsyncMock, return_value=mock_response
        ) as mock_llm:
            await rewriter.rewrite("그 제품?", history)
            await rewriter.rewrite("그 제품?", history)
            assert mock_llm.call_count == 1  # 두 번째는 캐시

    @pytest.mark.asyncio
    async def test_rewrite_error_fallback(self, rewriter):
        """LLM 오류 시 원본 반환"""
        with patch(
            "src.rag.query_rewriter.acompletion",
            new_callable=AsyncMock,
            side_effect=Exception("API error"),
        ):
            result = await rewriter.rewrite(
                query="그 제품?",
                conversation_history=[{"role": "user", "content": "LANEIGE"}],
            )
        assert result.was_rewritten is False
        assert result.rewritten_query == "그 제품?"

    @pytest.mark.asyncio
    async def test_short_rewrite_needs_clarification(self, rewriter):
        """너무 짧은 재구성 → 명확화 요청"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "가격"  # 6자 미만

        with patch(
            "src.rag.query_rewriter.acompletion", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await rewriter.rewrite(
                query="그거?",
                conversation_history=[{"role": "user", "content": "LANEIGE"}],
            )
        assert result.needs_clarification is True


# ---------------------------------------------------------------------------
# _clean_response
# ---------------------------------------------------------------------------


class TestCleanResponse:
    """LLM 응답 후처리"""

    @pytest.fixture
    def rewriter(self):
        return QueryRewriter()

    def test_remove_prefix(self, rewriter):
        assert rewriter._clean_response("[재구성] LANEIGE 가격?") == "LANEIGE 가격?"

    def test_remove_result_prefix(self, rewriter):
        assert rewriter._clean_response("[재구성 결과] LANEIGE 가격?") == "LANEIGE 가격?"

    def test_remove_quotes(self, rewriter):
        assert rewriter._clean_response('"LANEIGE 가격?"') == "LANEIGE 가격?"

    def test_no_prefix_unchanged(self, rewriter):
        assert rewriter._clean_response("LANEIGE 가격?") == "LANEIGE 가격?"


# ---------------------------------------------------------------------------
# _extract_resolved
# ---------------------------------------------------------------------------


class TestExtractResolved:
    """해소된 지시어 추출"""

    @pytest.fixture
    def rewriter(self):
        return QueryRewriter()

    def test_extract_demonstrative(self, rewriter):
        resolved = rewriter._extract_resolved("그 제품 가격은?", "LANEIGE 가격은?")
        assert len(resolved) > 0

    def test_no_demonstratives(self, rewriter):
        resolved = rewriter._extract_resolved("LANEIGE 가격?", "LANEIGE 가격?")
        assert len(resolved) == 0


# ---------------------------------------------------------------------------
# clear_cache / create_rewrite_result_no_change
# ---------------------------------------------------------------------------


class TestUtilities:
    """유틸리티 함수"""

    def test_clear_cache(self):
        rewriter = QueryRewriter()
        rewriter._cache["key"] = "val"
        rewriter.clear_cache()
        assert len(rewriter._cache) == 0

    def test_create_no_change_result(self):
        result = create_rewrite_result_no_change("LANEIGE 분석")
        assert result.original_query == "LANEIGE 분석"
        assert result.rewritten_query == "LANEIGE 분석"
        assert result.was_rewritten is False
        assert result.confidence == 1.0

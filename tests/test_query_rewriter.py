"""
Query Rewriter 테스트
대화 맥락 기반 질문 재구성 모듈 테스트
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.rag.query_rewriter import QueryRewriter, RewriteResult, create_rewrite_result_no_change


@pytest.fixture
def rewriter():
    """QueryRewriter 인스턴스 fixture"""
    return QueryRewriter()


class TestNeedsRewrite:
    """needs_rewrite() 메서드 테스트 - LLM 호출 없이 지시어 감지"""

    def test_korean_demonstrative_standalone(self, rewriter):
        """한국어 지시어 단독 사용"""
        assert rewriter.needs_rewrite("그것 분석해줘") == True
        assert rewriter.needs_rewrite("이건 뭐야?") == True
        assert rewriter.needs_rewrite("그건 어때?") == True

    def test_korean_demonstrative_with_noun(self, rewriter):
        """한국어 지시어 + 명사"""
        assert rewriter.needs_rewrite("그 제품 가격은?") == True
        assert rewriter.needs_rewrite("해당 브랜드 분석") == True
        assert rewriter.needs_rewrite("이 카테고리 SoS는?") == True
        assert rewriter.needs_rewrite("해당 지표 해석해줘") == True

    def test_english_demonstrative(self, rewriter):
        """영어 지시어"""
        assert rewriter.needs_rewrite("What about it?") == True
        assert rewriter.needs_rewrite("Tell me about this") == True
        assert rewriter.needs_rewrite("How is that performing?") == True
        assert rewriter.needs_rewrite("the same product") == True

    def test_ellipsis_pattern(self, rewriter):
        """생략 패턴 (주어 없는 질문)"""
        assert rewriter.needs_rewrite("왜 떨어졌어?") == True
        assert rewriter.needs_rewrite("어떻게 개선해?") == True
        # "언제부터"는 패턴에 없음 - "언제 변했어?"로 테스트
        assert rewriter.needs_rewrite("언제 변했어?") == True
        assert rewriter.needs_rewrite("비교해줘") == True
        assert rewriter.needs_rewrite("분석해") == True

    def test_no_demonstrative(self, rewriter):
        """지시어 없는 일반 질문"""
        assert rewriter.needs_rewrite("LANEIGE SoS 분석해줘") == False
        assert rewriter.needs_rewrite("COSRX 가격 비교해줘") == False
        assert rewriter.needs_rewrite("Skin Care 카테고리 현황") == False
        assert rewriter.needs_rewrite("TIRTIR 브랜드 경쟁력") == False

    def test_mixed_content(self, rewriter):
        """혼합 컨텐츠"""
        # 지시어 포함
        assert rewriter.needs_rewrite("LANEIGE의 그 제품 가격은?") == True
        # 지시어 미포함
        assert rewriter.needs_rewrite("LANEIGE Lip Sleeping Mask 가격은?") == False


def create_mock_response(content: str):
    """Mock LLM 응답 생성 헬퍼"""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


class TestRewrite:
    """rewrite() 메서드 테스트 - LLM 기반 재구성 (Mock 사용)"""

    @pytest.mark.asyncio
    async def test_no_history(self, rewriter):
        """히스토리 없으면 원본 반환"""
        result = await rewriter.rewrite("그 제품 가격은?", [])

        assert result.original_query == "그 제품 가격은?"
        assert result.rewritten_query == "그 제품 가격은?"
        assert result.was_rewritten == False
        assert result.needs_clarification == False

    @pytest.mark.asyncio
    async def test_basic_rewrite(self, rewriter):
        """기본 재구성 테스트 (Mock)"""
        history = [
            {"role": "user", "content": "LANEIGE Lip Sleeping Mask 분석해줘"},
            {"role": "assistant", "content": "LANEIGE Lip Sleeping Mask는 Lip Care 카테고리에서..."}
        ]

        # Mock LLM 응답
        mock_response = create_mock_response("LANEIGE Lip Sleeping Mask의 가격은?")

        with patch('src.rag.query_rewriter.acompletion', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await rewriter.rewrite("그 제품 가격은?", history)

        assert result.original_query == "그 제품 가격은?"
        assert "LANEIGE" in result.rewritten_query
        assert "가격" in result.rewritten_query
        assert result.was_rewritten == True

    @pytest.mark.asyncio
    async def test_brand_comparison_rewrite(self, rewriter):
        """브랜드 비교 컨텍스트 재구성 (Mock)"""
        history = [
            {"role": "user", "content": "COSRX와 LANEIGE 비교해줘"},
            {"role": "assistant", "content": "두 브랜드를 비교하면, COSRX는..."}
        ]

        mock_response = create_mock_response("COSRX와 LANEIGE의 SoS 비교는?")

        with patch('src.rag.query_rewriter.acompletion', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await rewriter.rewrite("그럼 SoS는?", history)

        assert result.original_query == "그럼 SoS는?"
        assert "SoS" in result.rewritten_query

    @pytest.mark.asyncio
    async def test_category_context_rewrite(self, rewriter):
        """카테고리 컨텍스트 재구성 (Mock)"""
        history = [
            {"role": "user", "content": "Skin Care 카테고리 분석해줘"},
            {"role": "assistant", "content": "Skin Care 카테고리에서는 LANEIGE가..."}
        ]

        mock_response = create_mock_response("Skin Care 카테고리 SoS가 왜 떨어졌어?")

        with patch('src.rag.query_rewriter.acompletion', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await rewriter.rewrite("왜 떨어졌어?", history)

        assert result.original_query == "왜 떨어졌어?"
        assert "Skin Care" in result.rewritten_query

    @pytest.mark.asyncio
    async def test_caching(self, rewriter):
        """캐싱 테스트 (Mock)"""
        history = [
            {"role": "user", "content": "LANEIGE 분석해줘"},
            {"role": "assistant", "content": "LANEIGE는..."}
        ]

        mock_response = create_mock_response("LANEIGE의 경쟁사는?")

        with patch('src.rag.query_rewriter.acompletion', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            # 첫 번째 호출
            result1 = await rewriter.rewrite("그 브랜드 경쟁사는?", history)

            # 두 번째 호출 (캐시 히트 - LLM 호출 안 됨)
            result2 = await rewriter.rewrite("그 브랜드 경쟁사는?", history)

            # LLM은 한 번만 호출되어야 함
            assert mock_llm.call_count == 1

        # 동일한 결과
        assert result1.rewritten_query == result2.rewritten_query

    @pytest.mark.asyncio
    async def test_cache_clear(self, rewriter):
        """캐시 초기화 테스트"""
        history = [{"role": "user", "content": "테스트"}]

        mock_response = create_mock_response("테스트 재구성")

        with patch('src.rag.query_rewriter.acompletion', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            await rewriter.rewrite("그건?", history)

        assert len(rewriter._cache) > 0
        rewriter.clear_cache()
        assert len(rewriter._cache) == 0

    @pytest.mark.asyncio
    async def test_clarification_on_short_response(self, rewriter):
        """짧은 재구성 결과 시 명확화 요청"""
        history = [
            {"role": "user", "content": "안녕"},
            {"role": "assistant", "content": "안녕하세요"}
        ]

        # 6자 미만의 짧은 응답
        mock_response = create_mock_response("그건")

        with patch('src.rag.query_rewriter.acompletion', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await rewriter.rewrite("그건?", history)

        assert result.needs_clarification == True
        assert "구체적으로" in result.clarification_message

    @pytest.mark.asyncio
    async def test_llm_error_graceful_degradation(self, rewriter):
        """LLM 오류 시 원본 반환 (graceful degradation)"""
        history = [
            {"role": "user", "content": "LANEIGE 분석해줘"},
            {"role": "assistant", "content": "LANEIGE는..."}
        ]

        with patch('src.rag.query_rewriter.acompletion', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("API Error")
            result = await rewriter.rewrite("그 브랜드는?", history)

        # 오류 시 원본 반환
        assert result.rewritten_query == "그 브랜드는?"
        assert result.was_rewritten == False


class TestRewriteResult:
    """RewriteResult 데이터클래스 테스트"""

    def test_create_no_change(self):
        """create_rewrite_result_no_change 헬퍼 테스트"""
        result = create_rewrite_result_no_change("테스트 질문")

        assert result.original_query == "테스트 질문"
        assert result.rewritten_query == "테스트 질문"
        assert result.was_rewritten == False
        assert result.needs_clarification == False
        assert result.confidence == 1.0
        assert result.resolved_entities == []


class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_empty_query(self, rewriter):
        """빈 쿼리"""
        assert rewriter.needs_rewrite("") == False

    def test_very_short_query(self, rewriter):
        """매우 짧은 쿼리"""
        # 단독 "그"는 word boundary 패턴에 맞지 않음
        assert rewriter.needs_rewrite("그것") == True  # 지시어 감지
        assert rewriter.needs_rewrite("a") == False

    def test_long_query(self, rewriter):
        """긴 쿼리"""
        long_query = "LANEIGE Lip Sleeping Mask와 COSRX Lip Sleep Mask를 가격, 성분, 효과 측면에서 비교 분석해줘"
        assert rewriter.needs_rewrite(long_query) == False  # 지시어 없음

    @pytest.mark.asyncio
    async def test_history_truncation(self, rewriter):
        """긴 히스토리 잘림 테스트 (Mock)"""
        # 10턴의 히스토리
        history = []
        for i in range(10):
            history.append({"role": "user", "content": f"질문 {i}"})
            history.append({"role": "assistant", "content": f"응답 {i}" * 100})

        mock_response = create_mock_response("테스트 제품 분석")

        with patch('src.rag.query_rewriter.acompletion', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await rewriter.rewrite("그 제품은?", history)

        # 오류 없이 실행되어야 함
        assert result is not None

    @pytest.mark.asyncio
    async def test_special_characters(self, rewriter):
        """특수 문자 포함 쿼리 (Mock)"""
        history = [
            {"role": "user", "content": "LANEIGE's Lip Sleeping Mask (베리향) 분석해줘"},
            {"role": "assistant", "content": "LANEIGE's Lip Sleeping Mask는..."}
        ]

        mock_response = create_mock_response("LANEIGE's Lip Sleeping Mask (베리향)의 가격은?")

        with patch('src.rag.query_rewriter.acompletion', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await rewriter.rewrite("그 제품의 가격은?", history)

        assert result is not None


class TestIntegration:
    """통합 테스트"""

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, rewriter):
        """전체 대화 흐름 테스트 (Mock)"""
        # 1. 첫 질문 (재구성 불필요)
        assert rewriter.needs_rewrite("LANEIGE Lip Sleeping Mask 분석해줘") == False

        # 2. 히스토리 쌓임
        history = [
            {"role": "user", "content": "LANEIGE Lip Sleeping Mask 분석해줘"},
            {"role": "assistant", "content": "LANEIGE Lip Sleeping Mask는 Lip Care 카테고리에서 1위..."}
        ]

        # 3. 후속 질문 (재구성 필요)
        assert rewriter.needs_rewrite("그 제품 경쟁사는?") == True

        # 4. 재구성 실행 (Mock)
        mock_response1 = create_mock_response("LANEIGE Lip Sleeping Mask의 경쟁사는?")

        with patch('src.rag.query_rewriter.acompletion', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response1
            result = await rewriter.rewrite("그 제품 경쟁사는?", history)

        assert result.was_rewritten == True
        assert "LANEIGE" in result.rewritten_query or "Lip" in result.rewritten_query

        # 5. 히스토리 업데이트
        history.append({"role": "user", "content": "그 제품 경쟁사는?"})
        history.append({"role": "assistant", "content": "LANEIGE Lip Sleeping Mask의 경쟁 제품으로는..."})

        # 6. 또 다른 후속 질문 (Mock)
        mock_response2 = create_mock_response("LANEIGE Lip Sleeping Mask와 경쟁사의 가격 비교는?")

        with patch('src.rag.query_rewriter.acompletion', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response2
            result2 = await rewriter.rewrite("그럼 가격 비교는?", history)

        assert result2.was_rewritten == True

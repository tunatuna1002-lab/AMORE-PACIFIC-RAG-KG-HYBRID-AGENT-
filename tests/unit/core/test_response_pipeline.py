"""
ResponsePipeline 단위 테스트
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.models import (
    ConfidenceLevel,
    Context,
    Response,
    SystemState,
    ToolResult,
)
from src.core.response_pipeline import ResponsePipeline


@pytest.fixture
def pipeline():
    return ResponsePipeline(
        openai_client=MagicMock(),
        model="gpt-4o-mini",
        max_tokens=1500,
        temperature=0.3,
    )


@pytest.fixture
def mock_context():
    return Context(
        query="LANEIGE 순위 알려줘",
        entities={"brands": ["LANEIGE"], "categories": ["lip_care"]},
        rag_docs=[
            {
                "content": "LANEIGE Lip Sleeping Mask는 1위입니다",
                "metadata": {"title": "Lip Care Report"},
            }
        ],
        kg_facts=[],
        kg_inferences=[],
        system_state=SystemState(
            data_freshness="fresh",
            kg_initialized=True,
            kg_triple_count=50,
        ),
        summary="LANEIGE Lip Sleeping Mask가 Lip Care 카테고리에서 1위를 유지하고 있습니다.",
    )


class TestResponsePipeline:
    """ResponsePipeline 클래스 테스트"""

    def test_init(self, pipeline):
        assert pipeline.model == "gpt-4o-mini"
        assert pipeline.max_tokens == 1500
        assert pipeline.temperature == 0.3

    @pytest.mark.asyncio
    async def test_generate_with_llm(self, pipeline, mock_context):
        """LLM 응답 생성"""
        with patch.object(pipeline, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (
                "LANEIGE Lip Sleeping Mask는 현재 Lip Care 카테고리에서 1위입니다."
            )

            result = await pipeline.generate("LANEIGE 순위", mock_context)

        assert isinstance(result, Response)
        assert "1위" in result.text
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_generate_fallback_no_client(self, mock_context):
        """클라이언트 없을 때 폴백"""
        pipeline = ResponsePipeline(openai_client=None)

        with patch.dict("os.environ", {}, clear=True):
            result = await pipeline.generate("테스트 질문", mock_context)

        assert isinstance(result, Response)
        assert result.text  # 빈 문자열 아님

    @pytest.mark.asyncio
    async def test_generate_error_handling(self, pipeline, mock_context):
        """오류 발생 시 fallback 응답"""
        with patch.object(pipeline, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("API 오류")

            result = await pipeline.generate("테스트", mock_context)

        assert isinstance(result, Response)
        assert result.is_fallback or "오류" in result.text

    def test_post_process_empty(self, pipeline, mock_context):
        """빈 응답 후처리"""
        result = pipeline._post_process("", mock_context)
        assert "응답을 생성할 수 없습니다" in result

    def test_post_process_prefix_removal(self, pipeline, mock_context):
        """prefix 제거"""
        result = pipeline._post_process("답변: 테스트 응답입니다", mock_context)
        assert result == "테스트 응답입니다"

    def test_generate_suggestions_with_brands(self, pipeline, mock_context):
        """브랜드 기반 제안 생성"""
        suggestions = pipeline._generate_suggestions("LANEIGE", mock_context)
        assert len(suggestions) > 0
        assert len(suggestions) <= 3

    def test_extract_sources(self, pipeline, mock_context):
        """출처 추출"""
        sources = pipeline._extract_sources(mock_context)
        assert "Lip Care Report" in sources

    def test_infer_query_type(self, pipeline, mock_context):
        """질문 유형 추론"""
        assert pipeline._infer_query_type("SoS가 뭐야?", mock_context) == "definition"
        assert pipeline._infer_query_type("현재 순위 알려줘", mock_context) == "data_query"
        assert pipeline._infer_query_type("경쟁사 분석해줘", mock_context) == "analysis"
        assert pipeline._infer_query_type("크롤링 해줘", mock_context) == "action"
        assert pipeline._infer_query_type("안녕하세요", mock_context) == "general"

    def test_assess_confidence(self, pipeline, mock_context):
        """신뢰도 평가"""
        level = pipeline._assess_confidence(mock_context)
        assert isinstance(level, ConfidenceLevel)

    def test_calculate_confidence_score(self, pipeline, mock_context):
        """신뢰도 점수 계산"""
        score = pipeline._calculate_confidence_score(mock_context)
        assert 0 <= score <= 10.0

    def test_build_messages(self, pipeline, mock_context):
        """메시지 구성"""
        messages = pipeline._build_messages("테스트 질문", mock_context)
        assert len(messages) >= 3  # system + context + user
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "테스트 질문"

    def test_format_tool_result_crawl(self, pipeline):
        """크롤링 결과 포맷"""
        result = ToolResult(
            tool_name="crawl_amazon",
            success=True,
            data={"total_products": 100, "laneige_count": 5},
        )
        formatted = pipeline._format_tool_result(result)
        assert "100" in formatted
        assert "5" in formatted

    def test_format_tool_result_failed(self, pipeline):
        """실패 결과 포맷"""
        result = ToolResult(
            tool_name="query_data",
            success=False,
            error="DB 연결 실패",
        )
        formatted = pipeline._format_tool_result(result)
        assert "실패" in formatted

    def test_set_client(self, pipeline):
        """클라이언트 설정"""
        new_client = MagicMock()
        pipeline.set_client(new_client)
        assert pipeline.client == new_client

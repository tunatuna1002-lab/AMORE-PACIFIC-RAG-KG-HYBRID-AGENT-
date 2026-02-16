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


class TestSetTracer:
    """set_tracer 테스트"""

    def test_set_tracer(self, pipeline):
        """Tracer 설정"""
        mock_tracer = MagicMock()
        pipeline.set_tracer(mock_tracer)
        assert pipeline._tracer == mock_tracer


class TestGenerateHighConfidence:
    """HIGH confidence fast path 테스트"""

    @pytest.mark.asyncio
    async def test_generate_high_confidence_fast_path(self, pipeline, mock_context):
        """HIGH confidence일 때 _call_llm_fast 호출"""
        from src.core.models import Decision

        decision = Decision(
            tool="direct_answer",
            confidence=0.9,
            reason="HIGH confidence based on strong context",
        )

        with patch.object(pipeline, "_call_llm_fast", new_callable=AsyncMock) as mock_fast:
            mock_fast.return_value = "LANEIGE는 현재 1위입니다."

            result = await pipeline.generate("LANEIGE 순위", mock_context, decision=decision)

        mock_fast.assert_called_once()
        assert isinstance(result, Response)
        assert "1위" in result.text


class TestGenerateWithToolResult:
    """tool_result 포함 generate 테스트"""

    @pytest.mark.asyncio
    async def test_generate_with_tool_result_includes_tool_name(self, pipeline, mock_context):
        """tool_result가 있으면 tools_called에 포함"""
        tool_result = ToolResult(
            tool_name="calculate_metrics",
            success=True,
            data={"brand_metrics": [], "product_metrics": [], "alerts": []},
        )

        with patch.object(pipeline, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "지표 계산이 완료되었습니다."

            result = await pipeline.generate("지표 계산해줘", mock_context, tool_result=tool_result)

        assert "calculate_metrics" in result.tools_called

    @pytest.mark.asyncio
    async def test_generate_with_tool_result_creates_enhanced_context(self, pipeline, mock_context):
        """generate_with_tool_result가 도구 요약을 컨텍스트에 추가"""
        tool_result = ToolResult(
            tool_name="crawl_amazon",
            success=True,
            data={"total_products": 100, "laneige_count": 5},
        )

        with patch.object(pipeline, "generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = Response(text="크롤링 완료")

            await pipeline.generate_with_tool_result("크롤링 해줘", mock_context, tool_result)

        call_args = mock_gen.call_args
        enhanced_context = call_args[0][1]
        assert "[도구 실행 결과]" in enhanced_context.summary


class TestGenerateHallucinationCheck:
    """환각 감지 테스트"""

    @pytest.mark.asyncio
    async def test_hallucination_check_called_for_low_confidence(self, pipeline, mock_context):
        """confidence < 0.8이면 hallucination detector 호출"""
        from src.core.models import Decision

        decision = Decision(tool="direct_answer", confidence=0.5)

        with patch.object(pipeline, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "응답입니다."

            with patch.object(
                pipeline._hallucination_detector, "check", new_callable=AsyncMock
            ) as mock_check:
                mock_check.return_value = MagicMock(is_grounded=True, score=0.9)

                result = await pipeline.generate("질문", mock_context, decision=decision)

        mock_check.assert_called_once()
        assert isinstance(result, Response)

    @pytest.mark.asyncio
    async def test_hallucination_check_exception_graceful(self, pipeline, mock_context):
        """환각 감지 예외 발생 시 graceful 처리"""
        from src.core.models import Decision

        decision = Decision(tool="direct_answer", confidence=0.5)

        with patch.object(pipeline, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "응답입니다."

            with patch.object(
                pipeline._hallucination_detector, "check", new_callable=AsyncMock
            ) as mock_check:
                mock_check.side_effect = Exception("Hallucination check failed")

                result = await pipeline.generate("질문", mock_context, decision=decision)

        assert isinstance(result, Response)


class TestGenerateConfidenceMerge:
    """신뢰도 병합 테스트"""

    @pytest.mark.asyncio
    async def test_confidence_merge_with_decision(self, pipeline, mock_context):
        """decision.confidence가 있으면 max(calculated, decision.confidence)"""
        from src.core.models import Decision

        decision = Decision(confidence=8.0)

        with patch.object(pipeline, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "응답입니다."

            result = await pipeline.generate("질문", mock_context, decision=decision)

        # decision.confidence(8.0)가 calculated보다 크면 8.0
        assert result.confidence_score >= 8.0

    @pytest.mark.asyncio
    async def test_confidence_without_decision(self, pipeline, mock_context):
        """decision 없으면 calculated confidence만 사용"""
        with patch.object(pipeline, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "응답입니다."

            result = await pipeline.generate("질문", mock_context)

        # calculated confidence만 사용
        assert result.confidence_score > 0


class TestBuildMessagesWithDecision:
    """_build_messages decision 처리 테스트"""

    def test_build_messages_with_key_points(self, pipeline, mock_context):
        """decision.key_points가 있으면 메시지 추가"""
        from src.core.models import Decision

        decision = Decision(
            tool="direct_answer",
            key_points=["포인트 1", "포인트 2"],
        )

        messages = pipeline._build_messages("질문", mock_context, decision=decision)

        key_points_msg = next(
            (m for m in messages if "[응답 핵심 포인트]" in m.get("content", "")), None
        )
        assert key_points_msg is not None
        assert "포인트 1" in key_points_msg["content"]

    def test_build_messages_without_key_points(self, pipeline, mock_context):
        """decision.key_points 없으면 메시지 추가 안 함"""
        from src.core.models import Decision

        decision = Decision(tool="direct_answer", key_points=[])

        messages = pipeline._build_messages("질문", mock_context, decision=decision)

        key_points_msg = next(
            (m for m in messages if "[응답 핵심 포인트]" in m.get("content", "")), None
        )
        assert key_points_msg is None


class TestBuildMessagesWithToolResult:
    """_build_messages tool_result 처리 테스트"""

    def test_build_messages_with_successful_tool_result(self, pipeline, mock_context):
        """성공한 tool_result는 메시지 추가"""
        tool_result = ToolResult(
            tool_name="crawl_amazon",
            success=True,
            data={"total_products": 100},
        )

        messages = pipeline._build_messages("질문", mock_context, tool_result=tool_result)

        tool_msg = next((m for m in messages if "[도구 실행 결과]" in m.get("content", "")), None)
        assert tool_msg is not None

    def test_build_messages_with_failed_tool_result(self, pipeline, mock_context):
        """실패한 tool_result는 메시지 추가 안 함"""
        tool_result = ToolResult(
            tool_name="crawl_amazon",
            success=False,
            error="크롤링 실패",
        )

        messages = pipeline._build_messages("질문", mock_context, tool_result=tool_result)

        tool_msg = next((m for m in messages if "[도구 실행 결과]" in m.get("content", "")), None)
        assert tool_msg is None


class TestFormatContextWithoutSummary:
    """_format_context (summary 없을 때) 테스트"""

    def test_format_context_with_system_state(self, pipeline):
        """system_state만 있으면 데이터 상태 표시"""
        context = Context(
            query="질문",
            system_state=SystemState(
                data_freshness="fresh",
                kg_initialized=True,
                kg_triple_count=50,
            ),
            summary="",
        )

        result = pipeline._format_context(context)

        assert "데이터 상태: fresh" in result
        assert "KG: 50 트리플" in result

    def test_format_context_with_kg_inferences(self, pipeline):
        """kg_inferences 있으면 인사이트 표시"""
        context = Context(
            query="질문",
            kg_inferences=[
                {"insight": "인사이트 1"},
                {"insight": "인사이트 2"},
            ],
            summary="",
        )

        result = pipeline._format_context(context)

        assert "인사이트:" in result
        assert "인사이트 1" in result

    def test_format_context_with_kg_facts(self, pipeline):
        """kg_facts 있으면 관련 정보 표시"""
        from src.domain.entities.brain_models import KGFact

        context = Context(
            query="질문",
            kg_facts=[
                KGFact(fact_type="brand_info", entity="LANEIGE", data={}),
            ],
            summary="",
        )

        result = pipeline._format_context(context)

        assert "관련 정보:" in result
        assert "brand_info" in result
        assert "LANEIGE" in result

    def test_format_context_empty(self, pipeline):
        """빈 컨텍스트는 '컨텍스트 없음'"""
        context = Context(query="질문", summary="")

        result = pipeline._format_context(context)

        assert result == "컨텍스트 없음"


class TestFormatToolResultExtended:
    """_format_tool_result 확장 테스트"""

    def test_format_tool_result_calculate_metrics(self, pipeline):
        """calculate_metrics 결과 포맷"""
        result = ToolResult(
            tool_name="calculate_metrics",
            success=True,
            data={
                "brand_metrics": [1, 2, 3],
                "product_metrics": [1, 2],
                "alerts": [1],
            },
        )

        formatted = pipeline._format_tool_result(result)

        assert "3개 브랜드" in formatted
        assert "2개 제품" in formatted
        assert "1개 알림" in formatted

    def test_format_tool_result_query_data(self, pipeline):
        """query_data 결과는 JSON으로 (500자 제한)"""
        result = ToolResult(
            tool_name="query_data",
            success=True,
            data={"key": "value" * 200},  # 긴 데이터
        )

        formatted = pipeline._format_tool_result(result)

        assert len(formatted) <= 500

    def test_format_tool_result_unknown_tool(self, pipeline):
        """알 수 없는 도구는 to_summary() 호출"""
        result = ToolResult(
            tool_name="unknown_tool",
            success=True,
            data={"result": "success"},
        )

        formatted = pipeline._format_tool_result(result)

        # to_summary() 메서드가 없으므로 기본 동작
        assert isinstance(formatted, str)


class TestCallLlm:
    """_call_llm 테스트"""

    @pytest.mark.asyncio
    async def test_call_llm_success(self, pipeline):
        """LLM 호출 성공"""
        messages = [
            {"role": "system", "content": "시스템"},
            {"role": "user", "content": "질문"},
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "응답입니다"
            mock_acompletion.return_value = mock_response

            result = await pipeline._call_llm(messages)

        assert result == "응답입니다"

    @pytest.mark.asyncio
    async def test_call_llm_with_tracer(self, pipeline):
        """tracer가 있으면 llm_span 사용"""
        messages = [{"role": "user", "content": "질문"}]

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_span.attributes = {}
        mock_tracer.llm_span.return_value = mock_span
        mock_tracer.get_current_trace_id.return_value = "trace-123"

        pipeline.set_tracer(mock_tracer)

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "응답"
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
            mock_acompletion.return_value = mock_response

            result = await pipeline._call_llm(messages)

        mock_tracer.llm_span.assert_called_once()
        assert result == "응답"


class TestCallLlmFast:
    """_call_llm_fast 테스트"""

    @pytest.mark.asyncio
    async def test_call_llm_fast_success(self, pipeline, mock_context):
        """빠른 LLM 호출 성공"""
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "빠른 응답"
            mock_acompletion.return_value = mock_response

            result = await pipeline._call_llm_fast("질문", mock_context)

        assert result == "빠른 응답"

    @pytest.mark.asyncio
    async def test_call_llm_fast_failure_fallback(self, pipeline, mock_context):
        """빠른 호출 실패 시 fallback"""
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.side_effect = Exception("API 오류")

            result = await pipeline._call_llm_fast("질문", mock_context)

        # fallback 응답 생성
        assert isinstance(result, str)
        assert len(result) > 0


class TestGenerateFallbackResponseExtended:
    """_generate_fallback_response 확장 테스트"""

    def test_fallback_with_rag_docs(self, pipeline):
        """RAG 문서가 있으면 문서 내용 반환"""
        context = Context(
            query="질문",
            rag_docs=[
                {"content": "문서 내용 1"},
                {"content": "문서 내용 2"},
            ],
        )

        result = pipeline._generate_fallback_response("질문", context)

        assert "문서 내용 1" in result

    def test_fallback_with_kg_inferences(self, pipeline):
        """KG 추론이 있으면 인사이트 표시"""
        context = Context(
            query="질문",
            kg_inferences=[
                {"insight": "인사이트 1", "recommendation": "추천 1"},
                {"insight": "인사이트 2"},
            ],
        )

        result = pipeline._generate_fallback_response("질문", context)

        assert "분석 인사이트:" in result
        assert "인사이트 1" in result
        assert "추천 1" in result

    def test_fallback_with_kg_facts(self, pipeline):
        """KG 사실이 있으면 관련 정보 표시"""
        from src.domain.entities.brain_models import KGFact

        context = Context(
            query="질문",
            kg_facts=[
                KGFact(fact_type="brand_info", entity="LANEIGE", data={}),
            ],
        )

        result = pipeline._generate_fallback_response("질문", context)

        assert "관련 정보:" in result
        assert "LANEIGE" in result

    def test_fallback_empty_context(self, pipeline):
        """빈 컨텍스트는 기본 메시지"""
        context = Context(query="질문")

        result = pipeline._generate_fallback_response("질문", context)

        assert "추가 데이터가 필요합니다" in result


class TestGenerateSuggestionsExtended:
    """_generate_suggestions 확장 테스트"""

    def test_suggestions_with_kg_not_initialized(self, pipeline):
        """KG 미초기화 시 초기화 제안"""
        context = Context(
            query="질문",
            system_state=SystemState(kg_initialized=False),
        )

        suggestions = pipeline._generate_suggestions("질문", context)

        assert "지식 그래프 초기화해줘" in suggestions

    def test_suggestions_with_stale_data(self, pipeline):
        """데이터가 fresh 아니면 크롤링 제안"""
        context = Context(
            query="질문",
            system_state=SystemState(data_freshness="stale"),
        )

        suggestions = pipeline._generate_suggestions("질문", context)

        assert "최신 데이터 크롤링해줘" in suggestions

    def test_suggestions_no_brands_no_issues(self, pipeline):
        """브랜드 없고 시스템 이슈 없으면 기본 제안"""
        context = Context(query="질문")

        suggestions = pipeline._generate_suggestions("질문", context)

        assert "라네즈 현재 순위 알려줘" in suggestions

    def test_suggestions_limited_to_three(self, pipeline):
        """제안은 최대 3개"""
        context = Context(
            query="질문",
            entities={"brands": ["LANEIGE"]},
        )

        suggestions = pipeline._generate_suggestions("질문", context)

        assert len(suggestions) <= 3


class TestExtractSourcesExtended:
    """_extract_sources 확장 테스트"""

    def test_extract_sources_with_kg_facts(self, pipeline):
        """KG facts 있으면 'Knowledge Graph' 포함"""
        from src.domain.entities.brain_models import KGFact

        context = Context(
            query="질문",
            kg_facts=[
                KGFact(fact_type="brand_info", entity="LANEIGE", data={}),
            ],
        )

        sources = pipeline._extract_sources(context)

        assert "Knowledge Graph" in sources

    def test_extract_sources_with_kg_inferences(self, pipeline):
        """KG 추론 있으면 'Ontology Reasoning' 포함"""
        context = Context(
            query="질문",
            kg_inferences=[{"insight": "인사이트"}],
        )

        sources = pipeline._extract_sources(context)

        assert "Ontology Reasoning" in sources

    def test_extract_sources_deduplication(self, pipeline):
        """중복 제목 제거"""
        context = Context(
            query="질문",
            rag_docs=[
                {"metadata": {"title": "Report A"}},
                {"metadata": {"title": "Report A"}},  # 중복
                {"metadata": {"title": "Report B"}},
            ],
        )

        sources = pipeline._extract_sources(context)

        # 중복 제거되어 2개만
        assert sources.count("Report A") == 1
        assert "Report B" in sources

    def test_extract_sources_max_five(self, pipeline):
        """최대 5개 출처"""
        context = Context(
            query="질문",
            rag_docs=[{"metadata": {"title": f"Report {i}"}} for i in range(10)],
        )

        sources = pipeline._extract_sources(context)

        assert len(sources) <= 5


class TestAssessConfidenceLevels:
    """_assess_confidence 레벨 테스트"""

    def test_assess_confidence_high(self, pipeline):
        """score >= 5.0 → HIGH"""
        from src.domain.entities.brain_models import KGFact

        context = Context(
            query="질문",
            rag_docs=[{"content": "doc"} for _ in range(3)],
            kg_facts=[KGFact(fact_type="brand_info", entity="LANEIGE", data={})],
            system_state=SystemState(data_freshness="fresh", kg_initialized=True),
        )

        level = pipeline._assess_confidence(context)

        assert level == ConfidenceLevel.HIGH

    def test_assess_confidence_medium(self, pipeline):
        """score >= 3.0 → MEDIUM"""
        context = Context(
            query="질문",
            rag_docs=[{"content": "doc"} for _ in range(2)],
            system_state=SystemState(data_freshness="fresh"),
        )

        level = pipeline._assess_confidence(context)

        assert level == ConfidenceLevel.MEDIUM

    def test_assess_confidence_low(self, pipeline):
        """score >= 1.5 → LOW"""
        context = Context(
            query="질문",
            rag_docs=[{"content": "doc"}],
            system_state=SystemState(kg_initialized=True),
        )

        level = pipeline._assess_confidence(context)

        assert level == ConfidenceLevel.LOW

    def test_assess_confidence_unknown(self, pipeline):
        """score < 1.5 → UNKNOWN"""
        context = Context(query="질문")

        level = pipeline._assess_confidence(context)

        assert level == ConfidenceLevel.UNKNOWN


class TestCalculateConfidenceScoreDetailed:
    """_calculate_confidence_score 상세 테스트"""

    def test_score_rag_docs_contribution(self, pipeline):
        """RAG 문서는 min(count, 3) * 1.0"""
        context = Context(
            query="질문",
            rag_docs=[{"content": "doc"} for _ in range(5)],
        )

        score = pipeline._calculate_confidence_score(context)

        # 5개 문서 but 최대 3개까지만 카운트 = 3.0
        assert score == 3.0

    def test_score_kg_facts_contribution(self, pipeline):
        """KG facts는 min(count, 3) * 1.0"""
        from src.domain.entities.brain_models import KGFact

        context = Context(
            query="질문",
            kg_facts=[
                KGFact(fact_type="brand_info", entity=f"Brand{i}", data={}) for i in range(4)
            ],
        )

        score = pipeline._calculate_confidence_score(context)

        # 4개 facts but 최대 3개까지만 = 3.0
        assert score == 3.0

    def test_score_kg_inferences_contribution(self, pipeline):
        """KG 추론은 min(count, 2) * 1.5"""
        context = Context(
            query="질문",
            kg_inferences=[{"insight": f"insight{i}"} for i in range(3)],
        )

        score = pipeline._calculate_confidence_score(context)

        # 3개 inferences but 최대 2개까지만 = 2 * 1.5 = 3.0
        assert score == 3.0

    def test_score_fresh_data_contribution(self, pipeline):
        """fresh 데이터는 1.0 추가"""
        context = Context(
            query="질문",
            system_state=SystemState(data_freshness="fresh"),
        )

        score = pipeline._calculate_confidence_score(context)

        assert score == 1.0

    def test_score_kg_initialized_contribution(self, pipeline):
        """KG 초기화는 0.5 추가"""
        context = Context(
            query="질문",
            system_state=SystemState(kg_initialized=True),
        )

        score = pipeline._calculate_confidence_score(context)

        assert score == 0.5

    def test_score_capped_at_ten(self, pipeline):
        """최대 점수는 10.0"""
        from src.domain.entities.brain_models import KGFact

        context = Context(
            query="질문",
            rag_docs=[{"content": "doc"} for _ in range(10)],
            kg_facts=[
                KGFact(fact_type="brand_info", entity=f"Brand{i}", data={}) for i in range(10)
            ],
            kg_inferences=[{"insight": f"insight{i}"} for i in range(10)],
            system_state=SystemState(data_freshness="fresh", kg_initialized=True),
        )

        score = pipeline._calculate_confidence_score(context)

        assert score == 10.0


class TestPostProcessExtended:
    """_post_process 확장 테스트"""

    def test_post_process_removes_response_prefix(self, pipeline, mock_context):
        """'응답:' prefix 제거"""
        result = pipeline._post_process("응답: 테스트 내용입니다", mock_context)

        assert result == "테스트 내용입니다"

    def test_post_process_removes_analysis_prefix(self, pipeline, mock_context):
        """'분석 결과:' prefix 제거"""
        result = pipeline._post_process("분석 결과: 분석 내용입니다", mock_context)

        assert result == "분석 내용입니다"

    def test_post_process_whitespace_only_fallback(self, pipeline, mock_context):
        """공백만 있으면 fallback 메시지"""
        result = pipeline._post_process("   \n  \t  ", mock_context)

        assert "응답을 생성할 수 없습니다" in result


class TestInferQueryTypeInterpretation:
    """_infer_query_type interpretation 테스트"""

    def test_infer_query_type_interpretation(self, pipeline, mock_context):
        """'해석해줘' → interpretation"""
        query_type = pipeline._infer_query_type("SoS 해석해줘", mock_context)

        assert query_type == "interpretation"

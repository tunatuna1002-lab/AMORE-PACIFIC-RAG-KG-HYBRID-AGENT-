"""
RelevanceGrader 단위 테스트
===========================
관련성 판정, 임계값 처리, LLM 응답 파싱, 배치 판정 검증
(LLM 호출: mock)
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.relevance_grader import RelevanceGrader

# ---------------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------------


class TestRelevanceGraderInit:
    """RelevanceGrader 초기화 테스트"""

    def test_default_init(self):
        """기본 초기화"""
        grader = RelevanceGrader()
        assert grader.model == "gpt-4.1-mini"
        assert grader.temperature == 0.0
        assert grader.max_docs_per_call == 5
        assert grader.min_relevant_threshold == 2

    def test_custom_init(self):
        """커스텀 파라미터 초기화"""
        grader = RelevanceGrader(
            model="gpt-4o",
            temperature=0.1,
            max_docs_per_call=10,
            min_relevant_threshold=3,
        )
        assert grader.model == "gpt-4o"
        assert grader.temperature == 0.1
        assert grader.max_docs_per_call == 10
        assert grader.min_relevant_threshold == 3

    def test_stats_initialized(self):
        """통계 초기값"""
        grader = RelevanceGrader()
        assert grader._stats["total_graded"] == 0
        assert grader._stats["relevant"] == 0
        assert grader._stats["irrelevant"] == 0


# ---------------------------------------------------------------------------
# _parse_grades
# ---------------------------------------------------------------------------


class TestParseGrades:
    """LLM 응답 파싱 테스트"""

    def test_valid_json_response(self):
        """정상 JSON 응답 파싱"""
        grader = RelevanceGrader()
        response = json.dumps(
            {
                "grades": [
                    {"doc_index": 0, "relevant": True, "reason": "관련 있음"},
                    {"doc_index": 1, "relevant": False, "reason": "무관"},
                ]
            }
        )
        grades = grader._parse_grades(response)
        assert grades[0] is True
        assert grades[1] is False

    def test_json_with_surrounding_text(self):
        """JSON 전후에 텍스트가 있는 경우"""
        grader = RelevanceGrader()
        response = 'Here is the result:\n{"grades": [{"doc_index": 0, "relevant": true}]}\nDone.'
        grades = grader._parse_grades(response)
        assert grades[0] is True

    def test_invalid_json_returns_empty(self):
        """파싱 불가 응답은 빈 딕셔너리"""
        grader = RelevanceGrader()
        grades = grader._parse_grades("This is not JSON at all")
        assert grades == {}

    def test_empty_grades_array(self):
        """빈 grades 배열"""
        grader = RelevanceGrader()
        response = json.dumps({"grades": []})
        grades = grader._parse_grades(response)
        assert grades == {}

    def test_missing_grades_key(self):
        """grades 키가 없는 JSON"""
        grader = RelevanceGrader()
        response = json.dumps({"results": [{"doc_index": 0, "relevant": True}]})
        grades = grader._parse_grades(response)
        assert grades == {}

    def test_malformed_json(self):
        """불완전한 JSON"""
        grader = RelevanceGrader()
        response = '{"grades": [{"doc_index": 0, "relevant": true'
        grades = grader._parse_grades(response)
        assert grades == {}

    def test_default_relevant_true(self):
        """relevant 키 누락 시 기본값 True"""
        grader = RelevanceGrader()
        response = json.dumps({"grades": [{"doc_index": 0}]})
        grades = grader._parse_grades(response)
        assert grades.get(0) is True

    def test_multiple_documents(self):
        """다중 문서 파싱"""
        grader = RelevanceGrader()
        response = json.dumps(
            {
                "grades": [
                    {"doc_index": 0, "relevant": True},
                    {"doc_index": 1, "relevant": False},
                    {"doc_index": 2, "relevant": True},
                    {"doc_index": 3, "relevant": False},
                ]
            }
        )
        grades = grader._parse_grades(response)
        assert len(grades) == 4
        assert grades[0] is True
        assert grades[1] is False
        assert grades[2] is True
        assert grades[3] is False


# ---------------------------------------------------------------------------
# needs_rewrite
# ---------------------------------------------------------------------------


class TestNeedsRewrite:
    """needs_rewrite 메서드 테스트"""

    def test_below_threshold_needs_rewrite(self):
        """임계값 미만이면 재작성 필요"""
        grader = RelevanceGrader(min_relevant_threshold=2)
        assert grader.needs_rewrite(1) is True
        assert grader.needs_rewrite(0) is True

    def test_at_threshold_no_rewrite(self):
        """임계값 이상이면 재작성 불필요"""
        grader = RelevanceGrader(min_relevant_threshold=2)
        assert grader.needs_rewrite(2) is False
        assert grader.needs_rewrite(5) is False

    def test_custom_threshold(self):
        """커스텀 임계값"""
        grader = RelevanceGrader(min_relevant_threshold=5)
        assert grader.needs_rewrite(4) is True
        assert grader.needs_rewrite(5) is False


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    """get_stats 메서드 테스트"""

    def test_stats_initial(self):
        """초기 통계"""
        grader = RelevanceGrader()
        stats = grader.get_stats()
        assert stats["total_graded"] == 0
        assert stats["relevant"] == 0
        assert stats["irrelevant"] == 0
        assert stats["model"] == "gpt-4.1-mini"

    def test_stats_includes_model(self):
        """통계에 모델명 포함"""
        grader = RelevanceGrader(model="gpt-4o")
        stats = grader.get_stats()
        assert stats["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# grade_documents
# ---------------------------------------------------------------------------


class TestGradeDocuments:
    """grade_documents 비동기 메서드 테스트"""

    @pytest.mark.asyncio
    async def test_empty_documents(self):
        """빈 문서 리스트"""
        grader = RelevanceGrader()
        relevant, irrelevant = await grader.grade_documents("test", [])
        assert relevant == []
        assert irrelevant == []

    @pytest.mark.asyncio
    async def test_single_document_auto_relevant(self):
        """단일 문서는 LLM 호출 없이 관련으로 간주"""
        grader = RelevanceGrader()
        docs = [{"content": "LANEIGE Lip Sleeping Mask 분석"}]
        relevant, irrelevant = await grader.grade_documents("LANEIGE", docs)
        assert len(relevant) == 1
        assert len(irrelevant) == 0
        assert grader._stats["total_graded"] == 1
        assert grader._stats["relevant"] == 1

    @pytest.mark.asyncio
    async def test_grading_with_llm_mock(self):
        """LLM 판정 mock 테스트"""
        grader = RelevanceGrader()
        docs = [
            {"content": "LANEIGE is a top brand"},
            {"content": "Weather forecast for tomorrow"},
            {"content": "COSRX market analysis"},
        ]
        # Mock the LLM grading
        grader._call_llm_grading = AsyncMock(return_value={0: True, 1: False, 2: True})
        relevant, irrelevant = await grader.grade_documents("beauty brands", docs)
        assert len(relevant) == 2
        assert len(irrelevant) == 1
        assert grader._stats["relevant"] == 2
        assert grader._stats["irrelevant"] == 1

    @pytest.mark.asyncio
    async def test_grading_default_true_on_missing_grade(self):
        """grade 누락 시 기본값 True"""
        grader = RelevanceGrader()
        docs = [
            {"content": "Doc A"},
            {"content": "Doc B"},
        ]
        # doc_index=0만 반환, 1은 기본값 True 적용
        grader._call_llm_grading = AsyncMock(return_value={0: False})
        relevant, irrelevant = await grader.grade_documents("test", docs)
        assert len(relevant) == 1  # doc B (default True)
        assert len(irrelevant) == 1  # doc A (explicitly False)

    @pytest.mark.asyncio
    async def test_grading_exceeds_max_docs(self):
        """max_docs_per_call 초과 문서는 관련으로 간주"""
        grader = RelevanceGrader(max_docs_per_call=2)
        docs = [
            {"content": "Doc A"},
            {"content": "Doc B"},
            {"content": "Doc C (extra)"},
            {"content": "Doc D (extra)"},
        ]
        grader._call_llm_grading = AsyncMock(return_value={0: True, 1: False})
        relevant, irrelevant = await grader.grade_documents("test", docs)
        # Doc A (True) + Doc C, D (remaining, auto-relevant)
        assert len(relevant) == 3
        assert len(irrelevant) == 1

    @pytest.mark.asyncio
    async def test_grading_exception_passes_all(self):
        """LLM 호출 실패 시 모든 문서를 관련으로 간주"""
        grader = RelevanceGrader()
        docs = [
            {"content": "Doc A"},
            {"content": "Doc B"},
        ]
        grader._call_llm_grading = AsyncMock(side_effect=RuntimeError("LLM error"))
        relevant, irrelevant = await grader.grade_documents("test", docs)
        assert len(relevant) == 2
        assert len(irrelevant) == 0

    @pytest.mark.asyncio
    async def test_grading_stats_accumulate(self):
        """여러 호출에 걸친 통계 누적"""
        grader = RelevanceGrader()
        grader._call_llm_grading = AsyncMock(return_value={0: True, 1: True})
        await grader.grade_documents("q1", [{"content": "A"}, {"content": "B"}])
        grader._call_llm_grading = AsyncMock(return_value={0: False, 1: False})
        await grader.grade_documents("q2", [{"content": "C"}, {"content": "D"}])
        assert grader._stats["total_graded"] == 4
        assert grader._stats["relevant"] == 2
        assert grader._stats["irrelevant"] == 2


# ---------------------------------------------------------------------------
# _call_llm_grading
# ---------------------------------------------------------------------------


class TestCallLLMGrading:
    """_call_llm_grading LLM 호출 테스트"""

    @pytest.mark.asyncio
    async def test_call_llm_grading_formats_prompt(self):
        """프롬프트 포맷 및 호출"""
        grader = RelevanceGrader()
        docs = [
            {"content": "LANEIGE Lip Sleeping Mask is popular"},
            {"content": "Random unrelated content here"},
        ]
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "grades": [
                    {"doc_index": 0, "relevant": True, "reason": "관련"},
                    {"doc_index": 1, "relevant": False, "reason": "무관"},
                ]
            }
        )
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            grades = await grader._call_llm_grading("LANEIGE 분석", docs)
        assert grades[0] is True
        assert grades[1] is False
        mock_llm.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_call_llm_truncates_long_content(self):
        """500자 초과 문서는 잘림"""
        grader = RelevanceGrader()
        long_doc = {"content": "A" * 600}
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"grades": [{"doc_index": 0, "relevant": True}]}
        )
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            await grader._call_llm_grading("test", [long_doc])
            call_args = mock_llm.call_args
            prompt_content = call_args.kwargs["messages"][0]["content"]
            # 문서 내용이 500자 + "..." 로 잘림
            assert "..." in prompt_content

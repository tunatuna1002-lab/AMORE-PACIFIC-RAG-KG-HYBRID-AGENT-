"""
HallucinationDetector 단위 테스트
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.hallucination_detector import GroundednessResult, HallucinationDetector

# =============================================================================
# GroundednessResult 테스트
# =============================================================================


class TestGroundednessResult:
    """GroundednessResult 데이터클래스 테스트"""

    def test_create_grounded_result(self):
        """근거 있는 결과 생성"""
        result = GroundednessResult(is_grounded=True, score=0.95, method="heuristic")
        assert result.is_grounded is True
        assert result.score == 0.95
        assert result.method == "heuristic"
        assert result.details == {}

    def test_create_with_details(self):
        """상세 정보 포함 결과 생성"""
        details = {"checks": {"numbers": {"score": 1.0}}}
        result = GroundednessResult(is_grounded=False, score=0.3, method="llm", details=details)
        assert result.is_grounded is False
        assert result.details == details


# =============================================================================
# HallucinationDetector 초기화 테스트
# =============================================================================


class TestHallucinationDetectorInit:
    """HallucinationDetector 초기화 테스트"""

    def test_default_init(self):
        """기본값 초기화"""
        detector = HallucinationDetector()
        assert detector.model == "gpt-4.1-mini"
        assert detector.threshold == 0.6

    def test_custom_init(self):
        """커스텀 값 초기화"""
        detector = HallucinationDetector(model="gpt-4o", threshold=0.8)
        assert detector.model == "gpt-4o"
        assert detector.threshold == 0.8

    def test_initial_stats(self):
        """초기 통계"""
        detector = HallucinationDetector()
        stats = detector.get_stats()
        assert stats["total_checks"] == 0
        assert stats["heuristic_only"] == 0
        assert stats["llm_checks"] == 0
        assert stats["hallucinations_detected"] == 0


# =============================================================================
# check 메서드 테스트 (비동기)
# =============================================================================


class TestHallucinationDetectorCheck:
    """HallucinationDetector.check 비동기 테스트"""

    @pytest.mark.asyncio
    async def test_empty_response_returns_grounded(self):
        """빈 응답은 grounded로 처리"""
        detector = HallucinationDetector()
        result = await detector.check("", "some context")
        assert result.is_grounded is True
        assert result.score == 1.0
        assert result.method == "skip"

    @pytest.mark.asyncio
    async def test_empty_context_returns_grounded(self):
        """빈 컨텍스트는 grounded로 처리"""
        detector = HallucinationDetector()
        result = await detector.check("some response", "")
        assert result.is_grounded is True
        assert result.score == 1.0
        assert result.method == "skip"

    @pytest.mark.asyncio
    async def test_both_empty_returns_grounded(self):
        """응답과 컨텍스트 모두 빈 경우"""
        detector = HallucinationDetector()
        result = await detector.check("", "")
        assert result.is_grounded is True
        assert result.method == "skip"

    @pytest.mark.asyncio
    async def test_high_heuristic_skips_llm(self):
        """heuristic 점수 0.9 이상이면 LLM 스킵"""
        detector = HallucinationDetector()
        # 응답의 모든 숫자가 컨텍스트에 존재하는 케이스
        response = "LANEIGE ranks 1st with 25.5% market share"
        context = "LANEIGE ranks 1st with 25.5% market share in lip care"
        result = await detector.check(response, context)
        assert result.method == "heuristic"
        stats = detector.get_stats()
        assert stats["heuristic_only"] >= 1

    @pytest.mark.asyncio
    async def test_low_heuristic_triggers_llm(self):
        """heuristic 점수 낮으면 LLM 호출"""
        detector = HallucinationDetector()
        response = "LANEIGE has 45.2% market share and 100 products"
        context = "COSRX is a brand in lip care"

        # LLM 호출 모킹
        with patch.object(detector, "_llm_check", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = 0.8
            result = await detector.check(response, context)
            mock_llm.assert_called_once()
            assert result.method == "llm"

    @pytest.mark.asyncio
    async def test_llm_check_hallucination_detected(self):
        """LLM 검사에서 환각 감지"""
        detector = HallucinationDetector(threshold=0.6)
        response = "LANEIGE has 99% market share"
        context = "COSRX is leading"

        with patch.object(detector, "_llm_check", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = 0.1  # very low score
            result = await detector.check(response, context)
            assert result.is_grounded is False
            stats = detector.get_stats()
            assert stats["hallucinations_detected"] >= 1

    @pytest.mark.asyncio
    async def test_llm_check_grounded(self):
        """LLM 검사에서 grounded 판정"""
        detector = HallucinationDetector(threshold=0.6)
        response = "The brand has 30% share"
        context = "Some unrelated numbers"

        with patch.object(detector, "_llm_check", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = 0.9
            result = await detector.check(response, context)
            assert result.is_grounded is True
            assert "llm_score" in result.details

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_heuristic(self):
        """LLM 실패 시 heuristic으로 fallback"""
        detector = HallucinationDetector(threshold=0.6)
        response = "LANEIGE has 50% share"
        context = "The market is competitive"

        with patch.object(detector, "_llm_check", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("API error")
            result = await detector.check(response, context)
            assert result.method == "heuristic_fallback"
            assert "llm_error" in result.details

    @pytest.mark.asyncio
    async def test_stats_increment_total_checks(self):
        """총 검사 수 증가"""
        detector = HallucinationDetector()
        await detector.check("", "context")
        await detector.check("response", "")
        stats = detector.get_stats()
        assert stats["total_checks"] == 2


# =============================================================================
# _heuristic_check 테스트
# =============================================================================


class TestHallucinationDetectorHeuristic:
    """HallucinationDetector._heuristic_check 테스트"""

    def test_numbers_all_matched(self):
        """모든 숫자가 컨텍스트에 있으면 높은 점수"""
        detector = HallucinationDetector()
        # English-only to test number matching in isolation
        response = "Price is 25.5 dollars"
        context = "The product price is 25.5 dollars"
        score, details = detector._heuristic_check(response, context)
        assert score >= 0.9

    def test_numbers_none_matched(self):
        """숫자가 컨텍스트에 없으면 낮은 점수"""
        detector = HallucinationDetector()
        response = "가격은 99.9달러입니다"
        context = "제품 정보가 없습니다"
        score, details = detector._heuristic_check(response, context)
        assert score < 0.9

    def test_brands_matched(self):
        """브랜드명이 컨텍스트에 있으면 점수 기여"""
        detector = HallucinationDetector()
        response = "LANEIGE is a top brand"
        context = "LANEIGE ranks first in lip care"
        score, details = detector._heuristic_check(response, context)
        checks = details.get("checks", {})
        if "brands" in checks:
            assert checks["brands"]["score"] == 1.0

    def test_brands_not_matched(self):
        """브랜드명이 컨텍스트에 없으면 낮은 점수"""
        detector = HallucinationDetector()
        response = "LANEIGE is a top brand in cosrx category"
        context = "This is about skincare trends"
        score, details = detector._heuristic_check(response, context)
        checks = details.get("checks", {})
        if "brands" in checks:
            assert checks["brands"]["score"] < 1.0

    def test_korean_terms_matched(self):
        """한글 용어가 컨텍스트에 있으면 점수 기여"""
        detector = HallucinationDetector()
        # Use identical Korean terms so regex [가-힣]{3,} matches the same tokens
        response = "시장점유율 데이터를 분석합니다"
        context = "시장점유율 데이터에 따르면 분석결과"
        score, details = detector._heuristic_check(response, context)
        checks = details.get("checks", {})
        assert "terms" in checks
        assert checks["terms"]["matched"] > 0

    def test_no_checkable_elements(self):
        """검사 가능 요소 없으면 1.0 반환"""
        detector = HallucinationDetector()
        response = "ok"
        context = "yes"
        score, details = detector._heuristic_check(response, context)
        assert score == 1.0
        assert "no checkable elements" in details.get("note", "")

    def test_mixed_checks(self):
        """숫자 + 브랜드 + 한글 혼합 검사"""
        detector = HallucinationDetector()
        response = "LANEIGE 제품은 25달러이고 시장점유율이 높습니다"
        context = "LANEIGE 제품의 가격은 25달러이며 시장점유율이 상승 중입니다"
        score, details = detector._heuristic_check(response, context)
        assert score > 0.5


# =============================================================================
# _llm_check 테스트
# =============================================================================


class TestHallucinationDetectorLLMCheck:
    """HallucinationDetector._llm_check 테스트"""

    @pytest.mark.asyncio
    async def test_llm_check_parses_score(self):
        """LLM 응답에서 점수 파싱"""
        detector = HallucinationDetector()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.85"

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            score = await detector._llm_check("response text", "context text")
            assert score == 0.85

    @pytest.mark.asyncio
    async def test_llm_check_clamps_score(self):
        """LLM 점수가 0~1 범위를 벗어나면 클램핑"""
        detector = HallucinationDetector()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "1.5"

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            score = await detector._llm_check("response", "context")
            assert score == 1.0

    @pytest.mark.asyncio
    async def test_llm_check_parse_failure_returns_default(self):
        """LLM 점수 파싱 실패 시 0.5 반환"""
        detector = HallucinationDetector()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "cannot determine"

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            score = await detector._llm_check("response", "context")
            assert score == 0.5

    @pytest.mark.asyncio
    async def test_llm_check_extracts_number_from_text(self):
        """LLM 응답 텍스트에서 숫자 추출"""
        detector = HallucinationDetector()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Score: 0.72"

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            score = await detector._llm_check("response", "context")
            assert abs(score - 0.72) < 0.01


# =============================================================================
# get_stats 테스트
# =============================================================================


class TestHallucinationDetectorStats:
    """HallucinationDetector.get_stats 테스트"""

    def test_get_stats_returns_dict(self):
        """get_stats가 dict 반환"""
        detector = HallucinationDetector()
        stats = detector.get_stats()
        assert isinstance(stats, dict)
        assert "total_checks" in stats
        assert "heuristic_only" in stats
        assert "llm_checks" in stats
        assert "hallucinations_detected" in stats

"""
InsightVerifier 단위 테스트
============================
src/tools/intelligence/insight_verifier.py 테스트

테스트 구조:
1. VerificationIssue / VerificationResult - 데이터 클래스 테스트
2. InsightVerifier 초기화
3. _verify_rank_comparisons - 순위 비교 검증
4. _verify_brand_names - 브랜드명 정확성 검증
5. _verify_numeric_data - 수치 데이터 교차 검증
6. _verify_logic_consistency - LLM 논리 검증 (mocked)
7. verify_report - 종합 검증
8. auto_correct - 자동 수정
9. verify_with_tavily - Tavily 팩트체크 (mocked)
10. verify_insight_report - 편의 함수
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.intelligence.insight_verifier import (
    InsightVerifier,
    VerificationIssue,
    VerificationResult,
    verify_insight_report,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def verifier():
    """InsightVerifier 기본 인스턴스"""
    return InsightVerifier()


@pytest.fixture
def sample_analysis_data():
    """샘플 분석 데이터"""
    return {
        "brand": {
            "competitors": [
                {"brand": "LANEIGE", "sos": 12.0, "avg_rank": 5.2},
                {"brand": "COSRX", "sos": 8.0, "avg_rank": 15.3},
            ]
        },
        "categories": {"Lip Care": {"top_brands": ["LANEIGE", "Aquaphor"]}},
    }


@pytest.fixture
def clean_report():
    """이슈 없는 깨끗한 보고서"""
    return """
## LANEIGE Lip Care 카테고리 분석

LANEIGE Lip Sleeping Mask는 [Lip Care] 카테고리에서 4위를 유지하고 있습니다.
SoS: 12.0%로 안정적인 점유율을 보이고 있습니다.
HHI: 850으로 시장은 비교적 분산되어 있습니다.
Burt's Bees와의 경쟁이 심화되고 있습니다.
"""


@pytest.fixture
def problematic_report():
    """이슈가 있는 보고서"""
    return """
## 시장 분석 보고서

LANEIGE 제품이 4위에서 71위로 하락했습니다.
Unknown 브랜드가 시장에서 상위를 차지하고 있습니다.
SoS: 65.0%로 매우 높은 점유율을 보이고 있습니다.
HHI: 15000으로 매우 집중된 시장입니다.
"""


# =============================================================================
# 1. VerificationIssue / VerificationResult Tests
# =============================================================================


class TestVerificationIssue:
    """VerificationIssue 데이터 클래스 테스트"""

    def test_creation_with_all_fields(self):
        """모든 필드를 포함한 VerificationIssue 생성"""
        issue = VerificationIssue(
            severity="critical",
            category="rank_comparison",
            description="순위 비교 오류",
            location="보고서 1번째 줄",
            original_text="4위에서 71위로",
            suggested_fix="카테고리 확인 필요",
        )
        assert issue.severity == "critical"
        assert issue.category == "rank_comparison"
        assert issue.description == "순위 비교 오류"
        assert issue.location == "보고서 1번째 줄"
        assert issue.original_text == "4위에서 71위로"
        assert issue.suggested_fix == "카테고리 확인 필요"

    def test_creation_with_default_suggested_fix(self):
        """suggested_fix 기본값 테스트"""
        issue = VerificationIssue(
            severity="warning",
            category="brand_name",
            description="브랜드명 오류",
            location="전체",
            original_text="Burt's",
        )
        assert issue.suggested_fix == ""


class TestVerificationResult:
    """VerificationResult 데이터 클래스 테스트"""

    def test_creation_defaults(self):
        """기본값 생성 테스트"""
        result = VerificationResult(
            verified_at="2026-01-01T00:00:00",
            total_checks=4,
            passed_checks=4,
        )
        assert result.issues == []
        assert result.confidence_score == 1.0
        assert result.has_issues is False
        assert result.has_critical_issues is False

    def test_has_issues_true(self):
        """이슈가 있을 때 has_issues True"""
        issue = VerificationIssue(
            severity="warning",
            category="brand_name",
            description="test",
            location="test",
            original_text="test",
        )
        result = VerificationResult(
            verified_at="2026-01-01T00:00:00",
            total_checks=4,
            passed_checks=3,
            issues=[issue],
        )
        assert result.has_issues is True
        assert result.has_critical_issues is False

    def test_has_critical_issues_true(self):
        """critical 이슈가 있을 때"""
        critical_issue = VerificationIssue(
            severity="critical",
            category="data_accuracy",
            description="critical error",
            location="test",
            original_text="test",
        )
        result = VerificationResult(
            verified_at="2026-01-01T00:00:00",
            total_checks=4,
            passed_checks=3,
            issues=[critical_issue],
        )
        assert result.has_critical_issues is True

    def test_to_dict(self):
        """to_dict 직렬화 테스트"""
        issue = VerificationIssue(
            severity="warning",
            category="brand_name",
            description="브랜드명 오류",
            location="전체",
            original_text="Burt's",
            suggested_fix="Burt's Bees 사용",
        )
        result = VerificationResult(
            verified_at="2026-01-01T00:00:00",
            total_checks=4,
            passed_checks=3,
            issues=[issue],
            confidence_score=0.75,
        )
        d = result.to_dict()

        assert d["verified_at"] == "2026-01-01T00:00:00"
        assert d["total_checks"] == 4
        assert d["passed_checks"] == 3
        assert d["confidence_score"] == 0.75
        assert d["has_critical_issues"] is False
        assert len(d["issues"]) == 1
        assert d["issues"][0]["severity"] == "warning"
        assert d["issues"][0]["category"] == "brand_name"

    def test_to_dict_empty_issues(self):
        """이슈 없는 경우 to_dict"""
        result = VerificationResult(
            verified_at="2026-01-01T00:00:00",
            total_checks=4,
            passed_checks=4,
        )
        d = result.to_dict()
        assert d["issues"] == []
        assert d["has_critical_issues"] is False


# =============================================================================
# 2. InsightVerifier 초기화
# =============================================================================


class TestInsightVerifierInit:
    """InsightVerifier 초기화 테스트"""

    def test_default_model(self):
        """기본 모델 설정"""
        verifier = InsightVerifier()
        assert verifier.model == "gpt-4.1-mini"
        assert verifier._tavily_client is None

    def test_custom_model(self):
        """커스텀 모델 설정"""
        verifier = InsightVerifier(model="gpt-4")
        assert verifier.model == "gpt-4"


# =============================================================================
# 3. _verify_rank_comparisons Tests
# =============================================================================


class TestVerifyRankComparisons:
    """순위 비교 검증 테스트"""

    @pytest.mark.asyncio
    async def test_no_rank_patterns_returns_empty(self, verifier, sample_analysis_data):
        """순위 패턴이 없으면 빈 리스트 반환"""
        content = "LANEIGE는 좋은 성과를 보이고 있습니다."
        issues = await verifier._verify_rank_comparisons(content, sample_analysis_data)
        assert issues == []

    @pytest.mark.asyncio
    async def test_normal_rank_change_no_issue(self, verifier, sample_analysis_data):
        """정상적인 순위 변동 (30위 이하)은 이슈 없음"""
        content = "[Lip Care] LANEIGE가 4위에서 6위로 변동했습니다."
        issues = await verifier._verify_rank_comparisons(content, sample_analysis_data)
        assert len(issues) == 0

    @pytest.mark.asyncio
    async def test_large_rank_change_without_category_is_critical(
        self, verifier, sample_analysis_data
    ):
        """카테고리 없이 30위 이상 변동은 critical 이슈"""
        content = "LANEIGE 제품이 4위에서 71위로 하락했습니다."
        issues = await verifier._verify_rank_comparisons(content, sample_analysis_data)
        assert len(issues) >= 1
        assert issues[0].severity == "critical"
        assert issues[0].category == "rank_comparison"

    @pytest.mark.asyncio
    async def test_large_rank_change_with_category_no_issue(self, verifier, sample_analysis_data):
        """카테고리가 명시된 경우 이슈 아님"""
        content = "Lip Care 카테고리에서 LANEIGE 제품이 4위에서 71위로 하락했습니다."
        issues = await verifier._verify_rank_comparisons(content, sample_analysis_data)
        assert len(issues) == 0

    @pytest.mark.asyncio
    async def test_arrow_pattern_detection(self, verifier, sample_analysis_data):
        """화살표 패턴 (→) 감지"""
        content = "LANEIGE 5위 → 50위 변동 발생"
        issues = await verifier._verify_rank_comparisons(content, sample_analysis_data)
        assert len(issues) >= 1
        assert issues[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_colon_pattern_detection(self, verifier, sample_analysis_data):
        """순위: N → M 패턴 감지"""
        content = "순위: 3 → 60 변동 감지됨"
        issues = await verifier._verify_rank_comparisons(content, sample_analysis_data)
        assert len(issues) >= 1


# =============================================================================
# 4. _verify_brand_names Tests
# =============================================================================


class TestVerifyBrandNames:
    """브랜드명 정확성 검증 테스트"""

    def test_truncated_brand_detected(self, verifier, sample_analysis_data):
        """잘린 브랜드명 감지"""
        content = "Drunk 브랜드가 인기를 끌고 있습니다."
        issues = verifier._verify_brand_names(content, sample_analysis_data)
        found_drunk = any("Drunk" in i.original_text for i in issues)
        assert found_drunk

    def test_full_brand_name_no_issue(self, verifier, sample_analysis_data):
        """정식 브랜드명은 이슈 아님"""
        content = "Drunk Elephant 브랜드가 인기를 끌고 있습니다."
        issues = verifier._verify_brand_names(content, sample_analysis_data)
        drunk_issues = [i for i in issues if "Drunk" in i.original_text]
        assert len(drunk_issues) == 0

    def test_unknown_brand_detected_as_error(self, verifier, sample_analysis_data):
        """Unknown 브랜드명은 error 레벨 이슈"""
        content = "Unknown 브랜드가 상위를 차지하고 있습니다."
        issues = verifier._verify_brand_names(content, sample_analysis_data)
        unknown_issues = [
            i for i in issues if i.category == "brand_name" and "금지" in i.description
        ]
        assert len(unknown_issues) >= 1
        assert unknown_issues[0].severity == "error"

    def test_multiple_unknown_patterns(self, verifier, sample_analysis_data):
        """여러 Unknown 패턴 감지"""
        content = "Unknown 브랜드와 미확인 브랜드가 상위에 있습니다."
        issues = verifier._verify_brand_names(content, sample_analysis_data)
        unknown_issues = [i for i in issues if "금지" in i.description]
        assert len(unknown_issues) >= 1

    def test_no_brand_issues_in_clean_report(self, verifier, sample_analysis_data, clean_report):
        """깨끗한 보고서에서 브랜드 이슈 없음"""
        issues = verifier._verify_brand_names(clean_report, sample_analysis_data)
        # Burt's Bees는 full name이므로 이슈 아님
        critical_issues = [i for i in issues if i.severity in ("critical", "error")]
        assert len(critical_issues) == 0


# =============================================================================
# 5. _verify_numeric_data Tests
# =============================================================================


class TestVerifyNumericData:
    """수치 데이터 교차 검증 테스트"""

    def test_normal_sos_no_issue(self, verifier, sample_analysis_data):
        """정상 SoS 값 (50% 이하)은 이슈 없음"""
        content = "LANEIGE의 SoS: 12.5%를 기록했습니다."
        issues = verifier._verify_numeric_data(content, sample_analysis_data)
        assert len(issues) == 0

    def test_high_sos_warning(self, verifier, sample_analysis_data):
        """비정상적으로 높은 SoS (50% 초과) 경고"""
        content = "LANEIGE의 SoS: 65.0%를 기록했습니다."
        issues = verifier._verify_numeric_data(content, sample_analysis_data)
        sos_issues = [i for i in issues if "SoS" in i.description]
        assert len(sos_issues) >= 1
        assert sos_issues[0].severity == "warning"

    def test_normal_hhi_no_issue(self, verifier, sample_analysis_data):
        """정상 HHI 값 (10000 이하)은 이슈 없음"""
        content = "HHI: 850으로 시장은 분산되어 있습니다."
        issues = verifier._verify_numeric_data(content, sample_analysis_data)
        assert len(issues) == 0

    def test_impossible_hhi_critical(self, verifier, sample_analysis_data):
        """불가능한 HHI (10000 초과)은 critical"""
        content = "HHI: 15000으로 매우 집중된 시장입니다."
        issues = verifier._verify_numeric_data(content, sample_analysis_data)
        hhi_issues = [i for i in issues if "HHI" in i.description]
        assert len(hhi_issues) >= 1
        assert hhi_issues[0].severity == "critical"

    def test_multiple_sos_values(self, verifier, sample_analysis_data):
        """여러 SoS 값 검증"""
        content = "LANEIGE SoS: 12.5%, COSRX SoS: 8.0%, Unknown SoS: 55.0%"
        issues = verifier._verify_numeric_data(content, sample_analysis_data)
        # Only 55.0% should trigger a warning
        sos_warnings = [i for i in issues if "SoS" in i.description]
        assert len(sos_warnings) == 1

    def test_no_numeric_data_no_issue(self, verifier, sample_analysis_data):
        """수치 데이터가 없으면 이슈 없음"""
        content = "LANEIGE는 좋은 성과를 보이고 있습니다."
        issues = verifier._verify_numeric_data(content, sample_analysis_data)
        assert len(issues) == 0

    def test_boundary_sos_50_no_issue(self, verifier, sample_analysis_data):
        """SoS 정확히 50%는 이슈 아님"""
        content = "SoS: 50.0%를 기록"
        issues = verifier._verify_numeric_data(content, sample_analysis_data)
        sos_issues = [i for i in issues if "SoS" in i.description]
        assert len(sos_issues) == 0

    def test_boundary_hhi_10000_no_issue(self, verifier, sample_analysis_data):
        """HHI 정확히 10000은 이슈 아님"""
        content = "HHI: 10000"
        issues = verifier._verify_numeric_data(content, sample_analysis_data)
        hhi_issues = [i for i in issues if "HHI" in i.description]
        assert len(hhi_issues) == 0


# =============================================================================
# 6. _verify_logic_consistency Tests (LLM mocked)
# =============================================================================


class TestVerifyLogicConsistency:
    """LLM 기반 논리적 일관성 검증 테스트 (mocked)"""

    @pytest.mark.asyncio
    async def test_no_logic_issues(self, verifier):
        """LLM이 이슈 없음 반환"""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"issues": [], "is_consistent": true}'))
        ]
        with patch(
            "src.tools.intelligence.insight_verifier.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            issues = await verifier._verify_logic_consistency("좋은 보고서입니다.")
            assert issues == []

    @pytest.mark.asyncio
    async def test_logic_issues_detected(self, verifier):
        """LLM이 논리 이슈 감지"""
        llm_result = {
            "issues": [
                {
                    "description": "SoS 상승이라고 했으나 수치가 하락",
                    "severity": "warning",
                    "location": "2번째 문단",
                    "suggestion": "수치 확인 필요",
                }
            ],
            "is_consistent": False,
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(llm_result)))]
        with patch(
            "src.tools.intelligence.insight_verifier.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            issues = await verifier._verify_logic_consistency("테스트 보고서")
            assert len(issues) == 1
            assert issues[0].category == "logic"
            assert issues[0].severity == "warning"

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty(self, verifier):
        """LLM 호출 실패 시 빈 리스트 반환"""
        with patch(
            "src.tools.intelligence.insight_verifier.acompletion",
            new_callable=AsyncMock,
            side_effect=Exception("LLM API error"),
        ):
            issues = await verifier._verify_logic_consistency("테스트 보고서")
            assert issues == []

    @pytest.mark.asyncio
    async def test_llm_invalid_json_returns_empty(self, verifier):
        """LLM이 유효하지 않은 JSON 반환"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="이것은 JSON이 아닙니다"))]
        with patch(
            "src.tools.intelligence.insight_verifier.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            issues = await verifier._verify_logic_consistency("테스트 보고서")
            assert issues == []


# =============================================================================
# 7. verify_report - 종합 검증 Tests
# =============================================================================


class TestVerifyReport:
    """종합 보고서 검증 테스트"""

    @pytest.mark.asyncio
    async def test_clean_report_passes(self, verifier, clean_report, sample_analysis_data):
        """깨끗한 보고서는 이슈 없이 통과"""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"issues": [], "is_consistent": true}'))
        ]
        with patch(
            "src.tools.intelligence.insight_verifier.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await verifier.verify_report(clean_report, sample_analysis_data)

        assert isinstance(result, VerificationResult)
        assert result.total_checks == 4
        assert result.verified_at is not None

    @pytest.mark.asyncio
    async def test_problematic_report_finds_issues(
        self, verifier, problematic_report, sample_analysis_data
    ):
        """문제 있는 보고서에서 이슈 발견"""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"issues": [], "is_consistent": true}'))
        ]
        with patch(
            "src.tools.intelligence.insight_verifier.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await verifier.verify_report(problematic_report, sample_analysis_data)

        assert result.has_issues is True
        assert result.total_checks == 4

    @pytest.mark.asyncio
    async def test_confidence_score_calculation(self, verifier, sample_analysis_data):
        """신뢰도 점수 계산 검증"""
        content = "HHI: 15000 이고 4위에서 71위로 하락했습니다."
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"issues": [], "is_consistent": true}'))
        ]
        with patch(
            "src.tools.intelligence.insight_verifier.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await verifier.verify_report(content, sample_analysis_data)

        # critical 이슈가 있으면 confidence가 1.0보다 낮아야 함
        if result.has_critical_issues:
            assert result.confidence_score < 1.0

    @pytest.mark.asyncio
    async def test_verify_report_with_raw_data(self, verifier, sample_analysis_data):
        """raw_data 파라미터 전달 테스트"""
        content = "LANEIGE SoS: 12.0%"
        raw_data = [{"brand": "LANEIGE", "rank": 5}]
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"issues": [], "is_consistent": true}'))
        ]
        with patch(
            "src.tools.intelligence.insight_verifier.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await verifier.verify_report(content, sample_analysis_data, raw_data=raw_data)
        assert isinstance(result, VerificationResult)


# =============================================================================
# 8. auto_correct Tests
# =============================================================================


class TestAutoCorrect:
    """자동 수정 테스트"""

    @pytest.mark.asyncio
    async def test_auto_correct_critical_issues(self, verifier):
        """critical 이슈 자동 수정"""
        content = "HHI: 15000으로 매우 집중된 시장입니다."
        issues = [
            VerificationIssue(
                severity="critical",
                category="data_accuracy",
                description="HHI 불가능한 값",
                location="전체",
                original_text="15000",
                suggested_fix="HHI 계산을 확인하세요.",
            )
        ]
        corrected = await verifier.auto_correct(content, issues)
        assert "수정 필요" in corrected
        assert "HHI 계산을 확인하세요." in corrected

    @pytest.mark.asyncio
    async def test_auto_correct_skips_non_critical(self, verifier):
        """non-critical 이슈는 수정하지 않음"""
        content = "SoS: 55.0%입니다."
        issues = [
            VerificationIssue(
                severity="warning",
                category="data_accuracy",
                description="SoS 비정상",
                location="전체",
                original_text="55.0%",
                suggested_fix="확인 필요",
            )
        ]
        corrected = await verifier.auto_correct(content, issues)
        assert corrected == content  # warning은 수정하지 않음

    @pytest.mark.asyncio
    async def test_auto_correct_no_issues(self, verifier):
        """이슈 없으면 원본 반환"""
        content = "깨끗한 보고서입니다."
        corrected = await verifier.auto_correct(content, [])
        assert corrected == content

    @pytest.mark.asyncio
    async def test_auto_correct_missing_suggested_fix(self, verifier):
        """suggested_fix 없는 critical 이슈는 스킵"""
        content = "문제 있는 보고서"
        issues = [
            VerificationIssue(
                severity="critical",
                category="logic",
                description="논리 오류",
                location="전체",
                original_text="",
                suggested_fix="",
            )
        ]
        corrected = await verifier.auto_correct(content, issues)
        assert corrected == content


# =============================================================================
# 9. verify_with_tavily Tests
# =============================================================================


class TestVerifyWithTavily:
    """Tavily 팩트체크 테스트 (mocked)"""

    @pytest.mark.asyncio
    async def test_tavily_success(self, verifier):
        """Tavily 검증 성공"""
        mock_result = MagicMock()
        mock_result.url = "https://example.com"

        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value=[mock_result])

        mock_tavily_module = MagicMock()
        mock_tavily_module.TavilySearchClient = MagicMock(return_value=mock_client)

        with patch.dict(
            "sys.modules",
            {"src.tools.collectors.tavily_search": mock_tavily_module},
        ):
            verifier._tavily_client = None
            results = await verifier.verify_with_tavily(["LANEIGE is popular"])

        assert "LANEIGE is popular" in results
        assert results["LANEIGE is popular"]["verified"] is True

    @pytest.mark.asyncio
    async def test_tavily_failure_returns_empty(self, verifier):
        """Tavily 실패 시 빈 딕셔너리"""
        # Simulate import failure by making the module raise ImportError
        import sys

        # Remove cached module if present
        module_name = "src.tools.collectors.tavily_search"
        original = sys.modules.get(module_name)
        sys.modules[module_name] = None  # type: ignore[assignment]

        try:
            verifier._tavily_client = None
            results = await verifier.verify_with_tavily(["test claim"])
            assert results == {}
        finally:
            if original is not None:
                sys.modules[module_name] = original
            else:
                sys.modules.pop(module_name, None)

    @pytest.mark.asyncio
    async def test_tavily_max_5_claims(self, verifier):
        """최대 5개 클레임만 검증"""
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value=[])

        # Pre-set client so no import needed
        verifier._tavily_client = mock_client

        claims = [f"claim {i}" for i in range(10)]
        results = await verifier.verify_with_tavily(claims)

        # Should have called search at most 5 times
        assert mock_client.search.call_count == 5


# =============================================================================
# 10. verify_insight_report 편의 함수 Tests
# =============================================================================


class TestVerifyInsightReportFunction:
    """편의 함수 테스트"""

    @pytest.mark.asyncio
    async def test_convenience_function(self, sample_analysis_data):
        """verify_insight_report 편의 함수가 VerificationResult 반환"""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"issues": [], "is_consistent": true}'))
        ]
        with patch(
            "src.tools.intelligence.insight_verifier.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await verify_insight_report("LANEIGE SoS: 12.0%", sample_analysis_data)
        assert isinstance(result, VerificationResult)
        assert result.total_checks == 4

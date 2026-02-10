"""
ConfidenceScorer 단위 테스트
"""

import pytest

from src.tools.intelligence.claim_extractor import Claim, ClaimType
from src.tools.intelligence.claim_verifier import VerificationResult, VerificationStatus
from src.tools.intelligence.confidence_scorer import ConfidenceGrade, ConfidenceScorer


class TestConfidenceScorer:
    """ConfidenceScorer 테스트"""

    @pytest.fixture
    def scorer(self):
        """ConfidenceScorer 인스턴스"""
        return ConfidenceScorer()

    def _make_result(
        self,
        status: VerificationStatus,
        claim_type: ClaimType = ClaimType.NUMERICAL,
        confidence: float = 0.0,
    ) -> VerificationResult:
        """테스트용 VerificationResult 생성"""
        claim = Claim(text="test", claim_type=claim_type, value=1)
        return VerificationResult(claim=claim, status=status, confidence=confidence, reason="test")

    def test_calculate_all_verified(self, scorer):
        """모든 주장 검증됨 → GREEN"""
        results = [
            self._make_result(VerificationStatus.VERIFIED, confidence=1.0),
            self._make_result(VerificationStatus.VERIFIED, confidence=1.0),
            self._make_result(VerificationStatus.VERIFIED, confidence=1.0),
        ]
        report = scorer.calculate(results)

        assert report.grade == ConfidenceGrade.GREEN
        assert report.score >= 0.85
        assert report.verified_claims == 3
        assert report.unverified_claims == 0

    def test_calculate_mixed_results(self, scorer):
        """혼합 결과 → YELLOW"""
        results = [
            self._make_result(VerificationStatus.VERIFIED, confidence=1.0),
            self._make_result(VerificationStatus.PARTIALLY_VERIFIED, confidence=0.7),
            self._make_result(VerificationStatus.UNABLE, confidence=0.0),
        ]
        report = scorer.calculate(results)

        assert report.grade == ConfidenceGrade.YELLOW
        assert 0.60 <= report.score < 0.85
        assert report.verified_claims == 1
        assert report.partial_claims == 1
        assert report.unable_claims == 1

    def test_calculate_all_unverified(self, scorer):
        """모든 주장 검증 실패 → RED"""
        results = [
            self._make_result(VerificationStatus.UNVERIFIED, confidence=0.0),
            self._make_result(VerificationStatus.UNVERIFIED, confidence=0.0),
            self._make_result(VerificationStatus.UNVERIFIED, confidence=0.0),
        ]
        report = scorer.calculate(results)

        assert report.grade == ConfidenceGrade.RED
        assert report.score < 0.60
        assert report.unverified_claims == 3

    def test_calculate_empty_results(self, scorer):
        """빈 결과 → UNKNOWN"""
        report = scorer.calculate([])

        assert report.grade == ConfidenceGrade.UNKNOWN
        assert report.score == 0.0
        assert report.total_claims == 0

    def test_calculate_all_unable(self, scorer):
        """모든 주장 검증 불가 → 중립 점수"""
        results = [
            self._make_result(VerificationStatus.UNABLE),
            self._make_result(VerificationStatus.UNABLE),
        ]
        report = scorer.calculate(results)

        # 검증 불가는 중립 (0.5)
        assert report.unable_claims == 2
        assert report.score >= 0.4  # 중립 점수 부근

    def test_type_weights(self, scorer):
        """유형별 가중치 적용"""
        # 숫자 주장 (가중치 1.5)
        numerical_result = self._make_result(
            VerificationStatus.VERIFIED, claim_type=ClaimType.NUMERICAL, confidence=1.0
        )
        # 추론 주장 (가중치 1.2)
        logical_result = self._make_result(
            VerificationStatus.VERIFIED, claim_type=ClaimType.LOGICAL, confidence=1.0
        )

        results = [numerical_result, logical_result]
        report = scorer.calculate(results)

        # 상세 결과에서 가중치 확인
        assert len(report.details) == 2
        numerical_detail = [d for d in report.details if d["claim_type"] == "numerical"][0]
        logical_detail = [d for d in report.details if d["claim_type"] == "logical"][0]
        assert numerical_detail["weight"] == 1.5
        assert logical_detail["weight"] == 1.2

    def test_get_badge_info(self, scorer):
        """배지 정보 반환"""
        green_badge = scorer.get_badge_info(ConfidenceGrade.GREEN)
        assert green_badge["emoji"] == "🟢"
        assert green_badge["label"] == "높은 신뢰"

        yellow_badge = scorer.get_badge_info(ConfidenceGrade.YELLOW)
        assert yellow_badge["emoji"] == "🟡"

        red_badge = scorer.get_badge_info(ConfidenceGrade.RED)
        assert red_badge["emoji"] == "🔴"

        unknown_badge = scorer.get_badge_info(ConfidenceGrade.UNKNOWN)
        assert unknown_badge["emoji"] == "⚪"

    def test_format_report(self, scorer):
        """리포트 포맷팅"""
        results = [
            self._make_result(VerificationStatus.VERIFIED, confidence=1.0),
            self._make_result(VerificationStatus.UNVERIFIED, confidence=0.0),
        ]
        report = scorer.calculate(results)
        formatted = scorer.format_report(report)

        assert "검증 결과" in formatted
        assert "전체 주장" in formatted
        assert "검증됨" in formatted

    def test_determine_grade_boundaries(self, scorer):
        """등급 경계값 테스트"""
        assert scorer._determine_grade(0.85) == ConfidenceGrade.GREEN
        assert scorer._determine_grade(0.84) == ConfidenceGrade.YELLOW
        assert scorer._determine_grade(0.60) == ConfidenceGrade.YELLOW
        assert scorer._determine_grade(0.59) == ConfidenceGrade.RED
        assert scorer._determine_grade(0.0) == ConfidenceGrade.RED

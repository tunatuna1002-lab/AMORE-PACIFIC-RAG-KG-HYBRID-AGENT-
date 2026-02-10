"""
ClaimVerifier 단위 테스트
"""

import pytest

from src.tools.intelligence.claim_extractor import Claim, ClaimType
from src.tools.intelligence.claim_verifier import (
    ClaimVerifier,
    LogicVerifier,
    NumericalVerifier,
    TemporalVerifier,
    VerificationStatus,
)


class TestNumericalVerifier:
    """NumericalVerifier 테스트"""

    @pytest.fixture
    def verifier(self, tmp_path):
        """NumericalVerifier 인스턴스 (임시 DB 경로)"""
        return NumericalVerifier(db_path=str(tmp_path / "test.db"))

    @pytest.mark.asyncio
    async def test_verify_rank_claim_type_check(self, verifier):
        """숫자가 아닌 주장 처리"""
        claim = Claim(text="오늘", claim_type=ClaimType.TEMPORAL, value="today")
        result = await verifier.verify(claim)
        assert result.status == VerificationStatus.UNABLE

    @pytest.mark.asyncio
    async def test_verify_rank_no_brand(self, verifier):
        """브랜드 없는 순위 주장"""
        claim = Claim(text="3위", claim_type=ClaimType.NUMERICAL, value=3, unit="위", entities=[])
        result = await verifier.verify(claim)
        # 브랜드를 식별할 수 없으면 검증 불가
        assert result.status == VerificationStatus.UNABLE

    @pytest.mark.asyncio
    async def test_verify_rank_no_db(self, verifier):
        """DB 없을 때 순위 검증"""
        claim = Claim(
            text="3위", claim_type=ClaimType.NUMERICAL, value=3, unit="위", entities=["LANEIGE"]
        )
        result = await verifier.verify(claim)
        # DB가 없으면 검증 불가
        assert result.status == VerificationStatus.UNABLE


class TestTemporalVerifier:
    """TemporalVerifier 테스트"""

    @pytest.fixture
    def verifier(self, tmp_path):
        """TemporalVerifier 인스턴스"""
        return TemporalVerifier(db_path=str(tmp_path / "test.db"))

    @pytest.mark.asyncio
    async def test_verify_temporal_claim_type_check(self, verifier):
        """시간이 아닌 주장 처리"""
        claim = Claim(text="3위", claim_type=ClaimType.NUMERICAL, value=3)
        result = await verifier.verify(claim)
        assert result.status == VerificationStatus.UNABLE

    @pytest.mark.asyncio
    async def test_verify_temporal_no_db(self, verifier):
        """DB 없을 때 시간 검증"""
        claim = Claim(text="오늘", claim_type=ClaimType.TEMPORAL, value="today")
        result = await verifier.verify(claim)
        # DB가 없으면 검증 불가
        assert result.status == VerificationStatus.UNABLE


class TestLogicVerifier:
    """LogicVerifier 테스트"""

    @pytest.fixture
    def verifier(self, tmp_path):
        """LogicVerifier 인스턴스"""
        return LogicVerifier(db_path=str(tmp_path / "test.db"))

    @pytest.mark.asyncio
    async def test_verify_logical_claim_type_check(self, verifier):
        """추론이 아닌 주장 처리"""
        claim = Claim(text="3위", claim_type=ClaimType.NUMERICAL, value=3)
        result = await verifier.verify(claim)
        assert result.status == VerificationStatus.UNABLE

    @pytest.mark.asyncio
    async def test_verify_logical_no_brand(self, verifier):
        """브랜드 없는 추론 주장"""
        claim = Claim(
            text="경쟁력 있음",
            claim_type=ClaimType.LOGICAL,
            value="competitive_positive",
            entities=[],
        )
        result = await verifier.verify(claim)
        # 브랜드를 식별할 수 없으면 검증 불가
        assert result.status == VerificationStatus.UNABLE

    def test_get_sos_level(self, verifier):
        """SoS 레벨 판정 테스트"""
        assert verifier._get_sos_level(3.0) == "weak"
        assert verifier._get_sos_level(7.0) == "moderate"
        assert verifier._get_sos_level(15.0) == "strong"
        assert verifier._get_sos_level(25.0) == "dominant"


class TestClaimVerifier:
    """통합 검증기 테스트"""

    @pytest.fixture
    def verifier(self, tmp_path):
        """ClaimVerifier 인스턴스"""
        return ClaimVerifier(db_path=str(tmp_path / "test.db"))

    @pytest.mark.asyncio
    async def test_verify_routes_correctly(self, verifier):
        """주장 유형에 따른 라우팅 테스트"""
        # 숫자 주장
        numerical_claim = Claim(text="3위", claim_type=ClaimType.NUMERICAL, value=3, unit="위")
        result = await verifier.verify(numerical_claim)
        assert result.claim.claim_type == ClaimType.NUMERICAL

        # 시간 주장
        temporal_claim = Claim(text="오늘", claim_type=ClaimType.TEMPORAL, value="today")
        result = await verifier.verify(temporal_claim)
        assert result.claim.claim_type == ClaimType.TEMPORAL

        # 추론 주장
        logical_claim = Claim(
            text="경쟁력 있음", claim_type=ClaimType.LOGICAL, value="competitive_positive"
        )
        result = await verifier.verify(logical_claim)
        assert result.claim.claim_type == ClaimType.LOGICAL

    @pytest.mark.asyncio
    async def test_verify_all(self, verifier):
        """전체 검증 테스트"""
        claims = [
            Claim(text="3위", claim_type=ClaimType.NUMERICAL, value=3, unit="위"),
            Claim(text="오늘", claim_type=ClaimType.TEMPORAL, value="today"),
            Claim(text="경쟁력 있음", claim_type=ClaimType.LOGICAL, value="competitive_positive"),
        ]
        results = await verifier.verify_all(claims)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_verify_unsupported_type(self, verifier):
        """지원하지 않는 주장 유형"""
        claim = Claim(text="2배", claim_type=ClaimType.COMPARATIVE, value=2.0)
        result = await verifier.verify(claim)
        assert result.status == VerificationStatus.UNABLE

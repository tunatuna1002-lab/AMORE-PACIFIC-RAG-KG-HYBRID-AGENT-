"""
ClaimExtractor 단위 테스트
"""

import pytest

from src.tools.claim_extractor import ClaimExtractor, ClaimType


class TestClaimExtractor:
    """ClaimExtractor 테스트"""

    @pytest.fixture
    def extractor(self):
        """ClaimExtractor 인스턴스"""
        return ClaimExtractor()

    def test_extract_rank(self, extractor):
        """순위 추출 테스트"""
        text = "LANEIGE는 현재 3위를 기록하고 있습니다."
        claims = extractor.extract(text)

        rank_claims = [c for c in claims if c.claim_type == ClaimType.NUMERICAL and c.unit == "위"]
        assert len(rank_claims) >= 1
        assert rank_claims[0].value == 3

    def test_extract_percent(self, extractor):
        """퍼센트 추출 테스트"""
        text = "SoS는 12.5%로 강한 경쟁력을 보여주고 있습니다."
        claims = extractor.extract(text)

        percent_claims = [
            c for c in claims if c.claim_type == ClaimType.NUMERICAL and c.unit == "%"
        ]
        assert len(percent_claims) >= 1
        assert percent_claims[0].value == 12.5

    def test_extract_sos(self, extractor):
        """SoS 지표 추출 테스트"""
        text = "LANEIGE의 SoS: 15.3% 입니다."
        claims = extractor.extract(text)

        sos_claims = [c for c in claims if "sos" in c.text.lower() or c.unit == "%"]
        assert len(sos_claims) >= 1

    def test_extract_temporal_today(self, extractor):
        """시간 표현 추출 - 오늘"""
        text = "오늘 기준으로 LANEIGE는 상위권에 있습니다."
        claims = extractor.extract(text)

        temporal_claims = [c for c in claims if c.claim_type == ClaimType.TEMPORAL]
        assert len(temporal_claims) >= 1
        assert temporal_claims[0].value == "today"

    def test_extract_temporal_recent(self, extractor):
        """시간 표현 추출 - 최근"""
        text = "최근 LANEIGE의 순위가 상승했습니다."
        claims = extractor.extract(text)

        temporal_claims = [c for c in claims if c.claim_type == ClaimType.TEMPORAL]
        assert len(temporal_claims) >= 1
        assert temporal_claims[0].value == "recent"

    def test_extract_logical_positive(self, extractor):
        """추론 표현 추출 - 긍정"""
        text = "LANEIGE는 경쟁력 있는 브랜드입니다."
        claims = extractor.extract(text)

        logical_claims = [c for c in claims if c.claim_type == ClaimType.LOGICAL]
        assert len(logical_claims) >= 1
        assert "competitive_positive" in logical_claims[0].value

    def test_extract_logical_negative(self, extractor):
        """추론 표현 추출 - 부정"""
        text = "이 브랜드는 경쟁력 없는 상태입니다."
        claims = extractor.extract(text)

        logical_claims = [c for c in claims if c.claim_type == ClaimType.LOGICAL]
        assert len(logical_claims) >= 1
        assert "competitive_negative" in logical_claims[0].value

    def test_extract_multiple_claims(self, extractor):
        """복합 주장 추출 테스트"""
        text = """
        LANEIGE Lip Sleeping Mask는 현재 Lip Care 카테고리에서 3위를 기록하고 있습니다.
        SoS는 12.5%로 경쟁력 있는 수준입니다.
        최근 순위가 상승하는 추세입니다.
        """
        claims = extractor.extract(text)

        # 최소 3개 이상의 주장이 추출되어야 함
        assert len(claims) >= 3

        # 유형별 확인
        numerical = [c for c in claims if c.claim_type == ClaimType.NUMERICAL]
        temporal = [c for c in claims if c.claim_type == ClaimType.TEMPORAL]
        logical = [c for c in claims if c.claim_type == ClaimType.LOGICAL]

        assert len(numerical) >= 2  # 3위, 12.5%
        assert len(temporal) >= 1  # 최근
        assert len(logical) >= 1  # 경쟁력 있는

    def test_entity_linking(self, extractor):
        """엔티티 연결 테스트"""
        text = "LANEIGE는 Lip Care 카테고리에서 3위입니다."
        claims = extractor.extract(text)

        # 순위 주장에 브랜드/카테고리가 연결되어야 함
        rank_claims = [c for c in claims if c.unit == "위"]
        assert len(rank_claims) >= 1

        # 엔티티 확인 (문맥에서 추출)
        claim = rank_claims[0]
        assert claim.context is not None

    def test_get_summary(self, extractor):
        """요약 생성 테스트"""
        text = "LANEIGE는 3위이며 SoS 12.5%입니다. 최근 경쟁력 있는 브랜드입니다."
        claims = extractor.extract(text)
        summary = extractor.get_summary(claims)

        assert "total" in summary
        assert "by_type" in summary
        assert summary["total"] >= 3

    def test_empty_text(self, extractor):
        """빈 텍스트 처리"""
        claims = extractor.extract("")
        assert claims == []

    def test_no_claims(self, extractor):
        """주장 없는 텍스트"""
        text = "안녕하세요. 무엇을 도와드릴까요?"
        claims = extractor.extract(text)
        # 주장이 없거나 매우 적어야 함
        assert len(claims) <= 1

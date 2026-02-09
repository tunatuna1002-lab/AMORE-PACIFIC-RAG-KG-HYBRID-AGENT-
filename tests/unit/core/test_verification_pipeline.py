"""
ResponseVerificationPipeline 단위 테스트
"""

import pytest

from src.core.verification_pipeline import (
    ResponseVerificationPipeline,
    VerificationPipelineFactory,
)
from src.tools.confidence_scorer import ConfidenceGrade


class TestResponseVerificationPipeline:
    """ResponseVerificationPipeline 테스트"""

    @pytest.fixture
    def pipeline(self, tmp_path):
        """Pipeline 인스턴스 (임시 DB 경로)"""
        return ResponseVerificationPipeline(db_path=str(tmp_path / "test.db"))

    @pytest.mark.asyncio
    async def test_verify_simple_response(self, pipeline):
        """간단한 응답 검증"""
        response = "LANEIGE는 현재 3위를 기록하고 있습니다."
        verified = await pipeline.verify(response)

        assert verified.original_response == response
        assert verified.grade is not None
        assert 0 <= verified.score <= 1
        assert len(verified.claims) >= 1

    @pytest.mark.asyncio
    async def test_verify_complex_response(self, pipeline):
        """복합 응답 검증"""
        response = """
        LANEIGE Lip Sleeping Mask는 현재 Lip Care 카테고리에서 3위를 기록하고 있습니다.
        SoS는 12.5%로 경쟁력 있는 수준입니다.
        최근 순위가 상승하는 추세입니다.
        """
        verified = await pipeline.verify(response)

        assert len(verified.claims) >= 3
        assert verified.report.total_claims >= 3

    @pytest.mark.asyncio
    async def test_verify_empty_response(self, pipeline):
        """빈 응답 검증"""
        verified = await pipeline.verify("")

        assert verified.grade == ConfidenceGrade.UNKNOWN
        assert verified.score == 0.0
        assert len(verified.claims) == 0

    @pytest.mark.asyncio
    async def test_verify_no_claims_response(self, pipeline):
        """주장 없는 응답 검증"""
        response = "안녕하세요. 무엇을 도와드릴까요?"
        verified = await pipeline.verify(response)

        # 주장이 없거나 매우 적음
        assert verified.report.total_claims <= 1

    @pytest.mark.asyncio
    async def test_verify_with_context(self, pipeline):
        """컨텍스트 포함 검증"""
        response = "현재 3위입니다."
        context = {"category": "Lip Care", "brand": "LANEIGE"}
        verified = await pipeline.verify(response, context=context)

        assert verified.grade is not None

    @pytest.mark.asyncio
    async def test_enriched_response_contains_badge(self, pipeline):
        """enriched 응답에 배지 포함"""
        response = "LANEIGE는 3위입니다."
        verified = await pipeline.verify(response, include_details=True)

        # enriched 응답에 검증 결과 섹션이 포함되어야 함
        assert "검증 결과" in verified.enriched_response
        assert verified.original_response in verified.enriched_response

    @pytest.mark.asyncio
    async def test_verification_summary(self, pipeline):
        """검증 요약 테스트"""
        response = "LANEIGE는 3위이며 SoS 12.5%입니다."
        verified = await pipeline.verify(response)
        summary = pipeline.get_verification_summary(verified)

        assert "grade" in summary
        assert "grade_emoji" in summary
        assert "score" in summary
        assert "total_claims" in summary
        assert "verified_claims" in summary


class TestVerificationPipelineFactory:
    """Factory 테스트"""

    def test_singleton_instance(self):
        """싱글톤 패턴 테스트"""
        VerificationPipelineFactory.reset()

        instance1 = VerificationPipelineFactory.get_instance()
        instance2 = VerificationPipelineFactory.get_instance()

        assert instance1 is instance2

    def test_reset(self):
        """리셋 테스트"""
        VerificationPipelineFactory.reset()
        instance1 = VerificationPipelineFactory.get_instance()

        VerificationPipelineFactory.reset()
        instance2 = VerificationPipelineFactory.get_instance()

        assert instance1 is not instance2

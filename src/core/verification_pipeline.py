"""
Response Verification Pipeline
==============================
응답의 사실 검증을 위한 통합 파이프라인

역할:
- ClaimExtractor로 주장 추출
- ClaimVerifier로 검증
- ConfidenceScorer로 점수 산출
- 응답에 검증 결과 추가 (enrichment)

연결 파일:
- tools/claim_extractor.py: 주장 추출
- tools/claim_verifier.py: 주장 검증
- tools/confidence_scorer.py: 신뢰도 점수
- agents/hybrid_chatbot_agent.py: 챗봇 통합
"""

import logging
from dataclasses import dataclass
from typing import Any

from src.tools.claim_extractor import Claim, ClaimExtractor
from src.tools.claim_verifier import ClaimVerifier, VerificationResult
from src.tools.confidence_scorer import ConfidenceGrade, ConfidenceReport, ConfidenceScorer

logger = logging.getLogger(__name__)


@dataclass
class VerifiedResponse:
    """검증된 응답"""

    original_response: str  # 원본 응답
    enriched_response: str  # 검증 정보 추가된 응답
    grade: ConfidenceGrade  # 신뢰도 등급
    score: float  # 신뢰도 점수
    claims: list[Claim]  # 추출된 주장
    results: list[VerificationResult]  # 검증 결과
    report: ConfidenceReport  # 신뢰도 리포트


class ResponseVerificationPipeline:
    """
    응답 검증 파이프라인

    HybridChatbotAgent의 응답을 검증하고 신뢰도 배지를 추가

    Usage:
        pipeline = ResponseVerificationPipeline()
        verified = await pipeline.verify(response_text, context)
        print(verified.enriched_response)
    """

    def __init__(self, db_path: str | None = None):
        """
        Args:
            db_path: SQLite DB 경로 (검증용)
        """
        self.extractor = ClaimExtractor()
        self.verifier = ClaimVerifier(db_path)
        self.scorer = ConfidenceScorer()

    async def verify(
        self, response: str, context: dict[str, Any] | None = None, include_details: bool = True
    ) -> VerifiedResponse:
        """
        응답 검증 수행

        Args:
            response: 검증할 응답 텍스트
            context: 추가 컨텍스트 (카테고리, 브랜드 등)
            include_details: 상세 결과 포함 여부

        Returns:
            검증된 응답
        """
        logger.info("Starting response verification pipeline")

        # 1. 주장 추출
        claims = self.extractor.extract(response)
        logger.info(f"Extracted {len(claims)} claims")

        # 2. 주장 검증
        results = await self.verifier.verify_all(claims, context)
        logger.info(f"Verified {len(results)} claims")

        # 3. 신뢰도 점수 계산
        report = self.scorer.calculate(results)
        logger.info(f"Confidence: {report.grade.value} ({report.score:.0%})")

        # 4. 응답 enrichment
        enriched = self._enrich_response(response, report, include_details)

        return VerifiedResponse(
            original_response=response,
            enriched_response=enriched,
            grade=report.grade,
            score=report.score,
            claims=claims,
            results=results,
            report=report,
        )

    def _enrich_response(
        self, response: str, report: ConfidenceReport, include_details: bool
    ) -> str:
        """응답에 검증 결과 추가"""
        badge_info = self.scorer.get_badge_info(report.grade)

        # 배지 라인
        badge_line = f"\n\n---\n{badge_info['emoji']} **검증 결과: {badge_info['label']}** (신뢰도: {report.score:.0%})"

        if include_details and report.details:
            # 검증된 주장만 표시
            verified_items = [d for d in report.details if d["status"] in ["verified", "partial"]]

            if verified_items:
                badge_line += "\n\n**확인된 정보:**"
                for item in verified_items[:5]:  # 최대 5개
                    status_emoji = "✅" if item["status"] == "verified" else "🔶"
                    badge_line += f"\n- {status_emoji} {item['claim_text']}"

            # 검증 실패 주장
            unverified_items = [d for d in report.details if d["status"] == "unverified"]

            if unverified_items:
                badge_line += "\n\n**주의 필요:**"
                for item in unverified_items[:3]:  # 최대 3개
                    badge_line += f"\n- ⚠️ {item['claim_text']} ({item['reason']})"

        return response + badge_line

    def get_verification_summary(self, verified: VerifiedResponse) -> dict[str, Any]:
        """검증 결과 요약 반환 (API 응답용)"""
        return {
            "grade": verified.grade.value,
            "grade_emoji": self.scorer.get_badge_info(verified.grade)["emoji"],
            "grade_label": self.scorer.get_badge_info(verified.grade)["label"],
            "score": verified.score,
            "total_claims": verified.report.total_claims,
            "verified_claims": verified.report.verified_claims,
            "partial_claims": verified.report.partial_claims,
            "unverified_claims": verified.report.unverified_claims,
            "details": verified.report.details if verified.report.details else [],
        }


class VerificationPipelineFactory:
    """검증 파이프라인 팩토리"""

    _instance: ResponseVerificationPipeline | None = None

    @classmethod
    def get_instance(cls, db_path: str | None = None) -> ResponseVerificationPipeline:
        """싱글톤 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = ResponseVerificationPipeline(db_path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """인스턴스 리셋 (테스트용)"""
        cls._instance = None

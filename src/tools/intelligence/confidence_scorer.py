"""
Confidence Scorer
=================
검증 결과를 집계하여 신뢰도 점수 및 등급 산출

역할:
- 개별 검증 결과 가중 집계
- 전체 신뢰도 점수 계산
- 등급 결정 (GREEN/YELLOW/RED)

연결 파일:
- tools/claim_verifier.py: 검증 결과
- core/verification_pipeline.py: 파이프라인 통합
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .claim_extractor import ClaimType
from .claim_verifier import VerificationResult, VerificationStatus

logger = logging.getLogger(__name__)


class ConfidenceGrade(Enum):
    """신뢰도 등급"""

    GREEN = "green"  # 높은 신뢰 (0.85+)
    YELLOW = "yellow"  # 부분 신뢰 (0.60-0.84)
    RED = "red"  # 낮은 신뢰 (<0.60)
    UNKNOWN = "unknown"  # 판단 불가


@dataclass
class ConfidenceReport:
    """신뢰도 리포트"""

    grade: ConfidenceGrade  # 등급
    score: float  # 점수 (0-1)
    total_claims: int  # 전체 주장 수
    verified_claims: int  # 검증된 주장 수
    partial_claims: int  # 부분 검증된 주장 수
    unverified_claims: int  # 검증 실패 주장 수
    unable_claims: int  # 검증 불가 주장 수
    details: list[dict[str, Any]]  # 상세 결과


class ConfidenceScorer:
    """
    신뢰도 점수 계산기

    검증 결과를 집계하여 최종 신뢰도 점수 및 등급 산출

    Usage:
        scorer = ConfidenceScorer()
        report = scorer.calculate(verification_results)
    """

    # 등급 임계값
    GRADE_THRESHOLDS = {
        ConfidenceGrade.GREEN: 0.85,
        ConfidenceGrade.YELLOW: 0.60,
        ConfidenceGrade.RED: 0.0,
    }

    # 검증 상태별 기본 점수
    STATUS_SCORES = {
        VerificationStatus.VERIFIED: 1.0,
        VerificationStatus.PARTIALLY_VERIFIED: 0.7,
        VerificationStatus.UNVERIFIED: 0.0,
        VerificationStatus.UNABLE: 0.5,  # 검증 불가는 중립
    }

    # 주장 유형별 가중치
    TYPE_WEIGHTS = {
        ClaimType.NUMERICAL: 1.5,  # 숫자는 검증 중요
        ClaimType.TEMPORAL: 1.0,  # 시간은 보통
        ClaimType.LOGICAL: 1.2,  # 추론도 중요
        ClaimType.COMPARATIVE: 0.8,  # 비교는 덜 중요
    }

    def __init__(self):
        """초기화"""
        pass

    def calculate(self, results: list[VerificationResult]) -> ConfidenceReport:
        """
        검증 결과로부터 신뢰도 점수 계산

        Args:
            results: 검증 결과 리스트

        Returns:
            신뢰도 리포트
        """
        if not results:
            return ConfidenceReport(
                grade=ConfidenceGrade.UNKNOWN,
                score=0.0,
                total_claims=0,
                verified_claims=0,
                partial_claims=0,
                unverified_claims=0,
                unable_claims=0,
                details=[],
            )

        # 상태별 집계
        verified = sum(1 for r in results if r.status == VerificationStatus.VERIFIED)
        partial = sum(1 for r in results if r.status == VerificationStatus.PARTIALLY_VERIFIED)
        unverified = sum(1 for r in results if r.status == VerificationStatus.UNVERIFIED)
        unable = sum(1 for r in results if r.status == VerificationStatus.UNABLE)

        # 가중 점수 계산
        total_weighted_score = 0.0
        total_weight = 0.0

        details = []

        for result in results:
            # 기본 점수
            base_score = self.STATUS_SCORES.get(result.status, 0.5)

            # 검증기 신뢰도 반영
            if result.confidence > 0:
                base_score = base_score * 0.7 + result.confidence * 0.3

            # 유형별 가중치
            weight = self.TYPE_WEIGHTS.get(result.claim.claim_type, 1.0)

            total_weighted_score += base_score * weight
            total_weight += weight

            # 상세 결과
            details.append(
                {
                    "claim_text": result.claim.text,
                    "claim_type": result.claim.claim_type.value,
                    "status": result.status.value,
                    "actual_value": result.actual_value,
                    "confidence": result.confidence,
                    "reason": result.reason,
                    "score": base_score,
                    "weight": weight,
                }
            )

        # 최종 점수 계산
        final_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

        # 등급 결정
        grade = self._determine_grade(final_score)

        return ConfidenceReport(
            grade=grade,
            score=round(final_score, 3),
            total_claims=len(results),
            verified_claims=verified,
            partial_claims=partial,
            unverified_claims=unverified,
            unable_claims=unable,
            details=details,
        )

    def _determine_grade(self, score: float) -> ConfidenceGrade:
        """점수로부터 등급 결정"""
        if score >= self.GRADE_THRESHOLDS[ConfidenceGrade.GREEN]:
            return ConfidenceGrade.GREEN
        elif score >= self.GRADE_THRESHOLDS[ConfidenceGrade.YELLOW]:
            return ConfidenceGrade.YELLOW
        else:
            return ConfidenceGrade.RED

    def get_badge_info(self, grade: ConfidenceGrade) -> dict[str, str]:
        """등급에 따른 배지 정보 반환"""
        badge_info = {
            ConfidenceGrade.GREEN: {
                "emoji": "🟢",
                "label": "높은 신뢰",
                "color": "#28a745",
                "description": "검증된 데이터에 기반한 응답입니다.",
            },
            ConfidenceGrade.YELLOW: {
                "emoji": "🟡",
                "label": "부분 신뢰",
                "color": "#ffc107",
                "description": "일부 정보의 정확성을 확인하세요.",
            },
            ConfidenceGrade.RED: {
                "emoji": "🔴",
                "label": "낮은 신뢰",
                "color": "#dc3545",
                "description": "정보 정확성에 주의가 필요합니다.",
            },
            ConfidenceGrade.UNKNOWN: {
                "emoji": "⚪",
                "label": "판단 불가",
                "color": "#6c757d",
                "description": "검증 가능한 주장이 없습니다.",
            },
        }
        return badge_info.get(grade, badge_info[ConfidenceGrade.UNKNOWN])

    def format_report(self, report: ConfidenceReport) -> str:
        """리포트를 사람이 읽기 쉬운 형태로 포맷"""
        badge = self.get_badge_info(report.grade)

        lines = [
            f"\n{badge['emoji']} **검증 결과: {badge['label']}** (점수: {report.score:.0%})",
            f"\n{badge['description']}",
            f"\n- 전체 주장: {report.total_claims}개",
            f"- 검증됨: {report.verified_claims}개",
            f"- 부분 검증: {report.partial_claims}개",
            f"- 검증 실패: {report.unverified_claims}개",
        ]

        if report.details:
            lines.append("\n**상세:**")
            for detail in report.details:
                status_emoji = {
                    "verified": "✅",
                    "partial": "🔶",
                    "unverified": "❌",
                    "unable": "⬜",
                }.get(detail["status"], "⬜")

                lines.append(f'- {status_emoji} "{detail["claim_text"]}" → {detail["reason"]}')

        return "\n".join(lines)

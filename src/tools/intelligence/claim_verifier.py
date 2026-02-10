"""
Claim Verifier
==============
추출된 주장(claim)을 검증

역할:
- NumericalVerifier: DB 쿼리로 숫자/순위/퍼센트 검증
- TemporalVerifier: 데이터 타임스탬프 검증
- LogicVerifier: 비즈니스 규칙 기반 추론 검증

연결 파일:
- tools/claim_extractor.py: 주장 추출
- core/verification_pipeline.py: 파이프라인 통합
- ontology/knowledge_graph.py: KG 데이터 조회
"""

import logging
import re
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from .claim_extractor import Claim, ClaimType

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """검증 상태"""

    VERIFIED = "verified"  # 검증됨 (일치)
    PARTIALLY_VERIFIED = "partial"  # 부분 검증 (근접)
    UNVERIFIED = "unverified"  # 검증 실패 (불일치)
    UNABLE = "unable"  # 검증 불가 (데이터 없음)


@dataclass
class VerificationResult:
    """검증 결과"""

    claim: Claim  # 원본 주장
    status: VerificationStatus  # 검증 상태
    actual_value: Any | None = None  # 실제 값
    confidence: float = 0.0  # 신뢰도 (0-1)
    reason: str = ""  # 검증 결과 설명
    source: str = ""  # 데이터 소스


class BaseVerifier(ABC):
    """검증기 기본 클래스"""

    @abstractmethod
    async def verify(self, claim: Claim, context: dict | None = None) -> VerificationResult:
        """주장 검증"""
        pass


class NumericalVerifier(BaseVerifier):
    """
    숫자 주장 검증기

    DB에서 실제 값을 조회하여 주장된 숫자와 비교
    """

    # 허용 오차 (%)
    TOLERANCE_PERCENT = 5.0

    # 순위 허용 오차 (위)
    TOLERANCE_RANK = 1

    def __init__(self, db_path: str | None = None):
        """
        Args:
            db_path: SQLite DB 경로 (None이면 기본 경로)
        """
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "data" / "amore_data.db"
        self.db_path = Path(db_path)

    async def verify(self, claim: Claim, context: dict | None = None) -> VerificationResult:
        """숫자 주장 검증"""
        if claim.claim_type != ClaimType.NUMERICAL:
            return VerificationResult(
                claim=claim, status=VerificationStatus.UNABLE, reason="Not a numerical claim"
            )

        # 단위에 따라 검증 방식 결정
        if claim.unit == "위":
            return await self._verify_rank(claim, context)
        elif claim.unit == "%":
            return await self._verify_percent(claim, context)
        else:
            return await self._verify_generic(claim, context)

    async def _verify_rank(self, claim: Claim, context: dict | None) -> VerificationResult:
        """순위 검증"""
        try:
            # 엔티티에서 브랜드/제품 추출
            brand = self._extract_brand(claim)
            category = self._extract_category(claim, context)

            if not brand:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.UNABLE,
                    reason="Cannot identify brand from claim",
                )

            # DB에서 실제 순위 조회
            actual_rank = self._query_rank(brand, category)

            if actual_rank is None:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.UNABLE,
                    reason=f"No rank data found for {brand}",
                )

            claimed_rank = int(claim.value)

            # 순위 비교
            if actual_rank == claimed_rank:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.VERIFIED,
                    actual_value=actual_rank,
                    confidence=1.0,
                    reason=f"Rank matches: {claimed_rank}위",
                    source="SQLite DB",
                )
            elif abs(actual_rank - claimed_rank) <= self.TOLERANCE_RANK:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.PARTIALLY_VERIFIED,
                    actual_value=actual_rank,
                    confidence=0.8,
                    reason=f"Rank close: claimed {claimed_rank}위, actual {actual_rank}위",
                    source="SQLite DB",
                )
            else:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.UNVERIFIED,
                    actual_value=actual_rank,
                    confidence=0.0,
                    reason=f"Rank mismatch: claimed {claimed_rank}위, actual {actual_rank}위",
                    source="SQLite DB",
                )

        except Exception as e:
            logger.error(f"Rank verification error: {e}")
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNABLE,
                reason=f"Verification error: {str(e)}",
            )

    async def _verify_percent(self, claim: Claim, context: dict | None) -> VerificationResult:
        """퍼센트 검증 (SoS 등)"""
        try:
            # SoS 검증인지 확인
            if "sos" in claim.text.lower() or "SoS" in claim.context if claim.context else "":
                return await self._verify_sos(claim, context)

            # 일반 퍼센트는 검증 불가
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNABLE,
                reason="Cannot verify generic percentage",
            )

        except Exception as e:
            logger.error(f"Percent verification error: {e}")
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNABLE,
                reason=f"Verification error: {str(e)}",
            )

    async def _verify_sos(self, claim: Claim, context: dict | None) -> VerificationResult:
        """SoS 검증"""
        try:
            brand = self._extract_brand(claim)
            category = self._extract_category(claim, context)

            if not brand:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.UNABLE,
                    reason="Cannot identify brand for SoS",
                )

            # DB에서 SoS 조회
            actual_sos = self._query_sos(brand, category)

            if actual_sos is None:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.UNABLE,
                    reason=f"No SoS data found for {brand}",
                )

            claimed_sos = float(claim.value)

            # 허용 오차 내 비교
            diff_percent = abs(actual_sos - claimed_sos) / max(actual_sos, 0.001) * 100

            if diff_percent <= self.TOLERANCE_PERCENT:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.VERIFIED,
                    actual_value=actual_sos,
                    confidence=1.0 - (diff_percent / 100),
                    reason=f"SoS matches: {claimed_sos}% (actual: {actual_sos:.1f}%)",
                    source="SQLite DB",
                )
            elif diff_percent <= self.TOLERANCE_PERCENT * 2:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.PARTIALLY_VERIFIED,
                    actual_value=actual_sos,
                    confidence=0.7,
                    reason=f"SoS close: claimed {claimed_sos}%, actual {actual_sos:.1f}%",
                    source="SQLite DB",
                )
            else:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.UNVERIFIED,
                    actual_value=actual_sos,
                    confidence=0.0,
                    reason=f"SoS mismatch: claimed {claimed_sos}%, actual {actual_sos:.1f}%",
                    source="SQLite DB",
                )

        except Exception as e:
            logger.error(f"SoS verification error: {e}")
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNABLE,
                reason=f"Verification error: {str(e)}",
            )

    async def _verify_generic(self, claim: Claim, context: dict | None) -> VerificationResult:
        """일반 숫자 검증"""
        return VerificationResult(
            claim=claim,
            status=VerificationStatus.UNABLE,
            reason="Generic number verification not supported",
        )

    def _extract_brand(self, claim: Claim) -> str | None:
        """주장에서 브랜드 추출"""
        if claim.entities:
            # 브랜드 패턴 매칭
            for entity in claim.entities:
                if any(
                    kw in entity.lower() for kw in ["laneige", "라네즈", "burt", "elf", "cerave"]
                ):
                    return entity
        return None

    def _extract_category(self, claim: Claim, context: dict | None) -> str | None:
        """주장에서 카테고리 추출"""
        if claim.entities:
            for entity in claim.entities:
                if any(kw in entity.lower() for kw in ["lip", "skin", "face", "powder", "beauty"]):
                    return entity

        if context and "category" in context:
            return context["category"]

        return None

    def _query_rank(self, brand: str, category: str | None) -> int | None:
        """DB에서 순위 조회"""
        if not self.db_path.exists():
            logger.warning(f"DB not found: {self.db_path}")
            return None

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # 최신 크롤링 데이터에서 순위 조회
            query = """
                SELECT rank
                FROM products
                WHERE LOWER(brand) LIKE ?
                ORDER BY crawled_at DESC
                LIMIT 1
            """
            brand_pattern = f"%{brand.lower()}%"
            cursor.execute(query, (brand_pattern,))
            result = cursor.fetchone()
            conn.close()

            return result[0] if result else None

        except Exception as e:
            logger.error(f"DB query error: {e}")
            return None

    def _query_sos(self, brand: str, category: str | None) -> float | None:
        """DB에서 SoS 조회"""
        if not self.db_path.exists():
            logger.warning(f"DB not found: {self.db_path}")
            return None

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # 최신 KPI 데이터에서 SoS 조회
            query = """
                SELECT sos
                FROM kpi_metrics
                WHERE LOWER(brand) LIKE ?
                ORDER BY date DESC
                LIMIT 1
            """
            brand_pattern = f"%{brand.lower()}%"
            cursor.execute(query, (brand_pattern,))
            result = cursor.fetchone()
            conn.close()

            return result[0] if result else None

        except Exception as e:
            logger.error(f"DB query error: {e}")
            return None


class TemporalVerifier(BaseVerifier):
    """
    시간 주장 검증기

    데이터의 최신성을 확인하여 시간 표현 검증
    """

    # 시간 표현 → 허용 기간 매핑
    TEMPORAL_MAPPING = {
        "today": timedelta(days=1),
        "yesterday": timedelta(days=2),
        "this_week": timedelta(days=7),
        "last_week": timedelta(days=14),
        "this_month": timedelta(days=31),
        "last_month": timedelta(days=62),
        "recent": timedelta(days=7),
        "current": timedelta(days=1),
    }

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "data" / "amore_data.db"
        self.db_path = Path(db_path)

    async def verify(self, claim: Claim, context: dict | None = None) -> VerificationResult:
        """시간 주장 검증"""
        if claim.claim_type != ClaimType.TEMPORAL:
            return VerificationResult(
                claim=claim, status=VerificationStatus.UNABLE, reason="Not a temporal claim"
            )

        try:
            # 데이터 최신성 확인
            last_crawled = self._get_last_crawled_time()

            if last_crawled is None:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.UNABLE,
                    reason="Cannot determine data freshness",
                )

            now = datetime.now()
            data_age = now - last_crawled

            # 시간 표현에 따른 허용 기간
            temporal_type = claim.value
            allowed_age = self.TEMPORAL_MAPPING.get(temporal_type, timedelta(days=7))

            if data_age <= allowed_age:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.VERIFIED,
                    actual_value=last_crawled.isoformat(),
                    confidence=1.0 - (data_age.total_seconds() / allowed_age.total_seconds()),
                    reason=f"Data is fresh: last crawled {data_age.days}d {data_age.seconds // 3600}h ago",
                    source="SQLite DB",
                )
            else:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.UNVERIFIED,
                    actual_value=last_crawled.isoformat(),
                    confidence=0.3,
                    reason=f"Data may be stale: last crawled {data_age.days} days ago",
                    source="SQLite DB",
                )

        except Exception as e:
            logger.error(f"Temporal verification error: {e}")
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNABLE,
                reason=f"Verification error: {str(e)}",
            )

    def _get_last_crawled_time(self) -> datetime | None:
        """마지막 크롤링 시간 조회"""
        if not self.db_path.exists():
            return None

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            query = """
                SELECT MAX(crawled_at)
                FROM products
            """
            cursor.execute(query)
            result = cursor.fetchone()
            conn.close()

            if result and result[0]:
                return datetime.fromisoformat(
                    result[0].replace("Z", "+00:00").replace("+00:00", "")
                )

            return None

        except Exception as e:
            logger.error(f"DB query error: {e}")
            return None


class LogicVerifier(BaseVerifier):
    """
    추론 주장 검증기

    비즈니스 규칙과 임계값을 기반으로 추론 표현 검증
    """

    # SoS 임계값
    SOS_THRESHOLDS = {
        "weak": (0, 5),  # 0-5%: 미미
        "moderate": (5, 10),  # 5-10%: 중간
        "strong": (10, 20),  # 10-20%: 강함
        "dominant": (20, 100),  # 20%+: 지배적
    }

    # 추론 표현 → 허용 SoS 레벨 매핑
    EXPRESSION_MAPPING = {
        "competitive_positive": ["strong", "dominant"],
        "position_positive": ["strong", "dominant"],
        "competitive_negative": ["weak"],
        "position_negative": ["weak", "moderate"],
        "neutral": ["moderate"],
    }

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "data" / "amore_data.db"
        self.db_path = Path(db_path)

    async def verify(self, claim: Claim, context: dict | None = None) -> VerificationResult:
        """추론 주장 검증"""
        if claim.claim_type != ClaimType.LOGICAL:
            return VerificationResult(
                claim=claim, status=VerificationStatus.UNABLE, reason="Not a logical claim"
            )

        try:
            expression_type = claim.value
            brand = self._extract_brand(claim)

            if not brand:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.UNABLE,
                    reason="Cannot identify brand for logical verification",
                )

            # SoS 조회
            actual_sos = self._query_sos(brand)

            if actual_sos is None:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.UNABLE,
                    reason=f"No SoS data found for {brand}",
                )

            # SoS 레벨 결정
            sos_level = self._get_sos_level(actual_sos)

            # 표현과 SoS 레벨 일치 확인
            allowed_levels = self.EXPRESSION_MAPPING.get(expression_type, [])

            if sos_level in allowed_levels:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.VERIFIED,
                    actual_value=actual_sos,
                    confidence=0.9,
                    reason=f"Expression valid: SoS {actual_sos:.1f}% is '{sos_level}'",
                    source="Business Rules",
                )
            else:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.UNVERIFIED,
                    actual_value=actual_sos,
                    confidence=0.2,
                    reason=f"Expression invalid: SoS {actual_sos:.1f}% ({sos_level}) doesn't match '{expression_type}'",
                    source="Business Rules",
                )

        except Exception as e:
            logger.error(f"Logic verification error: {e}")
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNABLE,
                reason=f"Verification error: {str(e)}",
            )

    def _extract_brand(self, claim: Claim) -> str | None:
        """주장에서 브랜드 추출"""
        if claim.entities:
            for entity in claim.entities:
                if any(
                    kw in entity.lower() for kw in ["laneige", "라네즈", "burt", "elf", "cerave"]
                ):
                    return entity

        # 문맥에서 추출 시도
        if claim.context:
            brand_patterns = [
                r"라네즈|LANEIGE",
                r"Burt'?s\s*Bees",
                r"e\.l\.f\.?",
            ]
            for pattern in brand_patterns:
                match = re.search(pattern, claim.context, re.IGNORECASE)
                if match:
                    return match.group(0)

        return None

    def _get_sos_level(self, sos: float) -> str:
        """SoS 값에 따른 레벨 결정"""
        for level, (min_val, max_val) in self.SOS_THRESHOLDS.items():
            if min_val <= sos < max_val:
                return level
        return "dominant" if sos >= 20 else "weak"

    def _query_sos(self, brand: str) -> float | None:
        """DB에서 SoS 조회"""
        if not self.db_path.exists():
            return None

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            query = """
                SELECT sos
                FROM kpi_metrics
                WHERE LOWER(brand) LIKE ?
                ORDER BY date DESC
                LIMIT 1
            """
            brand_pattern = f"%{brand.lower()}%"
            cursor.execute(query, (brand_pattern,))
            result = cursor.fetchone()
            conn.close()

            return result[0] if result else None

        except Exception as e:
            logger.error(f"DB query error: {e}")
            return None


class ClaimVerifier:
    """
    통합 검증기

    모든 유형의 주장을 적절한 검증기로 라우팅
    """

    def __init__(self, db_path: str | None = None):
        self.numerical_verifier = NumericalVerifier(db_path)
        self.temporal_verifier = TemporalVerifier(db_path)
        self.logic_verifier = LogicVerifier(db_path)

    async def verify(self, claim: Claim, context: dict | None = None) -> VerificationResult:
        """단일 주장 검증"""
        if claim.claim_type == ClaimType.NUMERICAL:
            return await self.numerical_verifier.verify(claim, context)
        elif claim.claim_type == ClaimType.TEMPORAL:
            return await self.temporal_verifier.verify(claim, context)
        elif claim.claim_type == ClaimType.LOGICAL:
            return await self.logic_verifier.verify(claim, context)
        else:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNABLE,
                reason=f"Unsupported claim type: {claim.claim_type}",
            )

    async def verify_all(
        self, claims: list[Claim], context: dict | None = None
    ) -> list[VerificationResult]:
        """모든 주장 검증"""
        results = []
        for claim in claims:
            result = await self.verify(claim, context)
            results.append(result)
        return results

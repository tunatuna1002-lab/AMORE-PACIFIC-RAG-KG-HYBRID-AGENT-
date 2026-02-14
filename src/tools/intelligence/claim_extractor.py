"""
Claim Extractor
===============
응답 텍스트에서 검증 가능한 주장(claim)을 추출

역할:
- 숫자/순위/퍼센트 추출 (정규식)
- 시간 표현 추출 ("오늘", "최근", "이번 주")
- 추론 표현 추출 ("경쟁력 있음", "우위", "약함")

연결 파일:
- tools/claim_verifier.py: 추출된 주장 검증
- core/verification_pipeline.py: 파이프라인 통합
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ClaimType(Enum):
    """주장 유형"""

    NUMERICAL = "numerical"  # 숫자, 순위, 퍼센트
    TEMPORAL = "temporal"  # 시간 표현
    LOGICAL = "logical"  # 추론/평가 표현
    COMPARATIVE = "comparative"  # 비교 표현


@dataclass
class Claim:
    """검증 가능한 주장"""

    text: str  # 원본 텍스트
    claim_type: ClaimType  # 주장 유형
    value: Any | None = None  # 추출된 값
    unit: str | None = None  # 단위 (%, 위, 개 등)
    entities: list[str] = field(default_factory=list)  # 관련 엔티티
    context: str | None = None  # 주변 문맥
    position: int = 0  # 텍스트 내 위치


class ClaimExtractor:
    """
    응답 텍스트에서 검증 가능한 주장 추출

    Usage:
        extractor = ClaimExtractor()
        claims = extractor.extract("LANEIGE는 3위이며 SoS 12.5%입니다")
    """

    # 숫자 패턴
    NUMERICAL_PATTERNS = [
        # 순위: "3위", "1등", "TOP 5"
        (r"(\d+)\s*위", "rank"),
        (r"(\d+)\s*등", "rank"),
        (r"[Tt][Oo][Pp]\s*(\d+)", "rank"),
        # 퍼센트: "12.5%", "10퍼센트"
        (r"(\d+\.?\d*)\s*%", "percent"),
        (r"(\d+\.?\d*)\s*퍼센트", "percent"),
        # SoS/HHI/CPI 지표: "SoS 12.5%", "HHI 0.15"
        (r"[Ss][Oo][Ss]\s*[:=]?\s*(\d+\.?\d*)\s*%?", "sos"),
        (r"[Hh][Hh][Ii]\s*[:=]?\s*(\d+\.?\d*)", "hhi"),
        (r"[Cc][Pp][Ii]\s*[:=]?\s*(\d+\.?\d*)", "cpi"),
        # 가격: "$29.99", "29,900원"
        (r"\$\s*(\d+\.?\d*)", "price_usd"),
        (r"(\d{1,3}(?:,\d{3})*)\s*원", "price_krw"),
        # 일반 숫자: "100개", "50명"
        (r"(\d+)\s*개", "count"),
        (r"(\d+)\s*명", "count"),
    ]

    # 시간 표현 패턴
    TEMPORAL_PATTERNS = [
        (r"오늘", "today"),
        (r"어제", "yesterday"),
        (r"이번\s*주", "this_week"),
        (r"지난\s*주", "last_week"),
        (r"이번\s*달", "this_month"),
        (r"지난\s*달", "last_month"),
        (r"최근", "recent"),
        (r"현재", "current"),
        (r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})", "date"),
        (r"(\d{1,2})월\s*(\d{1,2})일", "date_kr"),
    ]

    # 추론/평가 표현 패턴
    LOGICAL_PATTERNS = [
        # 긍정적 평가
        (r"경쟁력\s*(있|높|강)", "competitive_positive"),
        (r"(강세|우위|선두|지배)", "position_positive"),
        (r"(성장|상승|증가|개선)", "trend_positive"),
        # 부정적 평가
        (r"경쟁력\s*(없|낮|약)", "competitive_negative"),
        (r"(약세|열위|후발|미미)", "position_negative"),
        (r"(하락|감소|악화|저조)", "trend_negative"),
        # 중립적 평가
        (r"(보통|중간|평균|유지)", "neutral"),
    ]

    # 비교 표현 패턴
    COMPARATIVE_PATTERNS = [
        (r"보다\s*(높|낮|많|적|크|작)", "comparison"),
        (r"대비\s*(\d+\.?\d*)\s*%?\s*(증가|감소|상승|하락)", "change"),
        (r"(\d+\.?\d*)\s*배", "multiple"),
    ]

    # 브랜드 패턴 (엔티티 추출용)
    BRAND_PATTERNS = [
        r"라네즈|LANEIGE",
        r"설화수|Sulwhasoo",
        r"이니스프리|Innisfree",
        r"에뛰드|Etude",
        r"Burt'?s\s*Bees",
        r"e\.l\.f\.?",
        r"CeraVe",
        r"Neutrogena",
        r"Maybelline",
        r"L'Oreal|로레알",
        r"Clinique|클리니크",
    ]

    # 카테고리 패턴
    CATEGORY_PATTERNS = [
        r"Lip\s*Care|립\s*케어",
        r"Lip\s*Makeup|립\s*메이크업",
        r"Skin\s*Care|스킨\s*케어",
        r"Face\s*Powder|파우더",
        r"Beauty|뷰티",
    ]

    def __init__(self):
        """초기화"""
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """정규식 패턴 컴파일"""
        self._numerical_compiled = [
            (re.compile(pattern, re.IGNORECASE), label)
            for pattern, label in self.NUMERICAL_PATTERNS
        ]
        self._temporal_compiled = [
            (re.compile(pattern, re.IGNORECASE), label) for pattern, label in self.TEMPORAL_PATTERNS
        ]
        self._logical_compiled = [
            (re.compile(pattern, re.IGNORECASE), label) for pattern, label in self.LOGICAL_PATTERNS
        ]
        self._comparative_compiled = [
            (re.compile(pattern, re.IGNORECASE), label)
            for pattern, label in self.COMPARATIVE_PATTERNS
        ]
        self._brand_compiled = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.BRAND_PATTERNS
        ]
        self._category_compiled = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.CATEGORY_PATTERNS
        ]

    def extract(self, response: str) -> list[Claim]:
        """
        응답 텍스트에서 주장 추출

        Args:
            response: 응답 텍스트

        Returns:
            추출된 주장 리스트
        """
        claims = []

        # 숫자 주장 추출
        claims.extend(self._extract_numerical(response))

        # 시간 주장 추출
        claims.extend(self._extract_temporal(response))

        # 추론 주장 추출
        claims.extend(self._extract_logical(response))

        # 비교 주장 추출
        claims.extend(self._extract_comparative(response))

        # 엔티티 연결
        self._link_entities(claims, response)

        # 위치순 정렬
        claims.sort(key=lambda c: c.position)

        logger.info(f"Extracted {len(claims)} claims from response")
        return claims

    def _extract_numerical(self, text: str) -> list[Claim]:
        """숫자 주장 추출"""
        claims = []

        for pattern, label in self._numerical_compiled:
            for match in pattern.finditer(text):
                value = match.group(1) if match.groups() else match.group(0)

                # 숫자 변환
                try:
                    if "." in value or label in ["percent", "sos", "hhi", "cpi", "price_usd"]:
                        value = float(value.replace(",", ""))
                    else:
                        value = int(value.replace(",", ""))
                except ValueError:
                    continue

                # 단위 결정
                unit = self._get_unit(label)

                # 문맥 추출 (전후 50자)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]

                claim = Claim(
                    text=match.group(0),
                    claim_type=ClaimType.NUMERICAL,
                    value=value,
                    unit=unit,
                    context=context,
                    position=match.start(),
                )
                claims.append(claim)

        return claims

    def _extract_temporal(self, text: str) -> list[Claim]:
        """시간 주장 추출"""
        claims = []

        for pattern, label in self._temporal_compiled:
            for match in pattern.finditer(text):
                # 문맥 추출
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]

                claim = Claim(
                    text=match.group(0),
                    claim_type=ClaimType.TEMPORAL,
                    value=label,
                    context=context,
                    position=match.start(),
                )
                claims.append(claim)

        return claims

    def _extract_logical(self, text: str) -> list[Claim]:
        """추론 주장 추출"""
        claims = []

        for pattern, label in self._logical_compiled:
            for match in pattern.finditer(text):
                # 문맥 추출
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]

                claim = Claim(
                    text=match.group(0),
                    claim_type=ClaimType.LOGICAL,
                    value=label,
                    context=context,
                    position=match.start(),
                )
                claims.append(claim)

        return claims

    def _extract_comparative(self, text: str) -> list[Claim]:
        """비교 주장 추출"""
        claims = []

        for pattern, _label in self._comparative_compiled:
            for match in pattern.finditer(text):
                value = None
                if match.groups():
                    try:
                        value = float(match.group(1))
                    except (ValueError, IndexError):
                        logger.warning("Suppressed Exception", exc_info=True)

                # 문맥 추출
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]

                claim = Claim(
                    text=match.group(0),
                    claim_type=ClaimType.COMPARATIVE,
                    value=value,
                    context=context,
                    position=match.start(),
                )
                claims.append(claim)

        return claims

    def _link_entities(self, claims: list[Claim], text: str) -> None:
        """주장에 관련 엔티티 연결"""
        # 텍스트에서 브랜드 찾기
        brands = set()
        for pattern in self._brand_compiled:
            for match in pattern.finditer(text):
                brands.add(match.group(0))

        # 텍스트에서 카테고리 찾기
        categories = set()
        for pattern in self._category_compiled:
            for match in pattern.finditer(text):
                categories.add(match.group(0))

        # 각 주장에 엔티티 연결 (문맥 기반)
        for claim in claims:
            if claim.context:
                # 문맥에서 브랜드 찾기
                for brand in brands:
                    if brand.lower() in claim.context.lower():
                        claim.entities.append(brand)

                # 문맥에서 카테고리 찾기
                for category in categories:
                    if category.lower() in claim.context.lower():
                        claim.entities.append(category)

    def _get_unit(self, label: str) -> str | None:
        """라벨에서 단위 결정"""
        unit_map = {
            "rank": "위",
            "percent": "%",
            "sos": "%",
            "hhi": "",
            "cpi": "",
            "price_usd": "USD",
            "price_krw": "KRW",
            "count": "개",
        }
        return unit_map.get(label)

    def get_summary(self, claims: list[Claim]) -> dict[str, Any]:
        """주장 요약 생성"""
        summary = {
            "total": len(claims),
            "by_type": {},
            "entities": set(),
        }

        for claim in claims:
            claim_type = claim.claim_type.value
            if claim_type not in summary["by_type"]:
                summary["by_type"][claim_type] = 0
            summary["by_type"][claim_type] += 1

            for entity in claim.entities:
                summary["entities"].add(entity)

        summary["entities"] = list(summary["entities"])
        return summary

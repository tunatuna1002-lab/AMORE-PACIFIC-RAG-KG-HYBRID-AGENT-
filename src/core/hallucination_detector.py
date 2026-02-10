"""
Hallucination Detector
======================
LLM 응답의 근거 기반 검증 (Groundedness Check)

LangSmith 4차원 평가 중 Groundedness 차원을 런타임에 적용.

2단계 검사:
1. 빠른 휴리스틱: 응답 내 숫자/브랜드명이 컨텍스트에 존재하는지
2. LLM 검사 (score < 0.9인 경우만): gpt-4.1-mini로 groundedness 점수 산출
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GroundednessResult:
    """검증 결과"""

    is_grounded: bool
    score: float  # 0.0 ~ 1.0
    method: str  # "heuristic" or "llm"
    details: dict[str, Any] = field(default_factory=dict)


class HallucinationDetector:
    """
    응답 환각 감지기

    Usage:
        detector = HallucinationDetector()
        result = await detector.check(response_text, context_text)
        if not result.is_grounded:
            # 환각 경고 플래그
    """

    # 근거 판정 임계값
    GROUNDED_THRESHOLD = 0.6

    def __init__(self, model: str = "gpt-4.1-mini", threshold: float = 0.6):
        self.model = model
        self.threshold = threshold
        self._stats = {
            "total_checks": 0,
            "heuristic_only": 0,
            "llm_checks": 0,
            "hallucinations_detected": 0,
        }

    async def check(
        self,
        response_text: str,
        context_text: str,
    ) -> GroundednessResult:
        """
        응답 근거 검증

        Args:
            response_text: LLM 생성 응답
            context_text: 컨텍스트 (RAG + KG)

        Returns:
            GroundednessResult
        """
        self._stats["total_checks"] += 1

        if not response_text or not context_text:
            return GroundednessResult(is_grounded=True, score=1.0, method="skip")

        # Step 1: 빠른 휴리스틱 검사
        heuristic_score, heuristic_details = self._heuristic_check(response_text, context_text)

        if heuristic_score >= 0.9:
            # 높은 점수 → LLM 검사 스킵
            self._stats["heuristic_only"] += 1
            return GroundednessResult(
                is_grounded=True,
                score=heuristic_score,
                method="heuristic",
                details=heuristic_details,
            )

        # Step 2: LLM 기반 검사 (heuristic 점수 낮을 때만)
        try:
            self._stats["llm_checks"] += 1
            llm_score = await self._llm_check(response_text, context_text)

            # 최종 점수: heuristic과 LLM의 가중 평균
            final_score = 0.3 * heuristic_score + 0.7 * llm_score
            is_grounded = final_score >= self.threshold

            if not is_grounded:
                self._stats["hallucinations_detected"] += 1
                logger.warning(
                    f"Hallucination detected: score={final_score:.2f} "
                    f"(heuristic={heuristic_score:.2f}, llm={llm_score:.2f})"
                )

            return GroundednessResult(
                is_grounded=is_grounded,
                score=round(final_score, 4),
                method="llm",
                details={
                    **heuristic_details,
                    "heuristic_score": heuristic_score,
                    "llm_score": llm_score,
                },
            )

        except Exception as e:
            logger.error(f"LLM hallucination check failed: {e}")
            # LLM 실패 시 heuristic만으로 판단
            is_grounded = heuristic_score >= self.threshold
            return GroundednessResult(
                is_grounded=is_grounded,
                score=heuristic_score,
                method="heuristic_fallback",
                details={**heuristic_details, "llm_error": str(e)},
            )

    def _heuristic_check(self, response_text: str, context_text: str) -> tuple[float, dict]:
        """
        빠른 휴리스틱 검사

        검사 항목:
        1. 숫자 일치: 응답의 숫자가 컨텍스트에 존재하는지
        2. 브랜드명 일치: 응답의 브랜드가 컨텍스트에 존재하는지
        3. 핵심 용어 일치: 응답의 주요 용어가 컨텍스트에 존재하는지

        Returns:
            (score, details)
        """
        checks = {}
        scores = []

        # 1. 숫자 일치 검사
        response_numbers = set(re.findall(r"\d+\.?\d*%?", response_text))
        context_numbers = set(re.findall(r"\d+\.?\d*%?", context_text))

        if response_numbers:
            matched = response_numbers & context_numbers
            number_score = len(matched) / len(response_numbers)
            checks["numbers"] = {
                "total": len(response_numbers),
                "matched": len(matched),
                "score": round(number_score, 2),
            }
            scores.append(number_score)

        # 2. 브랜드명 일치 검사
        known_brands = [
            "laneige",
            "cosrx",
            "tirtir",
            "rare beauty",
            "innisfree",
            "sulwhasoo",
            "hera",
            "etude",
            "라네즈",
            "코스알엑스",
        ]
        response_lower = response_text.lower()
        context_lower = context_text.lower()

        response_brands = [b for b in known_brands if b in response_lower]
        if response_brands:
            context_brands = [b for b in response_brands if b in context_lower]
            brand_score = len(context_brands) / len(response_brands)
            checks["brands"] = {
                "total": len(response_brands),
                "matched": len(context_brands),
                "score": round(brand_score, 2),
            }
            scores.append(brand_score)

        # 3. 핵심 용어 일치 (3글자 이상 한글 단어)
        response_terms = set(re.findall(r"[가-힣]{3,}", response_text))
        if response_terms:
            context_terms = set(re.findall(r"[가-힣]{3,}", context_text))
            matched_terms = response_terms & context_terms
            if response_terms:
                term_score = min(len(matched_terms) / max(len(response_terms) * 0.3, 1), 1.0)
                checks["terms"] = {
                    "total": len(response_terms),
                    "matched": len(matched_terms),
                    "score": round(term_score, 2),
                }
                scores.append(term_score)

        # 종합 점수
        if not scores:
            return 1.0, {"checks": checks, "note": "no checkable elements"}

        final_score = sum(scores) / len(scores)
        return round(final_score, 4), {"checks": checks}

    async def _llm_check(self, response_text: str, context_text: str) -> float:
        """LLM 기반 groundedness 검사"""
        from litellm import acompletion

        prompt = f"""다음 응답이 주어진 컨텍스트에 근거하는지 평가하세요.

## 컨텍스트
{context_text[:3000]}

## 응답
{response_text[:1500]}

## 평가 기준
- 응답의 사실적 주장이 컨텍스트에서 뒷받침되는지 확인
- 컨텍스트에 없는 숫자나 통계가 응답에 포함되었는지 확인
- 과도한 일반화나 근거 없는 결론이 있는지 확인

0.0(완전 환각)~1.0(완전 근거 기반) 사이의 점수만 반환하세요.
숫자만 응답하세요.

점수:"""

        response = await acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )

        score_text = response.choices[0].message.content.strip()
        # 숫자 추출
        match = re.search(r"(\d+\.?\d*)", score_text)
        if match:
            score = float(match.group(1))
            return min(max(score, 0.0), 1.0)

        return 0.5  # 파싱 실패 시 기본값

    def get_stats(self) -> dict[str, Any]:
        return self._stats

"""
인사이트 검증기 (Insight Verifier)
==================================
LLM 기반 인사이트 최종 검증 파이프라인

## 기능
- 보고서 내 순위 비교 검증 (동일 카테고리 내에서만)
- 브랜드명/제품명 정확성 검증
- 수치 데이터 교차 검증
- 논리적 일관성 검증
- 외부 소스 (Tavily) 활용한 팩트체크

## 사용 예
```python
verifier = InsightVerifier()
result = await verifier.verify_report(report_content, analysis_data)
if result.has_issues:
    corrected = await verifier.auto_correct(report_content, result.issues)
```
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from litellm import acompletion

logger = logging.getLogger(__name__)


@dataclass
class VerificationIssue:
    """검증 이슈"""

    severity: str  # "critical", "warning", "info"
    category: str  # "rank_comparison", "brand_name", "data_accuracy", "logic"
    description: str
    location: str  # 보고서 내 위치
    original_text: str
    suggested_fix: str = ""


@dataclass
class VerificationResult:
    """검증 결과"""

    verified_at: str
    total_checks: int
    passed_checks: int
    issues: list[VerificationIssue] = field(default_factory=list)
    confidence_score: float = 1.0  # 0-1, 검증 신뢰도

    @property
    def has_critical_issues(self) -> bool:
        return any(i.severity == "critical" for i in self.issues)

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified_at": self.verified_at,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "issues": [
                {
                    "severity": i.severity,
                    "category": i.category,
                    "description": i.description,
                    "location": i.location,
                    "original_text": i.original_text,
                    "suggested_fix": i.suggested_fix,
                }
                for i in self.issues
            ],
            "confidence_score": self.confidence_score,
            "has_critical_issues": self.has_critical_issues,
        }


class InsightVerifier:
    """
    인사이트 검증기

    보고서의 정확성을 다각도로 검증하고 필요시 자동 수정합니다.
    """

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model
        self._tavily_client = None

    async def verify_report(
        self, report_content: str, analysis_data: dict[str, Any], raw_data: list[dict] | None = None
    ) -> VerificationResult:
        """
        보고서 종합 검증

        Args:
            report_content: 마크다운 보고서 텍스트
            analysis_data: PeriodAnalysis 데이터
            raw_data: 원본 크롤링 데이터 (선택)

        Returns:
            VerificationResult
        """
        issues = []
        checks = 0

        # 1. 순위 비교 검증 (카테고리 간 비교 감지)
        checks += 1
        rank_issues = await self._verify_rank_comparisons(report_content, analysis_data)
        issues.extend(rank_issues)
        logger.info(f"Rank verification: {len(rank_issues)} issues")

        # 2. 브랜드명 정확성 검증
        checks += 1
        brand_issues = self._verify_brand_names(report_content, analysis_data)
        issues.extend(brand_issues)
        logger.info(f"Brand verification: {len(brand_issues)} issues")

        # 3. 수치 데이터 교차 검증
        checks += 1
        data_issues = self._verify_numeric_data(report_content, analysis_data)
        issues.extend(data_issues)
        logger.info(f"Data verification: {len(data_issues)} issues")

        # 4. 논리적 일관성 검증
        checks += 1
        logic_issues = await self._verify_logic_consistency(report_content)
        issues.extend(logic_issues)
        logger.info(f"Logic verification: {len(logic_issues)} issues")

        # 신뢰도 점수 계산
        passed = checks - len([i for i in issues if i.severity == "critical"])
        confidence = passed / checks if checks > 0 else 1.0

        return VerificationResult(
            verified_at=datetime.now().isoformat(),
            total_checks=checks,
            passed_checks=passed,
            issues=issues,
            confidence_score=confidence,
        )

    async def _verify_rank_comparisons(
        self, content: str, analysis_data: dict[str, Any]
    ) -> list[VerificationIssue]:
        """순위 비교 검증 - 다른 카테고리 간 비교 감지"""
        issues = []

        # 순위 변동 패턴 감지 (예: "4위에서 71위로")
        rank_pattern = (
            r"(\d+)\s*위에서\s*(\d+)\s*위로|(\d+)위\s*→\s*(\d+)위|순위:\s*(\d+)\s*→\s*(\d+)"
        )
        matches = re.findall(rank_pattern, content)

        for match in matches:
            # 튜플에서 유효한 값 추출
            nums = [int(n) for n in match if n]
            if len(nums) >= 2:
                start_rank, end_rank = nums[0], nums[1]
                change = end_rank - start_rank

                # 급격한 변동 (30위 이상) 감지 - 카테고리 혼동 가능성
                if abs(change) > 30:
                    # 컨텍스트에서 카테고리 정보 확인
                    context_pattern = rf".{{0,100}}{start_rank}\s*위.{{0,100}}"
                    context_match = re.search(context_pattern, content)
                    context = context_match.group(0) if context_match else ""

                    # 카테고리가 명시되지 않으면 경고
                    categories = ["Lip Care", "Skin Care", "Beauty", "Face Powder", "Lip Makeup"]
                    has_category = any(cat.lower() in context.lower() for cat in categories)

                    if not has_category:
                        issues.append(
                            VerificationIssue(
                                severity="critical",
                                category="rank_comparison",
                                description=f"순위 변동 {start_rank}→{end_rank} ({change:+d})가 비정상적으로 큽니다. "
                                f"서로 다른 카테고리 순위를 비교한 것일 수 있습니다.",
                                location=context[:100],
                                original_text=f"{start_rank}위에서 {end_rank}위로",
                                suggested_fix="동일 카테고리 내 순위만 비교하세요. "
                                "예: [Lip Care] 4위 → 6위",
                            )
                        )

        return issues

    def _verify_brand_names(
        self, content: str, analysis_data: dict[str, Any]
    ) -> list[VerificationIssue]:
        """브랜드명 정확성 검증"""
        issues = []

        # 잘린 브랜드명 목록
        truncated_brands = {
            "Burt's": "Burt's Bees",
            "wet": "wet n wild",
            "Tree": "Tree Hut",
            "Summer": "Summer Fridays",
            "Rare": "Rare Beauty",
            "La": "La Roche-Posay",
            "Beauty": "Beauty of Joseon",
            "Tower": "Tower 28",
            "Drunk": "Drunk Elephant",
            "Paula's": "Paula's Choice",
            "The": "The Ordinary",
            "Glow": "Glow Recipe",
            "Youth": "Youth To The People",
        }

        for truncated, full in truncated_brands.items():
            # 단독으로 사용된 잘린 브랜드명 찾기
            pattern = rf"\b{re.escape(truncated)}\b(?!\s*Bees|\s*n\s*wild|\s*Hut|\s*Fridays|\s*Beauty|\s*Roche|\s*of\s*Joseon|\s*28|\s*Elephant|\s*Choice|\s*Ordinary|\s*Recipe|\s*To)"
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(
                    VerificationIssue(
                        severity="warning",
                        category="brand_name",
                        description=f"'{truncated}'는 잘린 브랜드명일 수 있습니다.",
                        location=f"브랜드: {truncated}",
                        original_text=truncated,
                        suggested_fix=f"정식 브랜드명 '{full}' 사용 권장",
                    )
                )

        # Unknown 브랜드 경고 (강화된 검증)
        # 인사이트에서 "Unknown" 표현은 절대 금지
        unknown_patterns = [
            r"\bUnknown\b",
            r"\b미확인\s*브랜드\b",
            r"\b기타\s*브랜드\s*\(Unknown\)",
        ]
        unknown_matches = []
        for pattern in unknown_patterns:
            unknown_matches.extend(re.findall(pattern, content, re.IGNORECASE))

        if unknown_matches:
            issues.append(
                VerificationIssue(
                    severity="error",  # warning → error로 강화
                    category="brand_name",
                    description=f"금지된 표현 '{unknown_matches[0]}'이(가) {len(unknown_matches)}회 발견되었습니다.",
                    location="전체",
                    original_text=unknown_matches[0],
                    suggested_fix="'Unknown/미확인 브랜드' 표현 절대 금지. 분석에서 제외하거나 '소규모 브랜드(점유율 1% 미만 합산)'으로 대체",
                )
            )

        return issues

    def _verify_numeric_data(
        self, content: str, analysis_data: dict[str, Any]
    ) -> list[VerificationIssue]:
        """수치 데이터 교차 검증"""
        issues = []

        # SoS 값 추출 및 검증
        sos_pattern = r"SoS[:\s]*(\d+\.?\d*)%"
        sos_matches = re.findall(sos_pattern, content)

        for sos_str in sos_matches:
            sos_val = float(sos_str)
            # SoS > 50%는 비정상적
            if sos_val > 50:
                issues.append(
                    VerificationIssue(
                        severity="warning",
                        category="data_accuracy",
                        description=f"SoS {sos_val}%가 비정상적으로 높습니다.",
                        location=f"SoS: {sos_val}%",
                        original_text=f"{sos_val}%",
                        suggested_fix="Amazon Top 100 기준 SoS는 일반적으로 0-20% 범위입니다.",
                    )
                )

        # HHI 값 검증
        hhi_pattern = r"HHI[:\s]*(\d+\.?\d*)"
        hhi_matches = re.findall(hhi_pattern, content)

        for hhi_str in hhi_matches:
            hhi_val = float(hhi_str)
            # HHI > 10000은 불가능 (이론적 최대값)
            if hhi_val > 10000:
                issues.append(
                    VerificationIssue(
                        severity="critical",
                        category="data_accuracy",
                        description=f"HHI {hhi_val}는 불가능한 값입니다 (최대 10000).",
                        location=f"HHI: {hhi_val}",
                        original_text=f"{hhi_val}",
                        suggested_fix="HHI 계산을 확인하세요.",
                    )
                )

        return issues

    async def _verify_logic_consistency(self, content: str) -> list[VerificationIssue]:
        """LLM 기반 논리적 일관성 검증"""
        issues = []

        prompt = f"""다음 보고서의 논리적 일관성을 검증하세요.

## 검증 항목
1. 순위 변동 분석이 동일 카테고리 내에서 이루어졌는지
2. SoS 상승/하락 해석이 수치와 일치하는지
3. 결론이 데이터와 일치하는지
4. 서로 모순되는 진술이 있는지

## 보고서 (일부)
{content[:3000]}

## 응답 형식 (JSON)
{{
  "issues": [
    {{
      "description": "문제 설명",
      "severity": "critical|warning|info",
      "location": "문제 위치",
      "suggestion": "수정 제안"
    }}
  ],
  "is_consistent": true/false
}}

문제가 없으면 "issues": []로 응답하세요."""

        try:
            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0,
            )

            result_text = response.choices[0].message.content.strip()

            # JSON 파싱
            json_match = re.search(r"\{[\s\S]*\}", result_text)
            if json_match:
                result = json.loads(json_match.group())
                for issue_data in result.get("issues", []):
                    issues.append(
                        VerificationIssue(
                            severity=issue_data.get("severity", "warning"),
                            category="logic",
                            description=issue_data.get("description", ""),
                            location=issue_data.get("location", ""),
                            original_text="",
                            suggested_fix=issue_data.get("suggestion", ""),
                        )
                    )

        except Exception as e:
            logger.error(f"Logic verification failed: {e}")

        return issues

    async def auto_correct(self, content: str, issues: list[VerificationIssue]) -> str:
        """이슈 자동 수정"""
        corrected = content

        for issue in issues:
            if issue.severity == "critical" and issue.suggested_fix:
                # 간단한 텍스트 교체
                if issue.original_text and issue.suggested_fix:
                    corrected = corrected.replace(
                        issue.original_text,
                        f"{issue.original_text} [수정 필요: {issue.suggested_fix}]",
                    )

        return corrected

    async def verify_with_tavily(self, claims: list[str]) -> dict[str, Any]:
        """Tavily API를 활용한 팩트체크 (외부 검증)"""
        try:
            from src.tools.tavily_search import TavilySearchClient

            if self._tavily_client is None:
                self._tavily_client = TavilySearchClient()

            results = {}
            for claim in claims[:5]:  # 최대 5개 검증
                search_results = await self._tavily_client.search(
                    query=claim, max_results=3, days=30
                )
                results[claim] = {
                    "verified": len(search_results) > 0,
                    "sources": [r.url for r in search_results[:3]],
                }
                await asyncio.sleep(0.3)

            return results

        except Exception as e:
            logger.error(f"Tavily verification failed: {e}")
            return {}


# 편의 함수
async def verify_insight_report(
    report_content: str, analysis_data: dict[str, Any]
) -> VerificationResult:
    """인사이트 보고서 검증 편의 함수"""
    verifier = InsightVerifier()
    return await verifier.verify_report(report_content, analysis_data)

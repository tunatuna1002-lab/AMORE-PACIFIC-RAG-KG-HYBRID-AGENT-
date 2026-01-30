"""
기간별 인사이트 에이전트
======================
증권사/리서치 기관 수준의 산업 분석 인사이트 생성

## 보고서 구조
1. Executive Summary - 핵심 요약 (3줄)
2. LANEIGE 심층 분석 - 타겟 브랜드 분석
3. 경쟁 환경 분석 - 경쟁사 동향
4. 시장 동향 - HHI, 트렌드
5. 외부 신호 분석 - TikTok/Reddit/뷰티전문지
6. 리스크 및 기회 요인
7. 전략 제언

## 사용 예시
```python
from src.agents.period_insight_agent import PeriodInsightAgent
from src.tools.period_analyzer import PeriodAnalyzer

# 기간 분석
analyzer = PeriodAnalyzer()
analysis = analyzer.analyze_period(start_date, end_date)

# 인사이트 생성
agent = PeriodInsightAgent()
report = await agent.generate_report(analysis, external_signals=signals)
```
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from litellm import acompletion

from src.monitoring.logger import AgentLogger
from src.tools.insight_formatter import format_insight

if TYPE_CHECKING:
    from src.tools.insight_verifier import VerificationResult

logger = AgentLogger(__name__)


@dataclass
class SectionInsight:
    """섹션별 인사이트"""

    section_id: str
    section_title: str
    content: str
    key_points: list[str] = field(default_factory=list)
    data_highlights: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "section_id": self.section_id,
            "section_title": self.section_title,
            "content": self.content,
            "key_points": self.key_points,
            "data_highlights": self.data_highlights,
        }


@dataclass
class PeriodReport:
    """기간별 분석 보고서"""

    start_date: str
    end_date: str
    generated_at: str

    executive_summary: SectionInsight | None = None
    laneige_analysis: SectionInsight | None = None
    competitive_analysis: SectionInsight | None = None
    market_trends: SectionInsight | None = None
    external_signals: SectionInsight | None = None
    risks_opportunities: SectionInsight | None = None
    strategic_recommendations: SectionInsight | None = None
    references: SectionInsight | None = None  # 참고자료 섹션 추가

    # 메타데이터
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "generated_at": self.generated_at,
            "executive_summary": self.executive_summary.to_dict()
            if self.executive_summary
            else None,
            "laneige_analysis": self.laneige_analysis.to_dict() if self.laneige_analysis else None,
            "competitive_analysis": self.competitive_analysis.to_dict()
            if self.competitive_analysis
            else None,
            "market_trends": self.market_trends.to_dict() if self.market_trends else None,
            "external_signals": self.external_signals.to_dict() if self.external_signals else None,
            "risks_opportunities": self.risks_opportunities.to_dict()
            if self.risks_opportunities
            else None,
            "strategic_recommendations": self.strategic_recommendations.to_dict()
            if self.strategic_recommendations
            else None,
            "metadata": self.metadata,
        }


class PeriodInsightAgent:
    """
    기간별 인사이트 생성 에이전트

    증권사 애널리스트 수준의 전문성으로 기간별 분석 데이터를 해석하고
    실행 가능한 전략 제언을 생성합니다.

    특징:
    - 데이터 기반의 객관적 분석
    - 구체적 수치 인용 필수
    - 실행 가능한 액션 제시
    - 한국어 보고서 + 영문 용어 병기
    """

    # LLM 설정
    MODEL = "gpt-4.1-mini"
    MAX_TOKENS = 2000
    TEMPERATURE = 0.7

    # 시스템 프롬프트 템플릿 (날짜 컨텍스트 동적 주입)
    SYSTEM_PROMPT_TEMPLATE = """당신은 화장품 산업 전문 애널리스트입니다.

### ⏰ 시점 정보 (매우 중요!)
- 오늘 날짜: {current_date}
- 분석 기간: {start_date} ~ {end_date}
- "현재", "최근" 등의 표현은 반드시 분석 종료일({end_date}) 기준

⚠️ 날짜 관련 필수 규칙:
- 분석 기간 외의 날짜(예: 2024년 6월)는 절대 언급 금지
- 미래 날짜는 언급하지 않음
- "현재 시점", "최근" 등은 {end_date} 기준으로 해석
- 모든 시점 언급은 위 날짜 정보 기준으로 작성

역할:
- 아모레퍼시픽 LANEIGE 브랜드의 Amazon US 시장 경쟁력 분석
- 증권사 리서치 보고서 수준의 전문성과 구체성 유지
- 데이터 기반의 객관적 분석과 실행 가능한 전략 제언

분석 프레임워크:
- SoS (Share of Shelf): 브랜드 점유율 (Top 100 내 제품 비율)
- HHI (Herfindahl-Hirschman Index): 시장 집중도 (낮을수록 경쟁적)
- CPI (Competitive Price Index): 가격 경쟁력

브랜드 해석 가이드:
- 주요 브랜드 리스트(220+개)에 포함되지 않은 브랜드:
  - Top 100에 진입했으나 트래킹 대상이 아닌 소규모/신흥 브랜드
  - 프라이빗 라벨, 아마존 자체 브랜드, 지역 브랜드 등 포함
  - 분석 시 "소규모/신흥 브랜드" 또는 "Non-major Brands"로 표현 권장
  - ⚠️ "Unknown", "기타 브랜드(Unknown)", "미확인 브랜드" 표현 절대 금지
- 소규모/신흥 브랜드 비중이 높으면:
  - 시장 분산도가 높음 (Long-tail 구조)
  - 신규 브랜드 진입이 활발함
  - 틈새 제품/전문 브랜드가 많음

응답 원칙:
1. 구체적 수치를 반드시 포함 (SoS %, 순위, 변화율 등)
2. "~으로 보인다" 대신 "~이다"로 단정적 표현
3. 한국어로 작성, 전문 용어는 영문 병기
4. 각 포인트는 bullet point (■)로 구분
5. 데이터 없이 추측하지 않음 - 주어진 데이터만 해석
6. 주요 브랜드 외 브랜드는 "소규모/신흥 브랜드" 또는 "Non-major Brands"로 표현
7. ⚠️ "Unknown", "기타 브랜드(Unknown)" 표현 절대 금지 - 대신 "소규모/신흥 브랜드" 사용

⚠️ 참고자료 인용 필수 규칙:
- 본문에서는 [1], [2], [3] 형태로만 출처 번호 표기
- URL, 기사 제목, 출처 전체 목록은 본문에 절대 포함 금지
- 전체 참고자료 목록은 8장 '참고자료 (References)' 섹션에서만 별도 작성
- 각 섹션(1~7장)에서 URL이나 기사 링크를 직접 나열하지 않음"""

    # 기본 시스템 프롬프트 (하위 호환성)
    SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE

    def __init__(self, model: str = None):
        """
        Args:
            model: LLM 모델명 (기본값: gpt-4.1-mini)
        """
        self.model = model or self.MODEL
        self._external_signals = None
        self._analysis_start_date = None
        self._analysis_end_date = None

        logger.info(f"PeriodInsightAgent initialized with model: {self.model}")

    def _get_system_prompt(self, start_date: str = None, end_date: str = None) -> str:
        """날짜 컨텍스트가 포함된 시스템 프롬프트 반환"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        return self.SYSTEM_PROMPT_TEMPLATE.format(
            current_date=current_date,
            start_date=start_date or self._analysis_start_date or "N/A",
            end_date=end_date or self._analysis_end_date or "N/A",
        )

    async def generate_report(
        self,
        analysis,  # PeriodAnalysis 객체
        external_signals: dict[str, Any] = None,
        verify_report: bool = True,  # 최종 검증 활성화 (기본값: True)
    ) -> PeriodReport:
        """
        전체 보고서 생성

        Args:
            analysis: PeriodAnalysis 객체 (src.tools.period_analyzer)
            external_signals: 외부 신호 데이터 (Reddit, TikTok 등)
            verify_report: 최종 검증 수행 여부 (기본값: True)

        Returns:
            PeriodReport 객체
        """
        logger.info(f"Generating period report: {analysis.start_date} ~ {analysis.end_date}")

        self._external_signals = external_signals or {}
        # 날짜 컨텍스트 저장 (LLM 호출 시 사용)
        self._analysis_start_date = analysis.start_date
        self._analysis_end_date = analysis.end_date

        report = PeriodReport(
            start_date=analysis.start_date,
            end_date=analysis.end_date,
            generated_at=datetime.now().isoformat(),
        )

        # 메타데이터 설정
        report.metadata = {
            "total_days": analysis.total_days,
            "model": self.model,
            "has_external_signals": bool(external_signals),
        }

        try:
            # 각 섹션 생성
            logger.debug("Generating executive summary...")
            report.executive_summary = await self._generate_executive_summary(analysis)

            logger.debug("Generating LANEIGE analysis...")
            report.laneige_analysis = await self._generate_laneige_analysis(analysis)

            logger.debug("Generating competitive analysis...")
            report.competitive_analysis = await self._generate_competitive_analysis(analysis)

            logger.debug("Generating market trends...")
            report.market_trends = await self._generate_market_trends(analysis)

            logger.debug("Generating external signals...")
            report.external_signals = await self._generate_external_signals(analysis)

            logger.debug("Generating risks and opportunities...")
            report.risks_opportunities = await self._generate_risks_opportunities(analysis)

            logger.debug("Generating strategic recommendations...")
            report.strategic_recommendations = await self._generate_strategic_recommendations(
                analysis
            )

            logger.debug("Generating references...")
            report.references = self._generate_references()

            # 최종 검증 단계 (선택적)
            if verify_report:
                logger.debug("Running final verification...")
                verification_result = await self._verify_report(report, analysis)
                report.metadata["verification"] = verification_result.to_dict()

                if verification_result.has_critical_issues:
                    logger.warning(
                        f"Report has {len(verification_result.issues)} verification issues"
                    )
                else:
                    logger.info("Report verification passed")

            logger.info("Period report generated successfully")

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise

        return report

    async def _verify_report(self, report: "PeriodReport", analysis) -> "VerificationResult":
        """보고서 최종 검증"""
        try:
            from src.tools.insight_verifier import InsightVerifier

            verifier = InsightVerifier(model=self.model)

            # 마크다운 변환 후 검증
            content = self.format_report_markdown(report)
            analysis_data = {
                "laneige_metrics": analysis.laneige_metrics,
                "market_metrics": analysis.market_metrics,
                "brand_performance": analysis.brand_performance,
                "category_analysis": analysis.category_analysis,
            }

            result = await verifier.verify_report(content, analysis_data)
            return result

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            # 검증 실패해도 보고서 생성은 계속
            from src.tools.insight_verifier import VerificationResult

            return VerificationResult(
                verified_at=datetime.now().isoformat(),
                total_checks=0,
                passed_checks=0,
                confidence_score=0.0,
            )

    async def _call_llm(self, prompt: str) -> str:
        """LLM 호출 (날짜 컨텍스트 포함)"""
        try:
            # 날짜 컨텍스트가 포함된 시스템 프롬프트 사용
            system_prompt = self._get_system_prompt()

            response = await acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"[인사이트 생성 실패: {str(e)}]"

    async def _generate_executive_summary(self, analysis) -> SectionInsight:
        """1. Executive Summary"""
        metrics = analysis.laneige_metrics
        market = analysis.market_metrics

        current_date = datetime.now().strftime("%Y-%m-%d")
        prompt = f"""⏰ 시점 정보:
- 오늘 날짜: {current_date}
- 분석 기간: {analysis.start_date} ~ {analysis.end_date} ({analysis.total_days}일)
- "현재"는 {analysis.end_date} 기준으로 해석

📊 핵심 지표:
- LANEIGE SoS: {metrics.get('start_sos', 0):.1f}% → {metrics.get('end_sos', 0):.1f}% (변화: {metrics.get('sos_change', 0):+.1f}%)
- 평균 SoS: {metrics.get('avg_sos', 0):.1f}%
- 평균 진입 제품 수: {metrics.get('avg_product_count', 0):.1f}개
- 시장 HHI: {market.get('avg_hhi', 0):.0f} ({market.get('hhi_interpretation', '분석 중')})

📈 주요 변동:
- 상승 제품: {len(metrics.get('rising_products', []))}개
- 하락 제품: {len(metrics.get('falling_products', []))}개

🌍 글로벌 K-Beauty 맥락 (필수 참조):
- K-Beauty가 2024년 미국 스킨케어 수입시장 1위 달성 (프랑스 추월)
- 아모레퍼시픽 2025년 3Q IR: Americas 매출 $156.8B (+6.9% YoY)
- LANEIGE는 그룹 내 글로벌 전략 브랜드로, Lip Sleeping Mask가 핵심 성장 동력

다음 형식으로 Executive Summary를 작성하세요 (임원 보고용):

■ 글로벌 맥락에서의 성과 (2-3줄)
  [K-Beauty 산업 내 LANEIGE 위상 및 Amazon 성과 요약]
  [아모레퍼시픽 IR 실적과의 연계성 - Americas 매출 성장과 Amazon 채널 기여도]

■ 핵심 인사이트
  [1문장: 전체 기간 LANEIGE 성과 요약]
  [1문장: 가장 주목할 변화]
  [1문장: 시장 환경 요약]

■ So What? (의사결정 포인트)
  [bullet points로 2-3개 - 임원이 알아야 할 핵심 시사점]
  [각 포인트에 "→ 권장 액션" 형태로 간단한 방향 제시]

■ 기간 내 주요 이벤트
  [bullet points로 2-3개]"""

        content = await self._call_llm(prompt)

        return SectionInsight(
            section_id="1",
            section_title="Executive Summary",
            content=content,
            key_points=[
                f"기간: {analysis.start_date} ~ {analysis.end_date}",
                f"SoS 변화: {metrics.get('sos_change', 0):+.1f}%",
                f"시장 집중도: {market.get('hhi_interpretation', 'N/A')}",
            ],
            data_highlights={
                "avg_sos": metrics.get("avg_sos", 0),
                "sos_change": metrics.get("sos_change", 0),
                "avg_hhi": market.get("avg_hhi", 0),
            },
        )

    async def _generate_laneige_analysis(self, analysis) -> SectionInsight:
        """2. LANEIGE 심층 분석"""
        metrics = analysis.laneige_metrics
        categories = analysis.category_analysis

        # Top 5 제품 정보 - 카테고리명 포함하여 혼동 방지
        top_products = metrics.get("top_products", [])[:5]
        top_products_str = (
            "\n".join(
                [
                    f"  - {p['title']} [{p.get('category_name', p.get('category', ''))}] "
                    f"(순위: {p['start_rank']} → {p['end_rank']}, 변동: {p['change']:+d})"
                    for p in top_products
                ]
            )
            if top_products
            else "  - 데이터 없음"
        )

        # 카테고리별 SoS
        cat_str = (
            "\n".join(
                [f"  - {cat}: {data.get('avg_sos', 0):.1f}%" for cat, data in categories.items()]
            )
            if categories
            else "  - 데이터 없음"
        )

        current_date = datetime.now().strftime("%Y-%m-%d")
        prompt = f"""⏰ 시점 정보: 오늘={current_date}, 분석 기간={analysis.start_date}~{analysis.end_date}

LANEIGE 심층 분석을 작성하세요. (임원 보고용 - 전략적 관점 필수)

## 📊 데이터
종합 성과:
- 기간 평균 SoS: {metrics.get('avg_sos', 0):.1f}%
- SoS 변화율: {metrics.get('sos_change_pct', 0):.1f}%
- 평균 진입 제품 수: {metrics.get('avg_product_count', 0):.1f}개

Top 5 제품 순위 변동:
{top_products_str}

카테고리별 SoS:
{cat_str}

## 🏢 IR 전략 맥락 (2025년 3Q 기준, 분석 시 반드시 참조)
- LANEIGE: 그룹 내 글로벌 전략 브랜드, "Premium Skincare" 포지셔닝
- 핵심 제품: Lip Sleeping Mask (글로벌 히어로 아이템), Water Bank 라인
- 채널 전략: Amazon (볼륨) vs Sephora (프리미엄) 이원화
- Americas 매출: $156.8B (+6.9% YoY) - Amazon이 주요 기여 채널

## 작성 형식

2.1 종합 성과 개요
■ [SoS 추이 해석 - 상승/하락/안정 판단 및 원인]
■ [경쟁사 대비 포지션 평가 - COSRX, TIRTIR 등과 비교]
■ [IR 실적과의 정합성 - Americas 매출 성장률과 Amazon SoS 변동 비교]

2.2 제품별 분석 (IR 전략 연계)
■ Top 5 제품 순위 변동 분석 (⚠️ 중요: 위 데이터에 표시된 [카테고리명] 기준으로 분석)
■ 급등/급락 제품 원인 분석
■ Lip Sleeping Mask 성과 심층 분석 (히어로 제품 모니터링)
⚠️ 주의: 순위 비교는 반드시 동일 카테고리 내에서만 유효합니다.
예: Lip Care 4위 → Lip Care 6위 = 2단계 하락 (유효)
예: Lip Care 4위 → Beauty 67위 = 비교 불가 (서로 다른 카테고리)

2.3 카테고리별 점유율
■ [강점 카테고리 - Lip Care vs Skin Care 비교]
■ [약점 카테고리]
■ [개선 필요 영역]

2.4 가격 경쟁력 (CPI) 및 채널 전략
■ [가격 포지셔닝 평가 - Premium 브랜드 포지션 유지 여부]
■ [Amazon 채널 내 가격 경쟁력 분석]

2.5 So What? (전략적 시사점)
■ [현재 성과가 IR 목표 대비 어느 수준인지 평가]
■ [향후 분기 실적에 미칠 영향 예측]

※ 참고자료: 본문에서는 [1], [2] 번호만 표기. URL, 기사 목록은 본문에 절대 포함 금지. 8장에서만 별도 작성."""

        content = await self._call_llm(prompt)

        return SectionInsight(
            section_id="2",
            section_title="LANEIGE 심층 분석",
            content=content,
            key_points=[
                f"평균 SoS: {metrics.get('avg_sos', 0):.1f}%",
                f"제품 수: {metrics.get('avg_product_count', 0):.1f}개",
            ],
            data_highlights={
                "top_products": top_products[:3],
                "category_sos": {k: v.get("avg_sos", 0) for k, v in categories.items()},
            },
        )

    async def _generate_competitive_analysis(self, analysis) -> SectionInsight:
        """3. 경쟁 환경 분석"""
        brands = analysis.brand_performance[:10]
        shifts = analysis.competitive_shifts

        brand_str = (
            "\n".join(
                [
                    f"  {i+1}. {b['brand']}: {b['avg_sos']:.1f}% (변화: {b['sos_change']:+.1f}%)"
                    for i, b in enumerate(brands)
                ]
            )
            if brands
            else "  - 데이터 없음"
        )

        current_date = datetime.now().strftime("%Y-%m-%d")
        prompt = f"""⏰ 시점 정보: 오늘={current_date}, 분석 기간={analysis.start_date}~{analysis.end_date}

경쟁 환경 분석을 작성하세요.

## 데이터
브랜드별 SoS (Top 10):
{brand_str}

신규 진입 브랜드: {shifts.get('new_entrants', [])[:5]}
이탈 브랜드: {shifts.get('exits', [])[:5]}
총 브랜드 수: {shifts.get('total_brands_start', 0)} → {shifts.get('total_brands_end', 0)}

## 작성 형식

3.1 주요 경쟁사 동향
■ [Top 3 경쟁사 분석 - COSRX, TIRTIR 등]
■ [점유율 변화가 큰 브랜드 분석]

3.2 신규 진입/이탈 브랜드
■ [신규 진입자 특성]
■ [이탈 브랜드 원인 추정]

3.3 경쟁 구도 변화
■ [시장 구조 변화 해석]
■ [LANEIGE 경쟁 포지션 평가]

※ 참고자료: 본문에서는 [1], [2] 번호만 표기. URL, 기사 목록은 본문에 절대 포함 금지. 8장에서만 별도 작성."""

        content = await self._call_llm(prompt)

        return SectionInsight(
            section_id="3",
            section_title="경쟁 환경 분석",
            content=content,
            key_points=[
                f"Top 브랜드: {brands[0]['brand'] if brands else 'N/A'}",
                f"신규 진입: {len(shifts.get('new_entrants', []))}개",
            ],
            data_highlights={
                "top_brands": brands[:5],
                "new_entrants": shifts.get("new_entrants", [])[:5],
            },
        )

    async def _generate_market_trends(self, analysis) -> SectionInsight:
        """4. 시장 동향"""
        market = analysis.market_metrics

        current_date = datetime.now().strftime("%Y-%m-%d")
        prompt = f"""⏰ 시점 정보: 오늘={current_date}, 분석 기간={analysis.start_date}~{analysis.end_date}

시장 동향 분석을 작성하세요. (글로벌 K-Beauty 산업 맥락에서 분석)

## 📊 데이터
시장 집중도 (HHI):
- 평균: {market.get('avg_hhi', 0):.0f}
- 시작일: {market.get('start_hhi', 0):.0f}
- 종료일: {market.get('end_hhi', 0):.0f}
- 해석: {market.get('hhi_interpretation', 'N/A')}

## 🌍 글로벌 K-Beauty 산업 맥락 (필수 참조)
1. K-Beauty 글로벌 위상:
   - 2024년 미국 스킨케어 수입시장 1위 달성 (프랑스 추월)
   - K-Beauty ODM/OEM 인프라가 글로벌 최고 수준
   - 미국 소비자 K-Beauty 인지도 및 신뢰도 상승 중

2. 아모레퍼시픽 IR (2025년 3Q):
   - 전사 매출: 분기 기준 성장세 유지
   - Americas: $156.8B (+6.9% YoY) - 전 지역 중 최고 성장률
   - 채널별: Amazon (볼륨 채널), Sephora/Ulta (프리미엄 채널)

3. 경쟁 구도:
   - COSRX: 가성비 K-Beauty 대표, Amazon 강세
   - TIRTIR: 신흥 브랜드, 빠른 성장세
   - Anua, Beauty of Joseon: 중저가 시장 공략 중

## 작성 형식

4.1 시장 집중도 (HHI) 분석
■ [HHI 추이 해석 - 경쟁 심화/완화 판단]
■ [시장 경쟁 강도 평가]
■ [K-Beauty 브랜드 간 점유율 집중도 분석]

4.2 카테고리 전체 트렌드
■ [성장 카테고리 - Lip Care, Skin Care 등]
■ [침체 카테고리]
■ [K-Beauty 제품이 강세인 카테고리 분석]

4.3 IR 실적 크로스 분석 (핵심 섹션)
■ [아모레퍼시픽 IR 실적과의 연관성]
  - Americas 매출 +6.9% YoY vs Amazon SoS 변동 비교
  - 분기 실적 추세와 일일 Amazon 순위 변동의 정합성
■ [채널 믹스 전략 평가]
  - Amazon (볼륨): 대중 시장 침투율
  - Sephora (프리미엄): 브랜드 이미지 관리
■ [IR 목표 달성도 평가]
  - "Americas 지역 두자릿수 성장" 목표 대비 현재 진행 상황

4.4 So What? (전략적 시사점)
■ [시장 구조 변화가 LANEIGE에 미치는 영향]
■ [다음 분기 실적 전망에 대한 시사점]

※ 참고자료: 본문에서는 [1], [2] 번호만 표기. URL, 기사 목록은 본문에 절대 포함 금지. 8장에서만 별도 작성."""

        content = await self._call_llm(prompt)

        return SectionInsight(
            section_id="4",
            section_title="시장 동향",
            content=content,
            key_points=[
                f"HHI: {market.get('avg_hhi', 0):.0f}",
                market.get("hhi_interpretation", "N/A"),
            ],
            data_highlights={"market_metrics": market},
        )

    async def _generate_external_signals(self, analysis) -> SectionInsight:
        """5. 외부 신호 분석 - 실제 뉴스 기사를 출처로 활용"""
        signals = self._external_signals

        # 신호 상세 정보 추출 (실제 뉴스 기사)
        news_articles = []
        reddit_posts = []
        other_signals = []

        if signals and signals.get("signals"):
            for s in signals["signals"]:
                source = getattr(s, "source", "")
                title = getattr(s, "title", "")
                url = getattr(s, "url", "")
                content = getattr(s, "content", "")[:200]
                published = getattr(s, "published_at", "")
                metadata = getattr(s, "metadata", {})

                signal_info = {
                    "title": title,
                    "source": source,
                    "url": url,
                    "content": content,
                    "date": published,
                    "domain": metadata.get("domain", source),
                }

                if source in [
                    "tavily_news",
                    "allure",
                    "byrdie",
                    "cosmetics_design_asia",
                    "cosmetics_business",
                    "vogue_beauty",
                    "wwd_beauty",
                ]:
                    news_articles.append(signal_info)
                elif source == "reddit":
                    reddit_posts.append(signal_info)
                else:
                    other_signals.append(signal_info)

        # 뉴스 기사 포맷팅 (실제 출처로 활용)
        news_section = ""
        if news_articles:
            news_items = []
            for i, article in enumerate(news_articles[:8], 1):  # 최대 8개
                news_items.append(
                    f"{i}. [{article['title']}]({article['url']})\n"
                    f"   - 출처: {article['domain']}, 날짜: {article['date']}\n"
                    f"   - 요약: {article['content']}"
                )
            news_section = "\n".join(news_items)
        else:
            news_section = "수집된 뉴스 기사 없음"

        # Reddit 포맷팅
        reddit_section = ""
        if reddit_posts:
            reddit_items = []
            for post in reddit_posts[:5]:
                reddit_items.append(f"- {post['title']} (r/{post.get('subreddit', 'unknown')})")
            reddit_section = "\n".join(reddit_items)
        else:
            reddit_section = "수집된 Reddit 포스트 없음"

        current_date = datetime.now().strftime("%Y-%m-%d")

        # 외부 신호 없을 때 명시적 메시지
        if not news_articles and not reddit_posts:
            no_signal_content = f"""■ 외부 신호 수집 상태

현재 외부 신호가 수집되지 않았습니다.

가능한 원인:
- Tavily API 키 미설정 (TAVILY_API_KEY 환경변수 필요)
- RSS 피드 접근 불가 (네트워크 문제)
- Reddit API 일시적 제한

권장 조치:
1. .env 파일에 TAVILY_API_KEY 설정 (월 1,000건 무료)
2. /api/signals/status 엔드포인트로 상태 확인
3. 네트워크 연결 확인

■ 분석 기간: {analysis.start_date} ~ {analysis.end_date}
■ 데이터 생성일: {current_date}
"""
            return SectionInsight(
                section_id="5",
                section_title="외부 신호 분석",
                content=no_signal_content,
                key_points=["외부 신호 미수집"],
                data_highlights={"total_signals": 0, "api_status": "unavailable"},
            )

        # 키워드 추출 (뉴스 기사 + Reddit에서)
        all_keywords = set()
        for article in news_articles:
            # 제목에서 주요 키워드 추출
            title_words = article.get("title", "").split()
            for word in title_words:
                clean_word = word.strip(".,!?\"'()[]{}").lower()
                if len(clean_word) > 3 and clean_word not in [
                    "the",
                    "and",
                    "for",
                    "with",
                    "that",
                    "this",
                    "from",
                    "about",
                ]:
                    if any(
                        kw in clean_word
                        for kw in [
                            "beauty",
                            "skin",
                            "lip",
                            "glow",
                            "hydra",
                            "korean",
                            "k-beauty",
                            "cosrx",
                            "laneige",
                            "trend",
                        ]
                    ):
                        all_keywords.add(clean_word.title())

        # 기본 바이럴 키워드 추가
        viral_keywords = (
            list(all_keywords)[:10]
            if all_keywords
            else ["K-Beauty", "Glowy Skin", "Hydration", "Clean Beauty", "Skincare Trends"]
        )
        viral_keywords_str = ", ".join([f"#{kw.replace(' ', '')}" for kw in viral_keywords[:8]])

        prompt = f"""⏰ 시점 정보: 오늘={current_date}, 분석 기간={analysis.start_date}~{analysis.end_date}

외부 신호 분석을 작성하세요. 상세하고 풍부한 내용으로 작성해주세요.

## 실제 수집된 뉴스 기사 (출처로 활용 필수)
{news_section}

## Reddit 트렌드
{reddit_section}

## 추출된 바이럴 키워드
{viral_keywords_str}

## 기존 외부 신호 보고서
{signals.get('report_section', '') if signals else ''}

## 작성 형식 (중요: 상세하고 풍부하게 작성, 각 섹션에 3-5개 bullet point 필수)

5.1 소셜 트렌드 (TikTok/Reddit)
■ [LANEIGE 관련 바이럴 현황]
  - 현재 수집된 소셜 신호는 총 N건이며, 모두 특정 매체/플랫폼에서 발표된 내용이다.
  - LANEIGE 브랜드에 대한 직접적인 바이럴 언급 확인 여부 분석
  - 아모레퍼시픽 그룹 내 브랜드로서 K-Beauty 트렌드 내 포함 여부

■ [주요 해시태그/키워드]
  - 위 바이럴 키워드 목록을 구체적으로 나열하고 설명
  - 예: #KBeauty2026, #COSRX, #GlowySkin, #Amorepacific, #BeautyInnovation, #SkincareTrends 등

5.2 뷰티 전문지 동향
■ [업계 주요 뉴스]
  - 실제 뉴스 기사 1-2개 상세 인용 (Tavily News, Allure, Byrdie 등)
  - "XXX가 발표한 'YYY 리포트'에 따르면..." 형식으로 구체적 인용
  - K-Beauty 시장 주요 동향 분석 (글로벌 대형 브랜드 경쟁 현황 포함)

■ [LANEIGE 언급 현황]
  - 아모레퍼시픽 내 LANEIGE 브랜드가 주요 기사에 반복적으로 언급되는지 여부
  - 그룹 차원의 혁신과 시장 확장 노력의 일부로 간접적 노출 분석

5.3 바이럴 키워드
■ [트렌드 키워드 분석]
  - "K-Beauty"가 2026년 뷰티 트렌드 핵심 키워드로 자리 잡고 있음
  - 'Glowy Skin', 'Hydration', 'Clean Beauty' 관련 키워드 연계 분석
  - COSRX와 같은 K-Beauty 대표 브랜드의 'Serious Glowy Skin' 카테고리 분석
  - 아모레퍼시픽의 LANEIGE 역시 '수분 공급(hydration)'과 '스킨케어 혁신' 키워드와 연계 가능성

⚠️ 중요 작성 원칙:
1. 반드시 위 뉴스 기사들을 실제 근거로 인용하세요. "XXX 매체에 따르면..." 형식
2. 각 섹션에 최소 3-5개의 구체적인 bullet point (■) 포함
3. 수집된 신호가 없거나 적은 경우에도, 해당 사실을 명시하고 시사점 분석
4. 날짜 규칙: 분석 기간({analysis.start_date}~{analysis.end_date}) 외의 날짜는 언급 금지
5. 구체적 수치, 브랜드명, 키워드를 반드시 포함

⚠️⚠️ 참고자료 인용 규칙 (매우 중요! 필독!):
- 본문에서는 [1], [2] 형태의 **번호만** 표기
- **URL을 본문에 직접 포함하지 마세요** (절대 금지!)
- **기사 제목 전체 목록을 본문에 나열하지 마세요** (절대 금지!)
- 전체 참고자료 목록(URL 포함)은 8장 참고자료 섹션에서만 별도 작성
- 예시:
  ✅ 올바름: "Allure에 따르면 K-Beauty 트렌드가 확대되고 있다[1]."
  ❌ 잘못됨: "참고: [1] https://allure.com/article/... [2] https://wwd.com/..."
  ❌ 잘못됨: 섹션 끝에 "참고 자료:" 목록 작성"""

        content = await self._call_llm(prompt)

        # key_points에 실제 뉴스 출처 추가
        key_sources = []
        if news_articles:
            key_sources = [f"{a['domain']}" for a in news_articles[:3]]

        return SectionInsight(
            section_id="5",
            section_title="외부 신호 분석",
            content=content,
            key_points=key_sources if key_sources else ["데이터 수집 중"],
            data_highlights={
                "news_articles": news_articles[:5],
                "reddit_posts": reddit_posts[:3],
                "total_signals": len(signals.get("signals", [])) if signals else 0,
            },
        )

    async def _generate_risks_opportunities(self, analysis) -> SectionInsight:
        """6. 리스크 및 기회 요인"""
        metrics = analysis.laneige_metrics
        market = analysis.market_metrics

        current_date = datetime.now().strftime("%Y-%m-%d")
        prompt = f"""⏰ 시점 정보: 오늘={current_date}, 분석 기간={analysis.start_date}~{analysis.end_date}

리스크 및 기회 요인을 분석하세요.

## 컨텍스트
- LANEIGE SoS 변화: {metrics.get('sos_change', 0):+.1f}%
- 시장 집중도: {market.get('hhi_interpretation', 'N/A')}
- 하락 제품 수: {len(metrics.get('falling_products', []))}개

## 작성 형식

6.1 리스크 요인
■ [단기 리스크 1-2개]
■ [중기 리스크 1-2개]

6.2 기회 요인
■ [단기 기회 1-2개]
■ [중기 기회 1-2개]

6.3 주요 불확실성
■ [모니터링 필요 요소]

※ 참고자료: 본문에서는 [1], [2] 번호만 표기. URL, 기사 목록은 본문에 절대 포함 금지. 8장에서만 별도 작성."""

        content = await self._call_llm(prompt)

        return SectionInsight(
            section_id="6",
            section_title="리스크 및 기회 요인",
            content=content,
            key_points=["리스크", "기회", "불확실성"],
            data_highlights={},
        )

    async def _generate_strategic_recommendations(self, analysis) -> SectionInsight:
        """7. 전략 제언"""
        metrics = analysis.laneige_metrics

        current_date = datetime.now().strftime("%Y-%m-%d")
        prompt = f"""⏰ 시점 정보: 오늘={current_date}, 분석 기간={analysis.start_date}~{analysis.end_date}

전략 제언을 작성하세요. (임원 보고용 - 구체적이고 실행 가능한 제언 필수)

## 📊 현황 데이터
- LANEIGE 현재 SoS: {metrics.get('end_sos', 0):.1f}%
- 상승 제품: {len(metrics.get('rising_products', []))}개
- 하락 제품: {len(metrics.get('falling_products', []))}개

## 🎯 IR 전략 프레임워크 (2025년 3Q 기준)
1. 그룹 전략 방향:
   - "글로벌 리딩 K-Beauty 기업" 비전
   - Americas 지역 두자릿수 성장 목표
   - 채널 다각화: Amazon + Sephora + Ulta

2. LANEIGE 브랜드 전략:
   - 포지셔닝: "Premium K-Beauty Skincare"
   - 히어로 제품: Lip Sleeping Mask (인지도 앵커)
   - 확장 전략: Water Bank 라인 강화

3. 경쟁 환경:
   - COSRX: 가성비 리더, Amazon 점유율 높음
   - TIRTIR: 빠른 성장, 젊은 층 공략
   - 대응 필요: 프리미엄 가치 차별화

## 작성 형식 (담당 부서 및 KPI 필수 명시)

7.1 즉시 실행 액션 (1-2주)
■ [액션 1: 구체적 내용]
  - 담당: [부서명]
  - KPI: [측정 가능한 지표]
  - 예상 효과: [수치화된 기대치]
■ [액션 2: 구체적 내용]
  - 담당: [부서명]
  - KPI: [측정 가능한 지표]

7.2 단기 전략 (1개월)
■ [전략 1: 채널 최적화]
  - Amazon 리스팅 최적화 (키워드, 이미지, A+ 콘텐츠)
  - 예상 SoS 개선: +X%p
■ [전략 2: 제품 포트폴리오]
  - 하락 제품 대응 방안
  - 신제품 런칭 고려 여부

7.3 중기 전략 (1-3개월)
■ [전략 방향 1: IR 목표 달성]
  - Americas 두자릿수 성장 기여 방안
  - 채널 믹스 최적화 (Amazon vs Sephora 비중)
■ [전략 방향 2: 경쟁 대응]
  - COSRX 대비 차별화 포인트 강화
  - 프리미엄 포지셔닝 유지 전략

7.4 KPI 목표 (수치화 필수)
■ SoS 목표: 현재 {metrics.get('end_sos', 0):.1f}% → 목표 [X%] (+X%p)
■ 제품 순위: Top 10 진입 제품 [N]개 → 목표 [M]개
■ 히어로 제품: Lip Sleeping Mask 순위 유지/개선 목표
■ IR 기여도: Amazon 채널 매출 성장률 [X%] 기여

7.5 리스크 대비
■ [리스크 시나리오]: 경쟁 심화 시 대응 방안
■ [플랜 B]: 목표 미달 시 대안 전략

※ 참고자료: 본문에서는 [1], [2] 번호만 표기. URL, 기사 목록은 본문에 절대 포함 금지. 8장에서만 별도 작성."""

        content = await self._call_llm(prompt)

        return SectionInsight(
            section_id="7",
            section_title="전략 제언",
            content=content,
            key_points=["단기 액션", "중기 전략", "KPI"],
            data_highlights={
                "current_sos": metrics.get("end_sos", 0),
                "target_sos": metrics.get("end_sos", 0) * 1.1,  # +10% 목표 예시
            },
        )

    def _generate_references(self) -> SectionInsight:
        """
        8. 참고자료 (References) - 통합 번호 체계

        본문에서 [1], [2] 등으로 인용된 출처가 이 섹션에 순차적으로 나열됩니다.
        번호는 외부 신호 수집 순서 기준으로 통일됩니다.
        """
        signals = self._external_signals

        # 모든 외부 신호를 통합하여 순차 번호 부여 (본문 인용과 일치)
        all_signals = signals.get("signals", []) if signals else []

        def extract_ref_info(signal_list):
            """신호 리스트에서 참고자료 정보 추출 (중복 제거)"""
            refs = []
            seen_urls = set()
            for s in signal_list:
                url = getattr(s, "url", "")
                title = getattr(s, "title", "")
                if url and title and url not in seen_urls:
                    seen_urls.add(url)
                    published = getattr(s, "published_at", "")
                    metadata = getattr(s, "metadata", {}) or {}
                    domain = metadata.get("domain", getattr(s, "source", ""))
                    refs.append(
                        {
                            "title": title,
                            "source": domain,
                            "url": url,
                            "date": published[:10] if published else "",
                        }
                    )
            return refs

        all_refs = extract_ref_info(all_signals)

        # 참고자료 섹션 내용 생성 (통합 번호 체계)
        content_lines = []

        # 본문 인용과 일치하는 번호로 뉴스/기사 출처 나열
        if all_refs:
            for i, ref in enumerate(all_refs, 1):
                date_str = f", {ref['date']}" if ref["date"] else ""
                # 제목 전체 표시 (절대 축약하지 않음)
                content_lines.append(f"[{i}] {ref['source']}{date_str}, \"{ref['title']}\"")
        else:
            content_lines.append("수집된 외부 신호 없음")

        content_lines.append("")

        # 면책사항 추가
        content_lines.append("")
        content_lines.append("### 면책사항")
        content_lines.append("- 본 보고서는 AI 기반 자동 생성 문서로, 참고용으로만 활용 바랍니다.")
        content_lines.append(
            "- Amazon 데이터는 크롤링 시점 기준이며, 실시간 변동이 있을 수 있습니다."
        )
        content_lines.append("- 전략적 의사결정 시 추가 검증을 권장합니다.")

        return SectionInsight(
            section_id="8",
            section_title="참고자료 (References)",
            content="\n".join(content_lines),
            key_points=[f"총 {len(all_refs)}개 출처 참조"],
            data_highlights={
                "total_references": len(all_refs),
                "sources": list({r["source"] for r in all_refs[:10]}),
            },
        )

    def format_report_markdown(self, report: PeriodReport) -> str:
        """
        보고서를 마크다운 형식으로 변환 (AMOREPACIFIC 스타일)

        Args:
            report: PeriodReport 객체

        Returns:
            마크다운 문자열
        """
        sections = [
            "# 라네즈 Amazon US 기간별 분석 보고서",
            "",
            f"**분석 기간**: {report.start_date} ~ {report.end_date}",
            f"**생성일시**: {report.generated_at}",
            "",
            "---",
            "",
        ]

        # 섹션 제목 매핑 (AMOREPACIFIC 스타일)
        section_titles = {
            "executive_summary": "Executive Summary",
            "laneige_analysis": "라네즈 심층 분석",
            "competitive_analysis": "경쟁 환경 분석",
            "market_trends": "시장 동향",
            "external_signals": "외부 신호 분석",
            "risks_opportunities": "리스크 및 기회 요인",
            "strategic_recommendations": "전략 제언",
        }

        # 각 섹션 추가
        for section_attr in [
            "executive_summary",
            "laneige_analysis",
            "competitive_analysis",
            "market_trends",
            "external_signals",
            "risks_opportunities",
            "strategic_recommendations",
            "references",  # 참고자료 섹션 추가
        ]:
            section = getattr(report, section_attr)
            if section:
                # 참고자료 섹션은 이미 헤딩이 포함되어 있음
                if section_attr == "references":
                    sections.append(section.content)
                else:
                    # AMOREPACIFIC 스타일 섹션 헤더
                    title = section_titles.get(section_attr, section.section_title)
                    sections.append(f"▎**{section.section_id}. {title}**")
                    sections.append("")
                    # 섹션 내용에 포맷터 적용
                    formatted_content = format_insight(section.content)
                    sections.append(formatted_content)
                sections.append("")
                sections.append("---")
                sections.append("")

        # 메타데이터
        sections.append("▎**보고서 메타데이터**")
        sections.append("")
        sections.append(f"• 분석 일수: **{report.metadata.get('total_days', 'N/A')}**일")
        sections.append(f"• AI 모델: {report.metadata.get('model', 'N/A')}")
        sections.append(
            f"• 외부 신호 포함: {'Yes' if report.metadata.get('has_external_signals') else 'No'}"
        )
        sections.append("")

        return "\n".join(sections)

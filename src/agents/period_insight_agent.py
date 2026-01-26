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
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from litellm import acompletion

from src.monitoring.logger import AgentLogger


logger = AgentLogger(__name__)


@dataclass
class SectionInsight:
    """섹션별 인사이트"""
    section_id: str
    section_title: str
    content: str
    key_points: List[str] = field(default_factory=list)
    data_highlights: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "section_id": self.section_id,
            "section_title": self.section_title,
            "content": self.content,
            "key_points": self.key_points,
            "data_highlights": self.data_highlights
        }


@dataclass
class PeriodReport:
    """기간별 분석 보고서"""
    start_date: str
    end_date: str
    generated_at: str

    executive_summary: Optional[SectionInsight] = None
    laneige_analysis: Optional[SectionInsight] = None
    competitive_analysis: Optional[SectionInsight] = None
    market_trends: Optional[SectionInsight] = None
    external_signals: Optional[SectionInsight] = None
    risks_opportunities: Optional[SectionInsight] = None
    strategic_recommendations: Optional[SectionInsight] = None
    references: Optional[SectionInsight] = None  # 참고자료 섹션 추가

    # 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "generated_at": self.generated_at,
            "executive_summary": self.executive_summary.to_dict() if self.executive_summary else None,
            "laneige_analysis": self.laneige_analysis.to_dict() if self.laneige_analysis else None,
            "competitive_analysis": self.competitive_analysis.to_dict() if self.competitive_analysis else None,
            "market_trends": self.market_trends.to_dict() if self.market_trends else None,
            "external_signals": self.external_signals.to_dict() if self.external_signals else None,
            "risks_opportunities": self.risks_opportunities.to_dict() if self.risks_opportunities else None,
            "strategic_recommendations": self.strategic_recommendations.to_dict() if self.strategic_recommendations else None,
            "metadata": self.metadata
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

    # 시스템 프롬프트
    SYSTEM_PROMPT = """당신은 화장품 산업 전문 애널리스트입니다.

역할:
- 아모레퍼시픽 LANEIGE 브랜드의 Amazon US 시장 경쟁력 분석
- 증권사 리서치 보고서 수준의 전문성과 구체성 유지
- 데이터 기반의 객관적 분석과 실행 가능한 전략 제언

분석 프레임워크:
- SoS (Share of Shelf): 브랜드 점유율 (Top 100 내 제품 비율)
- HHI (Herfindahl-Hirschman Index): 시장 집중도 (낮을수록 경쟁적)
- CPI (Competitive Price Index): 가격 경쟁력

브랜드 해석 가이드:
- "Unknown" 또는 "기타 브랜드": 데이터 인식 실패가 아님
  - Top 100에 진입했으나 주요 브랜드(220+개 트래킹 리스트)에 포함되지 않은 브랜드
  - 소규모/신규/지역 브랜드, 프라이빗 라벨, 아마존 자체 브랜드 등 포함
  - 분석 시 "기타 브랜드" 또는 "Non-major Brands"로 표현 권장
- 브랜드 점유율 분석 시 Unknown/기타 브랜드의 비중이 높으면:
  - 시장 분산도가 높음 (Long-tail 구조)
  - 신규 브랜드 진입이 활발함
  - 틈새 제품/전문 브랜드가 많음

응답 원칙:
1. 구체적 수치를 반드시 포함 (SoS %, 순위, 변화율 등)
2. "~으로 보인다" 대신 "~이다"로 단정적 표현
3. 한국어로 작성, 전문 용어는 영문 병기
4. 각 포인트는 bullet point (■)로 구분
5. 데이터 없이 추측하지 않음 - 주어진 데이터만 해석
6. Unknown 브랜드는 "기타 브랜드" 또는 "Non-major Brands"로 표현"""

    def __init__(self, model: str = None):
        """
        Args:
            model: LLM 모델명 (기본값: gpt-4.1-mini)
        """
        self.model = model or self.MODEL
        self._external_signals = None

        logger.info(f"PeriodInsightAgent initialized with model: {self.model}")

    async def generate_report(
        self,
        analysis,  # PeriodAnalysis 객체
        external_signals: Dict[str, Any] = None,
        verify_report: bool = True  # 최종 검증 활성화 (기본값: True)
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

        report = PeriodReport(
            start_date=analysis.start_date,
            end_date=analysis.end_date,
            generated_at=datetime.now().isoformat()
        )

        # 메타데이터 설정
        report.metadata = {
            "total_days": analysis.total_days,
            "model": self.model,
            "has_external_signals": bool(external_signals)
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
            report.strategic_recommendations = await self._generate_strategic_recommendations(analysis)

            logger.debug("Generating references...")
            report.references = self._generate_references()

            # 최종 검증 단계 (선택적)
            if verify_report:
                logger.debug("Running final verification...")
                verification_result = await self._verify_report(report, analysis)
                report.metadata["verification"] = verification_result.to_dict()

                if verification_result.has_critical_issues:
                    logger.warning(f"Report has {len(verification_result.issues)} verification issues")
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
                "category_analysis": analysis.category_analysis
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
                confidence_score=0.0
            )

    async def _call_llm(self, prompt: str) -> str:
        """LLM 호출"""
        try:
            response = await acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"[인사이트 생성 실패: {str(e)}]"

    async def _generate_executive_summary(self, analysis) -> SectionInsight:
        """1. Executive Summary"""
        metrics = analysis.laneige_metrics
        market = analysis.market_metrics

        prompt = f"""분석 기간: {analysis.start_date} ~ {analysis.end_date} ({analysis.total_days}일)

핵심 지표:
- LANEIGE SoS: {metrics.get('start_sos', 0):.1f}% → {metrics.get('end_sos', 0):.1f}% (변화: {metrics.get('sos_change', 0):+.1f}%)
- 평균 SoS: {metrics.get('avg_sos', 0):.1f}%
- 평균 진입 제품 수: {metrics.get('avg_product_count', 0):.1f}개
- 시장 HHI: {market.get('avg_hhi', 0):.0f} ({market.get('hhi_interpretation', '분석 중')})

주요 변동:
- 상승 제품: {len(metrics.get('rising_products', []))}개
- 하락 제품: {len(metrics.get('falling_products', []))}개

다음 형식으로 Executive Summary를 작성하세요:

■ 핵심 인사이트 (3줄 이내)
  [1문장: 전체 기간 LANEIGE 성과 요약]
  [1문장: 가장 주목할 변화]
  [1문장: 시장 환경 요약]

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
                f"시장 집중도: {market.get('hhi_interpretation', 'N/A')}"
            ],
            data_highlights={
                "avg_sos": metrics.get("avg_sos", 0),
                "sos_change": metrics.get("sos_change", 0),
                "avg_hhi": market.get("avg_hhi", 0)
            }
        )

    async def _generate_laneige_analysis(self, analysis) -> SectionInsight:
        """2. LANEIGE 심층 분석"""
        metrics = analysis.laneige_metrics
        categories = analysis.category_analysis

        # Top 5 제품 정보 - 카테고리명 포함하여 혼동 방지
        top_products = metrics.get("top_products", [])[:5]
        top_products_str = "\n".join([
            f"  - {p['title'][:30]}... [{p.get('category_name', p.get('category', ''))}] "
            f"(순위: {p['start_rank']} → {p['end_rank']}, 변동: {p['change']:+d})"
            for p in top_products
        ]) if top_products else "  - 데이터 없음"

        # 카테고리별 SoS
        cat_str = "\n".join([
            f"  - {cat}: {data.get('avg_sos', 0):.1f}%"
            for cat, data in categories.items()
        ]) if categories else "  - 데이터 없음"

        prompt = f"""LANEIGE 심층 분석을 작성하세요.

## 데이터
종합 성과:
- 기간 평균 SoS: {metrics.get('avg_sos', 0):.1f}%
- SoS 변화율: {metrics.get('sos_change_pct', 0):.1f}%
- 평균 진입 제품 수: {metrics.get('avg_product_count', 0):.1f}개

Top 5 제품 순위 변동:
{top_products_str}

카테고리별 SoS:
{cat_str}

## 작성 형식

2.1 종합 성과 개요
■ [SoS 추이 해석 - 상승/하락/안정 판단 및 원인]
■ [경쟁사 대비 포지션 평가]

2.2 제품별 분석
■ Top 5 제품 순위 변동 분석 (⚠️ 중요: 위 데이터에 표시된 [카테고리명] 기준으로 분석)
■ 급등/급락 제품 원인 분석
⚠️ 주의: 순위 비교는 반드시 동일 카테고리 내에서만 유효합니다.
예: Lip Care 4위 → Lip Care 6위 = 2단계 하락 (유효)
예: Lip Care 4위 → Beauty 67위 = 비교 불가 (서로 다른 카테고리)

2.3 카테고리별 점유율
■ [강점 카테고리]
■ [약점 카테고리]
■ [개선 필요 영역]

2.4 가격 경쟁력 (CPI)
■ [가격 포지셔닝 평가]"""

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
                "category_sos": {k: v.get("avg_sos", 0) for k, v in categories.items()}
            }
        )

    async def _generate_competitive_analysis(self, analysis) -> SectionInsight:
        """3. 경쟁 환경 분석"""
        brands = analysis.brand_performance[:10]
        shifts = analysis.competitive_shifts

        brand_str = "\n".join([
            f"  {i+1}. {b['brand']}: {b['avg_sos']:.1f}% (변화: {b['sos_change']:+.1f}%)"
            for i, b in enumerate(brands)
        ]) if brands else "  - 데이터 없음"

        prompt = f"""경쟁 환경 분석을 작성하세요.

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
■ [LANEIGE 경쟁 포지션 평가]"""

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
                "new_entrants": shifts.get("new_entrants", [])[:5]
            }
        )

    async def _generate_market_trends(self, analysis) -> SectionInsight:
        """4. 시장 동향"""
        market = analysis.market_metrics

        prompt = f"""시장 동향 분석을 작성하세요.

## 데이터
시장 집중도 (HHI):
- 평균: {market.get('avg_hhi', 0):.0f}
- 시작일: {market.get('start_hhi', 0):.0f}
- 종료일: {market.get('end_hhi', 0):.0f}
- 해석: {market.get('hhi_interpretation', 'N/A')}

## 작성 형식

4.1 시장 집중도 (HHI)
■ [HHI 추이 해석]
■ [시장 경쟁 강도 평가]

4.2 카테고리 전체 트렌드
■ [성장 카테고리]
■ [침체 카테고리]

4.3 IR 실적 크로스 분석
■ [아모레퍼시픽 IR 실적과의 연관성]
■ [Americas 매출과 Amazon 성과 상관관계]"""

        content = await self._call_llm(prompt)

        return SectionInsight(
            section_id="4",
            section_title="시장 동향",
            content=content,
            key_points=[
                f"HHI: {market.get('avg_hhi', 0):.0f}",
                market.get("hhi_interpretation", "N/A")
            ],
            data_highlights={"market_metrics": market}
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
                source = getattr(s, 'source', '')
                title = getattr(s, 'title', '')
                url = getattr(s, 'url', '')
                content = getattr(s, 'content', '')[:200]
                published = getattr(s, 'published_at', '')
                metadata = getattr(s, 'metadata', {})

                signal_info = {
                    "title": title,
                    "source": source,
                    "url": url,
                    "content": content,
                    "date": published,
                    "domain": metadata.get("domain", source)
                }

                if source in ["tavily_news", "allure", "byrdie", "cosmetics_design_asia",
                              "cosmetics_business", "vogue_beauty", "wwd_beauty"]:
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
                    f"{i}. [{article['title'][:80]}]({article['url']})\n"
                    f"   - 출처: {article['domain']}, 날짜: {article['date']}\n"
                    f"   - 요약: {article['content'][:150]}..."
                )
            news_section = "\n".join(news_items)
        else:
            news_section = "수집된 뉴스 기사 없음"

        # Reddit 포맷팅
        reddit_section = ""
        if reddit_posts:
            reddit_items = []
            for post in reddit_posts[:5]:
                reddit_items.append(f"- {post['title'][:60]}... (r/{post.get('subreddit', 'unknown')})")
            reddit_section = "\n".join(reddit_items)
        else:
            reddit_section = "수집된 Reddit 포스트 없음"

        prompt = f"""외부 신호 분석을 작성하세요.

## 실제 수집된 뉴스 기사 (출처로 활용 필수)
{news_section}

## Reddit 트렌드
{reddit_section}

## 기존 외부 신호 보고서
{signals.get('report_section', '') if signals else ''}

## 작성 형식 (중요: 실제 뉴스 기사를 인용하고 출처 명시)

5.1 업계 뉴스 동향
■ [실제 뉴스 기사 1-2개 인용, 출처와 날짜 명시]
■ [K-Beauty/LANEIGE 관련 주요 동향 분석]
■ [시장 영향도 평가]

5.2 소셜 미디어 트렌드 (Reddit/TikTok)
■ [소비자 반응 및 언급 현황]
■ [주요 키워드/해시태그]

5.3 시사점
■ [뉴스 기반 시장 동향 해석]
■ [LANEIGE 전략적 함의]

⚠️ 중요: 반드시 위 뉴스 기사들을 실제 근거로 인용하세요. "XXX 매체에 따르면..." 형식으로 작성."""

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
                "total_signals": len(signals.get("signals", [])) if signals else 0
            }
        )

    async def _generate_risks_opportunities(self, analysis) -> SectionInsight:
        """6. 리스크 및 기회 요인"""
        metrics = analysis.laneige_metrics
        market = analysis.market_metrics

        prompt = f"""리스크 및 기회 요인을 분석하세요.

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
■ [모니터링 필요 요소]"""

        content = await self._call_llm(prompt)

        return SectionInsight(
            section_id="6",
            section_title="리스크 및 기회 요인",
            content=content,
            key_points=["리스크", "기회", "불확실성"],
            data_highlights={}
        )

    async def _generate_strategic_recommendations(self, analysis) -> SectionInsight:
        """7. 전략 제언"""
        metrics = analysis.laneige_metrics

        prompt = f"""전략 제언을 작성하세요.

## 컨텍스트
- LANEIGE 현재 SoS: {metrics.get('end_sos', 0):.1f}%
- 상승 제품: {len(metrics.get('rising_products', []))}개
- 하락 제품: {len(metrics.get('falling_products', []))}개

## 작성 형식

7.1 단기 액션 (1-2주)
■ [즉시 실행 가능한 액션 2-3개]
■ [담당 부서 명시]

7.2 중기 전략 (1-3개월)
■ [중기 전략 방향 2-3개]
■ [예상 효과]

7.3 KPI 목표
■ SoS 목표: [현재 대비 +X%p]
■ 제품 순위 목표: [Top N 제품 수]
■ 카테고리 목표: [특정 카테고리 SoS]"""

        content = await self._call_llm(prompt)

        return SectionInsight(
            section_id="7",
            section_title="전략 제언",
            content=content,
            key_points=["단기 액션", "중기 전략", "KPI"],
            data_highlights={
                "current_sos": metrics.get("end_sos", 0),
                "target_sos": metrics.get("end_sos", 0) * 1.1  # +10% 목표 예시
            }
        )

    def _generate_references(self) -> SectionInsight:
        """8. 참고자료 (References) - 뉴스 기사 URL 및 데이터 출처 포함"""
        signals = self._external_signals
        references = []

        # 1. 뉴스 기사 출처 (Tavily/RSS)
        news_refs = []
        if signals and signals.get("signals"):
            for s in signals["signals"]:
                source = getattr(s, 'source', '')
                title = getattr(s, 'title', '')
                url = getattr(s, 'url', '')
                published = getattr(s, 'published_at', '')
                metadata = getattr(s, 'metadata', {})
                domain = metadata.get("domain", source)

                if url and title:
                    news_refs.append({
                        "title": title,
                        "source": domain,
                        "url": url,
                        "date": published
                    })

        # 중복 URL 제거
        seen_urls = set()
        unique_news = []
        for ref in news_refs:
            if ref["url"] not in seen_urls:
                seen_urls.add(ref["url"])
                unique_news.append(ref)

        # 참고자료 섹션 내용 생성
        content_lines = ["## 8. 참고자료 (References)\n"]

        # 8.1 뉴스 및 미디어 출처
        content_lines.append("### 8.1 뉴스 및 미디어 출처\n")
        if unique_news:
            for i, ref in enumerate(unique_news[:15], 1):  # 최대 15개
                date_str = f" ({ref['date']})" if ref['date'] else ""
                content_lines.append(
                    f"{i}. [{ref['title'][:80]}]({ref['url']}) - {ref['source']}{date_str}"
                )
        else:
            content_lines.append("- 뉴스 데이터 없음")

        content_lines.append("")

        # 8.2 데이터 출처
        content_lines.append("### 8.2 데이터 출처\n")
        content_lines.append("- **Amazon Best Sellers**: Amazon US 공식 베스트셀러 랭킹 (Top 100)")
        content_lines.append("- **카테고리**: Beauty & Personal Care, Skin Care, Lip Care, Lip Makeup, Face Powder")
        content_lines.append("- **크롤링 시간**: 매일 KST 22:00 (미국 피크 판매 반영)")
        content_lines.append("- **데이터 기간**: 보고서 상단 분석 기간 참조")
        content_lines.append("")

        # 8.3 분석 방법론
        content_lines.append("### 8.3 분석 방법론\n")
        content_lines.append("- **SoS (Share of Shelf)**: Top 100 내 브랜드 제품 수 비율")
        content_lines.append("- **HHI (Herfindahl-Hirschman Index)**: 시장 집중도 지수 (Σ 점유율²)")
        content_lines.append("- **순위 변동**: 동일 카테고리 내에서만 비교 (카테고리 간 순위 비교 불가)")
        content_lines.append("")

        # 8.4 면책사항
        content_lines.append("### 8.4 면책사항\n")
        content_lines.append("- 본 보고서는 AI 기반 자동 생성 문서로, 참고용으로만 활용 바랍니다.")
        content_lines.append("- Amazon 데이터는 크롤링 시점 기준이며, 실시간 변동이 있을 수 있습니다.")
        content_lines.append("- 전략적 의사결정 시 추가 검증을 권장합니다.")

        content = "\n".join(content_lines)

        return SectionInsight(
            section_id="8",
            section_title="참고자료 (References)",
            content=content,
            key_points=[f"뉴스 출처 {len(unique_news)}건", "Amazon Best Sellers", "HHI/SoS 방법론"],
            data_highlights={
                "news_count": len(unique_news),
                "news_refs": unique_news[:10]  # 상위 10개만 하이라이트에 저장
            }
        )

    def format_report_markdown(self, report: PeriodReport) -> str:
        """
        보고서를 마크다운 형식으로 변환

        Args:
            report: PeriodReport 객체

        Returns:
            마크다운 문자열
        """
        sections = [
            f"# LANEIGE Amazon US 기간별 분석 보고서",
            f"",
            f"**분석 기간**: {report.start_date} ~ {report.end_date}",
            f"**생성일시**: {report.generated_at}",
            f"",
            f"---",
            f""
        ]

        # 각 섹션 추가
        for section_attr in [
            "executive_summary",
            "laneige_analysis",
            "competitive_analysis",
            "market_trends",
            "external_signals",
            "risks_opportunities",
            "strategic_recommendations",
            "references"  # 참고자료 섹션 추가
        ]:
            section = getattr(report, section_attr)
            if section:
                # 참고자료 섹션은 이미 헤딩이 포함되어 있음
                if section_attr == "references":
                    sections.append(section.content)
                else:
                    sections.append(f"## {section.section_id}. {section.section_title}")
                    sections.append("")
                    sections.append(section.content)
                sections.append("")
                sections.append("---")
                sections.append("")

        # 메타데이터
        sections.append("## 보고서 메타데이터")
        sections.append("")
        sections.append(f"- 분석 일수: {report.metadata.get('total_days', 'N/A')}일")
        sections.append(f"- AI 모델: {report.metadata.get('model', 'N/A')}")
        sections.append(f"- 외부 신호 포함: {'Yes' if report.metadata.get('has_external_signals') else 'No'}")
        sections.append("")

        return "\n".join(sections)

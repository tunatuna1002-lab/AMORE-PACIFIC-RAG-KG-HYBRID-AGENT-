"""
Market Intelligence Engine
==========================
외부 데이터 수집기를 통합하여 4-Layer 인사이트를 생성하는 엔진

## 4-Layer 데이터 아키텍처

Layer 4: 거시경제 & 무역 (관세청 수출입 API, 환율, 관세 뉴스)
Layer 3: 산업 & 기업 (아모레퍼시픽 IR, 증권사 리포트, 전문기관)
Layer 2: 소비자 트렌드 (Reddit, TikTok, 뷰티 매체 RSS)
Layer 1: Amazon 데이터 (현재 시스템 - 순위, 가격, 리뷰)

## 사용 예시
```python
engine = MarketIntelligenceEngine()
await engine.initialize()

# 전체 레이어 데이터 수집
await engine.collect_all_layers()

# 인사이트 생성
insight = engine.generate_layered_insight()

# 특정 레이어만 수집
await engine.collect_layer(4)  # 거시경제
await engine.collect_layer(2)  # 소비자 트렌드
```

## 출력 형식
```markdown
# LANEIGE Amazon US 일일 인사이트

## 📌 오늘의 핵심
[가장 중요한 변화/발견 + 원인 연결]

## 🔍 원인 분석 (Why?)

### Layer 4: 거시경제/무역
• 화장품 대미 수출: $12.3B (+12% YoY) [1]
• 환율: USD/KRW 1,438원 (전주 대비 +12원)

### Layer 3: 산업/기업 동향
• 아모레퍼시픽 IR: 3Q 2025 Americas +6.9% [2]
...

## 📚 참고자료
[1] 관세청, 품목별 수출입통계, 2025.01
[2] 아모레퍼시픽 IR, "3Q 2025 Earnings Release", 2025.11.06
```
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

from .public_data_collector import PublicDataCollector, TradeData
from .ir_report_parser import IRReportParser, IRReport
from .external_signal_collector import (
    ExternalSignalCollector,
    ExternalSignal,
    SignalTier
)
from .source_manager import SourceManager, InsightSourceBuilder


logger = logging.getLogger(__name__)

# 한국 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))


class DataLayer:
    """데이터 레이어 상수"""
    LAYER_1_AMAZON = 1     # Amazon 데이터 (현재 시스템)
    LAYER_2_CONSUMER = 2   # 소비자 트렌드 (Reddit, TikTok, RSS)
    LAYER_3_INDUSTRY = 3   # 산업/기업 (IR, 증권사, 전문기관)
    LAYER_4_MACRO = 4      # 거시경제/무역 (관세청, 환율)


@dataclass
class LayerData:
    """
    레이어별 수집된 데이터

    Attributes:
        layer: 레이어 번호 (1-4)
        layer_name: 레이어 이름
        collected_at: 수집 시각
        data: 수집된 데이터
        sources: 출처 정보
    """
    layer: int
    layer_name: str
    collected_at: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    sources: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.collected_at:
            self.collected_at = datetime.now(KST).isoformat()


class MarketIntelligenceEngine:
    """
    시장 인텔리전스 엔진

    4-Layer 데이터 아키텍처를 기반으로
    외부 데이터를 수집하고 인사이트를 생성합니다.
    """

    def __init__(
        self,
        public_data_api_key: Optional[str] = None,
        data_dir: str = "./data/market_intelligence"
    ):
        """
        Args:
            public_data_api_key: 공공데이터 API 키
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 수집기 초기화
        self.public_collector = PublicDataCollector(
            api_key=public_data_api_key,
            data_dir=str(self.data_dir / "public_data")
        )
        self.ir_parser = IRReportParser(
            data_dir=str(self.data_dir / "ir_reports")
        )
        self.signal_collector = ExternalSignalCollector(
            data_dir=str(self.data_dir / "signals")
        )

        # 출처 관리자
        self.source_manager = SourceManager(
            data_dir=str(self.data_dir / "sources")
        )

        # 레이어별 데이터 저장
        self.layer_data: Dict[int, LayerData] = {}

        # 초기화 상태
        self._initialized = False

    async def initialize(self) -> None:
        """비동기 초기화"""
        if self._initialized:
            return

        await self.public_collector.initialize()
        await self.ir_parser.initialize()
        await self.signal_collector.initialize()

        self._initialized = True
        logger.info("MarketIntelligenceEngine initialized")

    async def close(self) -> None:
        """리소스 정리"""
        await self.public_collector.close()
        await self.ir_parser.close()
        await self.signal_collector.close()

        self._initialized = False

    # =========================================================================
    # 레이어별 데이터 수집
    # =========================================================================

    async def collect_layer_4_macro(
        self,
        year: Optional[str] = None,
        month: Optional[str] = None
    ) -> LayerData:
        """
        Layer 4: 거시경제/무역 데이터 수집

        - 관세청 수출입 통계 (화장품 HS Code 3304)
        - 미국 대상 수출 통계

        Args:
            year: 조회 연도 (기본: 현재 연도)
            month: 조회 월 (기본: 현재 월)

        Returns:
            LayerData
        """
        if not year:
            year = datetime.now(KST).strftime("%Y")
        if not month:
            month = datetime.now(KST).strftime("%m")

        data = {
            "us_export": None,
            "total_export": [],
            "trade_summary": None
        }
        sources = []

        try:
            # 미국 대상 화장품 수출
            us_export = await self.public_collector.fetch_us_cosmetics_export(year, month)
            if us_export:
                data["us_export"] = us_export.to_dict()
                sources.append(self.public_collector.create_source_reference(trade_data=us_export))

            # 전체 수출입 통계
            exports = await self.public_collector.fetch_cosmetics_trade(year, month, "export")
            if exports:
                data["total_export"] = [e.to_dict() for e in exports]

            # 요약
            data["trade_summary"] = self.public_collector.get_trade_summary(year, "export")

        except Exception as e:
            logger.error(f"Failed to collect Layer 4 data: {e}")

        layer_data = LayerData(
            layer=DataLayer.LAYER_4_MACRO,
            layer_name="거시경제/무역",
            data=data,
            sources=sources
        )

        self.layer_data[DataLayer.LAYER_4_MACRO] = layer_data
        return layer_data

    async def collect_layer_3_industry(
        self,
        year: Optional[str] = None,
        quarter: Optional[str] = None
    ) -> LayerData:
        """
        Layer 3: 산업/기업 데이터 수집

        - 아모레퍼시픽 IR 데이터
        - Americas 지역 실적
        - 브랜드별 하이라이트

        Args:
            year: 조회 연도 (기본: 최신)
            quarter: 조회 분기 (기본: 최신)

        Returns:
            LayerData
        """
        data = {
            "ir_report": None,
            "americas_insights": None,
            "brand_highlights": {}
        }
        sources = []

        try:
            # IR 보고서
            if year and quarter:
                report = self.ir_parser.get_quarterly_data(year, quarter)
            else:
                report = self.ir_parser.get_latest_report()

            if report:
                data["ir_report"] = report.to_dict()
                sources.append(self.ir_parser.create_source_reference(report.year, report.quarter))

                # Americas 인사이트
                data["americas_insights"] = self.ir_parser.get_americas_insights(
                    report.year, report.quarter
                )

                # LANEIGE 하이라이트
                laneige = self.ir_parser.get_brand_highlights("LANEIGE", report.year, report.quarter)
                if laneige:
                    data["brand_highlights"]["LANEIGE"] = [h.to_dict() for h in laneige]

                # COSRX 하이라이트
                cosrx = self.ir_parser.get_brand_highlights("COSRX", report.year, report.quarter)
                if cosrx:
                    data["brand_highlights"]["COSRX"] = [h.to_dict() for h in cosrx]

        except Exception as e:
            logger.error(f"Failed to collect Layer 3 data: {e}")

        layer_data = LayerData(
            layer=DataLayer.LAYER_3_INDUSTRY,
            layer_name="산업/기업",
            data=data,
            sources=sources
        )

        self.layer_data[DataLayer.LAYER_3_INDUSTRY] = layer_data
        return layer_data

    async def collect_layer_2_consumer(
        self,
        keywords: Optional[List[str]] = None
    ) -> LayerData:
        """
        Layer 2: 소비자 트렌드 데이터 수집

        - RSS 피드 (뷰티 전문 매체)
        - Reddit (스킨케어/뷰티 커뮤니티)

        Args:
            keywords: 필터링 키워드 (기본: K-Beauty 키워드)

        Returns:
            LayerData
        """
        data = {
            "kbeauty_news": [],
            "reddit_trends": [],
            "industry_signals": []
        }
        sources = []

        try:
            # K-Beauty 뉴스
            kbeauty_signals = await self.signal_collector.fetch_kbeauty_news(max_articles=15)
            if kbeauty_signals:
                data["kbeauty_news"] = [s.to_dict() for s in kbeauty_signals]
                for signal in kbeauty_signals[:3]:
                    sources.append(self.signal_collector.create_source_reference(signal))

            # Reddit 트렌드
            reddit_signals = await self.signal_collector.fetch_reddit_trends(
                subreddits=["SkincareAddiction", "AsianBeauty"],
                keywords=keywords,
                max_posts=10
            )
            if reddit_signals:
                data["reddit_trends"] = [s.to_dict() for s in reddit_signals]

            # 산업 전반 신호
            industry_signals = await self.signal_collector.fetch_industry_signals(keywords)
            if industry_signals:
                data["industry_signals"] = [s.to_dict() for s in industry_signals]

        except Exception as e:
            logger.error(f"Failed to collect Layer 2 data: {e}")

        layer_data = LayerData(
            layer=DataLayer.LAYER_2_CONSUMER,
            layer_name="소비자 트렌드",
            data=data,
            sources=sources
        )

        self.layer_data[DataLayer.LAYER_2_CONSUMER] = layer_data
        return layer_data

    async def collect_layer(self, layer_number: int, **kwargs) -> Optional[LayerData]:
        """
        특정 레이어 데이터 수집

        Args:
            layer_number: 레이어 번호 (1-4)
            **kwargs: 레이어별 추가 인자

        Returns:
            LayerData 또는 None
        """
        if layer_number == DataLayer.LAYER_4_MACRO:
            return await self.collect_layer_4_macro(**kwargs)
        elif layer_number == DataLayer.LAYER_3_INDUSTRY:
            return await self.collect_layer_3_industry(**kwargs)
        elif layer_number == DataLayer.LAYER_2_CONSUMER:
            return await self.collect_layer_2_consumer(**kwargs)
        elif layer_number == DataLayer.LAYER_1_AMAZON:
            # Layer 1은 기존 시스템에서 처리 (MetricsAgent 등)
            logger.info("Layer 1 (Amazon) data should be collected from existing system")
            return None
        else:
            logger.warning(f"Unknown layer: {layer_number}")
            return None

    async def collect_all_layers(self) -> Dict[int, LayerData]:
        """
        모든 레이어 데이터 수집 (Layer 1 제외)

        Returns:
            레이어별 데이터 딕셔너리
        """
        if not self._initialized:
            await self.initialize()

        # 병렬 수집
        await asyncio.gather(
            self.collect_layer_4_macro(),
            self.collect_layer_3_industry(),
            self.collect_layer_2_consumer()
        )

        return self.layer_data

    # =========================================================================
    # 인사이트 생성
    # =========================================================================

    def generate_layered_insight(
        self,
        amazon_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        4-Layer 기반 인사이트 생성

        Args:
            amazon_data: Layer 1 Amazon 데이터 (선택)

        Returns:
            인사이트 문자열
        """
        # 출처 관리자 초기화
        self.source_manager.reset_session()

        sections = []

        # 헤더
        today = datetime.now(KST).strftime("%Y-%m-%d")
        sections.append(f"# LANEIGE Amazon US 일일 인사이트")
        sections.append(f"> 생성일: {today}")
        sections.append("")

        # Layer 4: 거시경제/무역
        layer4 = self.layer_data.get(DataLayer.LAYER_4_MACRO)
        if layer4 and layer4.data:
            sections.append("## 🔍 원인 분석 (Why?)")
            sections.append("")
            sections.append("### Layer 4: 거시경제/무역")

            # 미국 수출 데이터
            us_export = layer4.data.get("us_export")
            if us_export:
                amount = us_export.get("amount_usd")
                yoy = us_export.get("yoy_change")

                if amount:
                    amount_str = f"${amount/1_000_000_000:.1f}B" if amount >= 1_000_000_000 else f"${amount/1_000_000:.1f}M"
                    yoy_str = f" (+{yoy:.1f}% YoY)" if yoy and yoy >= 0 else f" ({yoy:.1f}% YoY)" if yoy else ""

                    source = self.source_manager.add_source(
                        title=f"품목별 수출입통계 (HS Code 3304)",
                        publisher="관세청",
                        date=f"{us_export.get('year')}.{us_export.get('month')}",
                        source_type="government"
                    )
                    sections.append(f"• 화장품 대미 수출: {amount_str}{yoy_str} {source.to_citation()}")

            sections.append("")

        # Layer 3: 산업/기업
        layer3 = self.layer_data.get(DataLayer.LAYER_3_INDUSTRY)
        if layer3 and layer3.data:
            sections.append("### Layer 3: 산업/기업 동향")

            ir_report = layer3.data.get("ir_report")
            if ir_report:
                # Americas 실적
                americas = layer3.data.get("americas_insights", {})
                regional_perf = americas.get("regional_performance", [])

                if regional_perf:
                    perf = regional_perf[0]
                    revenue = perf.get("revenue_krw")
                    yoy = perf.get("revenue_yoy")

                    source = self.source_manager.add_source(
                        title=f"{ir_report.get('quarter')} {ir_report.get('year')} Earnings Release",
                        publisher="아모레퍼시픽 IR",
                        date=ir_report.get("release_date"),
                        url="https://www.apgroup.com/int/en/investors/amorepacific-corporation/ir-reports/quarterly-results/quarterly-results.html",
                        source_type="ir"
                    )

                    yoy_str = f"+{yoy:.1f}%" if yoy >= 0 else f"{yoy:.1f}%"
                    sections.append(f"• 아모레퍼시픽 Americas: {revenue:.1f}B KRW ({yoy_str} YoY) {source.to_citation()}")

                # 브랜드 하이라이트
                brand_highlights = layer3.data.get("brand_highlights", {})
                laneige = brand_highlights.get("LANEIGE", [])
                if laneige and laneige[0].get("highlights"):
                    sections.append(f"• LANEIGE: {laneige[0]['highlights'][0]}")

            sections.append("")

        # Layer 2: 소비자 트렌드
        layer2 = self.layer_data.get(DataLayer.LAYER_2_CONSUMER)
        if layer2 and layer2.data:
            sections.append("### Layer 2: 소비자 트렌드")

            # K-Beauty 뉴스
            kbeauty_news = layer2.data.get("kbeauty_news", [])
            if kbeauty_news:
                for news in kbeauty_news[:2]:
                    source = self.source_manager.add_source(
                        title=news.get("title", ""),
                        publisher=news.get("source", "").replace("_", " ").title(),
                        date=news.get("published_at", ""),
                        url=news.get("url"),
                        source_type="news"
                    )
                    title_short = news.get("title", "")[:50]
                    sections.append(f"• {title_short}... {source.to_citation()}")

            # Reddit 트렌드
            reddit = layer2.data.get("reddit_trends", [])
            if reddit:
                total_upvotes = sum(r.get("metadata", {}).get("upvotes", 0) for r in reddit[:5])
                if total_upvotes > 0:
                    sections.append(f"• Reddit: 최근 K-Beauty 관련 게시물 {len(reddit)}건 (누적 {total_upvotes:,} 업보트)")

            sections.append("")

        # Layer 1: Amazon (외부에서 주입)
        if amazon_data:
            sections.append("### Layer 1: Amazon 성과")

            if "laneige_rank" in amazon_data:
                sections.append(f"• Lip Sleeping Mask: {amazon_data['laneige_rank']}위")

            if "sos" in amazon_data:
                sections.append(f"• SoS: {amazon_data['sos']:.1f}%")

            sections.append("")

        # 참고자료
        refs = self.source_manager.generate_references_section()
        if refs:
            sections.append(refs)

        sections.append("")
        sections.append("---")
        sections.append("_본 리포트는 AI 분석 시스템에 의해 생성되었습니다._")

        return "\n".join(sections)

    def generate_layer_summary(self, layer_number: int) -> str:
        """
        특정 레이어 요약 생성

        Args:
            layer_number: 레이어 번호

        Returns:
            요약 문자열
        """
        layer_data = self.layer_data.get(layer_number)
        if not layer_data:
            return f"Layer {layer_number} 데이터가 수집되지 않았습니다."

        if layer_number == DataLayer.LAYER_4_MACRO:
            return self.public_collector.generate_insight_section()
        elif layer_number == DataLayer.LAYER_3_INDUSTRY:
            return self.ir_parser.generate_insight_section()
        elif layer_number == DataLayer.LAYER_2_CONSUMER:
            return self.signal_collector.generate_report_section()
        else:
            return f"Layer {layer_number} 요약 생성 불가"

    # =========================================================================
    # 데이터 저장 및 로드
    # =========================================================================

    def save_data(self) -> None:
        """수집된 데이터 저장"""
        filepath = self.data_dir / "layer_data.json"

        data = {
            "layers": {
                str(k): {
                    "layer": v.layer,
                    "layer_name": v.layer_name,
                    "collected_at": v.collected_at,
                    "data": v.data,
                    "sources": v.sources
                }
                for k, v in self.layer_data.items()
            },
            "updated_at": datetime.now(KST).isoformat()
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved layer data to {filepath}")

    def load_data(self) -> None:
        """저장된 데이터 로드"""
        filepath = self.data_dir / "layer_data.json"
        if not filepath.exists():
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            for layer_num, layer_info in data.get("layers", {}).items():
                self.layer_data[int(layer_num)] = LayerData(
                    layer=layer_info["layer"],
                    layer_name=layer_info["layer_name"],
                    collected_at=layer_info["collected_at"],
                    data=layer_info["data"],
                    sources=layer_info["sources"]
                )

            logger.info(f"Loaded {len(self.layer_data)} layers")

        except Exception as e:
            logger.warning(f"Failed to load layer data: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        return {
            "layers_collected": list(self.layer_data.keys()),
            "public_data_stats": self.public_collector.get_stats(),
            "ir_stats": self.ir_parser.get_stats(),
            "signal_stats": self.signal_collector.get_stats(),
            "source_stats": self.source_manager.get_stats(),
            "initialized": self._initialized
        }


# 편의 함수
async def create_market_intelligence_engine(
    api_key: Optional[str] = None
) -> MarketIntelligenceEngine:
    """
    MarketIntelligenceEngine 인스턴스 생성 및 초기화

    Args:
        api_key: 공공데이터 API 키

    Returns:
        초기화된 엔진
    """
    engine = MarketIntelligenceEngine(public_data_api_key=api_key)
    await engine.initialize()
    return engine

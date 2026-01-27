"""
Google Trends Collector
========================
Google Trends 데이터 수집기

## 사용 기술 (우선순위)
1. trendspyg (pytrends 대체, 무료, 오픈소스)
2. pytrends (레거시 fallback)

## 기능
- 뷰티/화장품 키워드 트렌드 수집
- LANEIGE 및 경쟁사 검색 관심도 추적
- 지역별 (US, KR, Global) 트렌드 분석

## 사용 예
```python
collector = GoogleTrendsCollector()
trends = await collector.fetch_beauty_trends()
print(trends)
```

## 주의사항
- pytrends는 2025년 4월 archived됨
- trendspyg 우선 사용 권장
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Type hints only (런타임에 import되지 않음)
if TYPE_CHECKING:
    from trendspyg import TrendsPyG
    from pytrends.request import TrendReq

# trendspyg (pytrends 대체, 권장)
try:
    from trendspyg import TrendsPyG
    TRENDSPYG_AVAILABLE = True
except ImportError:
    TrendsPyG = None  # type: ignore
    TRENDSPYG_AVAILABLE = False
    logger.debug("trendspyg not installed. Install with: pip install trendspyg")

# pytrends (레거시 fallback)
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    TrendReq = None  # type: ignore
    PYTRENDS_AVAILABLE = False
    logger.debug("pytrends not installed (archived April 2025)")


# 한국 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))


@dataclass
class TrendData:
    """트렌드 데이터"""
    keyword: str
    interest_over_time: List[Dict[str, Any]] = field(default_factory=list)
    related_queries: List[str] = field(default_factory=list)
    geo: str = "US"
    timeframe: str = "today 3-m"  # 최근 3개월
    collected_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GoogleTrendsCollector:
    """
    Google Trends 데이터 수집기

    trendspyg 라이브러리를 사용하여 Google Trends 데이터를 수집합니다.
    무료로 무제한 사용 가능합니다.
    """

    # 기본 뷰티 키워드
    DEFAULT_BEAUTY_KEYWORDS = [
        "LANEIGE",
        "COSRX",
        "Korean skincare",
        "Lip sleeping mask",
        "Glass skin"
    ]

    # LANEIGE vs 경쟁사 비교 키워드
    COMPETITOR_KEYWORDS = [
        "LANEIGE",
        "COSRX",
        "Beauty of Joseon",
        "Innisfree",
        "TIRTIR"
    ]

    # 카테고리별 키워드
    CATEGORY_KEYWORDS = {
        "lip_care": ["lip sleeping mask", "lip balm", "lip treatment", "LANEIGE lip"],
        "skin_care": ["Korean skincare", "glass skin", "snail mucin", "hyaluronic acid"],
        "face_makeup": ["cushion foundation", "Korean makeup", "TIRTIR cushion"]
    }

    def __init__(self, geo: str = "US", timeframe: str = "today 3-m"):
        """
        Args:
            geo: 지역 코드 (US, KR, 또는 빈 문자열=글로벌)
            timeframe: 기간 (today 3-m, today 12-m, today 5-y 등)
        """
        self.geo = geo
        self.timeframe = timeframe
        self._pytrends: Optional[TrendReq] = None
        self._trendspyg: Optional[TrendsPyG] = None
        self._enabled = os.getenv("ENABLE_GOOGLE_TRENDS", "true").lower() == "true"

        # 사용 가능한 백엔드 확인
        if TRENDSPYG_AVAILABLE:
            self._backend = "trendspyg"
        elif PYTRENDS_AVAILABLE:
            self._backend = "pytrends"
        else:
            self._backend = None

        # 데이터 저장 경로
        self.data_dir = Path("data/market_intelligence/trends")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Google Trends collector initialized with backend: {self._backend}")

    def _get_pytrends(self) -> Optional[TrendReq]:
        """TrendReq 인스턴스 반환 (legacy fallback)"""
        if not PYTRENDS_AVAILABLE:
            logger.warning("pytrends not available")
            return None

        if self._pytrends is None:
            self._pytrends = TrendReq(hl='en-US', tz=360)

        return self._pytrends

    def _get_trendspyg(self) -> Optional[TrendsPyG]:
        """TrendsPyG 인스턴스 반환 (권장)"""
        if not TRENDSPYG_AVAILABLE:
            logger.warning("trendspyg not available")
            return None

        if self._trendspyg is None:
            self._trendspyg = TrendsPyG()

        return self._trendspyg

    async def fetch_trends(self, keywords: List[str], geo: Optional[str] = None) -> List[TrendData]:
        """
        키워드 리스트의 트렌드 데이터 수집

        Args:
            keywords: 검색할 키워드 리스트 (최대 5개)
            geo: 지역 코드 (기본값: 인스턴스 설정)

        Returns:
            TrendData 리스트
        """
        if not self._enabled:
            logger.info("Google Trends collector disabled")
            return []

        if self._backend is None:
            logger.warning("No Google Trends backend available (install trendspyg or pytrends)")
            return []

        # Google Trends는 최대 5개 키워드만 비교 가능
        keywords = keywords[:5]
        geo = geo or self.geo

        trends = []
        now = datetime.now(KST).isoformat()

        try:
            # trendspyg 우선 사용
            if self._backend == "trendspyg":
                trends = await self._fetch_with_trendspyg(keywords, geo, now)
            elif self._backend == "pytrends":
                trends = await self._fetch_with_pytrends(keywords, geo, now)

            logger.info(f"Fetched trends for {len(keywords)} keywords (geo={geo}, backend={self._backend})")

        except Exception as e:
            logger.error(f"Error fetching Google Trends: {e}")
            # 빈 TrendData 반환
            for keyword in keywords:
                trends.append(TrendData(
                    keyword=keyword,
                    geo=geo,
                    timeframe=self.timeframe,
                    collected_at=now
                ))

        return trends

    async def _fetch_with_trendspyg(self, keywords: List[str], geo: str, now: str) -> List[TrendData]:
        """trendspyg로 트렌드 수집 (권장)"""
        trends = []
        trendspyg = self._get_trendspyg()

        if trendspyg is None:
            return trends

        def _fetch():
            result = {}
            for keyword in keywords:
                try:
                    # trendspyg는 키워드별로 개별 조회
                    data = trendspyg.interest_over_time(
                        keyword=keyword,
                        geo=geo if geo else "US",
                        timeframe=self.timeframe
                    )
                    result[keyword] = data
                except Exception as e:
                    logger.debug(f"trendspyg error for {keyword}: {e}")
                    result[keyword] = None
            return result

        results = await asyncio.get_event_loop().run_in_executor(None, _fetch)

        for keyword in keywords:
            interest_data = []
            data = results.get(keyword)

            if data is not None and hasattr(data, 'iterrows'):
                for date, row in data.iterrows():
                    interest_data.append({
                        "date": str(date.date()) if hasattr(date, 'date') else str(date),
                        "value": int(row.get(keyword, row.iloc[0]) if hasattr(row, 'get') else row)
                    })

            trends.append(TrendData(
                keyword=keyword,
                interest_over_time=interest_data,
                related_queries=[],  # trendspyg는 related queries 별도 조회 필요
                geo=geo,
                timeframe=self.timeframe,
                collected_at=now
            ))

        return trends

    async def _fetch_with_pytrends(self, keywords: List[str], geo: str, now: str) -> List[TrendData]:
        """pytrends로 트렌드 수집 (레거시 fallback)"""
        trends = []
        pytrends = self._get_pytrends()

        if pytrends is None:
            return trends

        def _fetch():
            pytrends.build_payload(
                kw_list=keywords,
                cat=0,
                timeframe=self.timeframe,
                geo=geo
            )
            interest_df = pytrends.interest_over_time()
            related = pytrends.related_queries()
            return interest_df, related

        interest_df, related = await asyncio.get_event_loop().run_in_executor(None, _fetch)

        for keyword in keywords:
            interest_data = []
            if not interest_df.empty and keyword in interest_df.columns:
                for date, row in interest_df.iterrows():
                    interest_data.append({
                        "date": str(date.date()),
                        "value": int(row[keyword])
                    })

            related_queries = []
            if keyword in related and related[keyword].get('top') is not None:
                top_queries = related[keyword]['top']
                if hasattr(top_queries, 'head'):
                    related_queries = top_queries.head(10)['query'].tolist()

            trends.append(TrendData(
                keyword=keyword,
                interest_over_time=interest_data,
                related_queries=related_queries,
                geo=geo,
                timeframe=self.timeframe,
                collected_at=now
            ))

        return trends

    async def fetch_beauty_trends(self) -> List[TrendData]:
        """기본 뷰티 키워드 트렌드 수집"""
        return await self.fetch_trends(self.DEFAULT_BEAUTY_KEYWORDS)

    async def fetch_competitor_trends(self) -> List[TrendData]:
        """LANEIGE vs 경쟁사 트렌드 비교"""
        return await self.fetch_trends(self.COMPETITOR_KEYWORDS)

    async def fetch_category_trends(self, category: str) -> List[TrendData]:
        """
        카테고리별 트렌드 수집

        Args:
            category: lip_care, skin_care, face_makeup
        """
        keywords = self.CATEGORY_KEYWORDS.get(category, self.DEFAULT_BEAUTY_KEYWORDS)
        return await self.fetch_trends(keywords)

    def generate_insight_section(self, trends: List[TrendData]) -> str:
        """
        인사이트 보고서용 섹션 생성

        Returns:
            마크다운 형식의 트렌드 인사이트
        """
        if not trends:
            return ""

        lines = ["### Google Trends 검색 관심도\n"]

        for trend in trends:
            if not trend.interest_over_time:
                continue

            # 최근 값과 변화율 계산
            recent_values = trend.interest_over_time[-4:]  # 최근 4주
            if len(recent_values) >= 2:
                latest = recent_values[-1]["value"]
                prev = recent_values[0]["value"]

                if prev > 0:
                    change = ((latest - prev) / prev) * 100
                    change_str = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"

                    # 트렌드 아이콘
                    icon = "📈" if change > 10 else "📉" if change < -10 else "➡️"

                    lines.append(f"- **{trend.keyword}**: {icon} {change_str} (최근 관심도: {latest})")

        # Related queries 추가
        all_related = []
        for trend in trends:
            all_related.extend(trend.related_queries[:3])

        if all_related:
            lines.append("\n**연관 검색어**: " + ", ".join(set(all_related)[:10]))

        return "\n".join(lines)

    async def save_trends(self, trends: List[TrendData], filename: Optional[str] = None) -> Path:
        """
        트렌드 데이터 저장

        Args:
            trends: 저장할 트렌드 데이터
            filename: 파일명 (기본값: trends_YYYY-MM-DD.json)

        Returns:
            저장된 파일 경로
        """
        if filename is None:
            date_str = datetime.now(KST).strftime("%Y-%m-%d")
            filename = f"trends_{date_str}.json"

        filepath = self.data_dir / filename

        data = {
            "collected_at": datetime.now(KST).isoformat(),
            "geo": self.geo,
            "timeframe": self.timeframe,
            "trends": [t.to_dict() for t in trends]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved trends to {filepath}")
        return filepath

    def load_latest_trends(self) -> Optional[Dict[str, Any]]:
        """가장 최근 트렌드 데이터 로드"""
        files = sorted(self.data_dir.glob("trends_*.json"), reverse=True)

        if not files:
            return None

        with open(files[0], 'r', encoding='utf-8') as f:
            return json.load(f)


# 테스트용 메인
if __name__ == "__main__":
    async def main():
        collector = GoogleTrendsCollector()

        print("Fetching beauty trends...")
        trends = await collector.fetch_beauty_trends()

        for trend in trends:
            print(f"\n{trend.keyword}:")
            if trend.interest_over_time:
                latest = trend.interest_over_time[-1]
                print(f"  Latest: {latest['date']} = {latest['value']}")
            if trend.related_queries:
                print(f"  Related: {', '.join(trend.related_queries[:5])}")

        # 인사이트 생성
        print("\n" + collector.generate_insight_section(trends))

        # 저장
        await collector.save_trends(trends)

    asyncio.run(main())

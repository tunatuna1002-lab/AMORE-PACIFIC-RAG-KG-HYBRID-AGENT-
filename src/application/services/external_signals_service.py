"""
External Signals Service
========================
뉴스/RSS/Reddit 외부 신호 수집과 3-Tier 시간 관련성 분류 (F6).

``_get_external_signals``는 원래 ``src/api/routes/export.py``에 있었고
``src/tools/exporters/export_handlers.py``가 그것을 역참조(tools -> api)했다.
이제 두 호출자가 모두 이 모듈로 온다. FastAPI에 의존하지 않는다.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SIGNAL_BRANDS = ["LANEIGE", "COSRX", "K-Beauty"]


class SignalRelevance(Enum):
    """외부 신호 시간적 관련성 분류"""

    TIER1_CORE = "core"  # 분석 기간 ±7일 - 직접 관련
    TIER2_BACKGROUND = "background"  # ±30일 또는 구조적 트렌드
    TIER3_ARCHIVE = "archive"  # 30일+ 이전, 장기 트렌드만


# 구조적 트렌드 키워드 (시간 민감도 낮음)
STRUCTURAL_TREND_KEYWORDS = [
    "trend",
    "growth",
    "market size",
    "industry",
    "expansion",
    "k-beauty",
    "clean beauty",
    "sustainable",
    "global",
    "market report",
    "forecast",
    "analysis",
]


def classify_signal_relevance(
    signal: Any, analysis_start: datetime, analysis_end: datetime
) -> SignalRelevance | None:
    """
    외부 신호의 시간적 관련성 분류 (3-Tier)

    Args:
        signal: ExternalSignal 객체
        analysis_start: 분석 시작일
        analysis_end: 분석 종료일

    Returns:
        SignalRelevance, 또는 제외 대상이면 None
    """
    # 신호 날짜 파싱
    signal_date = None
    published_at = getattr(signal, "published_at", None)

    if published_at:
        try:
            if isinstance(published_at, str):
                # ISO 형식 또는 다양한 형식 처리
                published_at = published_at.replace("Z", "+00:00")
                if "T" in published_at:
                    signal_date = datetime.fromisoformat(published_at).replace(tzinfo=None)
                else:
                    signal_date = datetime.strptime(published_at[:10], "%Y-%m-%d")
            elif isinstance(published_at, datetime):
                signal_date = published_at.replace(tzinfo=None)
        except Exception:
            signal_date = None

    # 구조적 트렌드 여부 확인
    title = getattr(signal, "title", "").lower()
    content = getattr(signal, "content", "").lower()
    combined_text = title + " " + content

    is_structural = any(keyword.lower() in combined_text for keyword in STRUCTURAL_TREND_KEYWORDS)

    # 날짜 기반 분류
    if signal_date:
        days_from_end = (signal_date.date() - analysis_end.date()).days

        # TIER 1: 분석 기간 ±7일
        if -7 <= days_from_end <= 7:
            return SignalRelevance.TIER1_CORE

        # TIER 2: ±30일 또는 구조적 트렌드
        if -30 <= days_from_end <= 30 or is_structural:
            return SignalRelevance.TIER2_BACKGROUND

        # TIER 3: 30일+ 이전이지만 구조적 트렌드인 경우만
        if is_structural:
            return SignalRelevance.TIER3_ARCHIVE

        # 이벤트성 뉴스 30일+ 외는 None 반환 (제외 대상)
        return None

    # 날짜 없으면 구조적 트렌드인 경우만 TIER 2
    if is_structural:
        return SignalRelevance.TIER2_BACKGROUND

    # 날짜 없고 구조적도 아니면 TIER 2로 분류 (보수적 접근)
    return SignalRelevance.TIER2_BACKGROUND


async def get_external_signals(
    days: int = 7,
    brands: list | None = None,
    include_tavily: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    External Signal 수집 및 3-Tier 분류

    Args:
        days: 검색 기간 (일)
        brands: 검색할 브랜드 리스트
        include_tavily: Tavily 뉴스 검색 포함 여부
        start_date: 분석 시작일 (YYYY-MM-DD) - 3-Tier 분류용
        end_date: 분석 종료일 (YYYY-MM-DD) - 3-Tier 분류용

    Returns:
        {"signals": [...], "classified": {tier1_core, tier2_background, tier3_archive},
         "report_section": str}
    """
    from src.tools.collectors.external_signal_collector import ExternalSignalCollector

    all_signals: list = []

    try:
        collector = ExternalSignalCollector()
        await collector.initialize()

        # 브랜드 기본값
        if not brands:
            brands = list(DEFAULT_SIGNAL_BRANDS)

        # 1. Tavily 뉴스 검색 (최적화)
        if include_tavily:
            try:
                # 검색 기간: 최소 14일, 최대 30일 (분석 기간에 맞춤)
                search_days = min(max(days, 14), 30)
                tavily_signals = await collector.fetch_tavily_news(
                    brands=brands[:3],
                    topics=["K-Beauty skincare", "Amazon beauty bestseller"],
                    days=search_days,
                    max_results=15,  # 더 많은 결과 수집
                )
                all_signals.extend(tavily_signals)
                logger.info(
                    f"Collected {len(tavily_signals)} Tavily news signals "
                    f"(search_days={search_days})"
                )
            except Exception as e:
                logger.warning(f"Tavily news fetch failed: {e}")

        # 2. RSS 피드 수집
        try:
            rss_signals = await collector.fetch_all_rss_feeds(
                keywords=brands + ["skincare", "lip care"]
            )
            all_signals.extend(rss_signals)
            logger.info(f"Collected {len(rss_signals)} RSS signals")
        except Exception as e:
            logger.warning(f"RSS fetch failed: {e}")

        # 3. Reddit 트렌드 수집
        try:
            reddit_signals = await collector.fetch_reddit_trends()
            all_signals.extend(reddit_signals)
            logger.info(f"Collected {len(reddit_signals)} Reddit signals")
        except Exception as e:
            logger.warning(f"Reddit fetch failed: {e}")

        # 기존 수집된 신호도 추가
        if collector.signals:
            all_signals.extend(collector.signals)

        # 중복 제거 (URL 기준)
        seen_urls = set()
        unique_signals = []
        for signal in all_signals:
            url = getattr(signal, "url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_signals.append(signal)
            elif not url:
                unique_signals.append(signal)

        # 3-Tier 분류
        classified: dict[str, list] = {
            "tier1_core": [],  # 본문 인용 + 참고자료
            "tier2_background": [],  # 참고자료만
            "tier3_archive": [],  # 배경 자료 섹션
        }

        if start_date and end_date:
            try:
                analysis_start = datetime.strptime(start_date, "%Y-%m-%d")
                analysis_end = datetime.strptime(end_date, "%Y-%m-%d")

                filtered_signals = []
                for signal in unique_signals:
                    relevance = classify_signal_relevance(signal, analysis_start, analysis_end)

                    if relevance == SignalRelevance.TIER1_CORE:
                        classified["tier1_core"].append(signal)
                        filtered_signals.append(signal)
                    elif relevance == SignalRelevance.TIER2_BACKGROUND:
                        classified["tier2_background"].append(signal)
                        filtered_signals.append(signal)
                    elif relevance == SignalRelevance.TIER3_ARCHIVE:
                        classified["tier3_archive"].append(signal)
                        # TIER3는 참고자료에만 포함, 본문 분석에서는 제외
                    # relevance가 None이면 제외 (이벤트성 + 30일+ 외)

                logger.info(
                    f"Signal classification: TIER1={len(classified['tier1_core'])}, "
                    f"TIER2={len(classified['tier2_background'])}, "
                    f"TIER3={len(classified['tier3_archive'])}"
                )

                # 필터링된 신호로 교체 (TIER3 제외)
                unique_signals = filtered_signals

            except Exception as e:
                logger.warning(f"Signal classification failed: {e}")
                # 분류 실패 시 전체를 TIER2로 처리
                classified["tier2_background"] = unique_signals

        # 보고서 섹션 생성
        if unique_signals:
            # 컬렉터에 신호 추가 후 보고서 생성
            for signal in unique_signals:
                if signal not in collector.signals:
                    collector.signals.append(signal)
            report_section = collector.generate_report_section(days=days)
        else:
            report_section = ""

        return {
            "signals": unique_signals,
            "classified": classified,
            "report_section": report_section,
        }

    except Exception as e:
        logger.warning(f"External signal collection failed: {e}")
        return {
            "signals": [],
            "classified": {"tier1_core": [], "tier2_background": [], "tier3_archive": []},
            "report_section": "",
        }
    finally:
        try:
            await collector.close()
        except Exception:
            logger.warning("Suppressed Exception", exc_info=True)


def signal_source_status() -> dict[str, Any]:
    """외부 신호 API 상태 (설정/사용 가능 여부)."""
    return {
        "tavily": {
            "configured": bool(os.getenv("TAVILY_API_KEY")),
            "description": "뉴스 검색 API (월 1,000건 무료)",
            "docs": "https://tavily.com",
        },
        "gnews": {
            "configured": bool(os.getenv("GNEWS_API_KEY")),
            "description": "뉴스 API (일 100건 무료)",
            "docs": "https://gnews.io",
        },
        "rss_feeds": {
            "available": True,
            "count": 10,
            "sources": [
                "Allure",
                "Byrdie",
                "Cosmetics Design Asia",
                "Cosmetics Business",
                "Vogue Beauty",
                "WWD Beauty",
                "Beautyindependent",
                "Global Cosmetics News",
                "Happi",
                "CosmeticsDesign Europe",
            ],
        },
        "reddit": {
            "available": True,
            "description": "JSON API (무료, 인증 불필요)",
            "subreddits": ["AsianBeauty", "SkincareAddiction", "MakeupAddiction"],
        },
        "public_data": {
            "customs_korea": {
                "configured": bool(os.getenv("DATA_GO_KR_API_KEY")),
                "description": "관세청 수출입통계",
            },
            "mfds_korea": {
                "configured": bool(os.getenv("DATA_GO_KR_API_KEY")),
                "description": "식약처 기능성화장품 DB",
            },
        },
        "signal_classification": {
            "tier1_core": "분석 기간 ±7일 - 직접 관련 뉴스",
            "tier2_background": "±30일 또는 구조적 트렌드",
            "tier3_archive": "30일+ 이전, 장기 트렌드만 포함",
        },
    }

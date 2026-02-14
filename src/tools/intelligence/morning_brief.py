"""
Morning Brief Generator
=======================
매일 아침 자동 발송되는 뉴스레터 스타일 인사이트 생성

주요 기능:
- 전날 크롤링 데이터 기반 시장 현황 요약
- LANEIGE 성과 분석
- 경쟁사 동향
- 오늘의 액션 포인트

발송 스케줄: 매일 아침 8:00 KST
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from litellm import acompletion

from src.shared.constants import KST

logger = logging.getLogger(__name__)


# 한국 시간대 (UTC+9)
@dataclass
class MorningBriefData:
    """Morning Brief에 들어갈 데이터"""

    date: str
    day_of_week: str

    # LANEIGE 성과
    laneige_products: list[dict] = field(default_factory=list)
    laneige_avg_rank: float = 0.0
    laneige_rank_change: float = 0.0  # 전일 대비
    laneige_top10_count: int = 0
    laneige_sos: float = 0.0  # Share of Shelf

    # 경쟁사 동향
    competitor_highlights: list[str] = field(default_factory=list)
    market_changes: list[str] = field(default_factory=list)

    # 카테고리별 현황
    category_stats: dict[str, dict] = field(default_factory=dict)

    # 알림 요약
    alerts_count: int = 0
    critical_alerts: list[str] = field(default_factory=list)

    # 액션 포인트 (LLM 생성)
    action_points: list[str] = field(default_factory=list)

    # AI 인사이트 (LLM 생성)
    ai_summary: str = ""
    ai_recommendations: list[str] = field(default_factory=list)


class MorningBriefGenerator:
    """
    Morning Brief 생성기

    매일 아침 시장 현황을 요약한 뉴스레터를 생성합니다.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        data_source: Any | None = None,  # MarketIntelligenceEngine
    ):
        self.model = model
        self.data_source = data_source
        self.temperature = float(os.getenv("LLM_TEMPERATURE_INSIGHT", "0.6"))

    async def generate(
        self,
        crawl_data: dict | None = None,
        metrics_data: dict | None = None,
        previous_data: dict | None = None,
    ) -> MorningBriefData:
        """
        Morning Brief 데이터 생성

        Args:
            crawl_data: 최신 크롤링 데이터
            metrics_data: KPI 메트릭 데이터
            previous_data: 전일 데이터 (비교용)

        Returns:
            MorningBriefData
        """
        now = datetime.now(KST)

        # 기본 데이터 구조
        brief = MorningBriefData(
            date=now.strftime("%Y.%m.%d"),
            day_of_week=self._get_korean_day(now.weekday()),
        )

        # 크롤링 데이터가 있으면 분석
        if crawl_data:
            await self._analyze_crawl_data(brief, crawl_data, previous_data)

        # 메트릭 데이터가 있으면 추가
        if metrics_data:
            self._add_metrics(brief, metrics_data)

        # LLM으로 인사이트 생성
        await self._generate_ai_insights(brief)

        return brief

    def _get_korean_day(self, weekday: int) -> str:
        """요일 한글 변환"""
        days = ["월", "화", "수", "목", "금", "토", "일"]
        return days[weekday]

    async def _analyze_crawl_data(
        self, brief: MorningBriefData, crawl_data: dict, previous_data: dict | None = None
    ) -> None:
        """크롤링 데이터 분석"""
        products = crawl_data.get("products", [])

        # LANEIGE 제품 필터링
        laneige_products = [p for p in products if p.get("brand", "").upper() == "LANEIGE"]

        brief.laneige_products = laneige_products

        if laneige_products:
            # 평균 순위
            ranks = [p.get("rank", 100) for p in laneige_products]
            brief.laneige_avg_rank = sum(ranks) / len(ranks)

            # Top 10 진입 제품 수
            brief.laneige_top10_count = len([r for r in ranks if r <= 10])

            # Share of Shelf (Top 100 기준)
            total_products = len(products)
            if total_products > 0:
                brief.laneige_sos = (len(laneige_products) / total_products) * 100

        # 전일 대비 변화
        if previous_data:
            prev_products = previous_data.get("products", [])
            prev_laneige = [p for p in prev_products if p.get("brand", "").upper() == "LANEIGE"]
            if prev_laneige:
                prev_ranks = [p.get("rank", 100) for p in prev_laneige]
                prev_avg = sum(prev_ranks) / len(prev_ranks)
                brief.laneige_rank_change = prev_avg - brief.laneige_avg_rank  # 양수면 상승

        # 경쟁사 동향 분석
        await self._analyze_competitors(brief, products, previous_data)

        # 카테고리별 통계
        self._calculate_category_stats(brief, crawl_data)

    async def _analyze_competitors(
        self, brief: MorningBriefData, products: list[dict], previous_data: dict | None = None
    ) -> None:
        """경쟁사 동향 분석"""
        # 주요 경쟁사 리스트 (K-Beauty + LANEIGE 직접 경쟁 브랜드)
        competitors = [
            # K-Beauty
            "E.L.F.",
            "COSRX",
            "ANUA",
            "BEAUTY OF JOSEON",
            "MEDICUBE",
            "SOME BY MI",
            "BIODANCE",
            "SKIN1004",
            "TORRIDEN",
            "MIXSOON",
            # Lip Care 직접 경쟁
            "AQUAPHOR",
            "BURT'S BEES",
            "SUMMER FRIDAYS",
            "LANEIGE",
            "NIVEA",
            "CARMEX",
            "CHAPSTICK",
            "VASELINE",
            # Skincare/Beauty 주요 브랜드
            "CERAVE",
            "THE ORDINARY",
            "NEUTROGENA",
            "OLAY",
            "NYX",
            "COVERGIRL",
            "L'OREAL",
            "MAYBELLINE",
            "HERO COSMETICS",
            "PAULA'S CHOICE",
        ]

        for product in products[:20]:  # Top 20만 분석
            brand = product.get("brand", "").upper()
            rank = product.get("rank", 0)
            name = (product.get("product_name") or product.get("title") or "")[:50]

            # LANEIGE가 아닌 주요 브랜드 하이라이트
            if brand != "LANEIGE" and any(c.upper() in brand for c in competitors):
                if rank <= 5:
                    brief.competitor_highlights.append(f"{brand} #{rank}: {name}")

        # 순위 변동이 큰 제품 감지
        if previous_data:
            prev_products = {
                p.get("asin"): p.get("rank", 100) for p in previous_data.get("products", [])
            }

            for product in products[:50]:
                asin = product.get("asin")
                current_rank = product.get("rank", 100)
                prev_rank = prev_products.get(asin, 100)

                change = prev_rank - current_rank
                if abs(change) >= 10:
                    brand = product.get("brand", "Unknown")
                    name = (product.get("product_name") or product.get("title") or "")[:40]
                    direction = "상승" if change > 0 else "하락"
                    arrow = "🔺" if change > 0 else "🔻"
                    brief.market_changes.append(
                        f"{arrow} {brand} {abs(change)}등 {direction} (#{prev_rank}→#{current_rank}) {name}"
                    )

    def _calculate_category_stats(self, brief: MorningBriefData, crawl_data: dict) -> None:
        """카테고리별 통계"""
        category = crawl_data.get("category", "Unknown")
        products = crawl_data.get("products", [])

        laneige_in_cat = [p for p in products if p.get("brand", "").upper() == "LANEIGE"]

        brief.category_stats[category] = {
            "total_products": len(products),
            "laneige_count": len(laneige_in_cat),
            "laneige_best_rank": min([p.get("rank", 100) for p in laneige_in_cat])
            if laneige_in_cat
            else None,
            "top_brand": products[0].get("brand") if products else None,
        }

    def _add_metrics(self, brief: MorningBriefData, metrics_data: dict) -> None:
        """KPI 메트릭 추가"""
        if "sos" in metrics_data:
            brief.laneige_sos = metrics_data["sos"]
        if "alerts" in metrics_data:
            brief.alerts_count = len(metrics_data["alerts"])
            brief.critical_alerts = [
                a.get("message", "")
                for a in metrics_data.get("alerts", [])
                if a.get("severity") == "critical"
            ][:3]  # 최대 3개

    async def _generate_ai_insights(self, brief: MorningBriefData) -> None:
        """LLM으로 AI 인사이트 생성"""
        prompt = f"""
당신은 AMOREPACIFIC의 시장 분석 전문가입니다.
아래 데이터를 바탕으로 간결하고 액션 가능한 인사이트를 생성해주세요.

## 현재 상황 ({brief.date} {brief.day_of_week}요일)

### LANEIGE 성과
- 평균 순위: {brief.laneige_avg_rank:.1f}등
- 전일 대비 변화: {"+" if brief.laneige_rank_change > 0 else ""}{brief.laneige_rank_change:.1f}등
- Top 10 진입 제품: {brief.laneige_top10_count}개
- Share of Shelf: {brief.laneige_sos:.1f}%

### 경쟁사 동향
{chr(10).join(["- " + h for h in brief.competitor_highlights[:5]]) or "- 특이사항 없음"}

### 시장 변화
{chr(10).join(["- " + m for m in brief.market_changes[:5]]) or "- 큰 변동 없음"}

## 요청
1. **오늘의 핵심 요약** (2-3문장): 가장 중요한 시장 동향
2. **액션 포인트** (3개): 오늘 실행할 구체적인 행동
3. **주의 사항** (1-2개): 모니터링해야 할 리스크

JSON 형식으로 응답해주세요:
{{
    "summary": "핵심 요약 문장",
    "action_points": ["액션1", "액션2", "액션3"],
    "warnings": ["주의사항1"]
}}
"""

        try:
            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )

            import json

            result = json.loads(response.choices[0].message.content)

            brief.ai_summary = result.get("summary", "")
            brief.action_points = result.get("action_points", [])
            brief.ai_recommendations = result.get("warnings", [])

        except Exception as e:
            logger.error(f"AI insight generation failed: {e}")
            brief.ai_summary = "AI 분석을 생성하지 못했습니다."
            brief.action_points = ["데이터 확인 필요", "수동 분석 권장"]


# =============================================================================
# HTML 템플릿
# =============================================================================

MORNING_BRIEF_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AMORE Daily Brief</title>
</head>
<body style="margin: 0; padding: 0; font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f0f4f8; line-height: 1.6;">
    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">

        <!-- Header -->
        <div style="background: linear-gradient(135deg, #001C58 0%, #1F5795 100%); padding: 30px; border-radius: 16px 16px 0 0; text-align: center;">
            <div style="font-size: 32px; margin-bottom: 8px;">☀️</div>
            <h1 style="margin: 0; color: white; font-size: 24px; font-weight: 700; letter-spacing: -0.5px;">
                AMORE Daily Brief
            </h1>
            <p style="margin: 8px 0 0 0; color: rgba(255,255,255,0.9); font-size: 16px;">
                {date} ({day_of_week})
            </p>
        </div>

        <!-- AI Summary -->
        <div style="background: white; padding: 24px;">
            <h2 style="margin: 0 0 12px 0; color: #001C58; font-size: 16px; font-weight: 600;">
                💡 오늘의 핵심
            </h2>
            <p style="margin: 0; color: #334155; font-size: 15px; line-height: 1.7;">
                {ai_summary}
            </p>
        </div>

        <!-- LANEIGE Performance -->
        <div style="background: white; padding: 24px; margin-top: 2px;">
            <h2 style="margin: 0 0 16px 0; color: #001C58; font-size: 16px; font-weight: 600;">
                📊 LANEIGE 성과
            </h2>

            <div style="display: flex; flex-wrap: wrap; gap: 12px;">
                <!-- Avg Rank -->
                <div style="flex: 1; min-width: 120px; background: #f8fafc; padding: 16px; border-radius: 12px; text-align: center;">
                    <div style="font-size: 28px; font-weight: 700; color: #001C58;">{avg_rank}</div>
                    <div style="font-size: 12px; color: #64748b; margin-top: 4px;">평균 순위</div>
                    <div style="font-size: 13px; color: {rank_change_color}; margin-top: 4px;">{rank_change_text}</div>
                </div>

                <!-- Top 10 -->
                <div style="flex: 1; min-width: 120px; background: #f8fafc; padding: 16px; border-radius: 12px; text-align: center;">
                    <div style="font-size: 28px; font-weight: 700; color: #1F5795;">{top10_count}</div>
                    <div style="font-size: 12px; color: #64748b; margin-top: 4px;">Top 10 제품</div>
                </div>

                <!-- SoS -->
                <div style="flex: 1; min-width: 120px; background: #f8fafc; padding: 16px; border-radius: 12px; text-align: center;">
                    <div style="font-size: 28px; font-weight: 700; color: #059669;">{sos}%</div>
                    <div style="font-size: 12px; color: #64748b; margin-top: 4px;">Share of Shelf</div>
                </div>
            </div>
        </div>

        <!-- Competitor Watch -->
        <div style="background: white; padding: 24px; margin-top: 2px;">
            <h2 style="margin: 0 0 16px 0; color: #001C58; font-size: 16px; font-weight: 600;">
                👀 경쟁사 동향
            </h2>
            <div style="background: #fef3c7; padding: 16px; border-radius: 10px; border-left: 4px solid #f59e0b;">
                {competitor_section}
            </div>
        </div>

        <!-- Market Changes -->
        {market_changes_section}

        <!-- Action Points -->
        <div style="background: white; padding: 24px; margin-top: 2px;">
            <h2 style="margin: 0 0 16px 0; color: #001C58; font-size: 16px; font-weight: 600;">
                🎯 오늘의 액션 포인트
            </h2>
            <div style="background: #ecfdf5; padding: 16px; border-radius: 10px;">
                {action_points_html}
            </div>
        </div>

        <!-- Warnings -->
        {warnings_section}

        <!-- Footer -->
        <div style="background: #001C58; padding: 20px; border-radius: 0 0 16px 16px; text-align: center;">
            <p style="margin: 0; color: rgba(255,255,255,0.7); font-size: 12px;">
                AMORE Market Intelligence Agent<br>
                Amazon US Market Analysis
            </p>
            <p style="margin: 12px 0 0 0; color: rgba(255,255,255,0.5); font-size: 11px;">
                이 메일은 자동으로 생성되었습니다. 문의: tunatuna1002@gmail.com
            </p>
        </div>

    </div>
</body>
</html>
"""


def render_morning_brief_html(brief: MorningBriefData) -> str:
    """Morning Brief 데이터를 HTML로 렌더링"""

    # 순위 변화 텍스트
    if brief.laneige_rank_change > 0:
        rank_change_text = f"▲ {brief.laneige_rank_change:.1f}"
        rank_change_color = "#059669"  # green
    elif brief.laneige_rank_change < 0:
        rank_change_text = f"▼ {abs(brief.laneige_rank_change):.1f}"
        rank_change_color = "#dc2626"  # red
    else:
        rank_change_text = "━ 0"
        rank_change_color = "#64748b"  # gray

    # 경쟁사 섹션
    if brief.competitor_highlights:
        competitor_items = "<br>".join([f"• {h}" for h in brief.competitor_highlights[:5]])
        competitor_section = (
            f'<p style="margin: 0; color: #92400e; font-size: 14px;">{competitor_items}</p>'
        )
    else:
        competitor_section = (
            '<p style="margin: 0; color: #92400e; font-size: 14px;">특이사항 없음</p>'
        )

    # 액션 포인트
    if brief.action_points:
        action_items = "".join(
            [
                f'<table cellpadding="0" cellspacing="0" border="0" style="margin-bottom: 8px; width: 100%;"><tr>'
                f'<td style="width: 28px; vertical-align: top; padding-top: 2px;">'
                f'<div style="background: #059669; color: white; font-size: 12px; width: 22px; height: 22px; border-radius: 50%; text-align: center; line-height: 22px;">{i + 1}</div>'
                f"</td>"
                f'<td style="color: #065f46; font-size: 14px; line-height: 1.5; vertical-align: top;">{point}</td>'
                f"</tr></table>"
                for i, point in enumerate(brief.action_points[:5])
            ]
        )
        action_points_html = action_items
    else:
        action_points_html = (
            '<p style="margin: 0; color: #065f46; font-size: 14px;">액션 포인트 없음</p>'
        )

    # 순위 변동 섹션
    if brief.market_changes:
        change_items = "".join(
            [
                f'<div style="display: flex; align-items: center; margin-bottom: 6px;">'
                f'<span style="font-size: 14px; color: #1e3a5f;">{change}</span>'
                f"</div>"
                for change in brief.market_changes[:8]
            ]
        )
        market_changes_section = f"""
        <div style="background: white; padding: 24px; margin-top: 2px;">
            <h2 style="margin: 0 0 16px 0; color: #001C58; font-size: 16px; font-weight: 600;">
                📈 주요 순위 변동
            </h2>
            <div style="background: #eff6ff; padding: 16px; border-radius: 10px; border-left: 4px solid #3b82f6;">
                {change_items}
            </div>
        </div>
        """
    else:
        market_changes_section = ""

    # 주의사항 섹션
    if brief.ai_recommendations or brief.critical_alerts:
        warnings = brief.ai_recommendations + brief.critical_alerts
        warning_items = "<br>".join([f"⚠️ {w}" for w in warnings[:3]])
        warnings_section = f"""
        <div style="background: white; padding: 24px; margin-top: 2px;">
            <h2 style="margin: 0 0 16px 0; color: #001C58; font-size: 16px; font-weight: 600;">
                ⚠️ 주의 사항
            </h2>
            <div style="background: #fef2f2; padding: 16px; border-radius: 10px; border-left: 4px solid #dc2626;">
                <p style="margin: 0; color: #991b1b; font-size: 14px;">{warning_items}</p>
            </div>
        </div>
        """
    else:
        warnings_section = ""

    # 템플릿 렌더링
    html = MORNING_BRIEF_TEMPLATE.format(
        date=brief.date,
        day_of_week=brief.day_of_week,
        ai_summary=brief.ai_summary or "데이터 분석 중입니다.",
        avg_rank=f"{brief.laneige_avg_rank:.1f}" if brief.laneige_avg_rank else "-",
        rank_change_text=rank_change_text,
        rank_change_color=rank_change_color,
        top10_count=brief.laneige_top10_count,
        sos=f"{brief.laneige_sos:.1f}" if brief.laneige_sos else "0",
        competitor_section=competitor_section,
        market_changes_section=market_changes_section,
        action_points_html=action_points_html,
        warnings_section=warnings_section,
    )

    return html

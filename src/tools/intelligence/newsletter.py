"""
Newsletter (Morning Brief + Insight Report)
===========================================
아침 뉴스레터와 인사이트 리포트 이메일 생성·발송.

Phase 4: ``src.core.brain`` 이 들고 있던 두 메서드를 옮겼다. 둘 다 brain 상태를
참조하지 않는 순수 흐름이었고(``self`` 참조 0), brain 은 초기화·DI·파사드만 남긴다.
brain 은 ``send_morning_brief()`` 의 False/예외를 받아 자신의 오류 카운터를 올린다.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.domain.brand import is_target_brand

logger = logging.getLogger(__name__)


async def send_morning_brief() -> bool:
    """
    Morning Brief 뉴스레터 생성 및 발송

    매일 아침 8시 KST에 자동 실행됩니다.
    전날 크롤링 데이터를 기반으로 시장 현황을 요약합니다.

    인사이트 리포트 이메일도 함께 발송합니다.

    Returns:
        Morning Brief 발송에 성공하면 True. 발송기 비활성·수신자 없음·예외는 False
        (호출자인 brain이 자신의 오류 카운터를 올린다).
    """

    logger.info("Generating Morning Brief...")

    try:
        # 1. 최신 크롤링 데이터 가져오기
        from src.tools.intelligence.market_intelligence import MarketIntelligenceEngine

        mi = MarketIntelligenceEngine()
        latest_data = await mi.get_latest_data()

        products = latest_data.get("products", [])
        crawl_data = {
            "products": products,
            "category": "All Categories",
        }

        # 2. Morning Brief 생성
        from src.tools.intelligence.morning_brief import (
            MorningBriefGenerator,
            render_morning_brief_html,
        )

        generator = MorningBriefGenerator()
        brief_data = await generator.generate(
            crawl_data=crawl_data,
            metrics_data=latest_data.get("metrics"),
        )

        # 3. HTML 렌더링
        html_content = render_morning_brief_html(brief_data)

        # 4. 이메일 발송
        from src.tools.notifications.email_sender import EmailSender

        sender = EmailSender()

        if not sender.is_enabled():
            logger.warning("Email sender is disabled, Morning Brief not sent")
            return False

        # 수신자 목록 (환경변수에서)
        recipients_str = os.getenv("ALERT_RECIPIENTS", "")
        recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]

        if not recipients:
            logger.warning("No recipients configured for Morning Brief")
            return False

        # 기존 Morning Brief 발송
        result = await sender.send_morning_brief(
            recipients=recipients,
            html_content=html_content,
            date_str=brief_data.date,
        )

        if result.success:
            logger.info(f"Morning Brief sent successfully to {len(result.sent_to)} recipients")
        else:
            logger.error(f"Morning Brief send failed: {result.message}")

        # 5. 인사이트 리포트 이메일도 발송
        await send_insight_report_email(
            products, recipients, sender, metrics_data=latest_data.get("metrics")
        )
        return bool(result.success)

    except Exception as e:
        logger.error(f"Morning Brief generation failed: {e}")
        return False


async def send_insight_report_email(
    products: list,
    recipients: list[str],
    sender,
    metrics_data: dict[str, Any] | None = None,
) -> None:
    """
    인사이트 리포트 이메일 발송 (자동)

    Morning Brief와 함께 발송되는 상세 인사이트 리포트입니다.

    Args:
        products: 최신 크롤링 제품 목록
        recipients: 수신자 목록
        sender: EmailSender (send_insight_report 제공)
        metrics_data: MetricsAgent 결과 (있으면 인사이트 생성에 전달)
    """
    from datetime import datetime

    try:
        logger.info("Sending Insight Report email...")

        # KPI 계산
        laneige_products = [p for p in products if is_target_brand(p.get("brand"))]
        avg_rank = (
            sum(p.get("rank", 100) for p in laneige_products) / len(laneige_products)
            if laneige_products
            else 0
        )

        # SoS 계산 (Top 100 기준)
        top100 = products[:100]
        laneige_in_top100 = len([p for p in top100 if is_target_brand(p.get("brand"))])
        sos = (laneige_in_top100 / len(top100) * 100) if top100 else 0

        # HHI 계산
        brand_counts = {}
        for p in top100:
            brand = p.get("brand", "Unknown")
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        hhi = (
            sum((count / len(top100) * 100) ** 2 for count in brand_counts.values())
            if top100
            else 0
        )

        # 인사이트 생성 (HybridInsightAgent 사용)
        insight_content = "<p>현재 생성된 인사이트가 없습니다.</p>"
        try:
            from src.agents.hybrid_insight_agent import HybridInsightAgent

            insight_agent = HybridInsightAgent()
            # HybridInsightAgent.execute(metrics_data, crawl_data, crawl_summary)
            # crawl_data는 categories -> rank_records 구조를 기대한다.
            sample_products = products[:50]
            now_iso = datetime.now().isoformat()
            crawl_summary = {
                "total_products": len(sample_products),
                "categories": ["All Categories"],
            }
            crawl_payload = {
                "collected_at": now_iso,
                "categories": {"all_categories": {"rank_records": sample_products}},
                "summary": crawl_summary,
            }
            metrics_payload: dict[str, Any] = dict(metrics_data or {})
            metrics_payload.setdefault(
                "metadata",
                {"data_date": datetime.now().strftime("%Y-%m-%d"), "generated_at": now_iso},
            )
            insight_result = await insight_agent.execute(
                metrics_data=metrics_payload,
                crawl_data=crawl_payload,
                crawl_summary=crawl_summary,
            )
            raw_insight = (insight_result or {}).get("daily_insight") or ""
            if raw_insight.strip():
                insight_content = raw_insight.replace("\n\n", "</p><p>").replace("\n", "<br>")
                insight_content = f"<p>{insight_content}</p>"
        except Exception as e:
            logger.warning(f"Failed to generate insight for email: {e}")

        # Top 10 제품 데이터
        top10_products = []
        for i, p in enumerate(products[:10]):
            top10_products.append(
                {
                    "rank": i + 1,
                    "name": p.get("title", "N/A"),
                    "brand": p.get("brand", "Unknown"),
                    "change": p.get("rank_change", 0),
                }
            )

        # 브랜드별 변동
        brand_changes = []
        for brand in ["LANEIGE", "e.l.f.", "Maybelline", "Summer Fridays", "COSRX"]:
            brand_products = [p for p in products if p.get("brand") == brand]
            if brand_products:
                avg_change = sum(p.get("rank_change", 0) for p in brand_products) / len(
                    brand_products
                )
                if avg_change > 0:
                    brand_changes.append(
                        {
                            "brand": brand,
                            "change_text": f"평균 ▲{avg_change:.1f} 상승",
                            "color": "#28a745",
                        }
                    )
                elif avg_change < 0:
                    brand_changes.append(
                        {
                            "brand": brand,
                            "change_text": f"평균 ▼{abs(avg_change):.1f} 하락",
                            "color": "#dc3545",
                        }
                    )

        # 리포트 날짜
        report_date = datetime.now().strftime("%Y년 %m월 %d일")

        # 대시보드 URL (Railway 자동 감지)
        def get_base_url() -> str:
            if url := os.getenv("DASHBOARD_URL"):
                return url.rstrip("/")
            if railway_domain := os.getenv("RAILWAY_PUBLIC_DOMAIN"):
                return f"https://{railway_domain}"
            return f"http://localhost:{os.getenv('PORT', '8001')}"

        dashboard_url = get_base_url() + "/dashboard"

        # 이메일 발송
        result = await sender.send_insight_report(
            recipients=recipients,
            report_date=report_date,
            avg_rank=avg_rank,
            sos=sos,
            hhi=hhi,
            insight_content=insight_content,
            top10_products=top10_products,
            brand_changes=brand_changes,
            dashboard_url=dashboard_url,
        )

        if result.success:
            logger.info(f"Insight Report sent successfully to {len(result.sent_to)} recipients")
        else:
            logger.error(f"Insight Report send failed: {result.message}")

    except Exception as e:
        logger.error(f"Insight Report email failed: {e}")

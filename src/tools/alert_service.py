"""
Alert Service
경쟁사 할인 알림 발송 서비스

지원 채널:
- Slack Webhook
- Email (SMTP)
- 대시보드 알림 (in-app)

Usage:
    service = AlertService()
    await service.send_alert(alert_data)
"""

import os
import json
import logging
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 한국 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))


@dataclass
class AlertConfig:
    """알림 설정"""
    # Slack
    slack_webhook_url: Optional[str] = None
    slack_channel: str = "#deals-alert"

    # Email
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    email_recipients: List[str] = None
    email_from: str = "deals-alert@amore.com"

    # 알림 조건
    min_discount_percent: float = 20.0  # 최소 할인율
    alert_brands: List[str] = None  # 모니터링 브랜드 (None이면 전체)

    def __post_init__(self):
        if self.email_recipients is None:
            self.email_recipients = []
        if self.alert_brands is None:
            self.alert_brands = []


class AlertService:
    """경쟁사 할인 알림 서비스"""

    # 주요 경쟁사 브랜드
    COMPETITOR_BRANDS = [
        "COSRX", "Beauty of Joseon", "SKIN1004", "ANUA",
        "medicube", "innisfree", "TIRTIR", "Torriden",
        "mixsoon", "Paula's Choice", "CeraVe", "La Roche-Posay"
    ]

    # 알림 타입
    ALERT_TYPES = {
        "lightning_deal": "⚡ Lightning Deal",
        "big_discount": "🔥 Big Discount",
        "deal_of_day": "🏆 Deal of the Day",
        "competitor_promo": "🎯 Competitor Promotion"
    }

    def __init__(self, config: Optional[AlertConfig] = None):
        """
        Args:
            config: 알림 설정 (없으면 환경변수에서 로드)
        """
        self.config = config or self._load_config_from_env()
        self._slack_enabled = bool(self.config.slack_webhook_url)
        self._email_enabled = bool(
            self.config.smtp_host and
            self.config.smtp_user and
            self.config.email_recipients
        )

    def _load_config_from_env(self) -> AlertConfig:
        """환경변수에서 설정 로드"""
        recipients = os.getenv("ALERT_EMAIL_RECIPIENTS", "")
        brands = os.getenv("ALERT_BRANDS", "")

        return AlertConfig(
            slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
            slack_channel=os.getenv("SLACK_CHANNEL", "#deals-alert"),
            smtp_host=os.getenv("SMTP_HOST"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_user=os.getenv("SMTP_USER"),
            smtp_password=os.getenv("SMTP_PASSWORD"),
            email_recipients=[e.strip() for e in recipients.split(",") if e.strip()],
            email_from=os.getenv("ALERT_EMAIL_FROM", "deals-alert@amore.com"),
            min_discount_percent=float(os.getenv("ALERT_MIN_DISCOUNT", "20.0")),
            alert_brands=[b.strip() for b in brands.split(",") if b.strip()] or None
        )

    async def process_deals_for_alerts(
        self,
        deals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        딜 목록에서 알림 대상 추출 및 발송

        Args:
            deals: DealRecord 리스트

        Returns:
            생성된 알림 목록
        """
        alerts = []

        for deal in deals:
            alert = self._check_deal_for_alert(deal)
            if alert:
                alerts.append(alert)

        # 알림 발송
        if alerts:
            await self._send_alerts_batch(alerts)

        return alerts

    def _check_deal_for_alert(self, deal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """개별 딜에 대한 알림 조건 체크"""
        brand = deal.get("brand", "").strip()
        discount = deal.get("discount_percent", 0) or 0
        deal_type = deal.get("deal_type", "")

        # 경쟁사 브랜드 확인
        is_competitor = False
        for comp_brand in self.COMPETITOR_BRANDS:
            if comp_brand.lower() in brand.lower():
                is_competitor = True
                break

        if not is_competitor:
            return None

        # 특정 브랜드만 모니터링하는 경우
        if self.config.alert_brands:
            if not any(b.lower() in brand.lower() for b in self.config.alert_brands):
                return None

        # 알림 타입 결정
        alert_type = None
        alert_message = ""

        if deal_type == "lightning":
            alert_type = "lightning_deal"
            alert_message = f"{brand}의 Lightning Deal 발견! {discount:.0f}% 할인"
        elif discount >= 30:
            alert_type = "big_discount"
            alert_message = f"{brand}가 {discount:.0f}% 대폭 할인 중!"
        elif deal_type == "deal_of_day":
            alert_type = "deal_of_day"
            alert_message = f"{brand} - 오늘의 딜 선정!"
        elif discount >= self.config.min_discount_percent:
            alert_type = "competitor_promo"
            alert_message = f"{brand} 할인 프로모션 진행 ({discount:.0f}%)"

        if not alert_type:
            return None

        return {
            "alert_datetime": datetime.now(KST).isoformat(),
            "brand": brand,
            "asin": deal.get("asin"),
            "product_name": deal.get("product_name"),
            "deal_type": deal_type,
            "discount_percent": discount,
            "deal_price": deal.get("deal_price"),
            "original_price": deal.get("original_price"),
            "time_remaining": deal.get("time_remaining"),
            "claimed_percent": deal.get("claimed_percent"),
            "product_url": deal.get("product_url"),
            "alert_type": alert_type,
            "alert_message": alert_message
        }

    async def _send_alerts_batch(self, alerts: List[Dict[str, Any]]) -> None:
        """알림 일괄 발송"""
        if not alerts:
            return

        # Slack 알림
        if self._slack_enabled:
            try:
                await self._send_slack_batch(alerts)
            except Exception as e:
                logger.error(f"Slack alert failed: {e}")

        # Email 알림
        if self._email_enabled:
            try:
                await self._send_email_batch(alerts)
            except Exception as e:
                logger.error(f"Email alert failed: {e}")

    async def _send_slack_batch(self, alerts: List[Dict[str, Any]]) -> bool:
        """Slack으로 알림 일괄 발송"""
        if not self.config.slack_webhook_url:
            return False

        # 알림 메시지 구성
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🚨 경쟁사 할인 알림 ({len(alerts)}건)",
                    "emoji": True
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"⏰ {datetime.now(KST).strftime('%Y-%m-%d %H:%M')} KST"
                    }
                ]
            },
            {"type": "divider"}
        ]

        # 각 알림 추가 (최대 10개)
        for alert in alerts[:10]:
            alert_icon = self.ALERT_TYPES.get(alert["alert_type"], "📢")
            discount = alert.get("discount_percent", 0)
            product_name = (alert.get("product_name", "")[:50] + "...") if len(alert.get("product_name", "")) > 50 else alert.get("product_name", "")

            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{alert_icon}*\n*{alert['brand']}* - {discount:.0f}% OFF\n_{product_name}_"
                },
                "accessory": {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "View Deal"
                    },
                    "url": alert.get("product_url") or f"https://amazon.com/dp/{alert.get('asin')}"
                }
            })

            # 추가 정보
            fields = []
            if alert.get("deal_price"):
                fields.append(f"💰 ${alert['deal_price']:.2f}")
            if alert.get("time_remaining"):
                fields.append(f"⏱️ {alert['time_remaining']}")
            if alert.get("claimed_percent"):
                fields.append(f"📊 {alert['claimed_percent']}% claimed")

            if fields:
                blocks.append({
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": " | ".join(fields)}
                    ]
                })

        if len(alerts) > 10:
            blocks.append({
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"_... 외 {len(alerts) - 10}건 더 있음_"}
                ]
            })

        # Webhook 전송
        payload = {
            "channel": self.config.slack_channel,
            "blocks": blocks
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.slack_webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    logger.info(f"Slack alert sent: {len(alerts)} deals")
                    return True
                else:
                    logger.error(f"Slack webhook failed: {response.status}")
                    return False

    async def _send_email_batch(self, alerts: List[Dict[str, Any]]) -> bool:
        """Email로 알림 일괄 발송"""
        if not self._email_enabled:
            return False

        # HTML 이메일 본문 구성
        html_content = self._build_email_html(alerts)

        # 이메일 메시지 생성
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[AMORE] 경쟁사 할인 알림 - {len(alerts)}건 ({datetime.now(KST).strftime('%m/%d')})"
        msg["From"] = self.config.email_from
        msg["To"] = ", ".join(self.config.email_recipients)

        # Plain text 버전
        plain_text = self._build_email_plain(alerts)
        msg.attach(MIMEText(plain_text, "plain", "utf-8"))

        # HTML 버전
        msg.attach(MIMEText(html_content, "html", "utf-8"))

        try:
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                server.sendmail(
                    self.config.email_from,
                    self.config.email_recipients,
                    msg.as_string()
                )
            logger.info(f"Email alert sent: {len(alerts)} deals to {len(self.config.email_recipients)} recipients")
            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False

    def _build_email_html(self, alerts: List[Dict[str, Any]]) -> str:
        """HTML 이메일 본문 생성"""
        rows = ""
        for alert in alerts[:20]:
            alert_icon = self.ALERT_TYPES.get(alert["alert_type"], "📢")
            discount = alert.get("discount_percent", 0)
            product_name = (alert.get("product_name", "")[:40] + "...") if len(alert.get("product_name", "")) > 40 else alert.get("product_name", "")
            url = alert.get("product_url") or f"https://amazon.com/dp/{alert.get('asin')}"

            rows += f"""
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #eee;">
                    <span style="font-size: 20px;">{alert_icon}</span>
                </td>
                <td style="padding: 12px; border-bottom: 1px solid #eee;">
                    <strong>{alert['brand']}</strong><br>
                    <span style="color: #666; font-size: 13px;">{product_name}</span>
                </td>
                <td style="padding: 12px; border-bottom: 1px solid #eee; text-align: center;">
                    <span style="color: #e74c3c; font-weight: bold; font-size: 18px;">{discount:.0f}%</span>
                </td>
                <td style="padding: 12px; border-bottom: 1px solid #eee; text-align: center;">
                    ${alert.get('deal_price', 0):.2f}
                </td>
                <td style="padding: 12px; border-bottom: 1px solid #eee; text-align: center;">
                    <a href="{url}" style="color: #3498db; text-decoration: none;">View →</a>
                </td>
            </tr>
            """

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 24px; text-align: center; }}
                .content {{ padding: 24px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th {{ background: #f8f9fa; padding: 12px; text-align: left; font-weight: 600; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1 style="margin: 0;">🚨 경쟁사 할인 알림</h1>
                <p style="margin: 8px 0 0 0; opacity: 0.9;">{len(alerts)}건의 새로운 딜이 감지되었습니다</p>
            </div>
            <div class="content">
                <table>
                    <thead>
                        <tr>
                            <th style="width: 50px;"></th>
                            <th>브랜드 / 제품</th>
                            <th style="width: 80px; text-align: center;">할인율</th>
                            <th style="width: 80px; text-align: center;">할인가</th>
                            <th style="width: 80px; text-align: center;">링크</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
                <p style="color: #666; font-size: 13px; margin-top: 24px;">
                    ⏰ 발송 시각: {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')} KST<br>
                    📊 AMORE Pacific Competitive Intelligence System
                </p>
            </div>
        </body>
        </html>
        """

    def _build_email_plain(self, alerts: List[Dict[str, Any]]) -> str:
        """Plain text 이메일 본문 생성"""
        lines = [
            "=" * 50,
            "🚨 경쟁사 할인 알림",
            f"발송 시각: {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')} KST",
            f"총 {len(alerts)}건의 새로운 딜",
            "=" * 50,
            ""
        ]

        for i, alert in enumerate(alerts[:20], 1):
            discount = alert.get("discount_percent", 0)
            url = alert.get("product_url") or f"https://amazon.com/dp/{alert.get('asin')}"

            lines.extend([
                f"{i}. {alert['brand']} - {discount:.0f}% OFF",
                f"   {alert.get('product_name', '')[:60]}",
                f"   할인가: ${alert.get('deal_price', 0):.2f}",
                f"   링크: {url}",
                ""
            ])

        if len(alerts) > 20:
            lines.append(f"... 외 {len(alerts) - 20}건")

        return "\n".join(lines)

    async def send_single_alert(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """단일 알림 발송"""
        results = {
            "slack": False,
            "email": False
        }

        if self._slack_enabled:
            results["slack"] = await self._send_slack_batch([alert])

        if self._email_enabled:
            results["email"] = await self._send_email_batch([alert])

        return results

    def get_status(self) -> Dict[str, Any]:
        """알림 서비스 상태 반환"""
        return {
            "slack_enabled": self._slack_enabled,
            "email_enabled": self._email_enabled,
            "email_recipients": len(self.config.email_recipients) if self.config.email_recipients else 0,
            "min_discount_threshold": self.config.min_discount_percent,
            "monitored_brands": self.config.alert_brands or "ALL",
            "competitor_brands": self.COMPETITOR_BRANDS
        }


# =============================================================================
# 싱글톤 인스턴스
# =============================================================================

_alert_service_instance: Optional[AlertService] = None


def get_alert_service() -> AlertService:
    """AlertService 싱글톤 반환"""
    global _alert_service_instance
    if _alert_service_instance is None:
        _alert_service_instance = AlertService()
    return _alert_service_instance

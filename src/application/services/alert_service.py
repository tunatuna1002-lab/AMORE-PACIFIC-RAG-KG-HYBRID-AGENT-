"""
Alert Application Service
=========================
Everything the alert routes do *after* request validation (F6): dispatching
pending alerts, reading/writing subscriptions, flipping an address to verified
and building the insight-report email payload.

The collaborators (SQLite storage, the alert dispatcher, the StateManager) are
passed in by the caller, so this module never imports the web layer and is
usable from a batch job or a test without FastAPI.

Note: ``src.tools.notifications.alert_service.AlertService`` is the *channel*
(Slack/SMTP sender). This module is the application-level orchestration around it.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src.domain.brand import is_target_brand

logger = logging.getLogger(__name__)

DEFAULT_ALERT_TYPES = ["rank_change", "important_insight", "daily_summary"]

# 인사이트 메일에서 순위 변동을 표시할 브랜드
INSIGHT_EMAIL_BRANDS = ["LANEIGE", "e.l.f.", "Maybelline", "Summer Fridays", "COSRX"]


# ---------------------------------------------------------------- dispatching
async def send_unsent_alerts(
    storage: Any, alert_sender: Any, alert_ids: list | None = None, limit: int = 50
) -> dict[str, Any]:
    """
    미발송 알림 발송.

    ``alert_ids``가 주어지면 해당 알림만, 없으면 미발송 전체를 발송한다.
    성공(슬랙 또는 이메일)한 알림은 발송 완료로 표시한다.
    """
    unsent_alerts = await storage.get_unsent_alerts(limit=limit)

    if not unsent_alerts:
        return {"success": True, "message": "No pending alerts to send", "sent_count": 0}

    if alert_ids:
        unsent_alerts = [a for a in unsent_alerts if a.get("id") in alert_ids]

    if not unsent_alerts:
        return {"success": True, "message": "No matching alerts found", "sent_count": 0}

    sent_count = 0
    for alert in unsent_alerts:
        result = await alert_sender.send_single_alert(alert)

        if result.get("slack") or result.get("email"):
            await storage.mark_alert_sent(alert["id"])
            sent_count += 1

    return {
        "success": True,
        "sent_count": sent_count,
        "total_pending": len(unsent_alerts),
        "channels": {
            "slack": alert_sender._slack_enabled,
            "email": alert_sender._email_enabled,
        },
    }


def build_test_alert(now: datetime | None = None) -> dict[str, Any]:
    """알림 채널 점검용 더미 알림."""
    return {
        "alert_datetime": (now or datetime.now()).isoformat(),
        "brand": "TEST BRAND",
        "asin": "B000TEST01",
        "product_name": "Test Product - Alert System Verification",
        "deal_type": "lightning",
        "discount_percent": 50.0,
        "deal_price": 19.99,
        "original_price": 39.99,
        "time_remaining": "2h 30m",
        "claimed_percent": 45,
        "product_url": "https://amazon.com/dp/B000TEST01",
        "alert_type": "lightning_deal",
        "alert_message": "Test Alert - 시스템 테스트 알림입니다",
    }


# --------------------------------------------------------------- subscriptions
def alert_settings_v3(state_manager: Any) -> dict[str, Any]:
    """첫 번째 등록 이메일의 설정 (v3는 단일 사용자만 지원)."""
    subscriptions = state_manager.get_all_subscriptions()

    if not subscriptions:
        return {"email": "", "consent": False, "alert_types": [], "consent_date": None}

    email, sub = next(iter(subscriptions.items()))
    return {
        "email": email,
        "consent": sub.consent,
        "alert_types": sub.alert_types,
        "consent_date": sub.consent_date.isoformat() if sub.consent_date else None,
    }


def save_alert_settings_v3(
    state_manager: Any, email: str, consent: bool, alert_types: list
) -> bool:
    """동의가 있으면 등록, 없으면 알림 유형만 갱신. 성공 여부를 돌려준다."""
    if consent:
        return bool(
            state_manager.register_email(email=email, consent=True, alert_types=alert_types)
        )
    state_manager.update_email_subscription(email=email, alert_types=alert_types)
    return True


def revoke_first_consent(state_manager: Any) -> bool:
    """첫 번째 등록 이메일의 동의 철회. 철회할 대상이 없으면 False."""
    subscriptions = state_manager.get_all_subscriptions()
    if not subscriptions:
        return False
    email = next(iter(subscriptions.keys()))
    state_manager.revoke_email_consent(email)
    return True


def subscription_payload(sub: Any, email: str | None = None) -> dict[str, Any]:
    """구독 객체 -> v4 응답 본문."""
    return {
        "found": True,
        "email": email or sub.email,
        "consent": sub.consent,
        "alert_types": sub.alert_types,
        "active": sub.active,
        "verified": sub.verified,
        "verified_at": sub.verified_at.isoformat() if sub.verified_at else None,
        "consent_date": sub.consent_date.isoformat() if sub.consent_date else None,
    }


def alert_settings_v4(state_manager: Any, email: str | None = None) -> dict[str, Any]:
    """이메일 지정 시 해당 구독, 없으면 첫 번째 구독자 (v3 호환 동작)."""
    if email:
        sub = state_manager.get_subscription(email)
        if not sub:
            return {"found": False, "email": email, "message": "등록되지 않은 이메일입니다."}
        return subscription_payload(sub)

    subscriptions = state_manager.get_all_subscriptions()
    if not subscriptions:
        return {"found": False, "email": "", "consent": False, "alert_types": []}

    email_key, sub = next(iter(subscriptions.items()))
    return subscription_payload(sub, email=email_key)


def existing_subscription_settings(sub: Any) -> dict[str, Any]:
    """이미 인증된 구독자에게 돌려줄 현재 설정."""
    return {
        "alert_types": sub.alert_types,
        "active": sub.active,
        "consent": sub.consent,
    }


def store_pending_subscription(state_manager: Any, email: str, alert_types: list) -> None:
    """
    인증 메일 발송 직후 호출: 선택한 alert_types를 미인증 상태로 저장해 둔다.
    (인증이 끝나면 :func:`mark_email_verified`가 활성화한다.)
    """
    from src.core.state_manager import EmailSubscription

    existing = state_manager.get_subscription(email)
    if existing:
        existing.alert_types = alert_types
    else:
        state_manager._email_subscriptions[email] = EmailSubscription(
            email=email,
            consent=False,
            alert_types=alert_types,
            active=False,
            verified=False,
        )
    state_manager._save_subscriptions()


def update_subscription(state_manager: Any, sub: Any, email: str, alert_types: list) -> bool:
    """인증된 구독자의 알림 유형 변경 (설정 수정 = 동의 유지)."""
    success = state_manager.update_email_subscription(
        email=email, alert_types=alert_types, active=True
    )

    if success and sub is not None and not sub.consent:
        sub.consent = True
        sub.consent_date = datetime.now()
        state_manager._save_subscriptions()

    return bool(success)


def mark_email_verified(state_manager: Any, email: str) -> None:
    """
    JWT 인증이 끝난 이메일을 verified/active/consent 상태로 영구 저장한다.

    (POST /api/alerts/verify-email 과 GET /api/alerts/confirm-email 이 같은 코드를
    각각 들고 있었다 — 이제 한 곳이다.)
    """
    existing = state_manager.get_subscription(email)

    if existing:
        existing.verified = True
        existing.verified_at = datetime.now()
        existing.consent = True
        existing.consent_date = datetime.now()
        existing.active = True
        state_manager._save_subscriptions()
        return

    state_manager.register_email(
        email=email,
        consent=True,
        alert_types=list(DEFAULT_ALERT_TYPES),
    )
    subscription = state_manager.get_subscription(email)
    if subscription:
        subscription.verified = True
        subscription.verified_at = datetime.now()
        state_manager._save_subscriptions()


def verification_status(state_manager: Any, email: str) -> dict[str, Any]:
    """이메일 인증 상태 (StateManager 기반)."""
    subscription = state_manager.get_subscription(email)

    if subscription:
        return {
            "verified": subscription.verified,
            "status": "verified" if subscription.verified else "pending",
            "verified_at": subscription.verified_at.isoformat()
            if subscription.verified_at
            else None,
        }

    return {"verified": False, "status": "not_found"}


# ------------------------------------------------------------- insight e-mail
def build_insight_email_payload(
    products: list[dict[str, Any]],
    latest_insight: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """
    정규화된 제품 목록 -> 인사이트 리포트 메일 본문 인자.

    ``products``는 호출자가 대시보드 JSON을 평탄화해서 넘긴다(순위 오름차순).

    Returns keys matching ``EmailSender.send_insight_report``:
    ``report_date``, ``avg_rank``, ``sos``, ``hhi``, ``insight_content``,
    ``top10_products``, ``brand_changes``.
    """
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
    brand_counts: dict[str, int] = {}
    for p in top100:
        brand = p.get("brand", "Unknown")
        brand_counts[brand] = brand_counts.get(brand, 0) + 1
    hhi = sum((count / len(top100) * 100) ** 2 for count in brand_counts.values()) if top100 else 0

    # 인사이트 가져오기 (캐시된 것 또는 안내 문구)
    insight_content = latest_insight
    if not insight_content:
        insight_content = (
            "<p>현재 생성된 인사이트가 없습니다. 대시보드에서 인사이트를 먼저 생성해주세요.</p>"
        )
    else:
        # 마크다운을 HTML로 간단 변환
        insight_content = insight_content.replace("\n\n", "</p><p>").replace("\n", "<br>")
        insight_content = f"<p>{insight_content}</p>"

    # Top 10 제품 데이터
    top10_products = [
        {
            "rank": i + 1,
            "name": p.get("title", "N/A"),
            "brand": p.get("brand", "Unknown"),
            "change": p.get("rank_change", 0),
        }
        for i, p in enumerate(products[:10])
    ]

    # 브랜드별 변동
    brand_changes = []
    for brand in INSIGHT_EMAIL_BRANDS:
        brand_products = [p for p in products if p.get("brand") == brand]
        if brand_products:
            avg_change = sum(p.get("rank_change", 0) for p in brand_products) / len(brand_products)
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

    return {
        "report_date": (now or datetime.now()).strftime("%Y년 %m월 %d일"),
        "avg_rank": avg_rank,
        "sos": sos,
        "hhi": hhi,
        "insight_content": insight_content,
        "top10_products": top10_products,
        "brand_changes": brand_changes,
    }

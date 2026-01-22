"""
Alerts Routes - Alert service and settings endpoints
"""
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from src.api.dependencies import verify_api_key
from src.tools.alert_service import AlertService, get_alert_service
from src.tools.sqlite_storage import get_sqlite_storage
from src.core.state_manager import StateManager, get_state_manager

router = APIRouter(prefix="/api", tags=["alerts"])


# ============= Alert Service Endpoints =============

class AlertSendRequest(BaseModel):
    """알림 발송 요청"""
    alert_ids: Optional[List[int]] = None  # 발송할 알림 ID (없으면 미발송 전체)


@router.get("/alerts/status")
async def get_alert_service_status():
    """알림 서비스 상태 조회"""
    try:
        service = get_alert_service()
        return {
            "success": True,
            **service.get_status()
        }
    except Exception as e:
        logging.error(f"Alert service status error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/alerts/send")
async def send_pending_alerts(request: Optional[AlertSendRequest] = None):
    """
    미발송 알림 발송

    특정 alert_ids를 지정하면 해당 알림만, 없으면 미발송 전체 발송
    """
    try:
        storage = get_sqlite_storage()
        await storage.initialize()

        alert_service = get_alert_service()

        # 미발송 알림 조회
        unsent_alerts = await storage.get_unsent_alerts(limit=50)

        if not unsent_alerts:
            return {
                "success": True,
                "message": "No pending alerts to send",
                "sent_count": 0
            }

        # 특정 ID 필터링
        if request and request.alert_ids:
            unsent_alerts = [a for a in unsent_alerts if a.get("id") in request.alert_ids]

        if not unsent_alerts:
            return {
                "success": True,
                "message": "No matching alerts found",
                "sent_count": 0
            }

        # 알림 발송
        sent_count = 0
        for alert in unsent_alerts:
            result = await alert_service.send_single_alert(alert)

            # 성공 시 발송 완료 표시
            if result.get("slack") or result.get("email"):
                await storage.mark_alert_sent(alert["id"])
                sent_count += 1

        return {
            "success": True,
            "sent_count": sent_count,
            "total_pending": len(unsent_alerts),
            "channels": {
                "slack": alert_service._slack_enabled,
                "email": alert_service._email_enabled
            }
        }

    except Exception as e:
        logging.error(f"Alert send error: {e}")
        return {
            "success": False,
            "error": str(e),
            "sent_count": 0
        }


@router.post("/alerts/test")
async def send_test_alert():
    """테스트 알림 발송"""
    try:
        alert_service = get_alert_service()

        test_alert = {
            "alert_datetime": datetime.now().isoformat(),
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
            "alert_message": "Test Alert - 시스템 테스트 알림입니다"
        }

        result = await alert_service.send_single_alert(test_alert)

        return {
            "success": True,
            "test_alert": test_alert,
            "send_result": result,
            "message": "Test alert sent successfully" if any(result.values()) else "No channels enabled"
        }

    except Exception as e:
        logging.error(f"Test alert error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# ============= Alert Settings Endpoints =============

# 싱글톤 State Manager
_state_manager: Optional[StateManager] = None


def get_app_state_manager() -> StateManager:
    """앱 레벨 State Manager 반환"""
    global _state_manager
    if _state_manager is None:
        _state_manager = get_state_manager()
    return _state_manager


class AlertSettingsRequest(BaseModel):
    """알림 설정 요청"""
    email: str
    consent: bool
    alert_types: List[str] = []


class AlertSettingsResponse(BaseModel):
    """알림 설정 응답"""
    email: str
    consent: bool
    alert_types: List[str]
    consent_date: Optional[str] = None


@router.get("/v3/alert-settings")
async def get_alert_settings():
    """
    현재 알림 설정 조회

    참고: 현재는 단일 사용자 설정만 지원 (첫 번째 등록된 이메일)
    """
    state_manager = get_app_state_manager()
    subscriptions = state_manager.get_all_subscriptions()

    if not subscriptions:
        return {
            "email": "",
            "consent": False,
            "alert_types": [],
            "consent_date": None
        }

    # 첫 번째 구독 반환
    email, sub = next(iter(subscriptions.items()))
    return {
        "email": email,
        "consent": sub.consent,
        "alert_types": sub.alert_types,
        "consent_date": sub.consent_date.isoformat() if sub.consent_date else None
    }


@router.post("/v3/alert-settings", dependencies=[Depends(verify_api_key)])
async def save_alert_settings(request: AlertSettingsRequest):
    """
    알림 설정 저장 (API Key 필요)

    중요: consent가 True일 때만 이메일 등록
    """
    state_manager = get_app_state_manager()

    if not request.email:
        raise HTTPException(status_code=400, detail="이메일 주소가 필요합니다.")

    if request.consent:
        # 이메일 등록 (명시적 동의)
        success = state_manager.register_email(
            email=request.email,
            consent=True,
            alert_types=request.alert_types
        )

        if not success:
            raise HTTPException(status_code=400, detail="이메일 등록 실패")

        return {"status": "ok", "message": "알림 설정이 저장되었습니다."}
    else:
        # 동의 없으면 업데이트만 (알림 유형 변경)
        success = state_manager.update_email_subscription(
            email=request.email,
            alert_types=request.alert_types
        )

        return {"status": "ok", "message": "설정이 업데이트되었습니다."}


@router.post("/v3/alert-settings/revoke", dependencies=[Depends(verify_api_key)])
async def revoke_alert_consent():
    """
    알림 동의 철회 (API Key 필요)

    첫 번째 등록된 이메일의 동의를 철회합니다.
    """
    state_manager = get_app_state_manager()
    subscriptions = state_manager.get_all_subscriptions()

    if not subscriptions:
        return {"status": "ok", "message": "철회할 동의가 없습니다."}

    # 첫 번째 이메일 철회
    email = next(iter(subscriptions.keys()))
    state_manager.revoke_email_consent(email)

    return {"status": "ok", "message": "동의가 철회되었습니다."}

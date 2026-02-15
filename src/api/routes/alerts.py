"""
Alert API Routes
================
알림 설정, 발송, 이메일 인증 관련 API 엔드포인트

주요 기능:
- 알림 서비스 상태 및 발송
- v3 알림 설정 (단일 이메일)
- v4 알림 설정 (뉴닉 스타일 구독 플로우 + JWT 인증)
- 이메일 인증 (JWT 기반)
- 인사이트 리포트 이메일 발송
"""

import logging
import os
import re
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from src.api.dependencies import get_base_url, limiter, load_dashboard_data, verify_api_key
from src.api.models import (
    AlertSendRequest,
    AlertSettingsRequest,
    SubscribeRequest,
    UpdateAlertSettingsRequest,
)
from src.core.state_manager import EmailSubscription, StateManager, get_state_manager
from src.tools.notifications.alert_service import get_alert_service
from src.tools.storage.sqlite_storage import get_sqlite_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["alerts"])

# JWT 설정
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
EMAIL_VERIFICATION_EXPIRES_MINUTES = 30  # 30분 만료


# =============================================================================
# Helper Functions
# =============================================================================

_state_manager_singleton: StateManager | None = None


def get_app_state_manager() -> StateManager:
    """
    앱 전역 StateManager 싱글톤 반환

    Note: dashboard_api.py의 get_app_state_manager()와 동일한 패턴
    """
    global _state_manager_singleton
    if _state_manager_singleton is None:
        _state_manager_singleton = get_state_manager()
    return _state_manager_singleton


def create_email_verification_token(
    email: str, expires_minutes: int = EMAIL_VERIFICATION_EXPIRES_MINUTES
) -> str:
    """
    이메일 인증용 JWT 토큰 생성

    Args:
        email: 인증할 이메일 주소
        expires_minutes: 토큰 만료 시간 (분)

    Returns:
        JWT 토큰 문자열

    Raises:
        ValueError: JWT_SECRET_KEY 미설정 시
    """
    if not JWT_SECRET_KEY:
        raise ValueError("JWT_SECRET_KEY 환경변수가 설정되지 않았습니다.")

    payload = {
        "email": email,
        "purpose": "email_verification",
        "exp": datetime.now(UTC) + timedelta(minutes=expires_minutes),
        "iat": datetime.now(UTC),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_jwt_email_token(token: str) -> dict:
    """
    JWT 이메일 인증 토큰 검증

    Args:
        token: JWT 토큰

    Returns:
        {"valid": True, "email": "..."} 또는 {"valid": False, "error": "..."}
    """
    if not JWT_SECRET_KEY:
        return {"valid": False, "error": "JWT_SECRET_KEY 환경변수가 설정되지 않았습니다."}

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        # purpose 검증
        if payload.get("purpose") != "email_verification":
            return {"valid": False, "error": "유효하지 않은 토큰입니다."}

        return {"valid": True, "email": payload["email"]}

    except jwt.ExpiredSignatureError:
        return {"valid": False, "error": "인증 토큰이 만료되었습니다. 다시 인증해주세요."}
    except jwt.InvalidTokenError:
        return {"valid": False, "error": "유효하지 않은 인증 토큰입니다."}


# =============================================================================
# Alert Service Endpoints
# =============================================================================


@router.get("/alerts/status")
async def get_alert_service_status():
    """알림 서비스 상태 조회"""
    try:
        service = get_alert_service()
        return {"success": True, **service.get_status()}
    except Exception as e:
        logger.error(f"Alert service status error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/alerts/send")
async def send_pending_alerts(request: AlertSendRequest | None = None):
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
            return {"success": True, "message": "No pending alerts to send", "sent_count": 0}

        # 특정 ID 필터링
        if request and request.alert_ids:
            unsent_alerts = [a for a in unsent_alerts if a.get("id") in request.alert_ids]

        if not unsent_alerts:
            return {"success": True, "message": "No matching alerts found", "sent_count": 0}

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
                "email": alert_service._email_enabled,
            },
        }

    except Exception as e:
        logger.error(f"Alert send error: {e}")
        return {"success": False, "error": str(e), "sent_count": 0}


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
            "alert_message": "Test Alert - 시스템 테스트 알림입니다",
        }

        result = await alert_service.send_single_alert(test_alert)

        return {
            "success": True,
            "test_alert": test_alert,
            "send_result": result,
            "message": "Test alert sent successfully"
            if any(result.values())
            else "No channels enabled",
        }

    except Exception as e:
        logger.error(f"Test alert error: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# Alert Settings v3 (Legacy)
# =============================================================================


@router.get("/v3/alert-settings")
async def get_alert_settings():
    """
    현재 알림 설정 조회

    참고: 현재는 단일 사용자 설정만 지원 (첫 번째 등록된 이메일)
    """
    state_manager = get_app_state_manager()
    subscriptions = state_manager.get_all_subscriptions()

    if not subscriptions:
        return {"email": "", "consent": False, "alert_types": [], "consent_date": None}

    # 첫 번째 구독 반환
    email, sub = next(iter(subscriptions.items()))
    return {
        "email": email,
        "consent": sub.consent,
        "alert_types": sub.alert_types,
        "consent_date": sub.consent_date.isoformat() if sub.consent_date else None,
    }


@router.post("/v3/alert-settings", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")  # 분당 5회 제한 (스팸 방지)
async def save_alert_settings(request: Request, settings: AlertSettingsRequest):
    """
    알림 설정 저장

    보안: API Key + Rate Limiting (IP당 분당 5회)
    중요: consent가 True일 때만 이메일 등록
    """
    state_manager = get_app_state_manager()

    if not settings.email:
        raise HTTPException(status_code=400, detail="이메일 주소가 필요합니다.")

    if settings.consent:
        # 이메일 등록 (명시적 동의)
        success = state_manager.register_email(
            email=settings.email, consent=True, alert_types=settings.alert_types
        )

        if not success:
            raise HTTPException(status_code=400, detail="이메일 등록 실패")

        return {"status": "ok", "message": "알림 설정이 저장되었습니다."}
    else:
        # 동의 없으면 업데이트만 (알림 유형 변경)
        success = state_manager.update_email_subscription(
            email=settings.email, alert_types=settings.alert_types
        )

        return {"status": "ok", "message": "설정이 업데이트되었습니다."}


@router.post("/v3/alert-settings/revoke", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")  # 분당 5회 제한
async def revoke_alert_consent(request: Request):
    """
    알림 동의 철회

    보안: API Key + Rate Limiting
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


# =============================================================================
# Alert Settings v4 (뉴닉 스타일 구독 플로우)
# =============================================================================


@router.post("/v4/subscribe")
@limiter.limit("3/minute")
async def subscribe_v4(request: Request, body: SubscribeRequest):
    """
    구독 시작 (v4 통합 엔드포인트)

    - 신규 이메일: JWT 인증 메일 발송 + alert_types 임시 저장
    - 기존 이메일 (already_verified): 현재 구독 설정 반환
    """
    email = body.email.strip()
    email_regex = r"^[^\s@]+@[^\s@]+\.[^\s@]+$"
    if not email or not re.match(email_regex, email):
        raise HTTPException(status_code=400, detail="올바른 이메일 주소를 입력해주세요.")

    if not body.alert_types:
        raise HTTPException(status_code=400, detail="최소 하나 이상의 알림 유형을 선택해주세요.")

    state_manager = get_state_manager()
    existing = state_manager.get_subscription(email)

    # 이미 인증된 이메일
    if existing and existing.verified:
        return {
            "success": True,
            "already_verified": True,
            "message": "이미 가입한 이메일이에요.",
            "current_settings": {
                "alert_types": existing.alert_types,
                "active": existing.active,
                "consent": existing.consent,
            },
        }

    # 신규 이메일 - JWT 인증 메일 발송
    try:
        token = create_email_verification_token(email)

        base_url = get_base_url()
        verify_url = f"{base_url}/api/alerts/confirm-email?token={token}&email={email}"

        from src.tools.notifications.email_sender import EmailSender

        email_sender = EmailSender()

        if not email_sender.is_enabled():
            raise HTTPException(status_code=503, detail="이메일 서비스가 설정되지 않았습니다.")

        result = await email_sender.send_verification_email(
            recipient=email, verify_url=verify_url, token=token
        )

        if result.success:
            # 인증 전이지만 선택한 alert_types를 미리 저장 (인증 완료 시 적용)
            if not existing:
                # 새 구독 생성 (아직 미인증, 미동의 상태)
                sub = EmailSubscription(
                    email=email,
                    consent=False,
                    alert_types=body.alert_types,
                    active=False,
                    verified=False,
                )
                state_manager._email_subscriptions[email] = sub
                state_manager._save_subscriptions()
            else:
                # 기존 미인증 구독 업데이트
                existing.alert_types = body.alert_types
                state_manager._save_subscriptions()

            logger.info(f"[v4] Verification email sent to {email}, alert_types={body.alert_types}")
            return {
                "success": True,
                "already_verified": False,
                "message": "인증 이메일이 발송되었습니다. (30분 내 인증해주세요)",
            }
        else:
            raise HTTPException(status_code=500, detail=f"이메일 발송 실패: {result.message}")

    except ValueError as e:
        logger.error(f"JWT configuration error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[v4] Subscribe error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/v4/alert-settings")
async def get_alert_settings_v4(email: str | None = None):
    """
    알림 설정 조회 (v4)

    Args:
        email: 조회할 이메일 (없으면 첫 번째 구독자)
    """
    state_manager = get_state_manager()

    if email:
        sub = state_manager.get_subscription(email)
        if not sub:
            return {"found": False, "email": email, "message": "등록되지 않은 이메일입니다."}
        return {
            "found": True,
            "email": sub.email,
            "consent": sub.consent,
            "alert_types": sub.alert_types,
            "active": sub.active,
            "verified": sub.verified,
            "verified_at": sub.verified_at.isoformat() if sub.verified_at else None,
            "consent_date": sub.consent_date.isoformat() if sub.consent_date else None,
        }

    # email 미지정 시 기존 v3 동작 (첫 번째 구독자)
    subscriptions = state_manager.get_all_subscriptions()
    if not subscriptions:
        return {"found": False, "email": "", "consent": False, "alert_types": []}

    email_key, sub = next(iter(subscriptions.items()))
    return {
        "found": True,
        "email": email_key,
        "consent": sub.consent,
        "alert_types": sub.alert_types,
        "active": sub.active,
        "verified": sub.verified,
        "verified_at": sub.verified_at.isoformat() if sub.verified_at else None,
        "consent_date": sub.consent_date.isoformat() if sub.consent_date else None,
    }


@router.put("/v4/alert-settings")
@limiter.limit("5/minute")
async def update_alert_settings_v4(request: Request, body: UpdateAlertSettingsRequest):
    """
    알림 설정 수정 (v4) - 기존 구독자 전용

    인증 완료된 이메일만 수정 가능
    """
    email = body.email.strip()
    if not email:
        raise HTTPException(status_code=400, detail="이메일 주소가 필요합니다.")

    state_manager = get_state_manager()
    sub = state_manager.get_subscription(email)

    if not sub:
        raise HTTPException(status_code=404, detail="등록되지 않은 이메일입니다.")

    if not sub.verified:
        raise HTTPException(status_code=403, detail="이메일 인증이 완료되지 않았습니다.")

    # alert_types 업데이트
    success = state_manager.update_email_subscription(
        email=email, alert_types=body.alert_types, active=True
    )

    # consent도 True로 설정 (설정 수정 = 동의 유지)
    if success and not sub.consent:
        sub.consent = True
        sub.consent_date = datetime.now()
        state_manager._save_subscriptions()

    if success:
        return {
            "status": "ok",
            "message": "알림 설정이 업데이트되었습니다.",
            "alert_types": body.alert_types,
        }
    else:
        raise HTTPException(status_code=500, detail="설정 업데이트 실패")


@router.delete("/v4/alert-settings")
@limiter.limit("5/minute")
async def delete_alert_settings_v4(request: Request, email: str):
    """
    구독 해지 (v4)

    Args:
        email: 해지할 이메일 주소
    """
    if not email:
        raise HTTPException(status_code=400, detail="이메일 주소가 필요합니다.")

    state_manager = get_state_manager()
    sub = state_manager.get_subscription(email)

    if not sub:
        raise HTTPException(status_code=404, detail="등록되지 않은 이메일입니다.")

    state_manager.revoke_email_consent(email)
    return {"status": "ok", "message": "구독이 해지되었습니다."}


# =============================================================================
# Alerts List v3
# =============================================================================


@router.get("/v3/alerts")
async def get_alerts(limit: int = 50, alert_type: str | None = None):
    """
    알림 목록 조회

    Args:
        limit: 최대 개수
        alert_type: 필터할 알림 유형
    """
    from src.agents.alert_agent import AlertAgent

    state_manager = get_app_state_manager()
    alert_agent = AlertAgent(state_manager)

    return {
        "alerts": alert_agent.get_alerts(limit=limit, alert_type=alert_type),
        "pending_count": alert_agent.get_pending_count(),
        "stats": alert_agent.get_stats(),
    }


# =============================================================================
# Email Verification Endpoints
# =============================================================================


@router.post("/alerts/send-verification")
@limiter.limit("3/minute")  # 분당 3회 제한 (스팸 방지)
async def send_verification_email(request: Request):
    """
    이메일 인증 요청 - 인증 이메일 발송 (JWT 방식)

    보안: Rate Limit으로 스팸 방지 (분당 3회)
    사용자가 이메일을 입력하고 '인증하기' 버튼을 누르면
    해당 이메일로 JWT 토큰이 포함된 인증 링크를 발송합니다.

    JWT 토큰은 30분간 유효하며, 서버 재시작과 무관하게 검증 가능합니다.
    """
    try:
        body = await request.json()
        email = body.get("email", "").strip()

        # 이메일 형식 검증
        email_regex = r"^[^\s@]+@[^\s@]+\.[^\s@]+$"
        if not email or not re.match(email_regex, email):
            raise HTTPException(status_code=400, detail="올바른 이메일 주소를 입력해주세요.")

        # 이미 인증된 이메일인지 확인
        state_manager = get_state_manager()
        existing = state_manager.get_subscription(email)
        if existing and existing.verified:
            return {
                "success": True,
                "already_verified": True,
                "message": "이미 인증 완료된 이메일입니다.",
            }

        # JWT 토큰 생성 (30분 유효)
        token = create_email_verification_token(email)

        # 인증 전용 페이지 URL 생성 (대시보드 대신 전용 페이지로 리다이렉트)
        base_url = get_base_url()
        verify_url = f"{base_url}/api/alerts/confirm-email?token={token}&email={email}"

        # EmailSender 직접 사용
        from src.tools.notifications.email_sender import EmailSender

        email_sender = EmailSender()

        if not email_sender.is_enabled():
            raise HTTPException(status_code=503, detail="이메일 서비스가 설정되지 않았습니다.")

        # 인증 이메일 발송
        result = await email_sender.send_verification_email(
            recipient=email, verify_url=verify_url, token=token
        )

        if result.success:
            logger.info(
                f"Verification email sent to {email} (JWT, expires in {EMAIL_VERIFICATION_EXPIRES_MINUTES}min)"
            )
            return {
                "success": True,
                "message": "인증 이메일이 발송되었습니다. (30분 내 인증해주세요)",
            }
        else:
            raise HTTPException(status_code=500, detail=f"이메일 발송 실패: {result.message}")

    except ValueError as e:
        # JWT_SECRET_KEY 미설정 에러
        logger.error(f"JWT configuration error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Send verification email error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/alerts/verify-email")
@limiter.limit("10/minute")  # 분당 10회 제한 (brute force 방지)
async def verify_email_token_endpoint(request: Request):
    """
    이메일 인증 토큰 검증 (JWT 방식)

    보안: Rate Limit으로 brute force 방지 (분당 10회)
    사용자가 이메일의 인증 버튼을 클릭하면
    JWT 토큰을 검증하고 이메일 인증 상태를 StateManager에 영구 저장합니다.

    JWT 토큰은 stateless이므로 서버 재시작과 무관하게 검증 가능합니다.
    """
    try:
        body = await request.json()
        token = body.get("token", "")
        email = body.get("email", "").strip()

        if not token or not email:
            raise HTTPException(status_code=400, detail="토큰과 이메일이 필요합니다.")

        # JWT 토큰 검증
        result = verify_jwt_email_token(token)

        if not result["valid"]:
            raise HTTPException(status_code=400, detail=result["error"])

        # 토큰의 이메일과 요청 이메일 일치 확인
        token_email = result["email"]
        if token_email != email:
            raise HTTPException(status_code=400, detail="이메일이 일치하지 않습니다.")

        # StateManager에 인증 완료 상태 영구 저장
        try:
            state_manager = get_state_manager()

            # 기존 구독 정보 확인
            existing = state_manager.get_subscription(email)

            if existing:
                # 기존 구독이 있으면 verified 상태 업데이트 + 활성화
                existing.verified = True
                existing.verified_at = datetime.now()
                existing.consent = True
                existing.consent_date = datetime.now()
                existing.active = True
                state_manager._save_subscriptions()
            else:
                # 새 구독 등록 (verified=True로 생성)
                state_manager.register_email(
                    email=email,
                    consent=True,
                    alert_types=["rank_change", "important_insight", "daily_summary"],
                )
                # verified 상태 추가 설정
                subscription = state_manager.get_subscription(email)
                if subscription:
                    subscription.verified = True
                    subscription.verified_at = datetime.now()
                    state_manager._save_subscriptions()

            logger.info(f"Email verified and saved to StateManager: {email}")
        except Exception as e:
            logger.warning(f"Failed to save verification status: {e}")

        return {"verified": True, "email": email, "message": "이메일 인증이 완료되었습니다!"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verify email error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/alerts/confirm-email")
async def confirm_email_page(token: str, email: str):
    """
    이메일 인증 확인 페이지 (GET 요청으로 접근)

    사용자가 이메일의 인증 링크를 클릭하면 이 페이지가 표시됩니다.
    토큰을 검증하고 인증 완료 상태를 저장한 후, 창을 닫아도 되는 안내 페이지를 반환합니다.
    원래 대시보드 탭은 폴링으로 인증 완료를 감지하여 자동으로 다음 단계로 이동합니다.
    """
    # JWT 토큰 검증
    result = verify_jwt_email_token(token)

    if not result["valid"]:
        error_html = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>인증 실패 - AMORE Pacific</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #001C58 0%, #1F5795 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 20px;
                }}
                .card {{
                    background: white;
                    border-radius: 20px;
                    padding: 48px;
                    max-width: 420px;
                    width: 100%;
                    text-align: center;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                }}
                .icon {{
                    width: 80px;
                    height: 80px;
                    background: #fee2e2;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 24px;
                }}
                .icon svg {{ width: 40px; height: 40px; color: #ef4444; }}
                h1 {{ color: #001C58; font-size: 24px; margin-bottom: 12px; }}
                p {{ color: #64748b; font-size: 15px; line-height: 1.6; }}
                .error-msg {{ color: #ef4444; font-size: 13px; margin-top: 16px; padding: 12px; background: #fef2f2; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <div class="card">
                <div class="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                    </svg>
                </div>
                <h1>인증 실패</h1>
                <p>이메일 인증 링크가 만료되었거나 유효하지 않습니다.</p>
                <div class="error-msg">{result.get("error", "토큰이 유효하지 않습니다.")}</div>
                <p style="margin-top: 20px; font-size: 13px;">대시보드에서 다시 인증을 요청해주세요.</p>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=400)

    # 토큰의 이메일과 요청 이메일 일치 확인
    token_email = result["email"]
    if token_email != email:
        return HTMLResponse(content="이메일이 일치하지 않습니다.", status_code=400)

    # StateManager에 인증 완료 상태 저장
    try:
        state_manager = get_state_manager()
        existing = state_manager.get_subscription(email)

        if existing:
            existing.verified = True
            existing.verified_at = datetime.now()
            existing.consent = True
            existing.consent_date = datetime.now()
            existing.active = True
            state_manager._save_subscriptions()
        else:
            state_manager.register_email(
                email=email,
                consent=True,
                alert_types=["rank_change", "important_insight", "daily_summary"],
            )
            subscription = state_manager.get_subscription(email)
            if subscription:
                subscription.verified = True
                subscription.verified_at = datetime.now()
                state_manager._save_subscriptions()

        logger.info(f"Email verified via confirm page: {email}")
    except Exception as e:
        logger.warning(f"Failed to save verification status: {e}")

    # 인증 성공 페이지 반환
    success_html = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>인증 완료 - AMORE Pacific</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #001C58 0%, #1F5795 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }}
            .card {{
                background: white;
                border-radius: 20px;
                padding: 48px;
                max-width: 420px;
                width: 100%;
                text-align: center;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }}
            .icon {{
                width: 80px;
                height: 80px;
                background: #d1fae5;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 24px;
                animation: pulse 2s infinite;
            }}
            @keyframes pulse {{
                0%, 100% {{ transform: scale(1); }}
                50% {{ transform: scale(1.05); }}
            }}
            .icon svg {{ width: 40px; height: 40px; color: #10b981; }}
            h1 {{ color: #001C58; font-size: 24px; margin-bottom: 12px; }}
            p {{ color: #64748b; font-size: 15px; line-height: 1.6; }}
            .email {{
                color: #1F5795;
                font-weight: 600;
                background: #f0f9ff;
                padding: 8px 16px;
                border-radius: 8px;
                display: inline-block;
                margin: 16px 0;
            }}
            .hint {{
                margin-top: 24px;
                padding: 16px;
                background: #f8fafc;
                border-radius: 12px;
                font-size: 13px;
                color: #475569;
            }}
            .close-btn {{
                margin-top: 24px;
                padding: 14px 32px;
                background: #001C58;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.2s;
            }}
            .close-btn:hover {{ background: #1F5795; }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="icon">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
                </svg>
            </div>
            <h1>이메일 인증 완료!</h1>
            <div class="email">{email}</div>
            <p>이메일 주소가 성공적으로 인증되었습니다.</p>
            <div class="hint">
                이 창은 닫아도 됩니다.<br>
                원래 대시보드 화면에서 자동으로 다음 단계로 이동합니다.
            </div>
            <button class="close-btn" onclick="window.close()">이 창 닫기</button>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=success_html)


@router.get("/alerts/verification-status")
async def get_verification_status(email: str):
    """
    이메일 인증 상태 확인 (StateManager 기반)

    JWT 방식으로 변경되어 인증 완료 상태는 StateManager에 영구 저장됩니다.
    """
    try:
        state_manager = get_state_manager()
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

    except Exception as e:
        logger.error(f"Get verification status error: {e}")
        return {"verified": False, "status": "error", "error": str(e)}


# =============================================================================
# Insight Email API
# =============================================================================


@router.post("/alerts/send-insight-report")
async def send_insight_report_email(request: Request):
    """
    인사이트 리포트 이메일 발송 (수동)

    대시보드에서 '이메일로 보내기' 버튼 클릭 시 호출됩니다.
    현재 인사이트와 KPI 데이터를 이메일로 발송합니다.

    StateManager 기반 인증 상태 확인 (JWT 방식 변경에 따른 업데이트)
    """
    try:
        body = await request.json()
        recipient_email = body.get("email", "").strip()

        if not recipient_email:
            raise HTTPException(status_code=400, detail="이메일 주소가 필요합니다.")

        # StateManager에서 이메일 인증 상태 확인
        state_manager = get_state_manager()
        subscription = state_manager.get_subscription(recipient_email)

        if not subscription or not subscription.verified:
            raise HTTPException(
                status_code=403, detail="이메일 인증이 필요합니다. 먼저 이메일을 인증해주세요."
            )

        # EmailSender 초기화
        from src.tools.notifications.email_sender import EmailSender

        email_sender = EmailSender()

        if not email_sender.is_enabled():
            raise HTTPException(status_code=503, detail="이메일 서비스가 설정되지 않았습니다.")

        # 현재 대시보드 데이터 로드
        dashboard_data = load_dashboard_data()
        if not dashboard_data:
            raise HTTPException(status_code=404, detail="대시보드 데이터가 없습니다.")

        # KPI 계산
        products = dashboard_data.get("products", [])
        laneige_products = [p for p in products if p.get("brand") == "LANEIGE"]
        avg_rank = (
            sum(p.get("rank", 100) for p in laneige_products) / len(laneige_products)
            if laneige_products
            else 0
        )

        # SoS 계산 (Top 100 기준)
        top100 = products[:100]
        laneige_in_top100 = len([p for p in top100 if p.get("brand") == "LANEIGE"])
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

        # 인사이트 가져오기 (캐시된 것 또는 새로 생성)
        insight_content = dashboard_data.get("latest_insight", "")
        if not insight_content:
            insight_content = (
                "<p>현재 생성된 인사이트가 없습니다. 대시보드에서 인사이트를 먼저 생성해주세요.</p>"
            )
        else:
            # 마크다운을 HTML로 간단 변환
            insight_content = insight_content.replace("\n\n", "</p><p>").replace("\n", "<br>")
            insight_content = f"<p>{insight_content}</p>"

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
        dashboard_url = get_base_url() + "/dashboard"

        # 이메일 발송
        result = await email_sender.send_insight_report(
            recipients=[recipient_email],
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
            logger.info(f"Insight report sent to {recipient_email}")
            return {
                "success": True,
                "message": f"인사이트 리포트가 {recipient_email}로 발송되었습니다.",
                "sent_to": result.sent_to,
            }
        else:
            raise HTTPException(status_code=500, detail=f"이메일 발송 실패: {result.message}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Send insight report error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

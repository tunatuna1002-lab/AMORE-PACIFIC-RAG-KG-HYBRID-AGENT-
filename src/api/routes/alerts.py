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
import re

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from src.api.dashboard_shape import products_as_list
from src.api.dependencies import (
    EMAIL_VERIFICATION_EXPIRES_MINUTES,
    create_email_verification_token,
    get_app_state_manager,
    get_base_url,
    limiter,
    load_dashboard_data,
    verify_api_key,
    verify_jwt_email_token,
)
from src.api.models import (
    AlertSendRequest,
    AlertSettingsRequest,
    SubscribeRequest,
    UpdateAlertSettingsRequest,
)
from src.api.templates import render as render_template
from src.application.services import alert_service as alert_svc
from src.core.state_manager import get_state_manager
from src.tools.notifications.alert_service import get_alert_service
from src.tools.storage.sqlite_storage import get_sqlite_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["alerts"])

# JWT helpers (create_email_verification_token / verify_jwt_email_token) and the
# app-level StateManager singleton are shared from src.api.dependencies.


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


@router.post("/alerts/send", dependencies=[Depends(verify_api_key)])
async def send_pending_alerts(request: AlertSendRequest | None = None):
    """
    미발송 알림 발송

    특정 alert_ids를 지정하면 해당 알림만, 없으면 미발송 전체 발송
    """
    try:
        storage = get_sqlite_storage()
        await storage.initialize()

        return await alert_svc.send_unsent_alerts(
            storage=storage,
            alert_sender=get_alert_service(),
            alert_ids=request.alert_ids if request else None,
        )

    except Exception as e:
        logger.error(f"Alert send error: {e}")
        return {"success": False, "error": str(e), "sent_count": 0}


@router.post("/alerts/test", dependencies=[Depends(verify_api_key)])
async def send_test_alert():
    """테스트 알림 발송"""
    try:
        alert_service = get_alert_service()

        test_alert = alert_svc.build_test_alert()

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
    return alert_svc.alert_settings_v3(get_app_state_manager())


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

    saved = alert_svc.save_alert_settings_v3(
        state_manager,
        email=settings.email,
        consent=settings.consent,
        alert_types=settings.alert_types,
    )

    if settings.consent:
        if not saved:
            raise HTTPException(status_code=400, detail="이메일 등록 실패")
        return {"status": "ok", "message": "알림 설정이 저장되었습니다."}

    return {"status": "ok", "message": "설정이 업데이트되었습니다."}


@router.post("/v3/alert-settings/revoke", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")  # 분당 5회 제한
async def revoke_alert_consent(request: Request):
    """
    알림 동의 철회

    보안: API Key + Rate Limiting
    첫 번째 등록된 이메일의 동의를 철회합니다.
    """
    if not alert_svc.revoke_first_consent(get_app_state_manager()):
        return {"status": "ok", "message": "철회할 동의가 없습니다."}

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
            "current_settings": alert_svc.existing_subscription_settings(existing),
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
            alert_svc.store_pending_subscription(state_manager, email, body.alert_types)

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
    return alert_svc.alert_settings_v4(get_state_manager(), email)


@router.put("/v4/alert-settings", dependencies=[Depends(verify_api_key)])
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

    success = alert_svc.update_subscription(state_manager, sub, email, body.alert_types)

    if success:
        return {
            "status": "ok",
            "message": "알림 설정이 업데이트되었습니다.",
            "alert_types": body.alert_types,
        }
    else:
        raise HTTPException(status_code=500, detail="설정 업데이트 실패")


@router.delete("/v4/alert-settings", dependencies=[Depends(verify_api_key)])
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
    from src.infrastructure.container import Container

    state_manager = get_app_state_manager()
    alert_agent = Container.get_alert_agent(state_manager=state_manager)

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
            alert_svc.mark_email_verified(get_state_manager(), email)
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
        error_html = render_template(
            "email_confirm_error",
            error_message=result.get("error", "토큰이 유효하지 않습니다."),
        )
        return HTMLResponse(content=error_html, status_code=400)

    # 토큰의 이메일과 요청 이메일 일치 확인
    token_email = result["email"]
    if token_email != email:
        return HTMLResponse(content="이메일이 일치하지 않습니다.", status_code=400)

    # StateManager에 인증 완료 상태 저장
    try:
        alert_svc.mark_email_verified(get_state_manager(), email)
        logger.info(f"Email verified via confirm page: {email}")
    except Exception as e:
        logger.warning(f"Failed to save verification status: {e}")

    # 인증 성공 페이지 반환
    success_html = render_template("email_confirm_success", email=email)
    return HTMLResponse(content=success_html)


@router.get("/alerts/verification-status")
async def get_verification_status(email: str):
    """
    이메일 인증 상태 확인 (StateManager 기반)

    JWT 방식으로 변경되어 인증 완료 상태는 StateManager에 영구 저장됩니다.
    """
    try:
        return alert_svc.verification_status(get_state_manager(), email)

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

        # KPI/본문 계산 (dashboard_data.json은 products를 ASIN 키 dict로 저장 → 리스트로 정규화)
        payload = alert_svc.build_insight_email_payload(
            products=products_as_list(dashboard_data),
            latest_insight=dashboard_data.get("latest_insight", ""),
        )

        # 대시보드 URL (Railway 자동 감지)
        dashboard_url = get_base_url() + "/dashboard"

        # 이메일 발송
        result = await email_sender.send_insight_report(
            recipients=[recipient_email],
            dashboard_url=dashboard_url,
            **payload,
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

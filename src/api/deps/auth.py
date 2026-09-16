"""
Auth dependencies
=================
API-key verification (``X-API-Key``), the shared slowapi limiter and the JWT
helpers used by the e-mail subscription flow.
"""

from __future__ import annotations

import hmac
import logging
import os
import re
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

# ============= API Key 인증 =============

# Startup guard only: production/staging must boot with a key configured.
# verify_api_key() itself reads API_KEY from the environment at call time so a
# re-configured process (or a test that sets the variable) needs no restart.
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    _env = os.getenv("RAILWAY_ENVIRONMENT", os.getenv("ENV", "development"))
    if _env in ("production", "staging"):
        raise RuntimeError(
            "API_KEY 환경변수가 설정되지 않았습니다. 프로덕션/스테이징 환경에서는 필수입니다."
        )
    logging.warning("API_KEY 환경변수가 설정되지 않았습니다. (개발 환경)")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_configured_api_key() -> str | None:
    """현재 프로세스에 설정된 API_KEY (호출 시점의 환경변수)"""
    return os.environ.get("API_KEY") or None


async def verify_api_key(api_key: str = Security(api_key_header)):
    """API Key 검증 (민감한 엔드포인트용)"""
    configured = get_configured_api_key()
    if configured is None:
        raise HTTPException(
            status_code=503,
            detail="Server not configured for authenticated access",
        )
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API Key가 필요합니다. 헤더에 X-API-Key를 추가하세요.",
        )
    if not hmac.compare_digest(api_key.encode(), configured.encode()):
        raise HTTPException(
            status_code=403,
            detail="유효하지 않은 API Key입니다.",
        )
    return api_key


# ============= Rate Limiter =============

limiter = Limiter(key_func=get_remote_address)


# ============= JWT Helpers =============

# Startup guard only (same pattern as API_KEY above): the helpers read the secret
# from the environment at call time via ``get_jwt_secret()``.
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
EMAIL_VERIFICATION_EXPIRES_MINUTES = 30

_jwt_env = os.getenv("RAILWAY_ENVIRONMENT", os.getenv("ENV", "development"))
if not JWT_SECRET_KEY and _jwt_env in ("production", "staging"):
    raise RuntimeError(
        "JWT_SECRET_KEY 환경변수가 설정되지 않았습니다. 프로덕션/스테이징 환경에서는 필수입니다."
    )

if JWT_SECRET_KEY:
    if len(JWT_SECRET_KEY) < 32:
        logging.warning(
            "JWT_SECRET_KEY가 32자 미만입니다. HS256은 최소 256비트(32바이트) 키를 권장합니다."
        )
    _weak_patterns = [
        r"^changeme",
        r"^secret",
        r"^password",
        r"^(.)\1+$",  # all same character
        r"^0123456789",  # sequential digits
    ]
    if any(re.match(p, JWT_SECRET_KEY, re.IGNORECASE) for p in _weak_patterns):
        logging.warning(
            "JWT_SECRET_KEY가 취약한 패턴을 사용하고 있습니다. 강력한 랜덤 키로 교체하세요."
        )


def get_jwt_secret() -> str | None:
    """현재 프로세스에 설정된 JWT_SECRET_KEY (호출 시점의 환경변수)"""
    return os.environ.get("JWT_SECRET_KEY") or None


def create_email_verification_token(
    email: str, expires_minutes: int = EMAIL_VERIFICATION_EXPIRES_MINUTES
) -> str:
    """이메일 인증용 JWT 토큰 생성"""
    secret = get_jwt_secret()
    if not secret:
        raise ValueError("JWT_SECRET_KEY 환경변수가 설정되지 않았습니다.")

    payload = {
        "email": email,
        "purpose": "email_verification",
        "exp": datetime.now(UTC) + timedelta(minutes=expires_minutes),
        "iat": datetime.now(UTC),
    }
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def verify_jwt_email_token(token: str) -> dict:
    """JWT 이메일 인증 토큰 검증"""
    secret = get_jwt_secret()
    if not secret:
        return {"valid": False, "error": "JWT_SECRET_KEY 환경변수가 설정되지 않았습니다."}

    try:
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])

        if payload.get("purpose") != "email_verification":
            return {"valid": False, "error": "유효하지 않은 토큰입니다."}

        return {"valid": True, "email": payload["email"]}

    except jwt.ExpiredSignatureError:
        return {"valid": False, "error": "인증 토큰이 만료되었습니다. 다시 인증해주세요."}
    except jwt.InvalidTokenError:
        return {"valid": False, "error": "유효하지 않은 인증 토큰입니다."}

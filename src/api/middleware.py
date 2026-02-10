"""
API Middleware
==============
FastAPI 미들웨어 모듈 (보안 헤더, 요청 로깅 등)
"""

from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """보안 헤더 추가 미들웨어"""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"  # iframe 임베딩은 같은 도메인만 허용
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

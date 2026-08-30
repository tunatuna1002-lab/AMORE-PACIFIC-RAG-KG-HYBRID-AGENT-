"""
FX 프록시 라우트 테스트 (§2.6)

대시보드가 외부 도메인을 직접 호출하면 CSP에 차단되므로 서버가 대신 조회한다.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.api.routes import fx as fx_routes


class _Req:
    """slowapi limiter가 요구하는 최소 Request 스텁"""

    class _Client:
        host = "testclient"

    client = _Client()
    headers: dict = {}


@pytest.mark.asyncio
async def test_returns_live_rates():
    with patch.object(fx_routes._service, "get_rate", new=AsyncMock(side_effect=[1374.55, 159.68])):
        result = await fx_routes.get_fx_rates.__wrapped__(_Req())

    assert result["base"] == "USD"
    assert result["rates"] == {"USD": 1.0, "KRW": 1374.55, "JPY": 159.68}
    assert result["is_fallback"] is False
    assert result["fallback_currencies"] == []


@pytest.mark.asyncio
async def test_flags_fallback_on_error():
    with patch.object(
        fx_routes._service, "get_rate", new=AsyncMock(side_effect=RuntimeError("net"))
    ):
        result = await fx_routes.get_fx_rates.__wrapped__(_Req())

    assert result["is_fallback"] is True
    assert set(result["fallback_currencies"]) == {"KRW", "JPY"}
    # 폴백이라도 값 자체는 돌려준다 (대시보드가 표시는 해야 함)
    assert result["rates"]["KRW"] == fx_routes._service.FALLBACK_RATES["KRW"]


@pytest.mark.asyncio
async def test_flags_fallback_when_service_returns_fallback_value():
    """서비스가 예외 없이 폴백 값을 돌려준 경우도 is_fallback으로 표시한다"""
    fallback_krw = fx_routes._service.FALLBACK_RATES["KRW"]
    fallback_jpy = fx_routes._service.FALLBACK_RATES["JPY"]
    with patch.object(
        fx_routes._service, "get_rate", new=AsyncMock(side_effect=[fallback_krw, fallback_jpy])
    ):
        result = await fx_routes.get_fx_rates.__wrapped__(_Req())

    assert result["is_fallback"] is True


def test_route_is_registered():
    from src.api.app_factory import create_app

    paths = {route.path for route in create_app().routes}
    assert "/api/fx/rates" in paths

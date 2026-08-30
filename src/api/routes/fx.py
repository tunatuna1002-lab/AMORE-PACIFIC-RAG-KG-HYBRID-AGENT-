"""
FX Routes
=========
환율 프록시 엔드포인트

대시보드가 frankfurter.app을 직접 호출하면 CSP `connect-src 'self'`에 차단되어
항상 폴백 환율(1350)이 쓰인다. CSP를 완화하는 대신 서버가 대신 조회하고
캐시한 값을 같은 오리진으로 돌려준다.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Request

from src.api.dependencies import limiter
from src.tools.calculators.exchange_rate import ExchangeRateService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["fx"])

# 대시보드가 쓰는 통화 (USD 기준)
SUPPORTED_CURRENCIES = ("KRW", "JPY")

_service = ExchangeRateService()


@router.get("/api/fx/rates")
@limiter.limit("60/minute")
async def get_fx_rates(request: Request):
    """USD 기준 환율 조회 (서버측 1시간 캐시).

    Returns:
        rates: {"USD": 1.0, "KRW": ..., "JPY": ...}
        is_fallback: 실시간 조회에 실패해 참고용 기본값을 쓴 통화가 있으면 True
    """
    rates: dict[str, float] = {"USD": 1.0}
    fallback_currencies: list[str] = []

    for currency in SUPPORTED_CURRENCIES:
        try:
            rate = await _service.get_rate("USD", currency)
        except Exception:
            logger.warning(f"환율 조회 실패: USD/{currency}", exc_info=True)
            rate = _service.FALLBACK_RATES.get(currency, 1.0)
            fallback_currencies.append(currency)
        else:
            if rate == _service.FALLBACK_RATES.get(currency):
                fallback_currencies.append(currency)
        rates[currency] = rate

    return {
        "base": "USD",
        "rates": rates,
        "is_fallback": bool(fallback_currencies),
        "fallback_currencies": fallback_currencies,
        "fetched_at": datetime.now().isoformat(),
    }

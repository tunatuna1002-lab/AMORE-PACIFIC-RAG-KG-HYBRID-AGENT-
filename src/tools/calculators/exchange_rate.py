"""
환율 API 유틸리티

무료 API 사용:
- frankfurter.app: 무제한, ECB 기준
- exchangerate-api.com: 월 1,500건 무료

사용법:
    from src.tools.calculators.exchange_rate import ExchangeRateService

    service = ExchangeRateService()
    rate = await service.get_rate("KRW", "USD")  # 1 KRW = ? USD
    usd_price = service.convert(34548, "KRW", "USD")  # KRW to USD
"""

import logging
from datetime import datetime, timedelta

import aiohttp

logger = logging.getLogger(__name__)


class ExchangeRateService:
    """실시간 환율 서비스"""

    # 캐시 유효 시간 (1시간)
    CACHE_TTL = timedelta(hours=1)

    # Fallback 환율 (API 실패 시 사용하는 참고용 기본값)
    # ⚠️ 주의: 실제 환율과 다를 수 있음. 실시간 API 실패 시에만 사용
    # 마지막 업데이트: 2026-01 기준 대략적인 환율
    FALLBACK_RATES = {
        "KRW": 1350.0,  # 1 USD ≈ 1350 KRW (참고용)
        "JPY": 150.0,  # 1 USD ≈ 150 JPY (참고용)
        "EUR": 0.92,  # 1 USD ≈ 0.92 EUR (참고용)
        "GBP": 0.79,  # 1 USD ≈ 0.79 GBP (참고용)
        "CNY": 7.2,  # 1 USD ≈ 7.2 CNY (참고용)
    }

    def __init__(self):
        self._cache: dict[str, tuple[float, datetime]] = {}
        self._base_currency = "USD"

    async def get_rate(self, from_currency: str, to_currency: str = "USD") -> float:
        """
        환율 조회 (캐시 사용)

        Args:
            from_currency: 원본 통화 (예: "KRW")
            to_currency: 대상 통화 (예: "USD")

        Returns:
            환율 (1 from_currency = ? to_currency)
        """
        # 같은 통화면 1 반환
        if from_currency.upper() == to_currency.upper():
            return 1.0

        cache_key = f"{from_currency}_{to_currency}"

        # 캐시 확인
        if cache_key in self._cache:
            rate, cached_at = self._cache[cache_key]
            if datetime.now() - cached_at < self.CACHE_TTL:
                return rate

        # API 호출
        rate = await self._fetch_rate(from_currency, to_currency)

        # 캐시 저장
        self._cache[cache_key] = (rate, datetime.now())

        return rate

    async def _fetch_rate(self, from_currency: str, to_currency: str) -> float:
        """API에서 환율 조회"""
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()

        # 1차: frankfurter.app API 시도
        try:
            rate = await self._fetch_from_frankfurter(from_currency, to_currency)
            if rate:
                logger.info(
                    f"Exchange rate fetched (frankfurter): 1 {from_currency} = {rate} {to_currency}"
                )
                return rate
        except Exception as e:
            logger.debug(f"Frankfurter API failed: {e}")

        # 2차: exchangerate-api.com 시도
        try:
            rate = await self._fetch_from_exchangerate_api(from_currency, to_currency)
            if rate:
                logger.info(
                    f"Exchange rate fetched (exchangerate-api): 1 {from_currency} = {rate} {to_currency}"
                )
                return rate
        except Exception as e:
            logger.debug(f"ExchangeRate API failed: {e}")

        # Fallback 환율 사용 (참고용 기본값 - 실제 환율과 다를 수 있음)
        fallback_rate = self._get_fallback_rate(from_currency, to_currency)
        logger.warning(
            f"⚠️ Using FALLBACK exchange rate for {from_currency}/{to_currency}: {fallback_rate} "
            f"(참고용 기본값, 실제 환율과 다를 수 있음)"
        )
        return fallback_rate

    async def _fetch_from_frankfurter(self, from_currency: str, to_currency: str) -> float | None:
        """frankfurter.app API 호출"""
        url = f"https://api.frankfurter.app/latest?from={from_currency}&to={to_currency}"

        try:
            import ssl

            import certifi

            ssl_context = ssl.create_default_context(cafile=certifi.where())

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10), ssl=ssl_context
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        rates = data.get("rates", {})
                        if to_currency in rates:
                            return rates[to_currency]
        except ImportError:
            # certifi 없으면 SSL 검증 비활성화 (개발용)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10), ssl=False
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        rates = data.get("rates", {})
                        if to_currency in rates:
                            return rates[to_currency]
        return None

    async def _fetch_from_exchangerate_api(
        self, from_currency: str, to_currency: str
    ) -> float | None:
        """exchangerate-api.com 무료 API (월 1,500건)"""
        # 무료 tier는 USD 기준만 지원
        if from_currency != "USD":
            # from -> USD -> to 경로로 계산
            url = f"https://open.er-api.com/v6/latest/{from_currency}"
        else:
            url = "https://open.er-api.com/v6/latest/USD"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10), ssl=False
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        rates = data.get("rates", {})
                        if to_currency in rates:
                            return rates[to_currency]
        except Exception as e:
            logger.debug(f"ExchangeRate API failed: {e}")
        return None

    def _get_fallback_rate(self, from_currency: str, to_currency: str) -> float:
        """Fallback 환율 계산"""
        # USD 기준 환율
        if to_currency == "USD":
            if from_currency in self.FALLBACK_RATES:
                return 1.0 / self.FALLBACK_RATES[from_currency]
            return 1.0 / 1350  # 기본값

        if from_currency == "USD":
            return self.FALLBACK_RATES.get(to_currency, 1.0)

        # 크로스 환율 (from -> USD -> to)
        from_to_usd = 1.0 / self.FALLBACK_RATES.get(from_currency, 1350)
        usd_to_to = self.FALLBACK_RATES.get(to_currency, 1.0)
        return from_to_usd * usd_to_to

    def convert(
        self, amount: float, from_currency: str, to_currency: str, rate: float | None = None
    ) -> float:
        """
        금액 변환 (동기 버전, 캐시된 환율 사용)

        Args:
            amount: 원본 금액
            from_currency: 원본 통화
            to_currency: 대상 통화
            rate: 환율 (None이면 fallback 사용)

        Returns:
            변환된 금액
        """
        if from_currency.upper() == to_currency.upper():
            return amount

        if rate is None:
            rate = self._get_fallback_rate(from_currency, to_currency)

        return round(amount * rate, 2)

    async def get_rates_for_currencies(
        self, currencies: list, base: str = "USD"
    ) -> dict[str, float]:
        """
        여러 통화의 환율 한 번에 조회

        Args:
            currencies: 통화 목록 (예: ["KRW", "JPY", "EUR"])
            base: 기준 통화

        Returns:
            {통화: 환율} 딕셔너리
        """
        result = {}
        for currency in currencies:
            rate = await self.get_rate(base, currency)
            result[currency] = rate
        return result

    def get_cached_rate(self, from_currency: str, to_currency: str = "USD") -> float | None:
        """캐시된 환율 반환 (API 호출 없음)"""
        cache_key = f"{from_currency}_{to_currency}"
        if cache_key in self._cache:
            rate, cached_at = self._cache[cache_key]
            if datetime.now() - cached_at < self.CACHE_TTL:
                return rate
        return None

    @property
    def fallback_rates(self) -> dict[str, float]:
        """Fallback 환율 (1 USD = X 통화)"""
        return self.FALLBACK_RATES.copy()


# 싱글톤 인스턴스
_exchange_service: ExchangeRateService | None = None


def get_exchange_service() -> ExchangeRateService:
    """싱글톤 환율 서비스 반환"""
    global _exchange_service
    if _exchange_service is None:
        _exchange_service = ExchangeRateService()
    return _exchange_service


async def get_usd_rate(currency: str = "KRW") -> float:
    """USD 환율 조회 (편의 함수)"""
    service = get_exchange_service()
    return await service.get_rate(currency, "USD")


def convert_to_usd(amount: float, currency: str, rate: float | None = None) -> float:
    """USD로 변환 (편의 함수)"""
    service = get_exchange_service()
    return service.convert(amount, currency, "USD", rate)

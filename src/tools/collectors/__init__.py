"""Data collection tools (non-crawling).

패키지 최상위 이름은 지연 로딩된다: ``src.tools.collectors`` 를 import 하는 것만으로
HTTP 클라이언트나 선택 의존성(pytrends)이 로딩되지 않게 한다.

``GoogleTrendsCollector`` / ``TrendData`` 는 선택 의존성이라 설치돼 있지 않으면
``None`` 으로 평가된다 (기존 동작 유지).
"""

import importlib

_LAZY: dict[str, str] = {
    "ExternalSignal": ".external_signal_collector",
    "ExternalSignalCollector": ".external_signal_collector",
    "CosmeticsProduct": ".public_data_collector",
    "PublicDataCollector": ".public_data_collector",
    "TradeData": ".public_data_collector",
    "TavilySearchClient": ".tavily_search",
    "GoogleTrendsCollector": ".google_trends_collector",
    "TrendData": ".google_trends_collector",
}
_OPTIONAL = frozenset({"GoogleTrendsCollector", "TrendData"})

__all__ = list(_LAZY)


def __getattr__(name: str):
    """지연 로딩: 하위 모듈은 실제로 접근할 때만 import한다."""
    if name in _LAZY:
        try:
            module = importlib.import_module(_LAZY[name], __name__)
        except ImportError:
            if name in _OPTIONAL:
                globals()[name] = None
                return None
            raise
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))

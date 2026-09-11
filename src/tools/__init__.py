"""
Tool modules for agent operations - organized into sub-packages

Sub-packages:
- scrapers: Web crawling tools (Amazon, deals)
- collectors: Data collection (trends, public data, external signals)
- calculators: Pure computation (metrics, period analysis)
- storage: Data storage (SQLite, Google Sheets)
- exporters: Export and reporting (dashboard, charts, reports)
- notifications: Alerts and messaging (email, Telegram)
- intelligence: Analysis and verification (market intel, claims, insights)
- utilities: General utilities (brand resolver, KG backup, data integrity)

패키지 최상위 이름은 지연 로딩된다: `from src.tools import AmazonScraper` 는 그 시점에만
playwright 를 import 한다. 하위 모듈 직접 import 를 권장한다.
"""

import importlib

_LAZY: dict[str, str] = {
    "AmazonScraper": ".scrapers.amazon_scraper",
    "AmazonDealsScraper": ".scrapers.deals_scraper",
    "SheetsWriter": ".storage.sheets_writer",
    "SQLiteStorage": ".storage.sqlite_storage",
    "PublicDataCollector": ".collectors.public_data_collector",
    "TradeData": ".collectors.public_data_collector",
    "CosmeticsProduct": ".collectors.public_data_collector",
    "ExternalSignalCollector": ".collectors.external_signal_collector",
    "ExternalSignal": ".collectors.external_signal_collector",
    "TavilySearchClient": ".collectors.tavily_search",
    "GoogleTrendsCollector": ".collectors.google_trends_collector",
    "TrendData": ".collectors.google_trends_collector",
    "IRReportParser": ".intelligence.ir_report_parser",
    "IRReport": ".intelligence.ir_report_parser",
    "QuarterlyFinancials": ".intelligence.ir_report_parser",
    "SourceManager": ".intelligence.source_manager",
    "Source": ".intelligence.source_manager",
    "InsightSourceBuilder": ".intelligence.source_manager",
    "MarketIntelligenceEngine": ".intelligence.market_intelligence",
    "InsightVerifier": ".intelligence.insight_verifier",
    "InsightVerificationResult": ".intelligence.insight_verifier",
    "verify_insight_report": ".intelligence.insight_verifier",
}
_OPTIONAL = frozenset(
    (
        "GoogleTrendsCollector",
        "TrendData",
        "InsightVerifier",
        "InsightVerificationResult",
        "verify_insight_report",
    )
)

__all__ = list(_LAZY)


def __getattr__(name: str):
    """지연 로딩: 하위 모듈은 실제로 접근할 때만 import한다 (무거운 의존성 로딩 방지)."""
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

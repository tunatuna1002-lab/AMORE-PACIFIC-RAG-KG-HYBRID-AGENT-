"""Data collection tools (non-crawling)"""

from .external_signal_collector import ExternalSignal, ExternalSignalCollector
from .public_data_collector import CosmeticsProduct, PublicDataCollector, TradeData
from .tavily_search import TavilySearchClient

# Google Trends (optional dependency)
try:
    from .google_trends_collector import GoogleTrendsCollector, TrendData
except ImportError:
    GoogleTrendsCollector = None
    TrendData = None

__all__ = [
    "ExternalSignalCollector",
    "ExternalSignal",
    "PublicDataCollector",
    "TradeData",
    "CosmeticsProduct",
    "TavilySearchClient",
    "GoogleTrendsCollector",
    "TrendData",
]

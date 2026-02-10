"""Data collection tools (non-crawling)"""

from .external_signal_collector import ExternalSignal, ExternalSignalCollector
from .instagram_collector import InstagramCollector
from .public_data_collector import CosmeticsProduct, PublicDataCollector, TradeData
from .reddit_collector import RedditCollector
from .tavily_search import TavilySearchClient

# Social media collectors
from .tiktok_collector import TikTokCollector
from .youtube_collector import YouTubeCollector

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
    "TikTokCollector",
    "InstagramCollector",
    "YouTubeCollector",
    "RedditCollector",
    "GoogleTrendsCollector",
    "TrendData",
]

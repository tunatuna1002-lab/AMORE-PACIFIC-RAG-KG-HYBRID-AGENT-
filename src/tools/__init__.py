"""
Tool modules for agent operations

Includes:
- Amazon scraping tools
- Data storage tools (Sheets, SQLite)
- Metric calculation
- Market intelligence (v2026.01.26)
  - Public data API collectors
  - IR report parser
  - External signal collector
  - Source manager
- Social Media Collectors (v2026.01.27)
  - TikTok (Playwright)
  - Instagram (Instaloader)
  - YouTube (yt-dlp)
  - Reddit (JSON API)
  - Google Trends (trendspyg/pytrends)
"""

from .amazon_scraper import AmazonScraper
from .sheets_writer import SheetsWriter
from .metric_calculator import MetricCalculator

# Market Intelligence System (v2026.01.26)
from .public_data_collector import PublicDataCollector, TradeData, CosmeticsProduct
from .ir_report_parser import IRReportParser, IRReport, QuarterlyFinancials
from .external_signal_collector import ExternalSignalCollector, ExternalSignal
from .source_manager import SourceManager, Source, InsightSourceBuilder
from .market_intelligence import MarketIntelligenceEngine

# Google Trends Collector (trendspyg or pytrends)
try:
    from .google_trends_collector import GoogleTrendsCollector, TrendData
except ImportError:
    GoogleTrendsCollector = None
    TrendData = None

# YouTube Collector (yt-dlp based)
try:
    from .youtube_collector import YouTubeCollector, YouTubeVideo
except ImportError:
    YouTubeCollector = None
    YouTubeVideo = None

# TikTok Collector (Playwright based) - v2026.01.27
try:
    from .tiktok_collector import TikTokCollector, TikTokPost
except ImportError:
    TikTokCollector = None
    TikTokPost = None

# Instagram Collector (Instaloader based) - v2026.01.27
try:
    from .instagram_collector import InstagramCollector, InstagramPost
except ImportError:
    InstagramCollector = None
    InstagramPost = None

# Reddit Collector (JSON API based) - v2026.01.27
try:
    from .reddit_collector import RedditCollector, RedditPost
except ImportError:
    RedditCollector = None
    RedditPost = None

# Legacy Apify (removed, kept for backward compatibility)
try:
    from .apify_amazon_scraper import ApifyAmazonScraper
except ImportError:
    ApifyAmazonScraper = None

# Insight Verifier (v2026.01.27)
try:
    from .insight_verifier import InsightVerifier, VerificationResult, verify_insight_report
except ImportError:
    InsightVerifier = None
    VerificationResult = None
    verify_insight_report = None

__all__ = [
    # Existing tools
    "AmazonScraper",
    "SheetsWriter",
    "MetricCalculator",

    # Public Data API
    "PublicDataCollector",
    "TradeData",
    "CosmeticsProduct",

    # IR Report Parser
    "IRReportParser",
    "IRReport",
    "QuarterlyFinancials",

    # External Signals
    "ExternalSignalCollector",
    "ExternalSignal",

    # Source Manager
    "SourceManager",
    "Source",
    "InsightSourceBuilder",

    # Market Intelligence Engine
    "MarketIntelligenceEngine",

    # Social Media Collectors (v2026.01.27)
    "GoogleTrendsCollector",
    "TrendData",
    "YouTubeCollector",
    "YouTubeVideo",
    "TikTokCollector",
    "TikTokPost",
    "InstagramCollector",
    "InstagramPost",
    "RedditCollector",
    "RedditPost",

    # Legacy
    "ApifyAmazonScraper",
]

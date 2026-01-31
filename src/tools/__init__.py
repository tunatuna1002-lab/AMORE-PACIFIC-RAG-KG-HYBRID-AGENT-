"""
Tool modules for agent operations

Includes:
- Amazon scraping tools
- Data storage tools (Sheets, SQLite)
- Market intelligence (v2026.01.26)
  - Public data API collectors
  - IR report parser
  - External signal collector
  - Source manager
- Google Trends Collector
"""

from .amazon_scraper import AmazonScraper
from .external_signal_collector import ExternalSignal, ExternalSignalCollector
from .ir_report_parser import IRReport, IRReportParser, QuarterlyFinancials
from .market_intelligence import MarketIntelligenceEngine

# Market Intelligence System (v2026.01.26)
from .public_data_collector import CosmeticsProduct, PublicDataCollector, TradeData
from .sheets_writer import SheetsWriter
from .source_manager import InsightSourceBuilder, Source, SourceManager

# Google Trends Collector (trendspyg or pytrends)
try:
    from .google_trends_collector import GoogleTrendsCollector, TrendData
except ImportError:
    GoogleTrendsCollector = None
    TrendData = None

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
    # Google Trends
    "GoogleTrendsCollector",
    "TrendData",
    # Insight Verifier
    "InsightVerifier",
    "VerificationResult",
    "verify_insight_report",
    # Legacy
    "ApifyAmazonScraper",
]

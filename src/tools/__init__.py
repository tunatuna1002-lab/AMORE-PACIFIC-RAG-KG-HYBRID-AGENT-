"""
Tool modules for agent operations - organized into sub-packages

Sub-packages:
- scrapers: Web crawling tools (Amazon, deals)
- collectors: Data collection (social media, trends, public data, external signals)
- calculators: Pure computation (metrics, period analysis, exchange rates)
- storage: Data storage (SQLite, Google Sheets)
- exporters: Export and reporting (dashboard, charts, reports)
- notifications: Alerts and messaging (email, Telegram)
- intelligence: Analysis and verification (market intel, claims, insights)
- utilities: General utilities (brand resolver, KG backup, data integrity)

Legacy imports maintained for backward compatibility.
"""

# ============================================================================
# Backward Compatibility Layer
# All imports below maintain existing import paths (e.g., from src.tools.amazon_scraper)
# ============================================================================

# Scrapers
# Collectors
from .collectors.external_signal_collector import ExternalSignal, ExternalSignalCollector
from .collectors.public_data_collector import CosmeticsProduct, PublicDataCollector, TradeData
from .collectors.tavily_search import TavilySearchClient
from .scrapers.amazon_product_scraper import AmazonProductScraper
from .scrapers.amazon_scraper import AmazonScraper
from .scrapers.deals_scraper import AmazonDealsScraper

# Google Trends (optional dependency)
try:
    from .collectors.google_trends_collector import GoogleTrendsCollector, TrendData
except ImportError:
    GoogleTrendsCollector = None
    TrendData = None

# Storage
from .intelligence.ir_report_parser import IRReport, IRReportParser, QuarterlyFinancials

# Intelligence
from .intelligence.market_intelligence import MarketIntelligenceEngine
from .intelligence.source_manager import InsightSourceBuilder, Source, SourceManager
from .storage.sheets_writer import SheetsWriter
from .storage.sqlite_storage import SQLiteStorage

# Insight Verifier (optional)
try:
    from .intelligence.insight_verifier import (
        InsightVerifier,
        VerificationResult,
        verify_insight_report,
    )
except ImportError:
    InsightVerifier = None
    VerificationResult = None
    verify_insight_report = None

# Legacy Apify (removed, kept for backward compatibility)
try:
    from .apify_amazon_scraper import ApifyAmazonScraper
except ImportError:
    ApifyAmazonScraper = None

__all__ = [
    # Scrapers
    "AmazonScraper",
    "AmazonProductScraper",
    "AmazonDealsScraper",
    # Storage
    "SheetsWriter",
    "SQLiteStorage",
    # Collectors - Public Data API
    "PublicDataCollector",
    "TradeData",
    "CosmeticsProduct",
    # Collectors - External Signals
    "ExternalSignalCollector",
    "ExternalSignal",
    "TavilySearchClient",
    # Collectors - Google Trends
    "GoogleTrendsCollector",
    "TrendData",
    # Intelligence - IR Report Parser
    "IRReportParser",
    "IRReport",
    "QuarterlyFinancials",
    # Intelligence - Source Manager
    "SourceManager",
    "Source",
    "InsightSourceBuilder",
    # Intelligence - Market Intelligence Engine
    "MarketIntelligenceEngine",
    # Intelligence - Insight Verifier
    "InsightVerifier",
    "VerificationResult",
    "verify_insight_report",
    # Legacy
    "ApifyAmazonScraper",
]

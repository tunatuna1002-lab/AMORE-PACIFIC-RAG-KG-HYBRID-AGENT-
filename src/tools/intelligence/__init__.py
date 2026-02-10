"""Analysis and verification tools"""

from .claim_extractor import ClaimExtractor
from .claim_verifier import ClaimVerifier
from .confidence_scorer import ConfidenceScorer
from .ir_report_parser import IRReport, IRReportParser, QuarterlyFinancials
from .market_intelligence import MarketIntelligenceEngine
from .morning_brief import MorningBriefGenerator
from .source_manager import InsightSourceBuilder, Source, SourceManager

# Insight Verifier (optional)
try:
    from .insight_verifier import InsightVerifier, VerificationResult, verify_insight_report
except ImportError:
    InsightVerifier = None
    VerificationResult = None
    verify_insight_report = None

__all__ = [
    "MarketIntelligenceEngine",
    "IRReportParser",
    "IRReport",
    "QuarterlyFinancials",
    "MorningBriefGenerator",
    "ClaimExtractor",
    "ClaimVerifier",
    "ConfidenceScorer",
    "SourceManager",
    "Source",
    "InsightSourceBuilder",
    "InsightVerifier",
    "VerificationResult",
    "verify_insight_report",
]

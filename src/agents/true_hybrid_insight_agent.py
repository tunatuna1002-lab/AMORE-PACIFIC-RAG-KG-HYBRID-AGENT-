"""
True Hybrid Insight Agent (Deprecated - Backward Compatibility Wrapper)
========================================================================

DEPRECATED: This module is kept for backward compatibility only.
Use HybridInsightAgent from hybrid_insight_agent.py instead.

The functionality from TrueHybridInsightAgent has been merged into HybridInsightAgent,
which now supports both legacy and true hybrid modes with optional OWL reasoning
and Entity Linking.

Migration Guide:
    # Old (deprecated):
    from src.agents.true_hybrid_insight_agent import TrueHybridInsightAgent
    agent = TrueHybridInsightAgent()

    # New (recommended):
    from src.agents.hybrid_insight_agent import HybridInsightAgent
    agent = HybridInsightAgent()

The merged HybridInsightAgent provides all features from both versions:
- Legacy mode: Basic Reasoner + HybridRetriever (default)
- True Hybrid mode: OWL Reasoner + TrueHybridRetriever + EntityLinker (optional)
- Market Intelligence integration
- Google Trends collection
- External signal collection
- RAG-to-KG extraction
"""

from src.agents.hybrid_insight_agent import HybridInsightAgent

# Re-export for backward compatibility
TrueHybridInsightAgent = HybridInsightAgent


# Also re-export the singleton getter if needed
def get_true_hybrid_insight_agent(**kwargs) -> HybridInsightAgent:
    """
    DEPRECATED: Use HybridInsightAgent directly.

    Returns HybridInsightAgent instance for backward compatibility.
    """
    return HybridInsightAgent(**kwargs)


__all__ = [
    "TrueHybridInsightAgent",
    "get_true_hybrid_insight_agent",
]

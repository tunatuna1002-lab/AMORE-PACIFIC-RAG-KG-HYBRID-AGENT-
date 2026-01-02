"""
AMORE RAG-Ontology Hybrid Agent System
Agent modules for Amazon bestseller tracking and analysis
Includes Hybrid Agents with Ontology reasoning capabilities

Recommended Usage:
    # Hybrid agents (recommended)
    from src.agents import HybridChatbotAgent, HybridInsightAgent

    # Core agents
    from src.agents import CrawlerAgent, StorageAgent, MetricsAgent

Deprecated:
    - InsightAgent → use HybridInsightAgent
    - ChatbotAgent → use HybridChatbotAgent
"""

# Core agents (always active)
from .crawler_agent import CrawlerAgent
from .storage_agent import StorageAgent
from .metrics_agent import MetricsAgent

# Hybrid agents (recommended)
from .hybrid_insight_agent import HybridInsightAgent
from .hybrid_chatbot_agent import HybridChatbotAgent, HybridChatbotSession

# Legacy agents (deprecated - lazy import to avoid warnings on module load)
def __getattr__(name):
    """Lazy import for deprecated modules to show warning only when used."""
    if name == "InsightAgent":
        from .insight_agent import InsightAgent
        return InsightAgent
    elif name == "ChatbotAgent":
        from .chatbot_agent import ChatbotAgent
        return ChatbotAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core Agents
    "CrawlerAgent",
    "StorageAgent",
    "MetricsAgent",
    # Hybrid Agents (Recommended)
    "HybridInsightAgent",
    "HybridChatbotAgent",
    "HybridChatbotSession",
    # Legacy (Deprecated)
    "InsightAgent",
    "ChatbotAgent",
]

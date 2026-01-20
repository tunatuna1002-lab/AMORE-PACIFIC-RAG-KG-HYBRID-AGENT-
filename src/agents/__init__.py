"""
AMORE RAG-Ontology Hybrid Agent System
Agent modules for Amazon bestseller tracking and analysis
Includes Hybrid Agents with Ontology reasoning capabilities

Usage:
    # Level 4 Agents (Brain에서 호출)
    from src.agents import QueryAgent, WorkflowAgent

    # Hybrid agents (recommended)
    from src.agents import HybridChatbotAgent, HybridInsightAgent

    # Core agents
    from src.agents import CrawlerAgent, StorageAgent, MetricsAgent

Note:
    InsightAgent and ChatbotAgent are DEPRECATED and will be removed.
    Use HybridInsightAgent and HybridChatbotAgent instead.
"""

# Core agents (always active)
from .crawler_agent import CrawlerAgent
from .storage_agent import StorageAgent
from .metrics_agent import MetricsAgent

# Level 4 Agents (Brain에서 호출)
from .query_agent import QueryAgent
from .workflow_agent import WorkflowAgent

# Hybrid agents (recommended)
from .hybrid_insight_agent import HybridInsightAgent
from .hybrid_chatbot_agent import HybridChatbotAgent, HybridChatbotSession

# Alert agent
from .alert_agent import AlertAgent


# Legacy agents (deprecated - lazy import with warning)
def __getattr__(name):
    """Lazy import for deprecated modules - shows warning when accessed."""
    if name == "InsightAgent":
        import warnings
        warnings.warn(
            "InsightAgent is deprecated and will be removed in a future version. "
            "Use HybridInsightAgent instead.",
            DeprecationWarning,
            stacklevel=2
        )
        from .insight_agent import InsightAgent
        return InsightAgent
    elif name == "ChatbotAgent":
        import warnings
        warnings.warn(
            "ChatbotAgent is deprecated and will be removed in a future version. "
            "Use HybridChatbotAgent instead.",
            DeprecationWarning,
            stacklevel=2
        )
        from .chatbot_agent import ChatbotAgent
        return ChatbotAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core Agents
    "CrawlerAgent",
    "StorageAgent",
    "MetricsAgent",
    # Level 4 Agents
    "QueryAgent",
    "WorkflowAgent",
    # Hybrid Agents (Recommended)
    "HybridInsightAgent",
    "HybridChatbotAgent",
    "HybridChatbotSession",
    # Alert Agent
    "AlertAgent",
]

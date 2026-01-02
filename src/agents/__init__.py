"""
AMORE RAG-Ontology Hybrid Agent System
Agent modules for Amazon bestseller tracking and analysis
Includes Hybrid Agents with Ontology reasoning capabilities

Recommended Usage:
    # Level 4 Agents (Brain에서 호출)
    from src.agents import QueryAgent, WorkflowAgent

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

# Level 4 Agents (Brain에서 호출)
from .query_agent import QueryAgent
from .workflow_agent import WorkflowAgent

# Hybrid agents (recommended)
from .hybrid_insight_agent import HybridInsightAgent
from .hybrid_chatbot_agent import HybridChatbotAgent, HybridChatbotSession

# Alert agent
from .alert_agent import AlertAgent

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
    # Level 4 Agents
    "QueryAgent",
    "WorkflowAgent",
    # Hybrid Agents (Recommended)
    "HybridInsightAgent",
    "HybridChatbotAgent",
    "HybridChatbotSession",
    # Alert Agent
    "AlertAgent",
    # Legacy (Deprecated)
    "InsightAgent",
    "ChatbotAgent",
]

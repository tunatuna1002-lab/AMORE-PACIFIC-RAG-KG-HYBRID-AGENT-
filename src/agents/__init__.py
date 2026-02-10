"""
AMORE RAG-Ontology Hybrid Agent System
Agent modules for Amazon bestseller tracking and analysis
Includes Hybrid Agents with Ontology reasoning capabilities

Usage:
    # Hybrid agents (recommended)
    from src.agents import HybridChatbotAgent, HybridInsightAgent

    # Core agents
    from src.agents import CrawlerAgent, StorageAgent, MetricsAgent

Note:
    InsightAgent, ChatbotAgent, QueryAgent, WorkflowAgent are REMOVED.
    Use HybridInsightAgent and HybridChatbotAgent instead.
    For workflow execution, use src.core.batch_workflow.BatchWorkflow.

Protocols:
    Each agent implements a corresponding Protocol from src.domain.interfaces:
    - CrawlerAgent → CrawlerAgentProtocol
    - StorageAgent → StorageAgentProtocol
    - MetricsAgent → MetricsAgentProtocol
    - HybridInsightAgent → InsightAgentProtocol
    - HybridChatbotAgent → ChatbotAgentProtocol
    - AlertAgent → AlertAgentProtocol
"""

# Core agents (always active)
# Alert agent
from .alert_agent import AlertAgent
from .crawler_agent import CrawlerAgent
from .hybrid_chatbot_agent import HybridChatbotAgent, HybridChatbotSession

# Hybrid agents (recommended)
from .hybrid_insight_agent import HybridInsightAgent
from .metrics_agent import MetricsAgent
from .storage_agent import StorageAgent

# Backward compatibility (deprecated)
from .true_hybrid_insight_agent import TrueHybridInsightAgent

__all__ = [
    # Core Agents
    "CrawlerAgent",
    "StorageAgent",
    "MetricsAgent",
    # Hybrid Agents (Recommended)
    "HybridInsightAgent",
    "HybridChatbotAgent",
    "HybridChatbotSession",
    # Alert Agent
    "AlertAgent",
    # Backward Compatibility (Deprecated)
    "TrueHybridInsightAgent",
]

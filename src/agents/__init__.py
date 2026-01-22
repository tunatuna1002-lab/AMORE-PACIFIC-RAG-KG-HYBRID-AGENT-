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
"""

# Core agents (always active)
from .crawler_agent import CrawlerAgent
from .storage_agent import StorageAgent
from .metrics_agent import MetricsAgent

# Hybrid agents (recommended)
from .hybrid_insight_agent import HybridInsightAgent
from .hybrid_chatbot_agent import HybridChatbotAgent, HybridChatbotSession

# Alert agent
from .alert_agent import AlertAgent


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
]

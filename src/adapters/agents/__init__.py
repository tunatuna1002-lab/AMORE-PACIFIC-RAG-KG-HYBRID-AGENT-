"""
Agent Adapters
==============
Agent implementations that fulfill Domain protocols.

For backward compatibility, agents are currently in src/agents/.
This module re-exports them.
"""

# Re-export from original location for backward compatibility
try:
    from src.agents.crawler_agent import CrawlerAgent
    from src.agents.storage_agent import StorageAgent
    from src.agents.metrics_agent import MetricsAgent
    from src.agents.alert_agent import AlertAgent
except ImportError:
    pass  # Original agents may not exist yet

__all__ = [
    "CrawlerAgent",
    "StorageAgent",
    "MetricsAgent",
    "AlertAgent",
]

"""
AMORE RAG-Ontology Hybrid Agent System — agent modules

Usage:
    from src.agents import HybridChatbotAgent, HybridInsightAgent
    from src.agents import CrawlerAgent, StorageAgent, MetricsAgent, AlertAgent

패키지 최상위 이름은 지연 로딩된다: CrawlerAgent 접근 전에는 playwright 가 import 되지 않는다.
"""

import importlib

_LAZY: dict[str, str] = {
    "AlertAgent": ".alert_agent",
    "CrawlerAgent": ".crawler_agent",
    "HybridChatbotAgent": ".hybrid_chatbot_agent",
    "HybridChatbotSession": ".hybrid_chatbot_agent",
    "HybridInsightAgent": ".hybrid_insight_agent",
    "MetricsAgent": ".metrics_agent",
    "StorageAgent": ".storage_agent",
}
_OPTIONAL = frozenset(())

__all__ = list(_LAZY)


def __getattr__(name: str):
    """지연 로딩: 하위 모듈은 실제로 접근할 때만 import한다 (무거운 의존성 로딩 방지)."""
    if name in _LAZY:
        try:
            module = importlib.import_module(_LAZY[name], __name__)
        except ImportError:
            if name in _OPTIONAL:
                globals()[name] = None
                return None
            raise
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))

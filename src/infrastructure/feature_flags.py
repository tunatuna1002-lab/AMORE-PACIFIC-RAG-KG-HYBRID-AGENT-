"""Feature Flags infrastructure for safe, incremental rollout of new features.

Supports three levels of override (highest priority first):
1. Environment variable: FF_{SECTION}_{KEY} (e.g., FF_RETRIEVER_USE_TRUE_HYBRID_RETRIEVER=true)
2. JSON config file: config/feature_flags.json
3. Default value passed to get_flag()

Usage:
    from src.infrastructure.feature_flags import FeatureFlags

    flags = FeatureFlags()
    if flags.use_true_hybrid_retriever():
        # new path
    else:
        # legacy path
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class FeatureFlags:
    """Thread-safe feature flag reader with ENV > JSON > default precedence."""

    _instance: FeatureFlags | None = None
    _config: dict[str, dict[str, Any]] = {}
    _config_path: Path | None = None

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is not None:
            self._config_path = Path(config_path)
        elif self._config_path is None:
            # Default: look for config/feature_flags.json relative to project root
            self._config_path = self._find_config_path()
        self._load_config()

    @staticmethod
    def _find_config_path() -> Path:
        """Find feature_flags.json by walking up from this file's location."""
        current = Path(__file__).resolve().parent
        for _ in range(5):  # max 5 levels up
            candidate = current / "config" / "feature_flags.json"
            if candidate.exists():
                return candidate
            current = current.parent
        # Fallback: relative to CWD
        return Path("config/feature_flags.json")

    def _load_config(self) -> None:
        """Load JSON config file. Silently uses empty config if file missing."""
        if self._config_path and self._config_path.exists():
            try:
                with open(self._config_path, encoding="utf-8") as f:
                    self._config = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._config = {}
        else:
            self._config = {}

    def reload(self) -> None:
        """Reload config from disk. Useful for testing."""
        self._load_config()

    def get_flag(self, section: str, key: str, default: bool = False) -> bool:
        """Get a feature flag value with ENV > JSON > default precedence.

        Args:
            section: Config section (e.g., "retriever", "cache")
            key: Flag key (e.g., "use_true_hybrid_retriever")
            default: Default value if not found anywhere

        Returns:
            Boolean flag value
        """
        # 1. Check environment variable (highest priority)
        env_key = f"FF_{section.upper()}_{key.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            return env_val.lower() in ("true", "1", "yes", "on")

        # 2. Check JSON config
        section_config = self._config.get(section, {})
        if key in section_config:
            return bool(section_config[key])

        # 3. Return default
        return default

    # ── Convenience methods ──────────────────────────────────────────

    def use_true_hybrid_retriever(self) -> bool:
        """Whether to use TrueHybridRetriever over legacy HybridRetriever."""
        return self.get_flag("retriever", "use_true_hybrid_retriever", default=True)

    def use_unified_retriever(self) -> bool:
        """Whether to use the UnifiedRetriever facade."""
        return self.get_flag("retriever", "use_unified_retriever", default=True)

    def use_unified_reasoner(self) -> bool:
        """Whether to use UnifiedReasoner over individual reasoners."""
        return self.get_flag("reasoner", "use_unified_reasoner", default=True)

    def use_owl_reasoner(self) -> bool:
        """Whether to enable OWL-based reasoning (requires owlready2)."""
        return self.get_flag("reasoner", "use_owl_reasoner", default=True)

    def use_ontology_kg(self) -> bool:
        """Whether to use OntologyKnowledgeGraph over plain KnowledgeGraph."""
        return self.get_flag("ontology", "use_ontology_kg", default=True)

    def use_sqlite_embedding_cache(self) -> bool:
        """Whether to use SQLite-backed embedding cache (vs in-memory dict)."""
        return self.get_flag("cache", "use_sqlite_embedding_cache", default=False)

    def use_centralized_prompts(self) -> bool:
        """Whether to use PromptRegistry for prompt loading."""
        return self.get_flag("prompts", "use_centralized_prompts", default=False)

    def use_decomposed_chatbot(self) -> bool:
        """Whether to use decomposed chatbot components."""
        return self.get_flag("agents", "use_decomposed_chatbot", default=False)

    # ── Singleton access ─────────────────────────────────────────────

    @classmethod
    def get_instance(cls, config_path: str | Path | None = None) -> FeatureFlags:
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config_path=config_path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton. Used in tests."""
        cls._instance = None
        cls._config = {}
        cls._config_path = None

"""Feature Flags infrastructure for safe, incremental rollout of new features.

Supports three levels of override (highest priority first):
1. Environment variable: FF_{SECTION}_{KEY} (e.g., FF_RETRIEVER_USE_RERANKER=true)
2. JSON config file: config/feature_flags.json
3. Default value passed to get_flag()

Usage:
    from src.infrastructure.feature_flags import FeatureFlags

    flags = FeatureFlags()
    if flags.use_reranker():
        # reranking path
    else:
        # no reranking
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
            key: Flag key (e.g., "use_reranker")
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

    def use_react_agent(self) -> bool:
        """Whether UnifiedBrain routes complex MEDIUM/LOW-confidence queries to ReActAgent.

        기본 OFF (2026-09): import 경로 오류로 추가된 날(a965437)부터 미연결이던 경로다.
        """
        return self.get_flag("agents", "use_react_agent", default=False)

    def react_shadow_mode(self) -> bool:
        """ReAct를 섀도 모드로 돌린다: 답변은 파이프라인 것을 쓰고 ReAct 결과는 기록만 한다.

        ``use_react_agent``가 OFF일 때만 의미가 있다 (둘 다 켜지면 ReAct가 실제 경로다).
        기본 OFF — 켜면 2홉 이상 문항마다 LLM 호출이 추가로 발생한다. 섀도 토큰 사용량은
        응답 metadata의 ``react_shadow.token_usage``에 따로 적힌다 (트랙 5-C).
        """
        return self.get_flag("agents", "react_shadow_mode", default=False)

    def use_router_llm_fallback(self) -> bool:
        """홉 수 라우터가 규칙으로 판정하지 못한 질문을 LLM에게 물어볼지 여부.

        기본 OFF — 라우팅은 규칙이 정본이고, LLM은 표지가 하나도 없는 질문에만 쓰는
        보조 수단이다. 판정 결과는 질문별로 캐시된다 (``src/core/router.py``).
        """
        return self.get_flag("router", "use_llm_fallback", default=False)

    # 아래 두 플래그는 이름과 달리 `HybridRetriever.retrieve`의 규칙 추론 on/off 스위치다
    # (둘 다 false여야 규칙 판정을 건너뛴다 — 평가 ablation `no-ontology`가 쓰는 스위치).

    def use_unified_reasoner(self) -> bool:
        """Whether to run unified/rule-based inference during retrieval (ablation: no-ontology)."""
        return self.get_flag("reasoner", "use_unified_reasoner", default=True)

    def use_owl_reasoner(self) -> bool:
        """Whether to enable OWL-based reasoning (requires owlready2; ablation: no-ontology)."""
        return self.get_flag("reasoner", "use_owl_reasoner", default=True)

    def use_ontology_kg(self) -> bool:
        """Whether to query the Knowledge Graph during retrieval (ablation: no-kg)."""
        return self.get_flag("ontology", "use_ontology_kg", default=True)

    def use_db_metric_facts(self) -> bool:
        """Whether to attach crawl-DB metric facts (SoS/HHI/rank/price) to retrieval context.

        SQLite가 지표의 정본이다. 끄면 수치 질문에 답할 근거가 컨텍스트에서 사라진다
        (사이클 10 이전 상태, ablation: no-db-metrics).
        """
        return self.get_flag("retriever", "use_db_metric_facts", default=True)

    def use_sqlite_embedding_cache(self) -> bool:
        """Whether to use SQLite-backed embedding cache (vs in-memory dict)."""
        return self.get_flag("cache", "use_sqlite_embedding_cache", default=False)

    def use_centralized_prompts(self) -> bool:
        """Whether to use PromptRegistry for prompt loading."""
        return self.get_flag("prompts", "use_centralized_prompts", default=False)

    def use_reranker(self) -> bool:
        """Whether to use relevance reranking (grading)."""
        return self.get_flag("retriever", "use_reranker", default=True)

    def use_query_rewriter(self) -> bool:
        """Whether to use query rewriting in chatbot agent."""
        return self.get_flag("agents", "use_query_rewriter", default=True)

    def use_confidence_fusion(self) -> bool:
        """Whether to use confidence fusion scoring."""
        return self.get_flag("retriever", "use_confidence_fusion", default=True)

    def use_external_signals(self) -> bool:
        """Whether to fetch external signals (Tavily, RSS, Reddit) per query."""
        return self.get_flag("agents", "use_external_signals", default=True)

    def numeric_verification_mode(self) -> str:
        """답변 수치 검증 모드 (설계 E8): ``off`` | ``annotate`` | ``enforce``.

        문자열 플래그라 ``get_flag``(bool)을 쓰지 않는다. 우선순위는 같다:
        ENV ``FF_RESPONSE_NUMERIC_VERIFICATION_MODE`` > JSON ``response.numeric_verification_mode``
        > 기본 ``annotate``. 알 수 없는 값은 경고를 남기고 기본값으로 둔다 — annotate는 답변을
        바꾸지 않는다. 기본값은 리드가 annotate로 효과를 측정한 뒤 정한다.
        """
        default = "annotate"
        env_name = "FF_RESPONSE_NUMERIC_VERIFICATION_MODE"
        raw = os.environ.get(env_name)
        source = env_name
        if raw is None:
            section = self._config.get("response")
            raw = section.get("numeric_verification_mode") if isinstance(section, dict) else None
            source = "config response.numeric_verification_mode"
        if raw is None:
            return default
        mode = str(raw).strip().lower()
        if mode in ("off", "annotate", "enforce"):
            return mode
        logger.warning(
            "Invalid numeric verification mode %r from %s; using %r", raw, source, default
        )
        return default

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

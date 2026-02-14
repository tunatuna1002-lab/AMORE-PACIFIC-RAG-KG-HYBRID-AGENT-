"""Centralized prompt registry for agent system prompts.

Loads prompt templates from prompts/agents/*.txt, injects shared components
(date, security guardrails, hallucination prevention), and caches results.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path


class PromptRegistry:
    """Registry for loading and composing agent prompts."""

    _prompts_dir: Path
    _cache: dict[str, str] = {}

    def __init__(self, prompts_dir: str | Path | None = None):
        if prompts_dir:
            self._prompts_dir = Path(prompts_dir)
        else:
            self._prompts_dir = self._find_prompts_dir()
        self._cache = {}

    @staticmethod
    def _find_prompts_dir() -> Path:
        """Find prompts/agents/ directory."""
        # Walk up from this file to find prompts/agents/
        current = Path(__file__).resolve().parent

        # First check if we're already in prompts/
        if current.name == "prompts":
            agents_dir = current / "agents"
            if agents_dir.exists():
                return agents_dir

        # Otherwise walk up to find project root, then down to prompts/agents/
        for _ in range(5):  # max 5 levels up
            candidate = current / "prompts" / "agents"
            if candidate.exists() and candidate.is_dir():
                return candidate
            current = current.parent

        # Fallback: relative to CWD
        return Path("prompts/agents")

    def get_system_prompt(
        self,
        agent_name: str,
        include_guardrails: bool = True,
        data_date: str | None = None,
        **kwargs,
    ) -> str:
        """Load and compose a system prompt for the given agent.

        Args:
            agent_name: One of "chatbot", "insight", "period_insight", "react"
            include_guardrails: Whether to append security guardrails
            data_date: Latest data date (defaults to today)
            **kwargs: Additional template variables

        Returns:
            Composed system prompt string
        """
        # 1. Load template from file
        template = self._load_template(agent_name)

        # 2. Inject shared components
        current_date = datetime.now().strftime("%Y-%m-%d")
        if data_date is None:
            data_date = current_date

        # Build date context
        from prompts.components import build_date_context

        date_context = build_date_context(
            current_date=current_date,
            data_date=data_date,
            start_date=kwargs.get("start_date"),
            end_date=kwargs.get("end_date"),
        )

        # Replace {current_date} placeholder
        template = template.replace("{current_date}", date_context)

        # 3. Inject guardrails if requested
        guardrails_text = ""
        if include_guardrails:
            guardrails_text = self._get_guardrails()

        template = template.replace("{guardrails}", guardrails_text)

        # 4. Inject other kwargs
        for key, value in kwargs.items():
            if key not in ["start_date", "end_date"]:  # Already handled above
                placeholder = f"{{{key}}}"
                if placeholder in template:
                    template = template.replace(placeholder, str(value))

        # 5. Clean up any remaining placeholders (replace with empty string)
        # This handles optional placeholders that weren't provided
        import re

        template = re.sub(r"\{[a-z_]+\}", "", template)

        return template

    def _load_template(self, agent_name: str) -> str:
        """Load a prompt template file."""
        if agent_name in self._cache:
            return self._cache[agent_name]

        path = self._prompts_dir / f"{agent_name}_system.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")

        text = path.read_text(encoding="utf-8")
        self._cache[agent_name] = text
        return text

    def _get_guardrails(self) -> str:
        """Return shared security and hallucination prevention text."""
        from prompts.components import get_hallucination_prevention, get_security_rules

        return get_hallucination_prevention() + "\n" + get_security_rules()

    def clear_cache(self) -> None:
        """Clear cached templates. Useful for testing."""
        self._cache.clear()

    @classmethod
    @lru_cache(maxsize=1)
    def get_instance(cls) -> PromptRegistry:
        """Singleton access."""
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton. Used in tests."""
        cls.get_instance.cache_clear()

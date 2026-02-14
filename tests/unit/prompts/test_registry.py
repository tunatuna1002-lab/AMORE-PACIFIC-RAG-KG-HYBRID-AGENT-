"""Tests for PromptRegistry centralized prompt system."""

from __future__ import annotations

from pathlib import Path

import pytest

from prompts.registry import PromptRegistry


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset PromptRegistry singleton before each test."""
    PromptRegistry.reset_instance()
    yield
    PromptRegistry.reset_instance()


@pytest.fixture()
def temp_prompts_dir(tmp_path: Path) -> Path:
    """Create a temporary prompts directory with test templates."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    chatbot = agents_dir / "chatbot_system.txt"
    chatbot.write_text("You are a chatbot.\n\n{current_date}\n\n{context}\n\n{guardrails}")

    insight = agents_dir / "insight_system.txt"
    insight.write_text(
        "You are an insight agent.\n\n{current_date}\n\n{analysis_data}\n\n{guardrails}"
    )

    period = agents_dir / "period_insight_system.txt"
    period.write_text(
        "Period analysis from {start_date} to {end_date}.\n\n{current_date}\n\n{guardrails}"
    )

    return agents_dir


class TestPromptRegistryTemplateLoading:
    """Test template file loading."""

    def test_loads_existing_template(self, temp_prompts_dir: Path) -> None:
        registry = PromptRegistry(prompts_dir=temp_prompts_dir)
        result = registry.get_system_prompt("chatbot", include_guardrails=False)
        assert "You are a chatbot" in result

    def test_missing_template_raises_error(self, temp_prompts_dir: Path) -> None:
        registry = PromptRegistry(prompts_dir=temp_prompts_dir)
        with pytest.raises(FileNotFoundError, match="Prompt template not found"):
            registry.get_system_prompt("nonexistent")

    def test_loads_insight_template(self, temp_prompts_dir: Path) -> None:
        registry = PromptRegistry(prompts_dir=temp_prompts_dir)
        result = registry.get_system_prompt(
            "insight", include_guardrails=False, analysis_data="test data"
        )
        assert "You are an insight agent" in result
        assert "test data" in result

    def test_loads_period_template(self, temp_prompts_dir: Path) -> None:
        registry = PromptRegistry(prompts_dir=temp_prompts_dir)
        result = registry.get_system_prompt(
            "period_insight",
            include_guardrails=False,
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
        assert "Period analysis" in result


class TestPromptRegistryVariableSubstitution:
    """Test template variable injection."""

    def test_injects_current_date(self, temp_prompts_dir: Path) -> None:
        registry = PromptRegistry(prompts_dir=temp_prompts_dir)
        result = registry.get_system_prompt("chatbot", include_guardrails=False)
        # current_date is replaced with build_date_context output
        assert "{current_date}" not in result

    def test_injects_custom_kwargs(self, temp_prompts_dir: Path) -> None:
        registry = PromptRegistry(prompts_dir=temp_prompts_dir)
        result = registry.get_system_prompt(
            "chatbot",
            include_guardrails=False,
            context="My custom context here",
        )
        assert "My custom context here" in result

    def test_data_date_defaults_to_today(self, temp_prompts_dir: Path) -> None:
        registry = PromptRegistry(prompts_dir=temp_prompts_dir)
        # Should not raise even without data_date
        result = registry.get_system_prompt("chatbot", include_guardrails=False)
        assert isinstance(result, str)

    def test_cleans_unused_placeholders(self, temp_prompts_dir: Path) -> None:
        registry = PromptRegistry(prompts_dir=temp_prompts_dir)
        result = registry.get_system_prompt("chatbot", include_guardrails=False)
        # {context} should be cleaned up if not provided
        assert "{context}" not in result


class TestPromptRegistryGuardrails:
    """Test guardrails inclusion/exclusion."""

    def test_includes_guardrails_by_default(self, temp_prompts_dir: Path) -> None:
        registry = PromptRegistry(prompts_dir=temp_prompts_dir)
        result = registry.get_system_prompt("chatbot")
        # When guardrails are included, {guardrails} should be replaced with content
        assert "{guardrails}" not in result

    def test_excludes_guardrails_when_disabled(self, temp_prompts_dir: Path) -> None:
        registry = PromptRegistry(prompts_dir=temp_prompts_dir)
        result = registry.get_system_prompt("chatbot", include_guardrails=False)
        assert "{guardrails}" not in result


class TestPromptRegistryCache:
    """Test caching behavior."""

    def test_caches_loaded_template(self, temp_prompts_dir: Path) -> None:
        registry = PromptRegistry(prompts_dir=temp_prompts_dir)
        # First call loads from disk
        registry.get_system_prompt("chatbot", include_guardrails=False)
        assert "chatbot" in registry._cache

    def test_clear_cache_removes_templates(self, temp_prompts_dir: Path) -> None:
        registry = PromptRegistry(prompts_dir=temp_prompts_dir)
        registry.get_system_prompt("chatbot", include_guardrails=False)
        assert len(registry._cache) > 0

        registry.clear_cache()
        assert len(registry._cache) == 0

    def test_serves_from_cache_on_second_call(self, temp_prompts_dir: Path) -> None:
        registry = PromptRegistry(prompts_dir=temp_prompts_dir)
        result1 = registry.get_system_prompt("chatbot", include_guardrails=False, context="v1")
        # Modify the cache directly to prove it's used
        registry._cache["chatbot"] = "CACHED VERSION {context} {current_date} {guardrails}"
        result2 = registry.get_system_prompt("chatbot", include_guardrails=False, context="v2")
        assert "CACHED VERSION" in result2
        assert "v2" in result2


class TestPromptRegistrySingleton:
    """Test singleton pattern."""

    def test_get_instance_returns_same_object(self) -> None:
        a = PromptRegistry.get_instance()
        b = PromptRegistry.get_instance()
        assert a is b

    def test_reset_instance_clears(self) -> None:
        a = PromptRegistry.get_instance()
        PromptRegistry.reset_instance()
        b = PromptRegistry.get_instance()
        assert a is not b


class TestPromptRegistryProjectTemplates:
    """Test loading from actual project templates."""

    def test_project_chatbot_template_loads(self) -> None:
        project_agents = Path(__file__).resolve().parents[3] / "prompts" / "agents"
        if project_agents.exists():
            registry = PromptRegistry(prompts_dir=project_agents)
            result = registry.get_system_prompt("chatbot", include_guardrails=False, context="test")
            assert len(result) > 50  # Should have substantial content
            assert "Amazon" in result or "베스트셀러" in result

    def test_all_project_templates_loadable(self) -> None:
        project_agents = Path(__file__).resolve().parents[3] / "prompts" / "agents"
        if project_agents.exists():
            registry = PromptRegistry(prompts_dir=project_agents)
            for name in ["chatbot", "insight", "period_insight", "react"]:
                template_path = project_agents / f"{name}_system.txt"
                if template_path.exists():
                    result = registry.get_system_prompt(name, include_guardrails=False)
                    assert isinstance(result, str)
                    assert len(result) > 0

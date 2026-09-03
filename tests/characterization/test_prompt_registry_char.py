"""
Prompt registry characterization
================================
Pins how prompts/registry.py (PromptRegistry) and prompts/version_manager.py
(PromptVersionManager) expose the templates in prompts/agents/ and
prompts/agents/variants/.
"""

import re
from pathlib import Path

import pytest

PLACEHOLDER_RE = re.compile(r"\{[a-z_]+\}")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = PROJECT_ROOT / "prompts" / "agents"
VARIANTS_DIR = AGENTS_DIR / "variants"

AGENT_FILES = sorted(AGENTS_DIR.glob("*.txt"))
VARIANT_FILES = sorted(VARIANTS_DIR.glob("*.txt"))

# Raw placeholder sets found on disk today (pinned exactly).
EXPECTED_AGENT_PLACEHOLDERS = {
    "chatbot": {"context", "current_date", "guardrails"},
    "insight": {"analysis_data", "current_date", "guardrails"},
    "period_insight": {"current_date", "guardrails", "period_data"},
    "react": {"available_tools", "current_date", "guardrails"},
}
EXPECTED_VARIANT_PLACEHOLDERS = {"context", "current_date", "guardrails"}


def _agent_name(path: Path) -> str:
    assert path.name.endswith("_system.txt"), path
    return path.name[: -len("_system.txt")]


@pytest.fixture
def registry():
    from prompts.registry import PromptRegistry

    PromptRegistry.reset_instance()
    reg = PromptRegistry()
    yield reg
    reg.clear_cache()
    PromptRegistry.reset_instance()


def test_agent_template_inventory_is_pinned():
    assert [_agent_name(p) for p in AGENT_FILES] == ["chatbot", "insight", "period_insight", "react"]
    assert [p.name for p in VARIANT_FILES] == [
        "chatbot_system_v0.txt",
        "chatbot_system_v1.txt",
        "chatbot_system_v1b.txt",
        "chatbot_system_v2.txt",
        "chatbot_system_v3.txt",
        "chatbot_system_v4.txt",
    ]


@pytest.mark.parametrize("path", AGENT_FILES, ids=lambda p: p.name)
def test_agent_raw_template_placeholders(path):
    raw = path.read_text(encoding="utf-8")
    assert raw.strip()
    found = set(PLACEHOLDER_RE.findall(raw))
    assert found == {f"{{{k}}}" for k in EXPECTED_AGENT_PLACEHOLDERS[_agent_name(path)]}


@pytest.mark.parametrize("path", AGENT_FILES, ids=lambda p: p.name)
def test_registry_renders_agent_with_defaults_and_no_unfilled_placeholders(registry, path):
    name = _agent_name(path)
    rendered = registry.get_system_prompt(name)
    assert rendered.strip()
    # Step 5 of get_system_prompt strips every remaining {snake_case} token, so nothing survives.
    assert PLACEHOLDER_RE.findall(rendered) == []
    # Guardrails (default include_guardrails=True) are appended from prompts.components.
    from prompts.components import get_hallucination_prevention, get_security_rules

    assert get_security_rules() in rendered
    assert get_hallucination_prevention() in rendered
    # Date context replaces {current_date}
    from datetime import datetime

    assert datetime.now().strftime("%Y-%m-%d") in rendered


def test_registry_without_guardrails_drops_component_text(registry):
    from prompts.components import get_security_rules

    with_g = registry.get_system_prompt("chatbot")
    without_g = registry.get_system_prompt("chatbot", include_guardrails=False)
    assert get_security_rules() in with_g
    assert get_security_rules() not in without_g
    assert len(without_g) < len(with_g)
    assert PLACEHOLDER_RE.findall(without_g) == []


def test_registry_injects_kwargs_and_blanks_missing_optional_placeholders(registry):
    rendered = registry.get_system_prompt("chatbot", context="__CTX_MARKER__")
    assert "__CTX_MARKER__" in rendered
    # Unprovided optional placeholder is replaced by the empty string, not left in place.
    rendered_default = registry.get_system_prompt("chatbot")
    assert "__CTX_MARKER__" not in rendered_default
    assert "{context}" not in rendered_default


def test_registry_data_date_overrides_today(registry):
    rendered = registry.get_system_prompt("chatbot", data_date="2020-01-01")
    assert "2020-01-01" in rendered


def test_registry_unknown_agent_raises_file_not_found(registry):
    with pytest.raises(FileNotFoundError):
        registry.get_system_prompt("does_not_exist")


def test_registry_cannot_load_variants_by_agent_name():
    """
    PINS CURRENT BEHAVIOR: PromptRegistry only resolves `<agent>_system.txt`, so the
    variant files (`chatbot_system_v*.txt`) are not reachable through it even when
    pointed at the variants directory.
    """
    from prompts.registry import PromptRegistry

    reg = PromptRegistry(prompts_dir=VARIANTS_DIR)
    with pytest.raises(FileNotFoundError):
        reg.get_system_prompt("chatbot")
    with pytest.raises(FileNotFoundError):
        reg.get_system_prompt("chatbot_v0")


@pytest.mark.parametrize("path", VARIANT_FILES, ids=lambda p: p.name)
def test_variant_raw_template_placeholders(path):
    raw = path.read_text(encoding="utf-8")
    assert raw.strip()
    found = set(PLACEHOLDER_RE.findall(raw))
    assert found == {f"{{{k}}}" for k in EXPECTED_VARIANT_PLACEHOLDERS}


@pytest.fixture
def version_manager(tmp_path):
    """PromptVersionManager pointed at a temp dir so metrics.json writes stay out of the repo."""
    from prompts.version_manager import PromptVersionManager

    vm = PromptVersionManager(prompts_dir=str(tmp_path))
    for path in VARIANT_FILES:
        version = path.stem.split("chatbot_system_")[1]  # v0, v1, v1b, ...
        vm.register_version("chatbot", version, path.read_text(encoding="utf-8"))
    return vm


def test_version_manager_returns_variant_content_verbatim(version_manager):
    for path in VARIANT_FILES:
        version = path.stem.split("chatbot_system_")[1]
        content = version_manager.get_prompt("chatbot", version=version)
        assert content == path.read_text(encoding="utf-8")
        # No formatting happened: the raw placeholders are still present.
        assert set(PLACEHOLDER_RE.findall(content)) == {f"{{{k}}}" for k in EXPECTED_VARIANT_PLACEHOLDERS}


def test_version_manager_latest_resolves_to_v4(version_manager):
    # "latest" = max(version.lstrip("v")) as a *string* -> "4" beats "1b".
    latest = version_manager.get_prompt("chatbot", version="latest")
    assert latest == (VARIANTS_DIR / "chatbot_system_v4.txt").read_text(encoding="utf-8")


def test_version_manager_format_kwargs_fill_variant_placeholders(version_manager):
    for path in VARIANT_FILES:
        version = path.stem.split("chatbot_system_")[1]
        content = version_manager.get_prompt(
            "chatbot", version=version, context="__C__", current_date="__D__", guardrails="__G__"
        )
        assert "__C__" in content and "__D__" in content and "__G__" in content
        assert PLACEHOLDER_RE.findall(content) == []


def test_version_manager_unknown_name_or_version_raises_key_error(version_manager):
    with pytest.raises(KeyError):
        version_manager.get_prompt("nope")
    with pytest.raises(KeyError):
        version_manager.get_prompt("chatbot", version="v99")

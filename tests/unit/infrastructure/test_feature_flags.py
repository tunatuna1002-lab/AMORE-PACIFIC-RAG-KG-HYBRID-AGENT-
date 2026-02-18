"""Tests for Feature Flags infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.infrastructure.feature_flags import FeatureFlags


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset FeatureFlags singleton before each test."""
    FeatureFlags.reset_instance()
    yield
    FeatureFlags.reset_instance()


@pytest.fixture()
def flags_config(tmp_path: Path) -> Path:
    """Create a temporary feature flags JSON config."""
    config = {
        "retriever": {
            "use_owl_strategy": True,
            "use_unified_retriever": False,
        },
        "cache": {
            "use_sqlite_embedding_cache": True,
        },
        "prompts": {
            "use_centralized_prompts": False,
        },
    }
    config_path = tmp_path / "feature_flags.json"
    config_path.write_text(json.dumps(config))
    return config_path


class TestFeatureFlagsJSONLoading:
    """Test JSON config file loading."""

    def test_loads_from_json(self, flags_config: Path) -> None:
        flags = FeatureFlags(config_path=flags_config)
        assert flags.get_flag("retriever", "use_owl_strategy") is True
        assert flags.get_flag("cache", "use_sqlite_embedding_cache") is True
        assert flags.get_flag("prompts", "use_centralized_prompts") is False

    def test_missing_file_uses_defaults(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.json"
        flags = FeatureFlags(config_path=missing)
        assert flags.get_flag("retriever", "use_owl_strategy", default=True) is True
        assert flags.get_flag("retriever", "use_owl_strategy", default=False) is False

    def test_invalid_json_uses_defaults(self, tmp_path: Path) -> None:
        bad_config = tmp_path / "bad.json"
        bad_config.write_text("not valid json{{{")
        flags = FeatureFlags(config_path=bad_config)
        assert flags.get_flag("retriever", "use_owl_strategy", default=True) is True

    def test_missing_section_uses_default(self, flags_config: Path) -> None:
        flags = FeatureFlags(config_path=flags_config)
        assert flags.get_flag("nonexistent", "some_key", default=True) is True

    def test_missing_key_in_section_uses_default(self, flags_config: Path) -> None:
        flags = FeatureFlags(config_path=flags_config)
        assert flags.get_flag("retriever", "nonexistent_key", default=False) is False


class TestFeatureFlagsENVOverride:
    """Test environment variable overrides."""

    def test_env_overrides_json_true(
        self, flags_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FF_PROMPTS_USE_CENTRALIZED_PROMPTS", "true")
        flags = FeatureFlags(config_path=flags_config)
        # JSON says False, but ENV says true
        assert flags.get_flag("prompts", "use_centralized_prompts") is True

    def test_env_overrides_json_false(
        self, flags_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FF_RETRIEVER_USE_OWL_STRATEGY", "false")
        flags = FeatureFlags(config_path=flags_config)
        # JSON says True, but ENV says false
        assert flags.get_flag("retriever", "use_owl_strategy") is False

    def test_env_accepts_various_truthy(
        self, flags_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for truthy in ("true", "True", "TRUE", "1", "yes", "on"):
            monkeypatch.setenv("FF_CACHE_USE_SQLITE_EMBEDDING_CACHE", truthy)
            flags = FeatureFlags(config_path=flags_config)
            assert (
                flags.get_flag("cache", "use_sqlite_embedding_cache") is True
            ), f"Failed for {truthy}"

    def test_env_accepts_various_falsy(
        self, flags_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for falsy in ("false", "False", "FALSE", "0", "no", "off", ""):
            monkeypatch.setenv("FF_CACHE_USE_SQLITE_EMBEDDING_CACHE", falsy)
            flags = FeatureFlags(config_path=flags_config)
            assert (
                flags.get_flag("cache", "use_sqlite_embedding_cache") is False
            ), f"Failed for {falsy}"

    def test_env_overrides_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FF_NEW_SECTION_NEW_KEY", "true")
        flags = FeatureFlags(config_path=tmp_path / "missing.json")
        assert flags.get_flag("new_section", "new_key", default=False) is True


class TestFeatureFlagsReload:
    """Test config reload functionality."""

    def test_reload_picks_up_changes(self, tmp_path: Path) -> None:
        config_path = tmp_path / "flags.json"
        config_path.write_text(json.dumps({"cache": {"use_sqlite_embedding_cache": False}}))

        flags = FeatureFlags(config_path=config_path)
        assert flags.get_flag("cache", "use_sqlite_embedding_cache") is False

        # Update config file
        config_path.write_text(json.dumps({"cache": {"use_sqlite_embedding_cache": True}}))
        flags.reload()
        assert flags.get_flag("cache", "use_sqlite_embedding_cache") is True


class TestFeatureFlagsConvenienceMethods:
    """Test convenience methods."""

    def test_use_owl_strategy(self, flags_config: Path) -> None:
        flags = FeatureFlags(config_path=flags_config)
        assert flags.use_owl_strategy() is True

    def test_use_unified_retriever(self, flags_config: Path) -> None:
        flags = FeatureFlags(config_path=flags_config)
        assert flags.use_unified_retriever() is False

    def test_use_sqlite_embedding_cache(self, flags_config: Path) -> None:
        flags = FeatureFlags(config_path=flags_config)
        assert flags.use_sqlite_embedding_cache() is True

    def test_use_centralized_prompts(self, flags_config: Path) -> None:
        flags = FeatureFlags(config_path=flags_config)
        assert flags.use_centralized_prompts() is False

    def test_convenience_defaults_when_no_config(self, tmp_path: Path) -> None:
        flags = FeatureFlags(config_path=tmp_path / "missing.json")
        # Defaults defined in each convenience method
        assert flags.use_owl_strategy() is True  # default True
        assert flags.use_unified_retriever() is True  # default True
        assert flags.use_unified_reasoner() is True  # default True
        assert flags.use_owl_reasoner() is True  # default True
        assert flags.use_ontology_kg() is True  # default True
        assert flags.use_sqlite_embedding_cache() is False  # default False
        assert flags.use_centralized_prompts() is False  # default False
        assert flags.use_decomposed_chatbot() is True  # default True


class TestFeatureFlagsSingleton:
    """Test singleton pattern."""

    def test_get_instance_returns_same_object(self, flags_config: Path) -> None:
        a = FeatureFlags.get_instance(config_path=flags_config)
        b = FeatureFlags.get_instance()
        assert a is b

    def test_reset_instance_clears(self, flags_config: Path) -> None:
        a = FeatureFlags.get_instance(config_path=flags_config)
        FeatureFlags.reset_instance()
        b = FeatureFlags.get_instance(config_path=flags_config)
        assert a is not b


class TestFeatureFlagsProjectConfig:
    """Test loading from actual project config/feature_flags.json."""

    def test_project_config_loads(self) -> None:
        project_config = Path(__file__).resolve().parents[3] / "config" / "feature_flags.json"
        if project_config.exists():
            flags = FeatureFlags(config_path=project_config)
            # Should load without error and have expected sections
            assert flags.use_owl_strategy() in (True, False)
            assert flags.use_sqlite_embedding_cache() in (True, False)

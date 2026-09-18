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
            "use_reranker": True,
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
        assert flags.get_flag("retriever", "use_reranker") is True
        assert flags.get_flag("cache", "use_sqlite_embedding_cache") is True
        assert flags.get_flag("prompts", "use_centralized_prompts") is False

    def test_missing_file_uses_defaults(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.json"
        flags = FeatureFlags(config_path=missing)
        assert flags.get_flag("retriever", "use_reranker", default=True) is True
        assert flags.get_flag("retriever", "use_reranker", default=False) is False

    def test_invalid_json_uses_defaults(self, tmp_path: Path) -> None:
        bad_config = tmp_path / "bad.json"
        bad_config.write_text("not valid json{{{")
        flags = FeatureFlags(config_path=bad_config)
        assert flags.get_flag("retriever", "use_reranker", default=True) is True

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
        monkeypatch.setenv("FF_RETRIEVER_USE_RERANKER", "false")
        flags = FeatureFlags(config_path=flags_config)
        # JSON says True, but ENV says false
        assert flags.get_flag("retriever", "use_reranker") is False

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

    def test_react_bypass_confidence_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FF_AGENTS_REACT_BYPASS_CONFIDENCE", "true")
        flags = FeatureFlags(config_path=tmp_path / "missing.json")
        assert flags.react_bypass_confidence() is True


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

    def test_use_reranker(self, flags_config: Path) -> None:
        flags = FeatureFlags(config_path=flags_config)
        assert flags.use_reranker() is True

    def test_use_sqlite_embedding_cache(self, flags_config: Path) -> None:
        flags = FeatureFlags(config_path=flags_config)
        assert flags.use_sqlite_embedding_cache() is True

    def test_use_centralized_prompts(self, flags_config: Path) -> None:
        flags = FeatureFlags(config_path=flags_config)
        assert flags.use_centralized_prompts() is False

    def test_convenience_defaults_when_no_config(self, tmp_path: Path) -> None:
        flags = FeatureFlags(config_path=tmp_path / "missing.json")
        # Defaults defined in each convenience method
        # 2026-09: ReAct는 수리 후 평가 확인 전까지 기본 OFF (결정 D1)
        assert flags.use_react_agent() is False
        assert flags.react_bypass_confidence() is False  # default False
        assert flags.use_unified_reasoner() is True  # default True
        assert flags.use_owl_reasoner() is True  # default True
        assert flags.use_ontology_kg() is True  # default True
        assert flags.use_sqlite_embedding_cache() is False  # default False
        assert flags.use_centralized_prompts() is False  # default False


class TestNumericVerificationMode:
    """response.numeric_verification_mode: off | annotate | enforce (기본 annotate)."""

    def test_default_is_annotate(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("FF_RESPONSE_NUMERIC_VERIFICATION_MODE", raising=False)
        flags = FeatureFlags(config_path=tmp_path / "missing.json")
        assert flags.numeric_verification_mode() == "annotate"

    def test_json_value(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("FF_RESPONSE_NUMERIC_VERIFICATION_MODE", raising=False)
        path = tmp_path / "flags.json"
        path.write_text(json.dumps({"response": {"numeric_verification_mode": "enforce"}}))
        assert FeatureFlags(config_path=path).numeric_verification_mode() == "enforce"

    def test_env_overrides_json(self, tmp_path: Path, monkeypatch) -> None:
        path = tmp_path / "flags.json"
        path.write_text(json.dumps({"response": {"numeric_verification_mode": "enforce"}}))
        monkeypatch.setenv("FF_RESPONSE_NUMERIC_VERIFICATION_MODE", " OFF ")
        assert FeatureFlags(config_path=path).numeric_verification_mode() == "off"

    def test_invalid_env_value_falls_back_to_annotate_with_warning(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        path = tmp_path / "flags.json"
        path.write_text(json.dumps({"response": {"numeric_verification_mode": "enforce"}}))
        monkeypatch.setenv("FF_RESPONSE_NUMERIC_VERIFICATION_MODE", "strict")
        flags = FeatureFlags(config_path=path)
        with caplog.at_level("WARNING", logger="src.infrastructure.feature_flags"):
            # JSON 값(enforce)이 아니라 기본값으로 — 잘못 쓴 ENV가 조용히 무시되지 않게
            assert flags.numeric_verification_mode() == "annotate"
        assert "strict" in caplog.text
        assert "FF_RESPONSE_NUMERIC_VERIFICATION_MODE" in caplog.text

    def test_invalid_json_value_falls_back_with_warning(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        monkeypatch.delenv("FF_RESPONSE_NUMERIC_VERIFICATION_MODE", raising=False)
        path = tmp_path / "flags.json"
        path.write_text(json.dumps({"response": {"numeric_verification_mode": True}}))
        with caplog.at_level("WARNING", logger="src.infrastructure.feature_flags"):
            assert FeatureFlags(config_path=path).numeric_verification_mode() == "annotate"
        assert "response.numeric_verification_mode" in caplog.text

    def test_malformed_response_section_uses_default(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("FF_RESPONSE_NUMERIC_VERIFICATION_MODE", raising=False)
        path = tmp_path / "flags.json"
        path.write_text(json.dumps({"response": "enforce"}))
        assert FeatureFlags(config_path=path).numeric_verification_mode() == "annotate"

    def test_repository_config_default(self, monkeypatch) -> None:
        monkeypatch.delenv("FF_RESPONSE_NUMERIC_VERIFICATION_MODE", raising=False)
        config = Path(__file__).resolve().parents[3] / "config" / "feature_flags.json"
        assert FeatureFlags(config_path=config).numeric_verification_mode() == "annotate"
        assert json.loads(config.read_text())["response"]["numeric_verification_mode"] == (
            "annotate"
        )


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
            assert flags.use_sqlite_embedding_cache() in (True, False)


class TestOWLStrategyFlagRemoved:
    """S4-1(트랙 4-C): OWL 검색 전략 플래그는 삭제됐다."""

    def test_feature_flags_has_no_use_owl_strategy(self) -> None:
        assert not hasattr(FeatureFlags, "use_owl_strategy")

    def test_project_config_has_no_owl_strategy_key(self) -> None:
        project_config = Path(__file__).resolve().parents[3] / "config" / "feature_flags.json"
        config = json.loads(project_config.read_text(encoding="utf-8"))
        assert "use_owl_strategy" not in config.get("retriever", {})

    def test_rule_inference_flags_are_kept(self, tmp_path: Path) -> None:
        """규칙 추론 on/off 스위치는 그대로 유지한다 (평가 ablation no-ontology)."""
        flags = FeatureFlags(config_path=tmp_path / "missing.json")
        assert flags.use_unified_reasoner() is True
        assert flags.use_owl_reasoner() is True

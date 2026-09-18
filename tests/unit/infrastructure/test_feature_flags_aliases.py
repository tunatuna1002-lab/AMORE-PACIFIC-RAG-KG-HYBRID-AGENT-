"""플래그 이름 바로잡기와 옛 이름 별칭 (설계 OE7, 트랙 O6) [2026-09 사후].

| 새 이름 | 옛 이름 |
|---|---|
| ``reasoner.enabled`` (ENV ``FF_REASONER_ENABLED``) | ``reasoner.use_owl_reasoner`` (``FF_REASONER_USE_OWL_REASONER``) |
| ``kg.enabled`` (ENV ``FF_KG_ENABLED``) | ``ontology.use_ontology_kg`` (``FF_ONTOLOGY_USE_ONTOLOGY_KG``) |

우선순위: ENV 새 > ENV 옛 > JSON 새 > JSON 옛 > 기본값(True). 옛 이름이 쓰이면 1회 경고.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.infrastructure import feature_flags as ff_module
from src.infrastructure.feature_flags import FeatureFlags

ENV_NAMES = (
    "FF_REASONER_ENABLED",
    "FF_REASONER_USE_OWL_REASONER",
    "FF_REASONER_USE_UNIFIED_REASONER",
    "FF_KG_ENABLED",
    "FF_ONTOLOGY_USE_ONTOLOGY_KG",
)

# (메서드, 새 섹션, 새 키, 옛 섹션, 옛 키, 새 ENV, 옛 ENV)
FLAGS = [
    pytest.param(
        "reasoner_enabled",
        "reasoner",
        "enabled",
        "reasoner",
        "use_owl_reasoner",
        "FF_REASONER_ENABLED",
        "FF_REASONER_USE_OWL_REASONER",
        id="reasoner",
    ),
    pytest.param(
        "kg_enabled",
        "kg",
        "enabled",
        "ontology",
        "use_ontology_kg",
        "FF_KG_ENABLED",
        "FF_ONTOLOGY_USE_ONTOLOGY_KG",
        id="kg",
    ),
]


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    FeatureFlags.reset_instance()
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    ff_module._warned_aliases.clear()
    yield
    FeatureFlags.reset_instance()
    ff_module._warned_aliases.clear()


def _flags(tmp_path: Path, config: dict) -> FeatureFlags:
    path = tmp_path / "feature_flags.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return FeatureFlags(config_path=path)


def _set(config: dict, section: str, key: str, value: bool) -> None:
    config.setdefault(section, {})[key] = value


@pytest.mark.parametrize("method,sec,key,osec,okey,env_new,env_old", FLAGS)
class TestAliasResolution:
    def test_default_is_true(self, tmp_path, method, sec, key, osec, okey, env_new, env_old):
        assert getattr(_flags(tmp_path, {}), method)() is True

    def test_json_new_only(self, tmp_path, method, sec, key, osec, okey, env_new, env_old):
        config: dict = {}
        _set(config, sec, key, False)
        assert getattr(_flags(tmp_path, config), method)() is False

    def test_json_old_only_is_alias(
        self, tmp_path, caplog, method, sec, key, osec, okey, env_new, env_old
    ):
        config: dict = {}
        _set(config, osec, okey, False)
        with caplog.at_level(logging.WARNING, logger="src.infrastructure.feature_flags"):
            assert getattr(_flags(tmp_path, config), method)() is False
        assert f"{osec}.{okey}" in caplog.text and "deprecated" in caplog.text

    def test_json_new_wins_over_json_old(
        self, tmp_path, method, sec, key, osec, okey, env_new, env_old
    ):
        config: dict = {}
        _set(config, sec, key, True)
        _set(config, osec, okey, False)
        assert getattr(_flags(tmp_path, config), method)() is True
        config2: dict = {}
        _set(config2, sec, key, False)
        _set(config2, osec, okey, True)
        assert getattr(_flags(tmp_path, config2), method)() is False

    def test_env_new_wins_over_everything(
        self, tmp_path, monkeypatch, method, sec, key, osec, okey, env_new, env_old
    ):
        config: dict = {}
        _set(config, sec, key, True)
        _set(config, osec, okey, True)
        monkeypatch.setenv(env_new, "false")
        monkeypatch.setenv(env_old, "true")
        assert getattr(_flags(tmp_path, config), method)() is False

    def test_env_old_wins_over_json_new(
        self, tmp_path, monkeypatch, caplog, method, sec, key, osec, okey, env_new, env_old
    ):
        """평가 스크립트의 옛 ENV(=false)는 JSON 새 키(true)보다 우선해야 한다."""
        config: dict = {}
        _set(config, sec, key, True)
        monkeypatch.setenv(env_old, "false")
        with caplog.at_level(logging.WARNING, logger="src.infrastructure.feature_flags"):
            assert getattr(_flags(tmp_path, config), method)() is False
        assert env_old in caplog.text

    def test_deprecated_method_matches_new(
        self, tmp_path, monkeypatch, method, sec, key, osec, okey, env_new, env_old
    ):
        old_method = {"reasoner_enabled": "use_owl_reasoner", "kg_enabled": "use_ontology_kg"}[
            method
        ]
        flags = _flags(tmp_path, {})
        for value in ("true", "false"):
            monkeypatch.setenv(env_old, value)
            assert getattr(flags, old_method)() is getattr(flags, method)()

    def test_warning_logged_once(
        self, tmp_path, monkeypatch, caplog, method, sec, key, osec, okey, env_new, env_old
    ):
        monkeypatch.setenv(env_old, "false")
        flags = _flags(tmp_path, {})
        with caplog.at_level(logging.WARNING, logger="src.infrastructure.feature_flags"):
            for _ in range(3):
                getattr(flags, method)()
        assert caplog.text.count("deprecated") == 1

    def test_no_warning_for_new_names(
        self, tmp_path, monkeypatch, caplog, method, sec, key, osec, okey, env_new, env_old
    ):
        config: dict = {}
        _set(config, sec, key, True)
        monkeypatch.setenv(env_new, "true")
        with caplog.at_level(logging.WARNING, logger="src.infrastructure.feature_flags"):
            getattr(_flags(tmp_path, config), method)()
        assert "deprecated" not in caplog.text


class TestRuleOffSwitchStillWorks:
    """평가의 규칙 off 명령 ``FF_REASONER_USE_OWL_REASONER=false FF_REASONER_USE_UNIFIED_REASONER=false``."""

    def test_old_env_pair_turns_rules_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        flags = FeatureFlags()  # 저장소 config/feature_flags.json (새 이름 사용)
        assert flags.use_unified_reasoner() or flags.reasoner_enabled()
        monkeypatch.setenv("FF_REASONER_USE_OWL_REASONER", "false")
        monkeypatch.setenv("FF_REASONER_USE_UNIFIED_REASONER", "false")
        assert not (flags.use_unified_reasoner() or flags.reasoner_enabled())

    def test_old_env_turns_kg_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        flags = FeatureFlags()
        assert flags.kg_enabled() is True
        monkeypatch.setenv("FF_ONTOLOGY_USE_ONTOLOGY_KG", "false")
        assert flags.kg_enabled() is False


class TestProjectConfigUsesNewNames:
    def test_new_keys_present_old_keys_absent(self) -> None:
        project_config = Path(__file__).resolve().parents[3] / "config" / "feature_flags.json"
        config = json.loads(project_config.read_text(encoding="utf-8"))
        assert config["reasoner"]["enabled"] is True
        assert config["kg"]["enabled"] is True
        assert "use_owl_reasoner" not in config["reasoner"]
        assert "use_ontology_kg" not in config.get("ontology", {})

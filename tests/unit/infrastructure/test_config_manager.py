"""
ConfigManager (AppConfig) 단위 테스트
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.infrastructure.config.config_manager import AppConfig

# =============================================================================
# AppConfig 기본 생성 테스트
# =============================================================================


class TestAppConfigDefaults:
    """AppConfig 기본값 테스트"""

    def test_default_values(self):
        """기본값으로 생성"""
        config = AppConfig()
        assert config.openai_api_key is None
        assert config.api_key is None
        assert config.use_sheets is True
        assert config.target_brands == ["LANEIGE"]
        assert config.competitor_brands == []
        assert config.auto_start_scheduler is False
        assert config.scheduler_cron == "0 6 * * *"
        assert config.host == "0.0.0.0"
        assert config.port == 8001

    def test_default_paths(self):
        """기본 경로 설정"""
        config = AppConfig()
        assert isinstance(config.base_path, Path)
        assert isinstance(config.data_path, Path)
        assert isinstance(config.docs_path, Path)
        assert isinstance(config.logs_path, Path)
        assert isinstance(config.config_path, Path)


# =============================================================================
# from_env 테스트
# =============================================================================


class TestAppConfigFromEnv:
    """AppConfig.from_env 테스트"""

    def test_loads_openai_key(self):
        """OPENAI_API_KEY 로드"""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-test123"},  # pragma: allowlist secret
            clear=False,
        ):
            config = AppConfig.from_env()
            assert config.openai_api_key == "sk-test123"

    def test_loads_api_key(self):
        """API_KEY 로드"""
        with patch.dict(
            os.environ,
            {"API_KEY": "my-api-key"},  # pragma: allowlist secret
            clear=False,
        ):
            config = AppConfig.from_env()
            assert config.api_key == "my-api-key"

    def test_loads_google_spreadsheet_id(self):
        """GOOGLE_SPREADSHEET_ID 로드"""
        with patch.dict(os.environ, {"GOOGLE_SPREADSHEET_ID": "sheet-123"}, clear=False):
            config = AppConfig.from_env()
            assert config.google_spreadsheet_id == "sheet-123"

    def test_loads_auto_start_scheduler_true(self):
        """AUTO_START_SCHEDULER=true"""
        with patch.dict(os.environ, {"AUTO_START_SCHEDULER": "true"}, clear=False):
            config = AppConfig.from_env()
            assert config.auto_start_scheduler is True

    def test_loads_auto_start_scheduler_false(self):
        """AUTO_START_SCHEDULER=false"""
        with patch.dict(os.environ, {"AUTO_START_SCHEDULER": "false"}, clear=False):
            config = AppConfig.from_env()
            assert config.auto_start_scheduler is False

    def test_loads_port(self):
        """PORT 환경변수 로드"""
        with patch.dict(os.environ, {"PORT": "9000"}, clear=False):
            config = AppConfig.from_env()
            assert config.port == 9000

    def test_default_port(self):
        """PORT 미설정 시 기본값"""
        env = os.environ.copy()
        env.pop("PORT", None)
        with patch.dict(os.environ, env, clear=True):
            config = AppConfig.from_env()
            assert config.port == 8001


# =============================================================================
# from_files 테스트
# =============================================================================


class TestAppConfigFromFiles:
    """AppConfig.from_files 테스트"""

    def test_from_files_with_config_dir(self, tmp_path):
        """커스텀 config 디렉토리 사용"""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        thresholds = {"thresholds": {"rank_change": 5}, "categories": {"lip_care": "url1"}}
        (config_dir / "thresholds.json").write_text(json.dumps(thresholds), encoding="utf-8")

        config = AppConfig.from_files(config_dir=config_dir)
        assert config.config_path == config_dir
        assert config.thresholds.get("rank_change") == 5
        assert config.categories.get("lip_care") == "url1"


# =============================================================================
# _load_thresholds 테스트
# =============================================================================


class TestAppConfigLoadThresholds:
    """AppConfig._load_thresholds 테스트"""

    def test_load_thresholds_from_file(self, tmp_path):
        """thresholds.json 로드"""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        thresholds = {
            "thresholds": {"rank_change": 5, "sos_drop": 10},
            "categories": {"lip_care": "https://example.com"},
        }
        (config_dir / "thresholds.json").write_text(json.dumps(thresholds), encoding="utf-8")

        config = AppConfig()
        config.config_path = config_dir
        config._load_thresholds()
        assert config.thresholds["rank_change"] == 5
        assert config.categories["lip_care"] == "https://example.com"

    def test_load_thresholds_no_file(self, tmp_path):
        """thresholds.json 없을 시"""
        config = AppConfig()
        config.config_path = tmp_path
        config._load_thresholds()
        assert config.thresholds == {}


# =============================================================================
# _load_brands 테스트
# =============================================================================


class TestAppConfigLoadBrands:
    """AppConfig._load_brands 테스트"""

    def test_load_brands_from_file(self, tmp_path):
        """brands.json 로드"""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        brands = {
            "target_brands": ["LANEIGE", "SULWHASOO"],
            "competitor_brands": ["COSRX", "TIRTIR"],
        }
        (config_dir / "brands.json").write_text(json.dumps(brands), encoding="utf-8")

        config = AppConfig()
        config.config_path = config_dir
        config._load_brands()
        assert config.target_brands == ["LANEIGE", "SULWHASOO"]
        assert config.competitor_brands == ["COSRX", "TIRTIR"]

    def test_load_brands_no_file(self, tmp_path):
        """brands.json 없을 시 기본값 유지"""
        config = AppConfig()
        config.config_path = tmp_path
        config._load_brands()
        assert config.target_brands == ["LANEIGE"]
        assert config.competitor_brands == []


# =============================================================================
# validate 테스트
# =============================================================================


class TestAppConfigValidate:
    """AppConfig.validate 테스트"""

    def test_validate_no_errors_with_api_key(self, tmp_path):
        """API 키 있으면 필수 검증 통과"""
        config = AppConfig()
        config.openai_api_key = "sk-test"  # pragma: allowlist secret  # pragma: allowlist secret
        config.data_path = tmp_path
        config.config_path = tmp_path
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_missing_openai_key(self):
        """OPENAI_API_KEY 누락 시 에러"""
        config = AppConfig()
        config.openai_api_key = None
        errors = config.validate()
        assert any("OPENAI_API_KEY" in e for e in errors)

    def test_validate_invalid_port_low(self):
        """포트 0 이하 에러"""
        config = AppConfig()
        config.openai_api_key = "sk-test"  # pragma: allowlist secret
        config.port = 0
        errors = config.validate()
        assert any("PORT" in e for e in errors)

    def test_validate_invalid_port_high(self):
        """포트 65536 이상 에러"""
        config = AppConfig()
        config.openai_api_key = "sk-test"  # pragma: allowlist secret
        config.port = 70000
        errors = config.validate()
        assert any("PORT" in e for e in errors)

    def test_validate_valid_port(self):
        """유효 포트"""
        config = AppConfig()
        config.openai_api_key = "sk-test"  # pragma: allowlist secret
        config.port = 8001
        errors = config.validate()
        port_errors = [e for e in errors if "PORT" in e]
        assert len(port_errors) == 0


# =============================================================================
# from_env_validated 테스트
# =============================================================================


class TestAppConfigFromEnvValidated:
    """AppConfig.from_env_validated 테스트"""

    def test_fail_fast_raises(self):
        """fail_fast=True일 때 검증 실패 시 RuntimeError"""
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("OPENAI_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(RuntimeError, match="설정 검증 실패"):
                    AppConfig.from_env_validated(fail_fast=True)

    def test_no_fail_fast_returns_config(self):
        """fail_fast=False일 때 검증 실패해도 config 반환"""
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            config = AppConfig.from_env_validated(fail_fast=False)
            assert isinstance(config, AppConfig)

    def test_valid_config_no_error(self):
        """유효 설정일 때 정상 반환"""
        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):  # pragma: allowlist secret
            config = AppConfig.from_env_validated(fail_fast=True)
            assert config.openai_api_key == "sk-test"


# =============================================================================
# 유틸리티 메서드 테스트
# =============================================================================


class TestAppConfigUtilities:
    """AppConfig 유틸리티 메서드 테스트"""

    def test_get_threshold(self):
        """get_threshold 조회"""
        config = AppConfig()
        config.thresholds = {"rank_change": 5, "sos_drop": 10}
        assert config.get_threshold("rank_change") == 5
        assert config.get_threshold("unknown") is None
        assert config.get_threshold("unknown", default=99) == 99

    def test_get_category_url(self):
        """get_category_url 조회"""
        config = AppConfig()
        config.categories = {"lip_care": "https://amazon.com/lip-care"}
        assert config.get_category_url("lip_care") == "https://amazon.com/lip-care"
        assert config.get_category_url("unknown") is None

    def test_to_dict(self):
        """to_dict 변환"""
        config = AppConfig()
        config.categories = {"lip_care": "url"}
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "base_path" in d
        assert "data_path" in d
        assert "use_sheets" in d
        assert "target_brands" in d
        assert d["categories"] == ["lip_care"]

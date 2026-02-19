"""
Centralized Configuration Manager
=================================
모든 설정을 중앙에서 관리합니다.

주요 기능:
- 환경변수 및 JSON 파일에서 설정 로드
- 시작 시 설정 검증 (validate)
- 환경 프로필 지원 (development, production)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """
    애플리케이션 설정

    환경변수와 설정 파일에서 로드합니다.
    """

    # Paths
    base_path: Path = field(default_factory=lambda: Path.cwd())
    data_path: Path = field(
        default_factory=lambda: Path("/data") if Path("/data").exists() else Path.cwd() / "data"
    )
    docs_path: Path = field(default_factory=lambda: Path.cwd() / "docs" / "guides")
    logs_path: Path = field(default_factory=lambda: Path.cwd() / "logs")
    config_path: Path = field(default_factory=lambda: Path.cwd() / "config")

    # API Keys (from env)
    openai_api_key: str | None = None
    api_key: str | None = None

    # Google Sheets
    google_spreadsheet_id: str | None = None
    google_credentials_path: str | None = None
    use_sheets: bool = True

    # Thresholds (from config/thresholds.json)
    thresholds: dict[str, Any] = field(default_factory=dict)

    # Categories (from config/thresholds.json or separate file)
    categories: dict[str, str] = field(default_factory=dict)

    # Brands
    target_brands: list[str] = field(default_factory=lambda: ["LANEIGE"])
    competitor_brands: list[str] = field(default_factory=list)

    # Scheduler
    auto_start_scheduler: bool = False
    scheduler_cron: str = "0 6 * * *"  # 06:00 KST daily

    # Server
    host: str = "0.0.0.0"
    port: int = 8001

    @classmethod
    def from_env(cls) -> "AppConfig":
        """환경변수에서 설정 로드"""
        config = cls()

        # Load from environment
        config.openai_api_key = os.environ.get("OPENAI_API_KEY")
        config.api_key = os.environ.get("API_KEY")
        config.google_spreadsheet_id = os.environ.get("GOOGLE_SPREADSHEET_ID")
        config.google_credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        config.auto_start_scheduler = os.environ.get("AUTO_START_SCHEDULER", "").lower() == "true"
        config.port = int(os.environ.get("PORT", "8001"))

        # Load from config files
        config._load_thresholds()
        config._load_brands()

        return config

    @classmethod
    def from_files(cls, config_dir: Path = None) -> "AppConfig":
        """설정 파일에서 로드"""
        config = cls.from_env()

        if config_dir:
            config.config_path = config_dir

        config._load_thresholds()
        config._load_brands()

        return config

    def _load_thresholds(self) -> None:
        """thresholds.json 로드"""
        thresholds_path = self.config_path / "thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path, encoding="utf-8") as f:
                data = json.load(f)
                self.thresholds = data.get("thresholds", data)
                self.categories = data.get("categories", {})

    def _load_brands(self) -> None:
        """brands.json 로드 (있는 경우)"""
        brands_path = self.config_path / "brands.json"
        if brands_path.exists():
            with open(brands_path, encoding="utf-8") as f:
                data = json.load(f)
                self.target_brands = data.get("target_brands", self.target_brands)
                self.competitor_brands = data.get("competitor_brands", [])

    def validate(self) -> list[str]:
        """설정 검증

        필수/선택 설정의 유효성을 검사하고, 오류 목록을 반환합니다.
        빈 리스트 반환 시 모든 검증 통과.

        Returns:
            오류 메시지 목록 (빈 리스트 = 정상)
        """
        errors: list[str] = []
        warnings: list[str] = []

        # === 필수 검증 ===
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY가 설정되지 않았습니다")

        # === 경로 검증 (경고만) ===
        if not self.data_path.exists():
            warnings.append(f"data 디렉토리가 없습니다: {self.data_path}")

        if not self.config_path.exists():
            warnings.append(f"config 디렉토리가 없습니다: {self.config_path}")
        else:
            # config 하위 파일 검증
            thresholds_path = self.config_path / "thresholds.json"
            if not thresholds_path.exists():
                warnings.append(f"thresholds.json을 찾을 수 없습니다: {thresholds_path}")

            hierarchy_path = self.config_path / "category_hierarchy.json"
            if not hierarchy_path.exists():
                warnings.append(f"category_hierarchy.json을 찾을 수 없습니다: {hierarchy_path}")

        # === 포트 범위 검증 ===
        if self.port < 1 or self.port > 65535:
            errors.append(f"PORT 범위 오류: 1-65535 필요, 현재 {self.port}")

        # === 경고 로깅 ===
        for w in warnings:
            logger.warning(f"[Config Warning] {w}")

        return errors

    @classmethod
    def from_env_validated(cls, fail_fast: bool = True) -> "AppConfig":
        """환경변수에서 설정 로드 + 검증

        Args:
            fail_fast: True면 필수 설정 누락 시 RuntimeError 발생.
                       False면 경고만 로깅하고 config 반환.

        Returns:
            검증된 AppConfig 인스턴스

        Raises:
            RuntimeError: fail_fast=True이고 필수 설정 누락 시
        """
        config = cls.from_env()
        errors = config.validate()

        if errors:
            error_msg = "설정 검증 실패:\n" + "\n".join(f"  - {e}" for e in errors)
            if fail_fast:
                raise RuntimeError(error_msg)
            else:
                logger.error(error_msg)

        return config

    def get_threshold(self, key: str, default: Any = None) -> Any:
        """특정 threshold 값 조회"""
        return self.thresholds.get(key, default)

    def get_category_url(self, category_id: str) -> str | None:
        """카테고리 URL 조회"""
        return self.categories.get(category_id)

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "base_path": str(self.base_path),
            "data_path": str(self.data_path),
            "docs_path": str(self.docs_path),
            "use_sheets": self.use_sheets,
            "target_brands": self.target_brands,
            "competitor_brands": self.competitor_brands,
            "auto_start_scheduler": self.auto_start_scheduler,
            "categories": list(self.categories.keys()),
        }

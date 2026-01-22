"""
Centralized Configuration Manager
=================================
모든 설정을 중앙에서 관리합니다.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class AppConfig:
    """
    애플리케이션 설정

    환경변수와 설정 파일에서 로드합니다.
    """
    # Paths
    base_path: Path = field(default_factory=lambda: Path.cwd())
    data_path: Path = field(default_factory=lambda: Path.cwd() / "data")
    docs_path: Path = field(default_factory=lambda: Path.cwd() / "docs" / "guides")
    logs_path: Path = field(default_factory=lambda: Path.cwd() / "logs")
    config_path: Path = field(default_factory=lambda: Path.cwd() / "config")

    # API Keys (from env)
    openai_api_key: Optional[str] = None
    api_key: Optional[str] = None

    # Google Sheets
    google_spreadsheet_id: Optional[str] = None
    google_credentials_path: Optional[str] = None
    use_sheets: bool = True

    # Thresholds (from config/thresholds.json)
    thresholds: Dict[str, Any] = field(default_factory=dict)

    # Categories (from config/thresholds.json or separate file)
    categories: Dict[str, str] = field(default_factory=dict)

    # Brands
    target_brands: List[str] = field(default_factory=lambda: ["LANEIGE"])
    competitor_brands: List[str] = field(default_factory=list)

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
            with open(thresholds_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.thresholds = data.get("thresholds", data)
                self.categories = data.get("categories", {})

    def _load_brands(self) -> None:
        """brands.json 로드 (있는 경우)"""
        brands_path = self.config_path / "brands.json"
        if brands_path.exists():
            with open(brands_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.target_brands = data.get("target_brands", self.target_brands)
                self.competitor_brands = data.get("competitor_brands", [])

    def get_threshold(self, key: str, default: Any = None) -> Any:
        """특정 threshold 값 조회"""
        return self.thresholds.get(key, default)

    def get_category_url(self, category_id: str) -> Optional[str]:
        """카테고리 URL 조회"""
        return self.categories.get(category_id)

    def to_dict(self) -> Dict[str, Any]:
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

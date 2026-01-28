"""
Prompt Version Manager
======================
프롬프트 버전 관리 및 A/B 테스트 지원

Usage:
    manager = PromptVersionManager()
    prompt = manager.get_prompt("insight_generation", version="v2")

    # A/B 테스트
    prompt = manager.get_prompt_ab("insight_generation", {"v1": 0.3, "v2": 0.7})
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PromptVersion:
    """프롬프트 버전 정보"""

    def __init__(
        self,
        name: str,
        version: str,
        content: str,
        created_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.name = name
        self.version = version
        self.content = content
        self.created_at = created_at or datetime.now().isoformat()
        self.metadata = metadata or {}

        # 메트릭
        self.usage_count = 0
        self.success_count = 0


class PromptVersionManager:
    """프롬프트 버전 관리자"""

    def __init__(self, prompts_dir: str = "./prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.versions: dict[str, dict[str, PromptVersion]] = {}
        self._metrics_file = self.prompts_dir / "metrics.json"
        self._load_metrics()

    def register_version(
        self, name: str, version: str, content: str, metadata: dict[str, Any] | None = None
    ) -> PromptVersion:
        """프롬프트 버전 등록"""
        if name not in self.versions:
            self.versions[name] = {}

        pv = PromptVersion(name, version, content, metadata=metadata)
        self.versions[name][version] = pv
        logger.info(f"Registered prompt: {name} {version}")
        return pv

    def get_prompt(self, name: str, version: str = "latest", **format_kwargs) -> str:
        """프롬프트 조회"""
        if name not in self.versions:
            raise KeyError(f"Prompt not found: {name}")

        versions = self.versions[name]

        if version == "latest":
            # 가장 높은 버전 선택
            version = max(versions.keys(), key=lambda v: v.lstrip("v"))

        if version not in versions:
            raise KeyError(f"Version not found: {name} {version}")

        pv = versions[version]
        pv.usage_count += 1

        content = pv.content
        if format_kwargs:
            content = content.format(**format_kwargs)

        return content

    def get_prompt_ab(
        self, name: str, weights: dict[str, float], **format_kwargs
    ) -> tuple[str, str]:
        """A/B 테스트로 프롬프트 선택

        Args:
            name: 프롬프트 이름
            weights: 버전별 가중치 (예: {"v1": 0.3, "v2": 0.7})
            **format_kwargs: 프롬프트 포맷팅 인자

        Returns:
            (formatted_content, selected_version)
        """
        versions = list(weights.keys())
        probs = list(weights.values())

        # 정규화
        total = sum(probs)
        probs = [p / total for p in probs]

        selected = random.choices(versions, weights=probs, k=1)[0]
        content = self.get_prompt(name, version=selected, **format_kwargs)

        return content, selected

    def record_success(self, name: str, version: str) -> None:
        """성공 기록"""
        if name in self.versions and version in self.versions[name]:
            self.versions[name][version].success_count += 1
            self._save_metrics()

    def get_metrics(self, name: str) -> dict[str, Any]:
        """버전별 메트릭 조회"""
        if name not in self.versions:
            return {}

        return {
            version: {
                "usage_count": pv.usage_count,
                "success_count": pv.success_count,
                "success_rate": pv.success_count / max(1, pv.usage_count),
            }
            for version, pv in self.versions[name].items()
        }

    def _load_metrics(self) -> None:
        """메트릭 파일 로드"""
        if self._metrics_file.exists():
            try:
                with open(self._metrics_file) as f:
                    data = json.load(f)

                    # 메트릭 복원
                    for name, version_metrics in data.items():
                        if name not in self.versions:
                            self.versions[name] = {}

                        for version, metrics in version_metrics.items():
                            if version not in self.versions[name]:
                                # 메트릭만 있고 실제 프롬프트가 없는 경우는 스킵
                                continue

                            pv = self.versions[name][version]
                            pv.usage_count = metrics.get("usage_count", 0)
                            pv.success_count = metrics.get("success_count", 0)

            except Exception as e:
                logger.warning(f"Failed to load metrics: {e}")

    def _save_metrics(self) -> None:
        """메트릭 파일 저장"""
        try:
            metrics = {}
            for name, versions in self.versions.items():
                metrics[name] = {
                    version: {"usage_count": pv.usage_count, "success_count": pv.success_count}
                    for version, pv in versions.items()
                }

            with open(self._metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")


# 싱글톤
_manager_instance: PromptVersionManager | None = None


def get_prompt_manager() -> PromptVersionManager:
    """PromptVersionManager 싱글톤"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = PromptVersionManager()
    return _manager_instance

"""
Oracle Cloud 프록시 매니저
무료 프록시 풀을 관리하고 로테이션합니다.

Usage:
    from src.tools.proxy_manager import get_proxy_manager

    proxy_manager = get_proxy_manager()
    proxy = proxy_manager.get_proxy()

    # Playwright에서 사용
    browser = await playwright.chromium.launch(proxy=proxy.playwright_config)

    # requests에서 사용
    response = requests.get(url, proxies=proxy.requests_config)
"""

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class ProxyConfig:
    """프록시 서버 설정"""

    name: str
    server: str
    username: str
    password: str
    region: str
    enabled: bool = True
    last_used: datetime | None = None
    success_count: int = 0
    failure_count: int = 0

    @property
    def playwright_config(self) -> dict:
        """Playwright용 프록시 설정 반환"""
        return {"server": self.server, "username": self.username, "password": self.password}

    @property
    def requests_config(self) -> dict:
        """requests 라이브러리용 프록시 설정 반환"""
        auth = f"{self.username}:{self.password}"
        server = self.server.replace("http://", "")
        return {"http": f"http://{auth}@{server}", "https": f"http://{auth}@{server}"}

    @property
    def aiohttp_config(self) -> str:
        """aiohttp용 프록시 URL 반환"""
        auth = f"{self.username}:{self.password}"
        server = self.server.replace("http://", "")
        return f"http://{auth}@{server}"

    @property
    def success_rate(self) -> float:
        """성공률 계산"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total


@dataclass
class ProxyPoolConfig:
    """프록시 풀 전체 설정"""

    proxy_pool: list[dict] = field(default_factory=list)
    rotation_strategy: str = "random"  # random, round-robin, weighted
    retry_on_failure: bool = True
    max_retries: int = 3
    health_check_interval: int = 300  # 5분
    cooldown_after_failure: int = 60  # 실패 후 60초 대기


class ProxyManager:
    """
    Oracle Cloud 프록시 풀 매니저

    Features:
    - 프록시 로테이션 (random, round-robin, weighted)
    - 실패한 프록시 자동 비활성화 및 쿨다운
    - 헬스 체크
    - 성공률 기반 가중치 로테이션
    """

    def __init__(self, config_path: str = "config/proxy_config.json"):
        self.config_path = Path(config_path)
        self.proxies: list[ProxyConfig] = []
        self.current_index = 0
        self.config = ProxyPoolConfig()
        self._cooldown_until: dict[str, datetime] = {}
        self._load_config()

    def _load_config(self):
        """설정 파일 로드"""
        if not self.config_path.exists():
            logger.warning(f"프록시 설정 파일 없음: {self.config_path}")
            logger.info("프록시 없이 직접 연결 모드로 동작합니다.")
            return

        try:
            with open(self.config_path, encoding="utf-8") as f:
                data = json.load(f)

            self.config = ProxyPoolConfig(
                proxy_pool=data.get("proxy_pool", []),
                rotation_strategy=data.get("rotation_strategy", "random"),
                retry_on_failure=data.get("retry_on_failure", True),
                max_retries=data.get("max_retries", 3),
                health_check_interval=data.get("health_check_interval", 300),
                cooldown_after_failure=data.get("cooldown_after_failure", 60),
            )

            self.proxies = [
                ProxyConfig(
                    name=proxy.get("name", f"proxy-{i}"),
                    server=proxy["server"],
                    username=proxy["username"],
                    password=proxy["password"],
                    region=proxy.get("region", "unknown"),
                    enabled=proxy.get("enabled", True),
                )
                for i, proxy in enumerate(self.config.proxy_pool)
            ]

            logger.info(
                f"프록시 {len(self.proxies)}개 로드됨 " f"(전략: {self.config.rotation_strategy})"
            )

        except json.JSONDecodeError as e:
            logger.error(f"프록시 설정 파일 파싱 오류: {e}")
        except Exception as e:
            logger.error(f"프록시 설정 로드 실패: {e}")

    def _is_in_cooldown(self, proxy_name: str) -> bool:
        """프록시가 쿨다운 중인지 확인"""
        if proxy_name not in self._cooldown_until:
            return False
        return datetime.now() < self._cooldown_until[proxy_name]

    def _get_available_proxies(self) -> list[ProxyConfig]:
        """사용 가능한 프록시 목록 반환"""
        return [p for p in self.proxies if p.enabled and not self._is_in_cooldown(p.name)]

    def get_proxy(self) -> ProxyConfig | None:
        """
        사용 가능한 프록시 반환

        Returns:
            ProxyConfig or None (프록시 없으면 None)
        """
        available = self._get_available_proxies()

        if not available:
            logger.warning("사용 가능한 프록시 없음 - 직접 연결 사용")
            return None

        proxy = None

        if self.config.rotation_strategy == "random":
            proxy = random.choice(available)

        elif self.config.rotation_strategy == "round-robin":
            proxy = available[self.current_index % len(available)]
            self.current_index += 1

        elif self.config.rotation_strategy == "weighted":
            # 성공률 기반 가중치 선택
            weights = [p.success_rate for p in available]
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                proxy = random.choices(available, weights=weights, k=1)[0]
            else:
                proxy = random.choice(available)

        if proxy:
            proxy.last_used = datetime.now()
            logger.debug(f"프록시 선택: {proxy.name} ({proxy.region})")

        return proxy

    def report_failure(self, proxy_name: str):
        """
        프록시 실패 보고
        - 실패 카운트 증가
        - max_retries 초과 시 비활성화
        - 쿨다운 적용
        """
        for proxy in self.proxies:
            if proxy.name == proxy_name:
                proxy.failure_count += 1

                # 쿨다운 적용
                cooldown = timedelta(seconds=self.config.cooldown_after_failure)
                self._cooldown_until[proxy_name] = datetime.now() + cooldown

                # 연속 실패 시 비활성화
                if proxy.failure_count >= self.config.max_retries:
                    proxy.enabled = False
                    logger.warning(f"프록시 비활성화 (실패 {proxy.failure_count}회): {proxy_name}")
                else:
                    logger.info(
                        f"프록시 실패 ({proxy.failure_count}/{self.config.max_retries}): {proxy_name}"
                    )

                break

    def report_success(self, proxy_name: str):
        """프록시 성공 보고"""
        for proxy in self.proxies:
            if proxy.name == proxy_name:
                proxy.success_count += 1
                # 성공 시 실패 카운트 리셋
                proxy.failure_count = 0
                # 쿨다운 해제
                self._cooldown_until.pop(proxy_name, None)
                logger.debug(f"프록시 성공: {proxy_name}")
                break

    def reset_proxy(self, proxy_name: str):
        """프록시 상태 리셋 (다시 활성화)"""
        for proxy in self.proxies:
            if proxy.name == proxy_name:
                proxy.enabled = True
                proxy.failure_count = 0
                self._cooldown_until.pop(proxy_name, None)
                logger.info(f"프록시 리셋: {proxy_name}")
                break

    def reset_all(self):
        """모든 프록시 상태 리셋"""
        for proxy in self.proxies:
            proxy.enabled = True
            proxy.failure_count = 0
        self._cooldown_until.clear()
        logger.info("모든 프록시 리셋됨")

    async def health_check(self, proxy: ProxyConfig) -> bool:
        """
        프록시 헬스 체크

        Returns:
            True if healthy, False otherwise
        """
        test_url = "https://httpbin.org/ip"

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(test_url, proxy=proxy.aiohttp_config) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"헬스 체크 성공: {proxy.name} -> {data.get('origin')}")
                        return True
                    else:
                        logger.warning(f"헬스 체크 실패 (HTTP {response.status}): {proxy.name}")
                        return False

        except Exception as e:
            logger.warning(f"헬스 체크 실패 ({type(e).__name__}): {proxy.name}")
            return False

    async def health_check_all(self) -> dict[str, bool]:
        """모든 프록시 헬스 체크"""
        results = {}

        tasks = [self.health_check(proxy) for proxy in self.proxies]

        check_results = await asyncio.gather(*tasks, return_exceptions=True)

        for proxy, result in zip(self.proxies, check_results, strict=False):
            if isinstance(result, Exception):
                results[proxy.name] = False
                self.report_failure(proxy.name)
            elif result:
                results[proxy.name] = True
                self.report_success(proxy.name)
            else:
                results[proxy.name] = False
                self.report_failure(proxy.name)

        return results

    def get_stats(self) -> dict:
        """프록시 풀 통계"""
        available = self._get_available_proxies()

        return {
            "total": len(self.proxies),
            "active": len([p for p in self.proxies if p.enabled]),
            "available": len(available),
            "in_cooldown": len(
                [p for p in self.proxies if p.enabled and self._is_in_cooldown(p.name)]
            ),
            "rotation_strategy": self.config.rotation_strategy,
            "proxies": [
                {
                    "name": p.name,
                    "region": p.region,
                    "enabled": p.enabled,
                    "success_count": p.success_count,
                    "failure_count": p.failure_count,
                    "success_rate": f"{p.success_rate:.1%}",
                    "in_cooldown": self._is_in_cooldown(p.name),
                }
                for p in self.proxies
            ],
        }

    def has_proxies(self) -> bool:
        """프록시가 설정되어 있는지 확인"""
        return len(self.proxies) > 0


# 싱글톤 인스턴스
_proxy_manager: ProxyManager | None = None


def get_proxy_manager(config_path: str = "config/proxy_config.json") -> ProxyManager:
    """전역 프록시 매니저 인스턴스 반환"""
    global _proxy_manager
    if _proxy_manager is None:
        _proxy_manager = ProxyManager(config_path)
    return _proxy_manager


def reset_proxy_manager():
    """프록시 매니저 리셋 (테스트용)"""
    global _proxy_manager
    _proxy_manager = None

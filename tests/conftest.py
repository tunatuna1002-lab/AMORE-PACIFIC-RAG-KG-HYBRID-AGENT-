import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from dotenv import load_dotenv


def pytest_configure(config):
    """테스트 환경 격리: 실제 .env는 로드하지 않고 .env.test만 로드한다."""
    project_root = Path(__file__).parent.parent
    env_file = os.environ.get("ENV_FILE", ".env.test")
    env_path = project_root / env_file
    if env_path.exists():
        load_dotenv(env_path, override=True)
    # 실 데이터 파일 의존 차단: KG 영속 경로를 임시 디렉토리로
    os.environ.setdefault(
        "KG_PERSIST_PATH",
        os.path.join(tempfile.gettempdir(), "amore_test_kg", "knowledge_graph.json"),
    )
    os.environ.pop("RAILWAY_ENVIRONMENT", None)
    # OWL 추론은 기본 python 엔진으로 고정: Pellet/HermiT는 Java 런타임에 의존해
    # 환경마다 결과가 달라진다(특성화 핀이 흔들림). Java 엔진을 검증하는 테스트는
    # materialize(..., reasoner=...)로 직접 지정한다.
    os.environ.setdefault("AMORE_OWL_REASONER", "python")
    # LiteLLM 은 import 시점에 원격 모델 비용표를 받아오려 하고, 막힌 환경에서는
    # 3회 재시도(약 9초)를 기다린 뒤 로컬 백업으로 폴백한다. 단위 테스트는 처음부터
    # 로컬 백업을 쓰게 한다.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


# ---------------------------------------------------------------------------
# 네트워크 차단 (Phase 5)
# ---------------------------------------------------------------------------
# 단위 테스트는 오프라인이어야 한다. 실제로 RSS/LLM 엔드포인트를 때리던 테스트가
# 테스트당 ~9초씩 프록시 타임아웃을 기다리고 있었고, 그 시간만큼 "네트워크가 되는
# 환경에서는 다른 결과가 나올 수 있는" 테스트였다. 여기서 아웃바운드 소켓을 막아
# 즉시 실패시키고, 진짜로 네트워크가 필요한 테스트는 ``@pytest.mark.allow_network``
# 로 예외 처리한다. 루프백은 허용한다(로컬 서버·TestClient 호환).

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", ""})


class NetworkBlockedError(RuntimeError):
    """단위 테스트가 외부 네트워크에 접속하려 했을 때."""


def _is_loopback(address) -> bool:
    if not isinstance(address, tuple) or not address:
        return True  # AF_UNIX 등 - 로컬
    host = address[0]
    return isinstance(host, str) and host in _LOOPBACK_HOSTS


@pytest.fixture(autouse=True)
def _block_outbound_network(request, monkeypatch):
    """외부 소켓 연결 차단 (``allow_network`` 마커로 해제)."""
    if request.node.get_closest_marker("allow_network"):
        return

    import socket

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def guard(original):
        def wrapper(self, address, *args, **kwargs):
            if not _is_loopback(address):
                raise NetworkBlockedError(
                    f"outbound network blocked in unit tests: {address!r}. "
                    "Use a fake collaborator, or mark the test @pytest.mark.allow_network."
                )
            return original(self, address, *args, **kwargs)

        return wrapper

    monkeypatch.setattr(socket.socket, "connect", guard(real_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guard(real_connect_ex))


@pytest.fixture(autouse=True)
def _isolate_singletons():
    """모듈 전역 싱글턴을 테스트마다 초기화한다 (테스트 간 상태 누수 방지)."""
    yield
    try:
        from src.infrastructure.container import Container

        Container.reset()
    except Exception:
        pass
    try:
        from src.core.brain import reset_brain

        reset_brain()
    except Exception:
        pass
    try:
        from src.infrastructure.feature_flags import FeatureFlags

        FeatureFlags.reset_instance()
    except Exception:
        pass
    try:
        from src.core.state_manager import reset_state_manager

        reset_state_manager()
    except Exception:
        pass
    try:
        from src.api import dependencies as _deps

        _deps.conversation_memory.clear()
        _deps.session_last_activity.clear()
    except Exception:
        pass


@pytest.fixture
def results():
    """테스트 결과 객체 (인라인 정의, 외부 의존 제거)"""

    class _TestResult:
        def __init__(self):
            self.passed = 0
            self.failed = 0
            self.errors = []

        def record_pass(self, test_name: str):
            self.passed += 1
            print(f"  ✅ {test_name}")

        def record_fail(self, test_name: str, error: str):
            self.failed += 1
            self.errors.append(f"{test_name}: {error}")
            print(f"  ❌ {test_name}: {error}")

        def summary(self) -> str:
            total = self.passed + self.failed
            return f"\n{'=' * 60}\n총 {total}개 테스트: ✅ {self.passed} 성공, ❌ {self.failed} 실패\n{'=' * 60}"

    return _TestResult()


@pytest_asyncio.fixture
async def retriever():
    """Mock retriever - 실제 ChromaDB/디스크 의존성 제거"""
    mock = MagicMock()
    mock.initialize = AsyncMock()
    mock.retrieve = AsyncMock(return_value=[])
    mock.get_relevant_documents = AsyncMock(return_value=[])
    return mock

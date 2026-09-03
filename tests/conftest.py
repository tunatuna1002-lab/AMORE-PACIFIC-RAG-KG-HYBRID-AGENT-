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

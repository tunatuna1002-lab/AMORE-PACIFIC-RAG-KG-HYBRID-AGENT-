import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from dotenv import load_dotenv


def pytest_configure(config):
    """테스트 시작 전 환경 설정 로드"""
    project_root = Path(__file__).parent.parent

    main_env_path = project_root / ".env"
    if main_env_path.exists():
        load_dotenv(main_env_path, override=False)
        print(f"\n[conftest] Loaded base environment from: {main_env_path}")

    env_file = os.environ.get("ENV_FILE", ".env.test")
    env_path = project_root / env_file

    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"[conftest] Applied test overrides from: {env_path}")
    else:
        print(f"[conftest] No {env_file} found, using base environment only")


@pytest.fixture
def results():
    """테스트 결과 객체 (인라인 정의, 외부 의존 제거)"""

    class TestResult:
        def __init__(self):
            self.passed = 0
            self.failed = 0
            self.errors = []

    return TestResult()


@pytest_asyncio.fixture
async def retriever():
    """Mock retriever - 실제 ChromaDB/디스크 의존성 제거"""
    mock = MagicMock()
    mock.initialize = AsyncMock()
    mock.retrieve = AsyncMock(return_value=[])
    mock.get_relevant_documents = AsyncMock(return_value=[])
    return mock
